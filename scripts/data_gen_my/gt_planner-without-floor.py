from typing import Any, Dict, List, Tuple

from pydantic import constr

from habitat_llm.planner.planner import Planner
from habitat_llm.world_model import Room
from habitat_llm.world_model import Floor


class GroundTruthPlanner(Planner):
    """
    GT Planner 最终修正版 (Ver 4.0):
    1. [Fix] 修复 AttributeError (正确遍历 PARTNR 自定义 Graph 结构)
    2. [Feat] 增加 Handle -> NodeName 映射
    3. [Feat] 保持 '忙碌等待' 机制
    """

    def __init__(self, plan_config: Any, env_interface: Any):
        super().__init__(plan_config, env_interface)
        self.action_queue = []
        self.current_episode_id = None
        self.current_action_tuple = None
        self.handle_to_name_map = {}

    def reset(self):
        self.action_queue = []
        self.current_episode_id = None
        self.current_action_tuple = None
        self.handle_to_name_map = {}

    def build_handle_map(self):
        """
        构建 Sim Handle 到 WorldGraph Node Name 的映射表。
        [核心修复]: 适应 PARTNR 的 Graph 结构 (dict of nodes)，而不是 NetworkX。
        """
        self.handle_to_name_map = {}
        # full_world_graph 是 WorldGraph 实例
        # WorldGraph.graph 是一个 dict {NodeEntity: {Neighbor: Edge}}
        world_graph = self.env_interface.full_world_graph

        # 直接遍历字典的键，键就是 Entity 对象
        for node in world_graph.graph:
            # 检查节点是否有 sim_handle 属性
            if hasattr(node, "sim_handle"):
                self.handle_to_name_map[node.sim_handle] = node.name

        print(f"[GT Planner] Built handle map with {len(self.handle_to_name_map)} entries.")

    def get_node_name(self, handle: str) -> str:
        if not handle: return "None"
        node = self.env_interface.full_world_graph.get_node_from_sim_handle(handle)
        return node.name if node else "None"

    def get_next_action(
            self,
            instruction: str,
            observations: Dict[str, Any],
            world_graph: Dict[int, Any]
    ) -> Tuple[Dict[int, Any], Dict[str, Any], bool]:

        # 1. 获取 Current Episode
        try:
            current_ep = self.env_interface.env.env.env._env.current_episode
        except AttributeError:
            try:
                current_ep = self.env_interface.env.env._env.current_episode
            except AttributeError:
                raise AttributeError("Could not find current_episode in env_interface stack")

        # 2. 检查是否切换了 Episode
        if self.current_episode_id != current_ep.episode_id:
            self.current_episode_id = current_ep.episode_id
            # 重新构建映射表
            self.build_handle_map()
            self.generate_plan_from_gt(current_ep)
            self.current_action_tuple = None

            # 3. 状态机逻辑
        if self.current_action_tuple is None:
            if not self.action_queue:
                return {}, {"thought": "All GT actions executed."}, True

            self.current_action_tuple = self.action_queue.pop(0)
            print(f"[GT Planner] Starting NEW action: {self.current_action_tuple}")

        # 4. 构建高级动作
        high_level_actions = {
            0: self.current_action_tuple,
            1: ("Wait", None, None)
        }

        # 5. 转换为低级控制信号
        low_level_actions, responses = self.process_high_level_actions(
            high_level_actions,
            observations
        )

        # 6. 检查完成状态
        agent_response = responses.get(0, "")

        if agent_response:
            print(f"[GT Planner] Action Finished/Stopped. Result: {agent_response}")
            self.current_action_tuple = None

        thought = f"Executing: {high_level_actions[0][0]} | Status: {'Running' if not agent_response else 'Done'}"
        info = {"thought": thought}
        if responses:
            info.update(responses)

        return low_level_actions, info, False

    def _requires_faucet(self, obj_handle: str) -> bool:
        """
        [智能判断] 检查该物体是否需要水源才能清洁。
        逻辑源自: oracle_clean_skills.py
        """
        sim = self.env_interface.sim
        world_graph = self.env_interface.full_world_graph

        # 1. 获取物体节点以查类别
        obj_node = world_graph.get_node_from_sim_handle(obj_handle)
        if not obj_node:
            return False

        # 获取语义类别 (通常存储在 properties['category'] 或直接是 node.category)
        # 优先从 properties 获取，如果没有则尝试属性
        obj_category = obj_node.properties.get("category", getattr(obj_node, "category", "unknown"))

        # 2. 查找 sim 中的 ObjectIsClean 状态定义
        if not hasattr(sim, "object_state_machine"):
            return False

        clean_states = [s for s in sim.object_state_machine.active_states if s.name=="is_clean"]
        if not clean_states:
            return False

        clean_state_def = clean_states[0]

        # 3. 检查类别是否在强制水源列表中
        if hasattr(clean_state_def, "requires_faucet_semantic_classes"):
            return obj_category in clean_state_def.requires_faucet_semantic_classes

        return False

    def find_nearest_receptacle_by_category(self, category_keywords: List[str]) -> str:
        """
        查找场景中包含特定关键词的容器（如 sink, basin）。
        """
        world_graph = self.env_interface.full_world_graph
        all_receptacles = world_graph.get_all_receptacles()

        for rec in all_receptacles:
            name_lower = rec.name.lower()
            if any(k in name_lower for k in category_keywords):
                return rec.name
        return None

    def get_valid_receptacle_in_room(self, room_id: str) -> str:
        """
        [is_in_room 专用] 在指定房间内查找一个合法的放置表面（Table, Counter等）。
        """
        world_graph = self.env_interface.full_world_graph

        # 1. 查找房间节点 (模糊匹配以应对 'kitchen' vs 'kitchen_0' 的差异)
        room_node = None
        for r in world_graph.get_all_rooms():
            if room_id in r.name:
                room_node = r
                break

        if not room_node:
            print(f"  [Warn] Room '{room_id}' not found in WorldGraph.")
            return None

        # 2. 获取房间内的家具
        furnitures = world_graph.get_furniture_in_room(room_node)
        if not furnitures:
            return None

        # 3. 筛选策略：优先选桌子、柜台
        preferred = []
        fallback = []
        for f in furnitures:
            name_lower = f.name.lower()
            if any(x in name_lower for x in ['table', 'counter', 'desk', 'shelf']):
                preferred.append(f)
            else:
                fallback.append(f)

        candidates = preferred if preferred else fallback
        return candidates[0].name if candidates else None


    def check_if_prop_satisfied(self, function_name: str, obj_handle: str) -> bool:
        """
        [状态预检] 检查 proposition 是否已经满足，避免生成多余动作。
        """
        world_graph = self.env_interface.full_world_graph
        obj_node = world_graph.get_node_from_sim_handle(obj_handle)
        if not obj_node:
            return False

        # 状态存储在 properties["states"] 字典中
        current_states = obj_node.properties.get("states", {})

        if function_name == "is_clean":
            return current_states.get("is_clean", False)
        elif function_name == "is_filled":
            return current_states.get("is_filled", False)
        elif function_name == "is_powered_on":
            return current_states.get("is_powered_on", False)
        elif function_name == "is_powered_off":
            return not current_states.get("is_powered_on", False)

        return False

    def generate_plan_from_gt(self, episode):
        """
        核心逻辑：根据 Evaluation Propositions 生成动作序列。
        此方法应在 episode 开始时被调用一次。
        """
        print(f"[GT Planner] Generating plan for Episode {episode.episode_id}...")

        if not hasattr(episode, "evaluation_propositions"):
            print(f"WARNING: Episode {episode.episode_id} has no evaluation_propositions!")
            return

        props = episode.evaluation_propositions
        self.action_queue = []

        # [核心] 全局已分配对象集合，防止多指令冲突
        used_objects = set()

        for prop in props:
            # 兼容字典或对象属性访问
            fname = prop.function_name if hasattr(prop, "function_name") else prop["function_name"]
            args = prop.args if hasattr(prop, "args") else prop["args"]

            if fname == "is_next_to": # is_next_to通常作为is_on_floor/on_top的约束条件出现，不作为单独子任务
                continue

            # [核心修复 1] 严格解析目标数量
            obj_handles_list = args.get("object_handles", [])
            number_required = args.get("number", len(obj_handles_list))

            # 预解析目标位置参数
            recep_handles = args.get("receptacle_handles", [])
            room_ids = args.get("room_ids", [])

            target_recep_name = "None"
            if fname in ["is_on_top", "is_inside"] and recep_handles:
                target_recep_name = self.get_node_name(recep_handles[0])
            elif fname == "is_in_room" and room_ids:
                # [核心修复 2] 解析房间内的具体家具
                target_room = room_ids[0]
                found_recep = self.get_valid_receptacle_in_room(target_room)
                if found_recep:
                    target_recep_name = found_recep
                    print(f"  [Resolved] Room '{target_room}' -> Receptacle '{target_recep_name}'")
                else:
                    print(f"  [Warn] Could not resolve receptacle in room '{target_room}'")
                    # 如果找不到放置点，该任务大概率无法执行，跳过
                    continue

            actions_generated_for_prop = 0

            for obj_handle in obj_handles_list:
                # [核心修复 1] 数量检查
                if actions_generated_for_prop >= number_required:
                    break

                # 跳过已被占用的对象
                if obj_handle in used_objects:
                    continue

                obj_name = self.get_node_name(obj_handle)
                if obj_name == "None": continue

                # 获取 obj_node 用于后续判断 (这就是你问的 obj_node)
                obj_node = self.env_interface.full_world_graph.get_node_from_sim_handle(obj_handle)

                # [预检] 如果状态已满足，跳过 (可选，但建议开启以减少冗余动作)
                if self.check_if_prop_satisfied(fname, obj_handle):
                    print(f"  [Skip] {obj_name} already satisfies {fname}")
                    # 即使跳过，也算作"已处理"一个名额
                    # 除非是空间位置类任务，通常建议强制 Rearrange 以确保视频效果
                    if fname not in ["is_on_top", "is_inside", "is_in_room", "is_on_floor"]:
                        print(f"    [Skip] {obj_name} already satisfies {fname}")
                        actions_generated_for_prop += 1
                        used_objects.add(obj_handle)
                        continue

                current_task_actions = []

                if fname == "is_clean":
                    # [核心修复 3] 智能水源判断
                    needs_sink = self._requires_faucet(obj_handle)

                    if needs_sink:
                        # 需要水源：搬运 -> 清洗
                        sink_name = self.find_nearest_receptacle_by_category(['sink', 'basin', 'bathtub'])
                        if sink_name:
                            print(f"  [Plan] {obj_name} requires faucet. Moving to {sink_name}.")
                            # Rearrange 格式: "Obj, Relation, Recep, Handle1, Handle2"
                            action_args = f"{obj_name}, inside, {sink_name}, None, None"
                            current_task_actions.append(("Rearrange", action_args, None))
                            current_task_actions.append(("OracleCleanSkill", f"{obj_name}", None))
                        else:
                            print(f"  [Warn] No sink found for {obj_name}. Trying in-place.")
                            current_task_actions.append(("OracleCleanSkill", f"{obj_name}", None))
                    else:
                        # 不需要水源 (如桌子)：直接原地清洗
                        print(f"  [Plan] {obj_name} does not require faucet. Cleaning in-place.")
                        current_task_actions.append(("OracleCleanSkill", f"{obj_name}", None))

                elif fname == "is_filled":
                    # Fill 几乎总是需要水源
                    sink_name = self.find_nearest_receptacle_by_category(['sink', 'basin', 'bathtub'])
                    if sink_name:
                        action_args = f"{obj_name}, inside, {sink_name}, None, None"
                        current_task_actions.append(("Rearrange", action_args, None))
                        current_task_actions.append(("OracleFillSkill", f"{obj_name}", None))
                    else:
                        current_task_actions.append(("OracleFillSkill", f"{obj_name}", None))

                elif fname in ["is_powered_on", "is_powered_off"]:
                    skill = "OraclePowerOnSkill" if fname == "is_powered_on" else "OraclePowerOffSkill"
                    current_task_actions.append((skill, f"{obj_name}", None))

                elif fname in ["is_on_top", "is_inside", "is_in_room"]:
                    if target_recep_name and target_recep_name != "None":
                        relation = "in" if fname == "is_inside" else "on"
                        action_args = f"{obj_name}, {relation}, {target_recep_name}, None, None"
                        current_task_actions.append(("Rearrange", action_args, None))
                    else:
                        print(f"  [Skip] No valid target for {fname} on {obj_name}")
                        continue


                # 将生成的动作加入总队列
                if current_task_actions:
                    self.action_queue.extend(current_task_actions)
                    used_objects.add(obj_handle)
                    actions_generated_for_prop += 1

        print(f"[GT Planner] Plan generated: {len(self.action_queue)} steps.")
        for i in range(len(self.action_queue)):
            print(f"{i+1}. {self.action_queue[i]}")