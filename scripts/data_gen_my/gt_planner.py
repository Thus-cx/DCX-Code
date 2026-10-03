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
        self.ERROR = []

    def reset(self):
        self.action_queue = []
        self.current_episode_id = None
        self.current_action_tuple = None
        self.handle_to_name_map = {}

    # def build_handle_map(self):
    #     """
    #     构建 Sim Handle 到 WorldGraph Node Name 的映射表。
    #     [核心修复]: 适应 PARTNR 的 Graph 结构 (dict of nodes)，而不是 NetworkX。
    #     """
    #     self.handle_to_name_map = {}
    #     # full_world_graph 是 WorldGraph 实例
    #     # WorldGraph.graph 是一个 dict {NodeEntity: {Neighbor: Edge}}
    #     world_graph = self.env_interface.full_world_graph
    #
    #     # 直接遍历字典的键，键就是 Entity 对象
    #     for node in world_graph.graph:
    #         # 检查节点是否有 sim_handle 属性
    #         if hasattr(node, "sim_handle"):
    #             self.handle_to_name_map[node.sim_handle] = node.name
    #
    #     print(f"[GT Planner] Built handle map with {len(self.handle_to_name_map)} entries.")

    def build_handle_map(self):
        self.handle_to_name_map = {}
        # 正确遍历 WorldGraph 的所有节点（Key 是 Node 对象）
        for node in self.env_interface.full_world_graph.graph.keys():
            # 确保是物体节点且具有 sim_handle
            if hasattr(node, "sim_handle") and node.sim_handle:
                # 关键：访问 PARTNR 实体定义的 category 属性
                # 如果没有 category 属性，再尝试从 handle 提取
                raw_category = getattr(node, "type", "unknown")

                if raw_category == "unknown":
                    # 备用方案：通过底层元数据接口获取（如果 node 属性没初始化完全）
                    # 这里的逻辑应确保能拿到 'pan' 或 'kettle' 字符串
                    pass

                self.handle_to_name_map[node.sim_handle] = str(raw_category).lower()

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
            if ("Successful execution" not in agent_response) or "Unexpected failure" in agent_response:
                self.ERROR.append(agent_response)
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
        obj_category = obj_node.properties.get("type", getattr(obj_node, "type", "unknown"))
        # obj_category = self.env_interface.sim.metadata_interface.get_object_category(obj_handle)
        # for node in self.env_interface.full_world_graph.graph.keys():
        #     if hasattr(node, "sim_handle") and node.sim_handle:
        #         obj_category = getattr(node, "category", "unknown").lower()
        #         self.handle_to_name_map[obj_handle] = obj_category
        print(f"obj_category: {obj_category}")

        # 2. 查找 sim 中的 ObjectIsClean 状态定义
        if not hasattr(sim, "object_state_machine"):
            return False

        clean_states = [s for s in sim.object_state_machine.active_states if s.name == "is_clean"]
        if not clean_states:
            return False

        clean_state_def = clean_states[0]

        # 3. 检查类别是否在强制水源列表中
        if hasattr(clean_state_def, "requires_faucet_semantic_classes"):
            print(f"Clean Require Faucet Semantic Classes:\n{clean_state_def.requires_faucet_semantic_classes}")
            return obj_category in clean_state_def.requires_faucet_semantic_classes

        return False

    def find_water_source(self) -> str:
        """
        查找场景中包含特定关键词的容器（如 sink, basin）。
        """
        water_source_names = ['sink', 'faucet', 'basin']
        # world_graph = self.env_interface.full_world_graph
        # all_receptacle_nodes = world_graph.get_all_receptacles()
        # from habitat_llm.utils.sim import get_faucet_points
        # all_faucet_nodes = get_faucet_points(self.env_interface.sim).keys()
        all_furnitures = self.env_interface.full_world_graph.get_all_furnitures()
        faucet_furnitures = [fur for fur in all_furnitures
                             if "components" in fur.properties
                             and "faucet" in fur.properties["components"]]
        print(f"Find Faucet: {faucet_furnitures}")
        if len(faucet_furnitures) > 0:
            return faucet_furnitures[0].name
        else:
            receptacle_nodes = self.env_interface.full_world_graph.get_all_receptacles()
            receptacle_names = []
            for receptacle_node in receptacle_nodes:
                receptacle_names.append(receptacle_node.name)
            print(f"receptacle names:\n{receptacle_names}")
            for water_source_name in water_source_names:
                for receptacle_name in receptacle_names:
                    if water_source_name in receptacle_name:
                        return receptacle_name
        return None

    def get_valid_receptacle_in_room(self, room_id: str) -> str:
        """
        [is_in_room 专用] 在指定房间内查找一个合法的放置表面（Table, Counter等）。
        """
        world_graph = self.env_interface.full_world_graph

        # 1. 查找房间节点 (模糊匹配以应对 'kitchen' vs 'kitchen_0' 的差异)
        room_node = None
        for r in world_graph.get_all_rooms():
            if room_id in r.name or room_id.replace(" ", "_") in r.name:
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

    def reorder_props(self, props):
        state_changes_fnames = ["is_clean", "is_filled", "is_powered_on", "is_powered_off"]
        pos_change_fnames = ["is_on_top", "is_inside", "is_on_floor", "is_in_room"]
        reordered_props = []
        for i in range(len(props)):
            current_prop = props[i]
            fname = current_prop.function_name if hasattr(current_prop, "function_name") else current_prop[
                "function_name"]
            args = current_prop.args if hasattr(current_prop, "args") else current_prop["args"]
            obj_handles_list = args.get("object_handles", [])
            number_required = args.get("number", len(obj_handles_list))
            if fname in pos_change_fnames:
                final_obj_handles_list = obj_handles_list[:number_required]
                for obj in final_obj_handles_list:
                    for j in range(len(props)):
                        if j == i:
                            continue
                        other_prop = props[j]
                        other_prop_fname = other_prop.function_name if hasattr(other_prop, "function_name") else \
                        other_prop["function_name"]
                        other_prop_args = other_prop.args if hasattr(other_prop, "args") else other_prop["args"]
                        other_prop_obj = other_prop_args.get("object_handles", [])
                        other_number_required = other_prop_args.get("number", len(other_prop_obj))
                        other_final_obj_list = other_prop_obj[:other_number_required]
                        if obj in other_final_obj_list:
                            if other_prop_fname in state_changes_fnames:
                                if other_prop not in reordered_props:
                                    reordered_props.append(other_prop)
                if current_prop not in reordered_props:
                    reordered_props.append(current_prop)
            else:
                if current_prop not in reordered_props:
                    reordered_props.append(current_prop)
        return reordered_props

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
        reordered_props = self.reorder_props(props)

        # [核心] 全局已分配对象集合，防止多指令冲突
        used_objects = set()

        for prop in reordered_props:
            # 兼容字典或对象属性访问
            fname = prop.function_name if hasattr(prop, "function_name") else prop["function_name"]
            args = prop.args if hasattr(prop, "args") else prop["args"]

            if fname == "is_next_to":  # is_next_to通常作为is_on_floor/on_top的约束条件出现，不作为单独子任务
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
                target_recep_handle = recep_handles[0]
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
                # if obj_handle in used_objects and fname in ["is_on_top", "is_inside", "is_in_room", "is_on_floor"]:
                #     continue

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
                        sink_name = self.find_water_source()
                        if sink_name:
                            print(f"  [Plan] {obj_name} requires faucet. Moving to {sink_name}.")
                            # Rearrange 格式: "Obj, Relation, Recep, Handle1, Handle2"
                            action_args = f"{obj_name}, on, {sink_name}, None, None"
                            current_task_actions.append(("Rearrange", action_args, None))
                            current_task_actions.append(("Wait", f"SET_CLEAN:{obj_handle}", None))
                            # current_task_actions.append(("CleanInPlace", f"{obj_name}", None))
                        else:
                            print(f"  [Warn] No sink found for {obj_name}. Trying in-place.")
                            current_task_actions.append(("Clean", f"{obj_name}", None))
                    else:
                        # 不需要水源 (如桌子)：直接原地清洗
                        print(f"  [Plan] {obj_name} does not require faucet. Cleaning in-place.")
                        current_task_actions.append(("Clean", f"{obj_name}", None))

                elif fname == "is_filled":
                    # Fill 几乎总是需要水源
                    sink_name = self.find_water_source()
                    if sink_name:
                        action_args = f"{obj_name}, on, {sink_name}, None, None"
                        current_task_actions.append(("Rearrange", action_args, None))
                        current_task_actions.append(("Wait", f"SET_FILL:{obj_handle}", None))
                    else:
                        print(f"    [Warn] No sink found for {obj_name}.")
                        current_task_actions.append(("Fill", f"{obj_name}", None))

                elif fname in ["is_powered_on", "is_powered_off"]:
                    skill = "PowerOn" if fname == "is_powered_on" else "PowerOff"
                    current_task_actions.append((skill, f"{obj_name}", None))

                elif fname in ["is_on_top", "is_in_room"]:
                    if target_recep_name and target_recep_name != "None":
                        relation = "in" if fname == "is_inside" else "on"
                        action_args = f"{obj_name}, {relation}, {target_recep_name}, None, None"
                        current_task_actions.append(("Rearrange", action_args, None))
                    else:
                        print(f"  [Skip] No valid target for {fname} on {obj_name}")
                        continue
                elif fname == "is_inside":
                    if target_recep_name and target_recep_name != "None":
                        target_recep_node = self.env_interface.full_world_graph.get_node_from_sim_handle(
                            target_recep_handle)
                        if hasattr(target_recep_node.properties["states"], "is_open"):
                            if not target_recep_node.properties["states"]["is_open"]:
                                current_task_actions.append(("Navigate", f"{obj_name}", None))
                                current_task_actions.append(("Pick", f"{obj_name}", None))
                                current_task_actions.append(("Navigate", f"{target_recep_name}", None))
                                current_task_actions.append(("Open", f"{target_recep_name}", None))
                                relation = "within"
                                action_args = f"{obj_name}, {relation}, {target_recep_name}, None, None"
                                current_task_actions.append(("Place", action_args, None))
                                current_task_actions.append(("Close", f"{target_recep_name}", None))
                        else:
                            relation = "within"
                            action_args = f"{obj_name}, {relation}, {target_recep_name}, None, None"
                            current_task_actions.append(("Rearrange", f"{action_args}", None))
                    else:
                        print(f"    [Skip] No valid target for {fname} on {obj_name}")
                        continue

                elif fname == "is_on_floor":
                    # 1. 获取目标物体
                    # 兼容对象属性访问和字典访问
                    args = prop.args if hasattr(prop, "args") else prop["args"]
                    obj_handle = args["object_handles"][0]
                    obj_name = self.get_node_name(obj_handle)

                    if obj_name == "None":
                        print(f"  [Skip] Object handle {obj_handle} not found in graph.")
                        continue

                    # 2. 扫描同一 Episode 中的关联命题 (确定房间和参照物)
                    target_room_id = None
                    ref_obj_handle = None
                    ref_obj_name = None

                    # 使用 props (即 episode.evaluation_propositions) 进行遍历
                    for p in props:
                        # 兼容处理
                        p_name = p.function_name if hasattr(p, "function_name") else p["function_name"]
                        p_args = p.args if hasattr(p, "args") else p["args"]

                        # 检查 is_in_room
                        if p_name == "is_in_room" and p_args.get("object_handles", [])[0] == obj_handle:
                            if "room_ids" in p_args:
                                target_room_id = p_args["room_ids"][0]

                        # 检查 is_next_to
                        if p_name == "is_next_to":
                            handles_a = p_args.get("entity_handles_a", [])
                            if handles_a and handles_a[0] == obj_handle:
                                handles_b = p_args.get("entity_handles_b", [])
                                if handles_b:
                                    for h_b in handles_b:
                                        name_b = self.get_node_name(h_b)
                                        if name_b != "None":
                                            ref_obj_handle = h_b
                                            ref_obj_name = name_b
                                            break

                    # 3. 寻找合适的地板 (Floor Node)
                    all_nodes = self.env_interface.full_world_graph.graph
                    floor_nodes = [n for n in all_nodes if isinstance(n, Floor)]
                    target_floor_name = None

                    # 策略 A: 优先匹配房间
                    if target_room_id:
                        for floor in floor_nodes:
                            if target_room_id.replace(" ", "_") in floor.name:
                                target_floor_name = floor.name
                                break

                    # 策略 B: 通过参照物找房间
                    if not target_floor_name and ref_obj_handle:
                        ref_node = next((n for n in all_nodes if n.properties.get("handle") == ref_obj_handle), None)
                        if ref_node:
                            neighbors = self.env_interface.full_world_graph.graph[ref_node]
                            for nbr in neighbors:
                                if isinstance(nbr, Room):
                                    room_floors = [f for f in floor_nodes if f.name.startswith(f"floor_{nbr.name}")]
                                    if room_floors:
                                        target_floor_name = room_floors[0].name
                                        break
                                elif isinstance(nbr, Floor):
                                    target_floor_name = nbr.name
                                    break

                    # 策略 C: 兜底
                    if not target_floor_name and floor_nodes:
                        target_floor_name = floor_nodes[0].name

                    if target_floor_name:
                        if ref_obj_name:
                            # [核心修正] 使用正确的 YAML 定义的 Tool Name
                            print(
                                f"  [GT Planner] Decomposing is_on_floor + next_to for {obj_name} near {ref_obj_name}")

                            # 1. Pick (对应 OraclePickSkill)
                            current_task_actions.append(("Pick", obj_name, None))

                            # 2. Navigate (对应 OracleNavSkill)
                            # 导航到参照物，确保物理上靠近
                            current_task_actions.append(("Navigate", ref_obj_name, None))

                            # 3. Place (对应 OraclePlaceSkill)
                            # 在地板上放置，但不传 spatial_constraint，防止触发底层报错
                            # 参数: object, on, target, constraint, reference
                            place_args = f"{obj_name}, on, {target_floor_name}, None, None"
                            current_task_actions.append(("Place", place_args, None))
                        else:
                            # 没有特殊约束，Rearrange 通常可以处理，或者也可以拆解
                            # Rearrange 在 yaml 中注册名为 "Rearrange"
                            action_args = f"{obj_name}, on, {target_floor_name}, None, None"
                            current_task_actions.append(("Rearrange", action_args, None))

                        used_objects.add(obj_handle)
                        actions_generated_for_prop += 1

                # 将生成的动作加入总队列
                if current_task_actions:
                    self.action_queue.extend(current_task_actions)
                    used_objects.add(obj_handle)
                    actions_generated_for_prop += 1

        print(f"[GT Planner] Plan generated: {len(self.action_queue)} steps.")
        for i in range(len(self.action_queue)):
            print(f"{i + 1}. {self.action_queue[i]}")