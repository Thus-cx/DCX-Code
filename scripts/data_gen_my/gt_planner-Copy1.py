# from typing import Any, Dict, List, Tuple
# from habitat_llm.planner.planner import Planner


# class GroundTruthPlanner(Planner):
#     """
#     GT Planner 最终修正版 (Ver 4.0):
#     1. [Fix] 修复 AttributeError (正确遍历 PARTNR 自定义 Graph 结构)
#     2. [Feat] 增加 Handle -> NodeName 映射
#     3. [Feat] 保持 '忙碌等待' 机制
#     """

#     def __init__(self, plan_config: Any, env_interface: Any):
#         super().__init__(plan_config, env_interface)
#         self.action_queue = []
#         self.current_episode_id = None
#         self.current_action_tuple = None
#         self.handle_to_name_map = {}

#     def reset(self):
#         self.action_queue = []
#         self.current_episode_id = None
#         self.current_action_tuple = None
#         self.handle_to_name_map = {}

#     def build_handle_map(self):
#         """
#         构建 Sim Handle 到 WorldGraph Node Name 的映射表。
#         [核心修复]: 适应 PARTNR 的 Graph 结构 (dict of nodes)，而不是 NetworkX。
#         """
#         self.handle_to_name_map = {}
#         # full_world_graph 是 WorldGraph 实例
#         # WorldGraph.graph 是一个 dict {NodeEntity: {Neighbor: Edge}}
#         world_graph = self.env_interface.full_world_graph

#         # 直接遍历字典的键，键就是 Entity 对象
#         for node in world_graph.graph:
#             # 检查节点是否有 sim_handle 属性
#             if hasattr(node, "sim_handle"):
#                 self.handle_to_name_map[node.sim_handle] = node.name

#         print(f"[GT Planner] Built handle map with {len(self.handle_to_name_map)} entries.")

#     def get_node_name(self, handle):
#         """
#         将 val.json 中的 handle (如 '9dfbc...') 转换为 node name (如 'candle_0')
#         """
#         if handle in self.handle_to_name_map:
#             return self.handle_to_name_map[handle]

#         # 处理可能的后缀差异
#         base_handle = handle.split(":")[0] if ":" in handle else handle
#         for k, v in self.handle_to_name_map.items():
#             if k.startswith(base_handle):
#                 return v

#         return handle

#     def get_next_action(
#             self,
#             instruction: str,
#             observations: Dict[str, Any],
#             world_graph: Dict[int, Any]
#     ) -> Tuple[Dict[int, Any], Dict[str, Any], bool]:

#         # 1. 获取 Current Episode
#         try:
#             current_ep = self.env_interface.env.env.env._env.current_episode
#         except AttributeError:
#             try:
#                 current_ep = self.env_interface.env.env._env.current_episode
#             except AttributeError:
#                 raise AttributeError("Could not find current_episode in env_interface stack")

#         # 2. 检查是否切换了 Episode
#         if self.current_episode_id != current_ep.episode_id:
#             self.current_episode_id = current_ep.episode_id
#             # 重新构建映射表
#             self.build_handle_map()
#             self.generate_plan_from_gt(current_ep)
#             self.current_action_tuple = None

#             # 3. 状态机逻辑
#         if self.current_action_tuple is None:
#             if not self.action_queue:
#                 return {}, {"thought": "All GT actions executed."}, True

#             self.current_action_tuple = self.action_queue.pop(0)
#             print(f"[GT Planner] Starting NEW action: {self.current_action_tuple}")

#         # 4. 构建高级动作
#         high_level_actions = {
#             0: self.current_action_tuple,
#             1: ("Wait", None, None)
#         }

#         # 5. 转换为低级控制信号
#         low_level_actions, responses = self.process_high_level_actions(
#             high_level_actions,
#             observations
#         )

#         # 6. 检查完成状态
#         agent_response = responses.get(0, "")

#         if agent_response:
#             print(f"[GT Planner] Action Finished/Stopped. Result: {agent_response}")
#             self.current_action_tuple = None

#         thought = f"Executing: {high_level_actions[0][0]} | Status: {'Running' if not agent_response else 'Done'}"
#         info = {"thought": thought}
#         if responses:
#             info.update(responses)

#         return low_level_actions, info, False

#     # def generate_plan_from_gt(self, episode):
#     #     print(f"[GT Planner] Generating plan for Episode {episode.episode_id}...")
#     #
#     #     if not hasattr(episode, "evaluation_propositions"):
#     #         print(f"WARNING: Episode {episode.episode_id} has no evaluation_propositions!")
#     #         return
#     #
#     #     props = episode.evaluation_propositions
#     #
#     #     state_tasks = []
#     #     rearrange_tasks = []
#     #
#     #     for prop in props:
#     #         fname = prop.function_name if hasattr(prop, "function_name") else prop["function_name"]
#     #         args = prop.args if hasattr(prop, "args") else prop["args"]
#     #
#     #         obj_handles = args.get("object_handles", [])
#     #         recep_handles = args.get("receptacle_handles", [])
#     #
#     #         target_recep_handle = recep_handles[0] if recep_handles else None
#     #         target_recep_name = self.get_node_name(target_recep_handle) if target_recep_handle else "None"
#     #
#     #         for obj_handle in obj_handles:
#     #             obj_name = self.get_node_name(obj_handle)
#     #
#     #             if fname in ["is_on_top", "is_inside"]:
#     #                 relation = "on" if fname == "is_on_top" else "in"
#     #                 action_args = f"{obj_name}, {relation}, {target_recep_name}, None, None"
#     #                 rearrange_tasks.append(("Rearrange", action_args, None))
#     #             elif fname == "is_clean":
#     #                 state_tasks.append(("OracleCleanSkill", f"{obj_name}", None))
#     #             elif fname == "is_filled":
#     #                 state_tasks.append(("OracleFillSkill", f"{obj_name}", None))
#     #             elif fname == "is_powered_on":
#     #                 state_tasks.append(("OraclePowerOnSkill", f"{obj_name}", None))
#     #             elif fname == "is_powered_off":
#     #                 state_tasks.append(("OraclePowerOffSkill", f"{obj_name}", None))
#     #
#     #     self.action_queue = state_tasks + rearrange_tasks
#     #
#     #     print(f"[GT Planner] Plan generated with mapped names: {len(self.action_queue)} steps.")
#     #     for i, act in enumerate(self.action_queue):
#     #         print(f"  {i + 1}. {act[0]} -> {act[1]}")
#     def generate_plan_from_gt(self, episode):
#         print(f"[GT Planner] Generating plan for Episode {episode.episode_id}...")

#         if not hasattr(episode, "evaluation_propositions"):
#             print(f"WARNING: Episode {episode.episode_id} has no evaluation_propositions!")
#             return

#         props = episode.evaluation_propositions

#         state_tasks = []
#         rearrange_tasks = []

#         # 获取全图引用，用于处理 is_next_to 等复杂空间逻辑
#         world_graph = self.env_interface.full_world_graph

#         for prop in props:
#             fname = prop.function_name if hasattr(prop, "function_name") else prop["function_name"]
#             args = prop.args if hasattr(prop, "args") else prop["args"]

#             obj_handles = args.get("object_handles", [])
#             recep_handles = args.get("receptacle_handles", [])

#             # --- [Fix 1] 获取数量约束 ---
#             # 如果没有 number 字段，默认为 1 (通常 rearrange 任务都有)
#             # 如果是 state 任务(clean all)，有时意味着全部，但通常 val.json 会明确
#             number_required = args.get("number", 1)

#             # --- [Fix 1] 截断对象列表 ---
#             # 只处理前 N 个满足条件的对象
#             # 注意：这里简单的取前 N 个。在更复杂的场景下可能需要择优，但在 GT 生成中通常任意 N 个都行
#             target_obj_handles = obj_handles[:number_required]

#             target_recep_handle = recep_handles[0] if recep_handles else None
#             target_recep_name = self.get_node_name(target_recep_handle) if target_recep_handle else "None"

#             for obj_handle in target_obj_handles:
#                 obj_name = self.get_node_name(obj_handle)

#                 # --- 1. 移动类任务 (Rearrange/Place) ---
#                 if fname in ["is_on_top", "is_inside", "is_on_floor"]:
#                     relation = "on"
#                     tgt_name = target_recep_name

#                     if fname == "is_inside": relation = "in"
#                     if fname == "is_on_floor": tgt_name = "floor"  # 特殊处理 floor

#                     action_args = f"{obj_name}, {relation}, {tgt_name}, None, None"
#                     rearrange_tasks.append(("Rearrange", action_args, None))

#                 # --- [Fix 2] 处理 is_in_room (移动到房间) ---
#                 elif fname == "is_in_room":
#                     # 策略：is_in_room 通常意味着放到该房间的任意位置
#                     # 我们尝试从 args 获取 room_id
#                     room_id = args.get("room_ids", [None])[0]
#                     if room_id:
#                         # 这是一个比较宽泛的任务。为了保证执行，我们转换成：
#                         # Rearrange obj -> 到该房间的某个家具上 (取第一个)
#                         # 或者如果 OracleRearrange 支持房间名，直接传房间名
#                         # 这里采取稳妥策略：找房间里的第一个家具
#                         # 注意：需要你实现简单的查找逻辑，或者简单地传 room_id 看 Oracle 是否支持
#                         # 假设 OracleRearrangeSkill 支持 "room_name" 作为 target
#                         action_args = f"{obj_name}, in, {room_id}, None, None"
#                         rearrange_tasks.append(("Rearrange", action_args, None))

#                 # --- [Fix 2] 处理 is_next_to (在...旁边) ---
#                 elif fname == "is_next_to":
#                     # args 中会有 'proxim_entity_handles' 或类似字段指明在谁旁边
#                     # 通常取 args['second_object_handles'] (视具体 json 结构而定)
#                     # 简单策略：放到和参考物相同的容器上
#                     ref_handles = args.get("second_object_handles", [])
#                     if ref_handles:
#                         ref_obj_name = self.get_node_name(ref_handles[0])
#                         # 构造语义：把 A 放到 B 旁边 -> Rearrange A to (B's parent)
#                         # 这需要查图，如果嫌麻烦，暂时可以生成一个 "Near" 的伪指令
#                         # 或者更直接：OracleRearrange 甚至可能支持 "next_to" 关系？
#                         # 如果不支持，稳妥做法是：找到 ref_obj 的 receptacle
#                         ref_obj = world_graph.get_node_from_name(ref_obj_name)
#                         parent = world_graph.find_furniture_for_object(ref_obj)
#                         if parent:
#                             tgt_name = parent.name
#                             action_args = f"{obj_name}, on, {tgt_name}, None, None"
#                             rearrange_tasks.append(("Rearrange", action_args, None))

#                 # --- 2. 状态类任务 ---
#                 elif fname == "is_clean":
#                     state_tasks.append(("OracleCleanSkill", f"{obj_name}", None))
#                 elif fname == "is_filled":
#                     state_tasks.append(("OracleFillSkill", f"{obj_name}", None))
#                 elif fname == "is_powered_on":
#                     state_tasks.append(("OraclePowerOnSkill", f"{obj_name}", None))
#                 elif fname == "is_powered_off":
#                     state_tasks.append(("OraclePowerOffSkill", f"{obj_name}", None))
#                 elif fname == "is_open":
#                     state_tasks.append(("OracleOpenSkill", f"{obj_name}", None))
#                 elif fname == "is_closed":
#                     state_tasks.append(("OracleCloseSkill", f"{obj_name}", None))

#         self.action_queue = state_tasks + rearrange_tasks

#         print(f"[GT Planner] Plan generated: {len(self.action_queue)} steps (Filtered by 'number').")
#         for i, act in enumerate(self.action_queue):
#             print(f"  {i + 1}. {act[0]} -> {act[1]}")

from typing import Any, Dict, List, Tuple
from habitat_llm.planner.planner import Planner
# 必须导入 Room 类以便进行类型检查和邻居查找
from habitat_llm.world_model import Room 

class GroundTruthPlanner(Planner):
    """
    GT Planner 最终健壮版 (Ver 5.0):
    1. [Fix] 修复 'is_in_room' 导致的任务秒败 (自动查找房间内的家具作为目标)
    2. [Fix] 修复 'number' 约束 (只执行指定数量的任务)
    3. [Fix] 完善 'is_next_to', 'is_on_floor' 等边缘情况
    4. [Debug] 增加详细的 Failure Log，一眼看出为什么跳过任务
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
        self.handle_to_name_map = {}
        world_graph = self.env_interface.full_world_graph
        for node in world_graph.graph:
            if hasattr(node, "sim_handle"):
                self.handle_to_name_map[node.sim_handle] = node.name
        print(f"[GT Planner] Built handle map with {len(self.handle_to_name_map)} entries.")

    def get_node_name(self, handle):
        if handle in self.handle_to_name_map:
            return self.handle_to_name_map[handle]
        # 尝试去掉后缀匹配
        base_handle = handle.split(":")[0] if ":" in handle else handle
        for k, v in self.handle_to_name_map.items():
            if k.startswith(base_handle):
                return v
        return handle

    def get_next_action(self, instruction: str, observations: Dict[str, Any], world_graph: Dict[int, Any]) -> Tuple[Dict[int, Any], Dict[str, Any], bool]:
        # 1. 获取 Episode
        try:
            current_ep = self.env_interface.env.env.env._env.current_episode
        except AttributeError:
            try:
                current_ep = self.env_interface.env.env._env.current_episode
            except AttributeError:
                raise AttributeError("Could not find current_episode in env_interface stack")

        # 2. 切换 Episode 检测
        if self.current_episode_id != current_ep.episode_id:
            self.current_episode_id = current_ep.episode_id
            self.build_handle_map()
            self.generate_plan_from_gt(current_ep)
            self.current_action_tuple = None 

        # 3. 状态机：取任务
        if self.current_action_tuple is None:
            if not self.action_queue:
                return {}, {"thought": "All GT actions executed."}, True
            
            self.current_action_tuple = self.action_queue.pop(0)
            print(f"[GT Planner] Starting NEW action: {self.current_action_tuple}")

        # 4. 执行
        high_level_actions = {0: self.current_action_tuple, 1: ("Wait", None, None)}
        low_level_actions, responses = self.process_high_level_actions(high_level_actions, observations)

        # 5. 检查 Agent 反馈 (这是导致 fast run 的关键)
        agent_response = responses.get(0, "")
        if agent_response:
            # 如果不是 "Successful execution!"，那就是报错了
            if "Success" not in agent_response:
                print(f"[GT Planner WARNING] Action FAILED/SKIPPED immediately! Reason: {agent_response}")
                print(f"   -> Action was: {self.current_action_tuple}")
            else:
                print(f"[GT Planner] Action Finished Successfully.")
            
            self.current_action_tuple = None # 结束当前任务，准备下一个
        
        info = {"thought": f"Executing: {high_level_actions[0][0]} | Status: {agent_response}"}
        if responses: info.update(responses)

        return low_level_actions, info, False

    def generate_plan_from_gt(self, episode):
        print(f"[GT Planner] Generating plan for Episode {episode.episode_id}...")
        
        if not hasattr(episode, "evaluation_propositions"):
             print(f"WARNING: Episode {episode.episode_id} has no evaluation_propositions!")
             return

        props = episode.evaluation_propositions
        
        state_tasks = []
        rearrange_tasks = []

        # 获取全图引用
        world_graph = self.env_interface.full_world_graph

        for prop in props:
            fname = prop.function_name if hasattr(prop, "function_name") else prop["function_name"]
            args = prop.args if hasattr(prop, "args") else prop["args"]
            
            obj_handles = args.get("object_handles", [])
            recep_handles = args.get("receptacle_handles", [])
            
            # 获取数量约束
            number_required = args.get("number", 1)
            target_obj_handles = obj_handles[:number_required]

            target_recep_handle = recep_handles[0] if recep_handles else None
            target_recep_name = self.get_node_name(target_recep_handle) if target_recep_handle else "None"

            for obj_handle in target_obj_handles:
                obj_name = self.get_node_name(obj_handle)
                
                # --- 1. 标准移动 (is_on_top / is_inside) ---
                if fname in ["is_on_top", "is_inside"]:
                    relation = "on" if fname == "is_on_top" else "in"
                    action_args = f"{obj_name}, {relation}, {target_recep_name}, None, None"
                    rearrange_tasks.append(("Rearrange", action_args, None))

                # --- 2. [核心修复] 房间移动 (is_in_room) ---
                elif fname == "is_in_room":
                    room_ids = args.get("room_ids", [])
                    if room_ids:
                        target_room_id = room_ids[0].lower() # 例如 'kitchen'
                        found_furniture = None
                        
                        # 遍历所有家具，查找其所在的房间是否匹配
                        all_receps = world_graph.get_all_receptacles()
                        for rec in all_receps:
                            # 获取该家具所在的房间
                            rooms = world_graph.get_neighbors_of_type(rec, Room)
                            for r in rooms:
                                # [修改点]: 全方位匹配 (Name, Handle, Category)
                                r_name = r.name.lower()
                                r_handle = getattr(r, "sim_handle", "").lower()
                                
                                # 从 properties 中获取 category (这是解决问题的关键!)
                                r_category = ""
                                if hasattr(r, "properties") and "category" in r.properties:
                                    r_category = r.properties["category"].lower()
                                
                                # 只要命中任意一个属性，就算匹配成功
                                if (target_room_id in r_name) or \
                                   (target_room_id == r_handle) or \
                                   (target_room_id in r_category):
                                    found_furniture = rec.name
                                    # print(f"[Debug] Matched room '{r.name}' (Cat: {r_category}) with target '{target_room_id}'. Found furniture: {found_furniture}")
                                    break
                            if found_furniture: break
                        
                        if found_furniture:
                            action_args = f"{obj_name}, on, {found_furniture}, None, None"
                            rearrange_tasks.append(("Rearrange", action_args, None))
                        else:
                            print(f"[GT Planner Error] Could not find any furniture in room '{target_room_id}' for {obj_name}")

                # --- 3. 地板放置 (is_on_floor) ---
                elif fname == "is_on_floor":
                    # 直接尝试放到 "floor"
                    action_args = f"{obj_name}, on, floor, None, None"
                    rearrange_tasks.append(("Rearrange", action_args, None))

                # --- 4. 相对位置 (is_next_to) ---
                elif fname == "is_next_to":
                    ref_handles = args.get("second_object_handles", [])
                    if ref_handles:
                        ref_obj_name = self.get_node_name(ref_handles[0])
                        # 查找参照物的父容器
                        ref_entity = None
                        for node in world_graph.graph:
                            if node.name == ref_obj_name:
                                ref_entity = node
                                break
                        
                        parent_rec = None
                        if ref_entity:
                            parent_rec = world_graph.find_furniture_for_object(ref_entity)
                        
                        if parent_rec:
                            tgt_name = parent_rec.name
                            action_args = f"{obj_name}, on, {tgt_name}, None, None"
                            rearrange_tasks.append(("Rearrange", action_args, None))
                        else:
                            print(f"[GT Planner Warning] 'is_next_to': Reference object {ref_obj_name} has no parent furniture.")

                # --- 5. 状态改变 ---
                elif fname == "is_clean":
                    state_tasks.append(("OracleCleanSkill", f"{obj_name}", None))
                elif fname == "is_filled":
                    state_tasks.append(("OracleFillSkill", f"{obj_name}", None))
                elif fname == "is_powered_on":
                    state_tasks.append(("OraclePowerOnSkill", f"{obj_name}", None))
                elif fname == "is_powered_off":
                    state_tasks.append(("OraclePowerOffSkill", f"{obj_name}", None))
                elif fname == "is_open":
                    state_tasks.append(("OracleOpenSkill", f"{obj_name}", None))
                elif fname == "is_closed":
                    state_tasks.append(("OracleCloseSkill", f"{obj_name}", None))

        self.action_queue = state_tasks + rearrange_tasks
        
        print(f"[GT Planner] Plan generated: {len(self.action_queue)} steps (Filtered by 'number').")
        for i, act in enumerate(self.action_queue):
            print(f"  {i+1}. {act[0]} -> {act[1]}")