import sys
import os
import cv2
import numpy as np
import gzip
import json
from pathlib import Path
import logging
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from numba.cuda import atomic
from omegaconf import OmegaConf, open_dict

# Habitat & DCX 核心导入
import habitat
# 只需要导入 dataset 以触发注册，不需要手动干预
import habitat_llm.agent.env.dataset
from habitat_llm.agent.env import EnvironmentInterface, register_sensors, register_actions, register_measures
from habitat_llm.utils import fix_config, setup_config
from habitat_llm.utils.sim import init_agents
from habitat_llm.agent.env.actions import find_action_range
from habitat_llm.world_model import Room
from habitat_llm.agent.env.evaluation.predicate_wrappers import SimBasedPredicates
from habitat_llm.agent.env import sensors
from habitat_llm.planner.centralized_llm_planner import CentralizedLLMPlanner

# 1. 路径设置
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from spot_cameraman import SpotCameraman

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
dataset_overrides = [
    "habitat.dataset.data_path=data/datasets/partnr_episodes/v0_0/val_mini.json.gz",
    "habitat.dataset.scenes_dir=data/hssd-hab",
    "+habitat.dataset.metadata.metadata_folder=data/hssd-hab/metadata",
    "habitat.environment.iterator_options.shuffle=False",
    "habitat.environment.max_episode_steps=500000",
]
def get_object_by_sim_handle(world_graph, sim_handle):
    """
    在 WorldGraph 中根据 sim_handle 查找 object node
    """
    for obj in world_graph.get_all_objects():
        if getattr(obj, "sim_handle", None) == sim_handle:
            return obj
        # 兼容部分对象 handle 存在 properties 里的情况
        if hasattr(obj, "properties") and obj.properties.get("handle") == sim_handle:
            return obj
    return None

def parse_evaluation_propositions(episode):
    """
    :param episode: 一个episode
    :return: 把val数据集的episode中的evaluation_propostions数据打包出来，用作后续对照填充完成步骤
    """
    prop_list = []
    for idx, prop in enumerate(episode.evaluation_propositions):
        entry = {
            "idx": idx,
            "function": prop.function_name,
            "object_handles": prop.args.get("object_handles", []),  # 哈希值
            "receptacle_handles": prop.args.get("receptacle_handles", []),  # 哈希值
            "satisfied": False,
            "satisfied_step": None,
        }
        prop_list.append(entry)
    return prop_list
def check_proposition_satisfied(prop, world_graph):
    """
    :param prop: val中的成功状态，传入的是处理过的prop_list中的一个即prop_list[0]之类的
    :param world_graph: 世界地图现在状态
    :return: 是否完成这个状态
    """
    fn = prop["function"]
    obj_handles = prop["object_handles"]
    rec_handles = prop["receptacle_handles"]
    for obj_handle in obj_handles: # 哈希值
        # obj = world_graph.get_object_from_handle(obj_handle)
        obj = get_object_by_sim_handle(world_graph, obj_handle)
        print("obj is ", obj)
        if obj is None:
            return False
        if fn in ["is_on_top", "is_inside"]:
            parent = world_graph.find_furniture_for_object(obj) # table_30格式，obj is  Object[name=candle_0, type=candle]
            # parent is  Furniture[name=table_30, type=table]
            print("parent is ", parent)
            if parent is None:
                return False
            print("parent.sim_handle: ", parent.sim_handle, type(parent.sim_handle))
            print("rec_handles:", rec_handles)
            if parent.sim_handle not in rec_handles:
                return False
        elif fn == "is_clean":
            if not obj.properties.get("states", {}).get("clean", False):
                return False
        elif fn == "is_open":
            if not obj.properties.get("states", {}).get("open", False):
                return False
        elif fn == "is_closed":
            if not obj.properties.get("states", {}).get("open", False):
                return False
        else:
            return False
    return True


class ActionBasedSubtaskRecorder:
    """
    基于动作流监控 + 几何验证的子任务记录器
    """

    def __init__(self, prop_list, sim_instance, handle_to_name_map):
        self.prop_list = prop_list  # 待完成的任务列表 (Propositions)
        self.sim = sim_instance
        self.handle_to_name = handle_to_name_map
        self.completed_subtasks = []

        # 状态机：记录当前正在进行的动作
        # 结构: {'action': str, 'args': dict, 'start_step': int}
        self.current_action_state = None

        # 动作名到命题类型的映射 (根据你的 Oracle Skill 名称调整)
        self.skill_to_prop_map = {
            "OraclePlaceSkill": ["is_on_top", "is_inside", "is_in_room"],
            "OracleOpenSkill": ["is_open"],
            "OracleCloseSkill": ["is_closed"],
            "OracleCleanSkill": ["is_clean"],
            # 兼容可能的简称
            "Place": ["is_on_top", "is_inside"],
            "Open": ["is_open"],
            "Close": ["is_closed"]
        }

    def step(self, step_idx, atomic_details):
        """
        在每一步仿真后调用
        atomic_details: get_atomic_action_details() 的返回值
        """
        action_name = atomic_details['action']
        action_args = atomic_details['args']

        # 1. 检测动作变化 (Segmentation)
        if self.current_action_state:
            # 如果动作名变了，或者参数变了，说明上一个动作结束了
            if (self.current_action_state['action'] != action_name) or \
                    (self.current_action_state['args'] != action_args):

                # 结算上一个动作
                self._finalize_action(self.current_action_state, step_idx - 1)

                # 开始新动作 (忽略 Idle/Unknown)
                if action_name not in ["Idle", "Unknown", "Wait"]:
                    self.current_action_state = {
                        'action': action_name,
                        'args': action_args,
                        'start_step': step_idx
                    }
                else:
                    self.current_action_state = None
        else:
            # 当前没有记录动作，且来了新动作
            if action_name not in ["Idle", "Unknown", "Wait"]:
                self.current_action_state = {
                    'action': action_name,
                    'args': action_args,
                    'start_step': step_idx
                }

    def _finalize_action(self, action_state, end_step):
        """
        动作结束时的结算与验证逻辑 (增强版：支持字符串参数解析)
        """
        act_name = action_state['action']
        act_args = action_state['args']
        start_step = action_state['start_step']

        # --- [关键修改] 参数标准化处理 ---
        # Oracle Skill 传过来的 args 可能是字典，也可能是字符串，这里统一解析
        target_obj = None
        target_rec = None

        if isinstance(act_args, dict):
            # 如果已经是字典，直接获取
            target_obj = act_args.get('object') or act_args.get('object_handle') or act_args.get('handle')
            target_rec = act_args.get('receptacle') or act_args.get('receptacle_handle')

        elif isinstance(act_args, str):
            # 如果是字符串，根据动作类型解析 PARTNR 的参数格式
            # 格式通常是: "obj_name" 或者 "obj, relation, rec, constraint, ref"

            # 1. Place 动作：参数通常是 CSV 字符串
            if "Place" in act_name:
                parts = [p.strip() for p in act_args.split(',')]
                # OracleRearrangeSkill 格式: <obj>, <rel>, <rec>, <constraint>, <ref>
                if len(parts) >= 3:
                    target_obj = parts[0]
                    target_rec = parts[2]

            # 2. Pick / Open / Close 动作：参数通常只是物体名称
            elif any(k in act_name for k in ["Pick", "Open", "Close", "Clean"]):
                # 有些字符串可能包含逗号，取第一个部分通常是物体名
                target_obj = act_args.split(',')[0].strip()

        # 如果解析不出操作对象，无法进行后续验证，直接跳过
        if not target_obj:
            return

        # --- 以下逻辑保持不变 ---
        # 2. 意图匹配 (Intent Matching)
        possible_prop_types = self.skill_to_prop_map.get(act_name, [])

        for prop in self.prop_list:
            if prop['satisfied']: continue
            if prop['function'] not in possible_prop_types: continue

            # 匹配逻辑：动作的操作对象 必须在 命题的要求对象列表里
            # 注意：PARTNR 中的 handle 有时带有 "_0" 后缀，有时没有，建议做包含匹配或精准匹配
            # 这里简单处理，只要字符串包含即可 (更加鲁棒)
            obj_match = any(target_obj in h or h in target_obj for h in prop['object_handles'])

            # 对于 Place 任务，还需要匹配容器
            rec_match = True
            if 'receptacle_handles' in prop and prop['receptacle_handles']:
                if target_rec:
                    rec_match = any(target_rec in h or h in target_rec for h in prop['receptacle_handles'])
                else:
                    rec_match = False  # Place 任务必须有容器

            if obj_match and rec_match:
                # 3. 几何验证 (Verification)
                if self._verify_proposition(prop):
                    print(f"[Success] Subtask {prop['idx']} ({prop['function']}) verified at step {end_step}!")
                    prop['satisfied'] = True
                    prop['satisfied_step'] = end_step

                    self.completed_subtasks.append({
                        "subtask_index": prop['idx'],
                        "function": prop['function'],
                        "description": f"{prop['function']} {self._get_names(prop['object_handles'])}",
                        "start_step": start_step,
                        "end_step": end_step,
                        "object_names": self._get_names(prop['object_handles']),
                        "receptacle_names": self._get_names(prop['receptacle_handles'])
                    })
                    break

    def _verify_proposition(self, prop):
        """调用 SimBasedPredicates 进行验证"""
        try:
            func = getattr(SimBasedPredicates, prop['function'])
            # 构造 kwargs
            kwargs = {
                "sim": self.sim,
                "object_handles": prop['object_handles'],
                "number": 1  # 默认只要满足一个
            }
            if prop['receptacle_handles']:
                kwargs["receptacle_handles"] = prop['receptacle_handles']

            # 调用判定
            result = func(**kwargs)
            return result.is_satisfied
        except Exception as e:
            print(f"[Warning] Verification failed for {prop['function']}: {e}")
            return False

    def _get_names(self, handles):
        return [self.handle_to_name.get(h, h) for h in handles]

    def finish_episode(self, end_step):
        """Episode 结束时强制结算最后一个动作"""
        if self.current_action_state:
            self._finalize_action(self.current_action_state, end_step)


# ============ [在此处插入 WorldStateMonitor 类] ============
# class WorldStateMonitor:
#     def __init__(self, sim):
#         self.sim = sim
#         self.hash_to_id = {}
#         self.id_to_hash = {}
#         self.hash_to_name = {}
#         self.name_to_receptacle = {}
#
#     def reset_mappings(self, current_episode):
#         self.hash_to_id = {}
#         self.id_to_hash = {}
#         self.hash_to_name = {}
#         self.name_to_receptacle = getattr(current_episode, "name_to_receptacle", {})
#
#         rom = self.sim.get_rigid_object_manager()
#         for obj_handle in rom.get_object_handles():
#             obj = rom.get_object_by_handle(obj_handle)
#             semantic_name = obj_handle.split(".")[0]
#             for k, v in self.name_to_receptacle.items():
#                 if k in obj_handle: semantic_name = k.split("_")[0]
#
#             self.hash_to_id[obj_handle] = obj.object_id
#             self.id_to_hash[obj.object_id] = obj_handle
#             self.hash_to_name[obj_handle] = semantic_name
#
#     def get_semantic_name(self, handle):
#         return self.hash_to_name.get(handle, handle)
#
#     def check_proposition(self, prop):
#         func_name = prop.function_name
#         args = prop.args
#         obj_handle = args.get("object_handles", [])[0]
#         recep_handle = args.get("receptacle_handles", [])[0]
#
#         if recep_handle in self.name_to_receptacle:
#             recep_handle = self.name_to_receptacle[recep_handle].split("|")[0]
#
#         obj_id = next((oid for h, oid in self.hash_to_id.items() if obj_handle in h), None)
#         recep_id = next((oid for h, oid in self.hash_to_id.items() if recep_handle in h), None)
#
#         if obj_id is None or recep_id is None: return False
#
#         if func_name in ["is_on_top", "is_inside", "is_in"]:
#             rom = self.sim.get_rigid_object_manager()
#             obj = rom.get_object_by_id(obj_id)
#             recep = rom.get_object_by_id(recep_id)
#             if not obj or not recep: return False
#             dist = np.linalg.norm(obj.translation - recep.translation)
#             if dist < 1.5 and obj.translation[1] > recep.translation[1]: return True
#         return False
#
#     def is_in_view(self, handle, robot_state):
#         obj_id = next((oid for h, oid in self.hash_to_id.items() if handle in h), None)
#         if obj_id is None: return False
#
#         obj = self.sim.get_rigid_object_manager().get_object_by_id(obj_id)
#         if not obj: return False
#
#         # 简单视锥体计算
#         from habitat_sim.utils.common import quat_rotate_vector
#         rel_pos = obj.translation - robot_state.position
#         local_pos = quat_rotate_vector(robot_state.rotation.inverse(), rel_pos)
#
#         if local_pos[2] > -0.1: return False
#         import math
#         angle = math.degrees(math.atan2(abs(local_pos[0]), abs(local_pos[2])))
#         return angle < 45
# ========================================================

def get_config(config_file, overrides):
    with initialize(version_base=None, config_path="../../habitat_llm/conf"):
        config = compose(
            config_name=config_file,
            overrides=overrides,
        )
    HydraConfig().cfg = config
    with open_dict(config):
        config.hydra = {}
        config.hydra.runtime = {}
        config.hydra.runtime.output_dir = "outputs/data_gen_my"
    return config


def setup_env(config):
    register_sensors(config)
    register_actions(config)
    register_measures(config)
    env = EnvironmentInterface(config, init_wg=True, init_env=True)
    return env


def save_video(frames, filepath, fps=10):
    if len(frames) == 0:
        print(f"Warning: No frames to save for {filepath}")
        return
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    out = cv2.VideoWriter(filepath, fourcc, fps, (w, h))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"Video saved to {filepath}")


def get_world_state_snapshot(env_interface, human_uid=0):
    world_graph = env_interface.full_world_graph
    sim = env_interface.sim
    snapshot = {}
    all_objects = world_graph.get_all_objects()
    human_pos = sim.get_agent_state(human_uid).position
    for obj in all_objects:
        obj_name = obj.name
        obj_info = {
            "category": obj.properties.get("category", "unknown"),
            "states": obj.properties.get("states", {}),
            "relation": "unknown",
            "parent": "None"
        }
        parent_furniture = world_graph.find_furniture_for_object(obj)
        is_held_by_human = False
        obj_pos = obj.properties.get("translation", [0, 0, 0])
        dist_to_human = np.linalg.norm(np.array(human_pos) - np.array(obj_pos))

        if parent_furniture:
            obj_info["relation"] = "on"
            obj_info["parent"] = parent_furniture.name
            # 获取房间信息
            rooms = world_graph.get_neighbors_of_type(parent_furniture, Room)
            if rooms:
                obj_info["room"] = rooms[0].name
        elif dist_to_human < 0.5:
            # 如果不在家具上，且离人很近，标记为可能被抓取
            obj_info["relation"] = "held_by_human_candidate"
            obj_info["parent"] = "agent_0"
            obj_info["room"] = "near_agent_0"  # 后处理时替换为 Agent 所在的房间
        else:
            obj_info["relation"] = "floor_or_isolated"
            obj_info["parent"] = "floor"
        snapshot[obj_name] = obj_info
    return snapshot


def get_atomic_action_name(tool_instance):
    """
    深度递归查找当前正在执行的原子技能名称
    兼容 MotorSkillTool 封装、OracleRearrangeSkill (Dict结构) 和 CompoundSkill (List结构)
    """
    # 1. 基础情况：为空
    if tool_instance is None:
        return "Idle"
    # 2. 解包 MotorSkillTool，Partnr 的工具通常被 MotorSkillTool 包裹，真实的技能逻辑在 .skill 属性中
    real_skill = getattr(tool_instance, "skill", tool_instance)
    # 3. 处理 OracleRearrangeSkill (字典结构 + Enum索引)，特征：skills 是 dict，active_skill 是 Enum
    if hasattr(real_skill, "skills") and isinstance(real_skill.skills, dict) and hasattr(real_skill, "active_skill"):
        active_enum = real_skill.active_skill
        # 确保枚举值在字典中 (防止初始化时状态不一致)
        if active_enum in real_skill.skills:
            sub_skill = real_skill.skills[active_enum]
            return get_atomic_action_name(sub_skill)
    # 4. 处理 CompoundSkill (列表结构 + Int索引)，特征：skills 是 list，active_skill 是 int
    if hasattr(real_skill, "skills") and isinstance(real_skill.skills, list) and hasattr(real_skill, "active_skill"):
        idx = real_skill.active_skill
        # 确保索引越界检查
        if isinstance(idx, int) and 0 <= idx < len(real_skill.skills):
            sub_skill = real_skill.skills[idx]
            return get_atomic_action_name(sub_skill)
    # 5. 基础情况：到达叶子节点 (原子技能)，优先尝试获取配置中的名称 (通常更易读，如 "Nav", "Pick")
    if hasattr(real_skill, "config") and hasattr(real_skill.config, "name"):
        return real_skill.config.name
    # 兜底：返回类名 (如 "OracleNavSkill")
    return real_skill.__class__.__name__


def get_atomic_action_details(tool_instance):
    """
    [修改版] 深度查找原子技能，并提取完整参数 args
    """
    # 默认返回值增加 'args' 字段
    default_result = {"action": "Idle", "target": "None", "args": None}
    if tool_instance is None:
        return default_result
    # 解包 MotorSkillTool
    real_skill = getattr(tool_instance, "skill", tool_instance)

    # --- 情况 A: OracleRearrangeSkill (最关键的参数来源) ---
    if hasattr(real_skill, "skills") and isinstance(real_skill.skills, dict) and hasattr(real_skill, "active_skill"):
        active_enum = real_skill.active_skill
        # 1. [新增] 提取当前层级的完整参数，PARTNR 的技能通常在这里存储 {'object':..., 'receptacle':...} 或类似字符串
        current_args = None
        if hasattr(real_skill, "_skill_args") and active_enum in real_skill._skill_args:
            current_args = real_skill._skill_args[active_enum]
        # 2. 递归查找
        if active_enum in real_skill.skills:
            sub_skill = real_skill.skills[active_enum]
            sub_result = get_atomic_action_details(sub_skill)
            # 如果子层没有找到args，优先使用当前层提取到的 args
            final_args = sub_result.get("args")
            if final_args is None:
                final_args = current_args
            return {"action": sub_result["action"], "target": sub_result["target"], "args": final_args}

    # --- 情况 B: CompoundSkill (List结构) ---
    if hasattr(real_skill, "skills") and isinstance(real_skill.skills, list) and hasattr(real_skill, "active_skill"):
        idx = real_skill.active_skill
        if isinstance(idx, int) and 0 <= idx < len(real_skill.skills):
            return get_atomic_action_details(real_skill.skills[idx])

    # --- 情况 C: 原子技能 (Atomic Skill) ---
    action_name = "Unknown"
    if hasattr(real_skill, "config") and hasattr(real_skill.config, "name"):
        action_name = real_skill.config.name
    else:
        action_name = real_skill.__class__.__name__
    target_name = "None"
    if hasattr(real_skill, "target_handle"):
        target_name = str(real_skill.target_handle)
    elif hasattr(real_skill, "target"):
        target_name = str(real_skill.target)
    return {"action": action_name, "target": target_name, "args": None}


def main():
    obs_sensors = ["agent_0_third_rgb",
                   "agent_0_articulated_agent_arm_depth",
                   "agent_0_articulated_agent_arm_rgb",
                   "agent_0_head_panoptic",
                   "agent_0_head_depth",
                   "agent_0_head_rgb",
                   "agent_0_relative_resting_position",
                   "agent_0_joint",
                   "agent_0_ee_pos",
                   "agent_0_is_holding",
                   "agent_0_dynamic_obj_goal_sensor",
                   "agent_0_dynamic_goal_to_agent_gps_compass",
                   "agent_0_dynamic_obj_start_sensor",
                   "agent_1_third_rgb",
                   "agent_1_articulated_agent_arm_depth",
                   "agent_1_articulated_agent_arm_rgb",
                   "agent_1_articulated_agent_arm_panoptic",
                   "agent_1_head_depth",
                   "agent_1_head_rgb",
                   "agent_1_relative_resting_position",
                   "agent_1_joint",
                   "agent_1_ee_pos",
                   "agent_1_is_holding",
                   "agent_1_dynamic_obj_goal_sensor",
                   "agent_1_dynamic_goal_to_agent_gps_compass",
                   "agent_1_dynamic_obj_start_sensor",
                   "agent_1_goal_to_agent_gps_compass",
                   "agent_1_humanoid_detector_sensor",
                   "agent_1_articulated_agent_jaw_rgb",
                   "agent_1_articulated_agent_jaw_depth",
                   "agent_1_articulated_agent_jaw_panoptic",
                   ]
    config = get_config("examples/planner_multi_agent_demo_config_my1230.yaml",
                        overrides=[f"habitat.gym.obs_keys={obs_sensors}"] + dataset_overrides, )
    config = setup_config(config, 0)
    with open_dict(config):
        config.habitat.environment.max_episode_steps = 500000
        config.habitat.environment.max_episode_seconds = 50000000
    dataset = habitat.make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)
    print("Initializing EnvironmentInterface...")
    env_interface = setup_env(config)
    sim = env_interface.sim
    cameraman = SpotCameraman(sim, robot_agent_uid=1, human_agent_uid=0)
    print("Initializing Planner...")
    llm_planner_conf = config.evaluation.planner
    planner = instantiate(llm_planner_conf)
    planner = planner(env_interface=env_interface)
    agent_config = config.evaluation.agents
    planner.agents = init_agents(agent_config, env_interface)

    total_episodes = len(dataset.episodes)
    print(f"Total episodes to process: {total_episodes}")
    max_num_steps = 400000
    ego_key = "agent_1_articulated_agent_jaw_rgb"
    third_key = "agent_0_third_rgb"
    ego_save_dir = "my_videos/ego_videos"
    third_save_dir = "my_videos/third_videos"
    metadata_save_dir = "my_videos/metadata"
    os.makedirs(metadata_save_dir, exist_ok=True)

    # for i in range(total_episodes):
    for i in range(1):
        current_episode = env_interface.env.env.env._env.current_episode
        current_episode_id = current_episode.episode_id
        current_instruction = current_episode.instruction
        episode_name_to_receptacle = current_episode.name_to_receptacle
        print(f"----- Now processing Episode {current_episode_id} -----")
        print(f"The Instruction is '{current_instruction}'.")
        prop_list = parse_evaluation_propositions(current_episode)
        print("prop_list: ", prop_list)


        name_to_handle_map = {}
        handle_to_name_map = {}
        wg = env_interface.full_world_graph
        all_graph_objects = wg.get_all_objects()
        all_graph_receptacles = wg.get_all_receptacles()
        count_of_objects = 0
        for obj in all_graph_objects:
            semantic_name = obj.name
            handle = getattr(obj, "sim_handle", None)
            if not handle and hasattr(obj, "properties"):
                handle = obj.properties.get("handle")
            if handle:
                name_to_handle_map[semantic_name] = handle
                handle_to_name_map[handle] = semantic_name
            count_of_objects = len(name_to_handle_map)
        print(f"Mapped {len(name_to_handle_map)} objects")

        for rc in all_graph_receptacles:
            semantic_name = rc.name
            handle = getattr(rc, "sim_handle", None)
            if not handle and hasattr(rc, "properties"):
                handle = rc.properties.get("handle")
            if handle:
                name_to_handle_map[semantic_name] = handle
                handle_to_name_map[handle] = semantic_name
        print(f"Mapped {len(name_to_handle_map)-count_of_objects} receptacles")

        subtask_recorder = ActionBasedSubtaskRecorder(prop_list, sim, handle_to_name_map)

        human_agent_sim = sim.agents_mgr[0]
        video_ego_frames = []
        video_3rd_frames = []
        task_done = False
        cstep = 0
        cameraman.reset()
        planner.reset()
        observations = env_interface.get_observations()
        print(f"Spot Start Observing: {cameraman.start_step}")
        episode_metadata = []
        # episode_metadata_record = {
        #     "instruction": current_instruction,
        #     "object_handle_mapping": name_to_handle_map,
        #     "steps": None
        # }
        while not task_done and cstep < max_num_steps:
            low_level_actions, planner_info, task_done = planner.get_next_action(
                current_instruction, observations, env_interface.world_graph
            )

            if "responses" in planner_info and any(planner_info["responses"].values()):
                print(f"\n[CRITICAL DEBUG] Agent Responses: {planner_info['responses']}\n")
            if low_level_actions:
                action_range = find_action_range(env_interface.orig_action_space, "agent_1_base_velocity")
                lin_index = action_range[0]
                ang_index = action_range[1] - 1
                cam_action = cameraman.step()
                if cam_action is not None:
                    low_level_actions[1][lin_index] = cam_action[0]
                    low_level_actions[1][ang_index] = cam_action[1]
                else:
                    # 还没到时间，发送零速度
                    low_level_actions[1][lin_index] = 0.0
                    low_level_actions[1][ang_index] = 0.0
                human_tool_instance = planner.agents[0].last_used_tool
                high_level_action = human_tool_instance.name if human_tool_instance else "Idle"
                # atomic_action = get_atomic_action_name(human_tool_instance)
                atomic_details = get_atomic_action_details(human_tool_instance)
                subtask_recorder.step(cstep, atomic_details)
                is_holding = False
                held_obj_name = None
                is_holding = human_agent_sim.grasp_mgr.is_grasped
                if is_holding:
                    held_obj_name = f"obj_{human_agent_sim.grasp_mgr.snap_idx}"
                world_snapshot = get_world_state_snapshot(env_interface)

                robot_state = sim.get_agent_state(1)
                human_state = sim.get_agent_state(0)
                frame_record = {
                    "step": cstep,
                    "time_sec": cstep / 10.0,
                    "robot_active": cameraman.is_active,
                    "human_agent": {
                        "action": high_level_action,  # 之前为human_tool
                        "atomic_action": atomic_details["action"],
                        "action_args": atomic_details["args"],
                        "action_target": atomic_details["target"],
                        "is_holding": is_holding,
                        "held_obj": held_obj_name,
                        "pos": list(human_state.position),
                        "rot": list(human_state.rotation.components)
                    },
                    "robot_agent": {
                        "pos": list(robot_state.position),
                        "rot": list(robot_state.rotation.components)
                    },
                    "world_objects": world_snapshot
                }
                episode_metadata.append(frame_record)

                observations, reward, done, info = env_interface.step(low_level_actions)
                video_ego_frames.append(observations[ego_key].astype(np.uint8))
                video_3rd_frames.append(observations[third_key].astype(np.uint8))
                cstep += 1
        # episode_metadata_record["steps"] = episode_metadata
        subtask_recorder.finish_episode(cstep)
        # for subtask_id in range(len(subtask_recorder.completed_subtasks)):
        #     subtask = subtask_recorder.completed_subtasks[subtask_id]
        #     object_handles = subtask["object_handles"]
        #     rec_handles = subtask["receptacle_handles"]
        #     if subtask["proposition_idx"]==0:
        #         subtask_recorder.completed_subtasks[subtask_id]["start_step"] = 0
        #     else:
        #         last_subtask = subtask_recorder.completed_subtasks[subtask_id-1]
        #         subtask_recorder.completed_subtasks[subtask_id]["start_step"] = last_subtask["end_step"] + 1
        #     for object_handle in object_handles:
        #         if object_handle in handle_to_name_map.keys():
        #             subtask_recorder.completed_subtasks[subtask_id]["object_names"].append(handle_to_name_map[object_handle])
        #     for rec_handle in rec_handles:
        #         if rec_handle in handle_to_name_map.keys():
        #             subtask_recorder.completed_subtasks[subtask_id]["receptacles_names"].append(handle_to_name_map[rec_handle])
        episode_metadata_record = {
            "instruction": current_instruction,
            "object_and_receptacle_handle_mapping": name_to_handle_map,
            "subtasks": subtask_recorder.completed_subtasks,
            "evaluation_propositions": prop_list,
            "steps": episode_metadata
        }
        meta_path = os.path.join(metadata_save_dir, f"{current_episode_id}.json")
        with open(meta_path, "w") as f:
            json.dump(episode_metadata_record, f, indent=2)
        print(f"Saved metadata to {meta_path}")
        save_video(video_ego_frames, f"{ego_save_dir}/{current_episode_id}.mp4")
        save_video(video_3rd_frames, f"{third_save_dir}/{current_episode_id}.mp4")
        print(f"Episode {current_episode_id} End cstep: {cstep}")
        print(f"Episode {current_episode_id} End task_done: {task_done}")
        print(f"----- Saved Videos of Episode {current_episode_id} -----")
        if i < total_episodes - 1:
            env_interface.reset_environment()
            cameraman.reset()
            sim = env_interface.sim
            planner.reset()


if __name__ == "__main__":
    main()