import copy
import sys
import os
import cv2
import numpy as np
import gzip
import json
from pathlib import Path
import logging
import magnum as mn
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from numba.cuda import atomic
from numba.scripts.generate_lower_listing import description
from omegaconf import OmegaConf, open_dict
import imageio  # Added for streaming video write
import math

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
from habitat_llm.planner.planner import Planner
from gt_planner import GroundTruthPlanner
from object_annotator import ProjectionAnnotator, FurnitureVisibilityChecker
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
    "habitat.dataset.data_path=data/datasets/partnr_episodes/v0_0/val.json.gz",
    "habitat.dataset.scenes_dir=data/hssd-hab",
    "+habitat.dataset.metadata.metadata_folder=data/hssd-hab/metadata",
    "habitat.environment.iterator_options.shuffle=False",
    "habitat.environment.max_episode_steps=500000",
]

RECORDING_FPS = 10.0


def get_object_by_sim_handle(world_graph, sim_handle):
    # 在 WorldGraph 中根据 sim_handle 查找 object node
    for obj in world_graph.get_all_objects():
        if getattr(obj, "sim_handle", None) == sim_handle:
            return obj
        # 兼容部分对象 handle 存在 properties 里的情况
        if hasattr(obj, "properties") and obj.properties.get("handle") == sim_handle:
            return obj
    return None


def parse_evaluation_propositions(episode):
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
    fn = prop["function"]
    obj_handles = prop["object_handles"]
    rec_handles = prop["receptacle_handles"]
    for obj_handle in obj_handles:  # 哈希值
        # obj = world_graph.get_object_from_handle(obj_handle)
        obj = get_object_by_sim_handle(world_graph, obj_handle)
        print("obj is ", obj)
        if obj is None:
            return False
        if fn in ["is_on_top", "is_inside"]:
            parent = world_graph.find_furniture_for_object(obj)  # table_30格式，obj is  Object[name=candle_0, type=candle]
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
    def __init__(self, env_interface, prop_list, sim_instance, handle_to_name_map):
        self.env_interface = env_interface
        self.prop_list = prop_list
        self.sim = sim_instance
        self.handle_to_name = handle_to_name_map
        self.completed_subtasks = []
        self.current_action_state = None
        self.pending_context = None  # 用于存储中间状态，任务缓冲区

    def step(self, step_idx, atomic_details):
        action_name = atomic_details['action']
        action_args = atomic_details['args']

        # 状态机：检测动作切换
        if self.current_action_state:
            if (self.current_action_state['action'] != action_name) or (
                    self.current_action_state['args'] != action_args):

                # 结算上一个动作
                self._finalize_action(self.current_action_state, step_idx - 1)

                # 开始新动作 (忽略 Idle/Wait)
                # if action_name not in ["Idle", "Unknown", "Wait"]:
                is_cleaning_wait = (action_name == "Wait" and action_args and "SET_CLEAN" in str(action_args))
                if is_cleaning_wait or action_name not in ["Idle", "Unknown", "Wait"]:
                    self.current_action_state = {
                        'action': action_name,
                        'args': action_args,
                        'start_step': step_idx
                    }
                else:
                    self.current_action_state = None
        else:
            is_cleaning_wait = (action_name == "Wait" and action_args and "SET_CLEAN" in str(action_args))
            if is_cleaning_wait or action_name not in ["Idle", "Unknown", "Wait"]:
                self.current_action_state = {
                    'action': action_name,
                    'args': action_args,
                    'start_step': step_idx
                }

    def _finalize_action(self, action_state, end_step):
        act_name = action_state['action']
        act_args = action_state['args']
        start_step = action_state['start_step']

        # --- 1. 参数解析 ---
        target_obj = None
        target_rec = None

        if isinstance(act_args, dict):
            target_obj = act_args.get('object') or act_args.get('object_handle') or act_args.get('handle')
            target_rec = act_args.get('receptacle') or act_args.get('receptacle_handle')
        elif isinstance(act_args, str):
            if "," in act_args:
                parts = [p.strip() for p in act_args.split(',')]
                # 格式通常是: obj, relation, target, ...
                if len(parts) >= 1: target_obj = parts[0]
                if len(parts) >= 3:
                    # 第3个参数通常是 target (例如 table_10)
                    # 只有当它看起来像个物体名时才取用
                    if parts[2] and parts[2] != "None":
                        target_rec = parts[2]
            else:
                # 单一参数情况 (Clean, Fill, Power)
                target_obj = act_args.split(',')[0].strip()
                if "SET_CLEAN" in act_args:
                    target_obj = act_args.split(":", 1)[-1].strip()
                if "SET_FILL" in act_args:
                    target_obj = act_args.split(":", 1)[-1].strip()

        if not target_obj:
            return
        target_obj = self.handle_to_name.get(target_obj, target_obj)

        def is_water_source(rec_name):
            # if not rec_name: return False
            # keywords = ['sink', 'basin', 'faucet', 'bathtub']
            # return any(k in rec_name.lower() for k in keywords)
            all_furnitures = self.env_interface.full_world_graph.get_all_furnitures()
            faucet_furnitures = [fur for fur in all_furnitures
                                 if "components" in fur.properties
                                 and "faucet" in fur.properties["components"]]
            faucet_name = []
            for faucet_furniture in faucet_furnitures:
                faucet_name.append(faucet_furniture.name)
            if not rec_name: return False
            return any(k in rec_name.lower() for k in faucet_name)

        if "Rearrange" in act_name:
            if self.pending_context:  # 说明是单独的rearrange
                print(f"[Recorder] Alone Rearrange is recorded")
                self._save_record(self.pending_context)
                self.pending_context = None
            # if target_rec and is_water_source(target_rec):
            if target_obj:
                print(f"[Recorder] Detected Rearrange to Water Source ({target_rec}). Pending...")
                self.pending_context = {
                    "action_state": action_state,
                    "target_obj": target_obj,
                    "target_rec": target_rec,
                    "end_step": end_step  # 暂存结束时间
                }
                return
        # is_clean_action = ("Wait" in act_name and "SET_CLEAN" in str(act_args)) or "Clean" in act_name
        # is_fill_action = ("Wait" in act_name and "SET_FILL" in str(act_args)) or "Fill" in act_name
        is_clean_action = ("Wait" in act_name and "SET_CLEAN" in str(act_args))
        is_fill_action = ("Wait" in act_name and "SET_FILL" in str(act_args))
        if is_clean_action or is_fill_action:
            if self.pending_context:
                prev_obj = self.pending_context["target_obj"]
                match_obj = (target_obj == prev_obj) or (target_obj in prev_obj) or (prev_obj in target_obj)
                if match_obj:
                    merged_function = "Clean" if is_clean_action else "Fill"
                    print(f"[Recorder] Merging Rearrange + {merged_function} for {target_obj}")
                    start_step = self.pending_context["action_state"]["start_step"]
                    description = f"{merged_function} {target_obj} in the water source"
                    subtask_record = {
                        "function": merged_function,
                        "description": description,
                        "description_processed": description,
                        "start_step": start_step,
                        "end_step": end_step,
                        "object_names": [target_obj],
                        "receptacle_names": [self.pending_context["target_rec"]],
                        "matched_gt": False,
                        "gt_prop_index": -1
                    }
                    self._match_and_save(subtask_record)
                    self.pending_context = None
                    return
        if self.pending_context:
            self._save_record(self.pending_context)
            self.pending_context = None
        readable_act = act_name.replace("Oracle", "").replace("Skill", "")
        if "Wait" in act_name and "SET_CLEAN" in str(act_args):
            readable_act = "Clean"
        if "Wait" in act_name and "SET_FILL" in str(act_args):
            readable_act = "Fill"
        description = f"{readable_act} {target_obj}"
        if target_rec: description += f" with {target_rec}"
        # --- 2. 构建基础记录 ---
        subtask_record = {
            "function": readable_act,
            "description": description,
            "description_processed": description,
            "start_step": start_step,
            "end_step": end_step,
            "object_names": [target_obj],
            "receptacle_names": [target_rec] if target_rec else [],
            "matched_gt": False,
            "gt_prop_index": -1
        }
        self._match_and_save(subtask_record)

    def _save_record(self, pending_ctx):
        # 辅助函数：保存挂起的 Rearrange 任务 (当合并失败时)
        state = pending_ctx["action_state"]
        # 这里需要把原来 _finalize_action 里生成 Rearrange 记录的逻辑拷过来
        # 为了代码整洁，你可以把原有的记录生成逻辑封装成函数
        # 简单起见，这里手动构造：
        description = f"Rearrange {pending_ctx['target_obj']} to {pending_ctx['target_rec']}"
        subtask_record = {
            "function": "Rearrange",
            "description": description,
            "description_processed": description,
            "start_step": state['start_step'],
            "end_step": pending_ctx['end_step'],
            "object_names": [pending_ctx['target_obj']],
            "receptacle_names": [pending_ctx['target_rec']],
            "matched_gt": False,
            "gt_prop_index": -1
        }
        self._match_and_save(subtask_record)

    def _match_and_save(self, subtask_record):
        act_name = subtask_record["function"]
        target_obj = subtask_record["object_names"][0]
        target_rec = subtask_record["receptacle_names"][0] if subtask_record["receptacle_names"] else None

        # [辅助函数] 核心类别提取 (方案一的关键)
        def get_category(name):
            if not name: return ""
            # 1. 去掉 rec_ 前缀
            name = name.replace("rec_", "")
            # 2. 去掉 :0000 这种后缀 (GT Handle 常见)
            name = name.split(':')[0]
            # 3. 去掉 _0, _1 这种数字后缀 (Semantic Name 常见)
            # 策略：如果名字里有下划线，且最后一部分是纯数字，就去掉
            parts = name.split('_')
            if len(parts) > 1 and parts[-1].isdigit():
                return "_".join(parts[:-1]).lower()
            return name.lower()

        action_to_pred = {
            "Place": ["is_on_top", "is_inside", "is_in_room"],
            "Rearrange": ["is_on_top", "is_inside", "is_in_room"],
            "Fill": ["is_filled"],
            "Clean": ["is_clean"],
            "Power": ["is_powered_on", "is_powered_off"],
            "Open": ["is_open"],
            "Close": ["is_closed"]
        }

        possible_props = []
        for key, vals in action_to_pred.items():
            if key in act_name:
                possible_props = vals
                break

        matched = False
        for prop in self.prop_list:
            if prop['function'] not in possible_props: continue

            # 翻译 GT Handles -> Semantic Names
            gt_obj_names = [self.handle_to_name.get(h, h) for h in prop['object_handles']]

            gt_rec_names = []
            gt_rooms = []
            if 'receptacle_handles' in prop and prop['receptacle_handles']:
                gt_rec_names = [self.handle_to_name.get(h, h) for h in prop['receptacle_handles']]
            if 'room_ids' in prop:
                gt_rooms = prop['room_ids']

            # --- A. 物体匹配 (类别级) ---
            obj_match = False
            for gt_name in gt_obj_names:
                # 只要类别一样，就视为匹配！
                if get_category(target_obj) == get_category(gt_name):
                    obj_match = True
                    break

            # --- B. 目标匹配 (容器/房间) ---
            target_match = True  # 默认 True (对于不需要目标的动作如 Fill)

            if "Place" in act_name:
                # 情况 1: GT 要求特定容器
                if gt_rec_names:
                    if not target_rec:
                        target_match = False
                    else:
                        rec_hit = False
                        for gt_name in gt_rec_names:
                            # 只要类别一样，就视为匹配！
                            if get_category(target_rec) == get_category(gt_name):
                                rec_hit = True
                                break
                        target_match = rec_hit

                # 情况 2: GT 要求特定房间 (Move 任务)
                elif gt_rooms:
                    # 如果物体对上了，且执行了 Place，我们模糊地认为是在尝试完成这个 Move 任务
                    target_match = True

                    # --- 最终判定 ---
            if obj_match and target_match:
                print(f"[Success] Action {act_name} matched GT {prop['function']} (Index {prop['idx']})!")
                matched = True

                subtask_record["matched_gt"] = True
                subtask_record["gt_prop_index"] = prop['idx']
                subtask_record["function"] = prop['function']

                # 几何验证 (可选，验证失败不影响记录)
                if self._verify_proposition(prop):
                    subtask_record["verified_success"] = True
                    prop['satisfied'] = True  # 标记 GT 为已完成
                else:
                    subtask_record["verified_success"] = False

                break

        if not matched:
            print(f"[Info] Recorded action (No GT match).")

        self.completed_subtasks.append(subtask_record)

    def finish_episode(self, end_step):
        if self.current_action_state:
            self._finalize_action(self.current_action_state, end_step)
        # 最后检查是否有遗留的挂起任务
        if self.pending_context:
            self._save_record(self.pending_context)
            self.pending_context = None

    def _verify_proposition(self, prop):
        try:
            func = getattr(SimBasedPredicates, prop['function'])
            kwargs = {
                "sim": self.sim,
                "object_handles": prop['object_handles'],
                "number": 1
            }
            if prop.get('receptacle_handles'):
                kwargs["receptacle_handles"] = prop['receptacle_handles']
            return func(**kwargs).is_satisfied
        except Exception:
            return False

    # def finish_episode(self, end_step):
    #     if self.current_action_state:
    #         self._finalize_action(self.current_action_state, end_step)


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


# [MODIFIED] Replaced list-based save with Streaming Video Writer logic
def init_video_writer(filepath, fps=RECORDING_FPS):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Using imageio for simpler streaming interface (no need to pre-allocate size)
    return imageio.get_writer(filepath, fps=fps, codec='libx264', format='FFMPEG', macro_block_size=1,
                              pixelformat='yuv420p', quality=8)


def append_frame(writer, frame_rgb):
    writer.append_data(frame_rgb)


def close_writer(writer):
    writer.close()


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
    # 深度递归查找当前正在执行的原子技能名称
    # 兼容 MotorSkillTool 封装、OracleRearrangeSkill (Dict结构) 和 CompoundSkill (List结构)
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
    # [增强版] 深度查找原子技能，并提取完整参数 args
    # 增加对 OraclePlaceSkill 属性的直接读取，防止 args 为 None
    default_result = {"action": "Idle", "target": "None", "args": None}
    if tool_instance is None:
        return default_result

    real_skill = getattr(tool_instance, "skill", tool_instance)

    # --- 情况 A: OracleRearrangeSkill (优先从 _skill_args 获取) ---
    if hasattr(real_skill, "skills") and isinstance(real_skill.skills, dict) and hasattr(real_skill, "active_skill"):
        active_enum = real_skill.active_skill
        current_args = None
        # 尝试获取字符串参数
        if hasattr(real_skill, "_skill_args") and active_enum in real_skill._skill_args:
            current_args = real_skill._skill_args[active_enum]

        # 递归获取子技能详情
        if active_enum in real_skill.skills:
            sub_skill = real_skill.skills[active_enum]
            sub_result = get_atomic_action_details(sub_skill)

            # 如果子技能没返回 args，就用当前层级的 args
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

    # [新增] 针对 OraclePlaceSkill 直接读取属性作为兜底
    # 即使 args 没传下来，也能构建出用于匹配的 args 字符串
    final_args = None
    if "Place" in action_name or "OraclePlace" in action_name:
        # 尝试读取内部属性 (参考 oracle_place_skill.py)
        obj = getattr(real_skill, "object_to_be_moved", None)
        rec = getattr(real_skill, "place_entity", None)
        if obj and rec:
            # 伪造一个符合格式的 args 字符串: "obj, on, rec"
            final_args = f"{obj.name}, on, {rec.name}"

    return {"action": action_name, "target": target_name, "args": final_args}


def get_high_level_action_details(tool_instance):
    # [修改版] 获取高级动作详情 (不进行递归深挖)
    # 目标：获取 Rearrange, Fill, Clean 等持续性动作，而不是底层的 Nav/Pick/Place
    default_result = {"action": "Idle", "target": "None", "args": None}
    if tool_instance is None:
        return default_result

    real_skill = getattr(tool_instance, "skill", tool_instance)

    # 获取技能名称
    action_name = "Unknown"
    if hasattr(real_skill, "config") and hasattr(real_skill.config, "name"):
        action_name = real_skill.config.name
    else:
        action_name = real_skill.__class__.__name__

    # 提取参数 (Args)
    # 对于 Rearrange/Fill/Clean 等高级技能，参数通常直接存储在 _skill_args 中
    # 或者我们需要解析它的目标
    final_args = None

    # 1. 尝试从 _skill_args 获取 (OracleRearrangeSkill 等常用)
    # 高级技能通常把参数存在某个特定的 key 下，或者直接是 _skill_args 的一部分
    if hasattr(real_skill, "_skill_args") and isinstance(real_skill._skill_args, dict):
        # Rearrange 技能通常把完整的参数字符串存在 'place' 键或者类似的键中
        # 或者它本身就缓存了当前的 args
        # 这是一个启发式搜索：找最长的那个字符串，通常就是完整指令
        for k, v in real_skill._skill_args.items():
            if isinstance(v, str) and "," in v:  # 包含逗号通常是完整参数 "obj, on, rec"
                final_args = v
                break
            if isinstance(v, str) and not final_args:  # 或者是物体名
                final_args = v

    if action_name == "Wait":
        if hasattr(real_skill, "config") and hasattr(real_skill.config, "step_action"):
            pass
        pass

    # 2. 如果没找到，尝试直接读取属性 (兜底)
    if not final_args:
        # 尝试读取 object_to_be_moved 等属性
        obj = getattr(real_skill, "object_to_be_moved", None)
        rec = getattr(real_skill, "place_entity", None)  # Rearrange/Place

        # 针对 Clean/Fill/Power 等
        if not obj:
            obj = getattr(real_skill, "interactive_object", None)

        if obj:
            obj_name = getattr(obj, "name", str(obj))
            if rec:
                rec_name = getattr(rec, "name", str(rec))
                final_args = f"{obj_name}, on, {rec_name}"
            else:
                final_args = obj_name

    # 过滤掉纯导航动作 (如果你只想看交互任务)
    # 如果你想保留导航作为子任务，可以注释掉下面这行
    if "Nav" in action_name and "Rearrange" not in action_name:
        # 如果当前不仅是导航，而是在执行更高级的任务，我们希望看到更高级的名字
        # 但在这里只能看到 Nav，说明 Planner 可能这一刻只暴露了 Nav
        # 不过通常 high-level tool 会保持 Rearrange
        pass

    return {"action": action_name, "target": "HighLevel", "args": final_args}


def annotate_agents_in_god_view(sim, rgb_img, human_pos, robot_pos):
    """
    鲁棒版上帝视角标注工具 (修复底层 Sensor 获取与解析逻辑)
    """
    import numpy as np
    import cv2

    annotated_img = rgb_img.copy()
    try:
        # =================================================================
        # 1. 严格从底层 habitat_sim 获取物理 Sensor
        # =================================================================
        sensor = None
        if hasattr(sim, 'agents') and len(sim.agents) > 0:
            agent_sensors = sim.agents[0]._sensors
            for k, v in agent_sensors.items():
                if "god_rgb" in k:
                    sensor = v
                    break

        if not sensor:
            print("[God View Annotator] Warning: Cannot find 'god_rgb' sensor in sim.agents[0]._sensors.")
            return annotated_img

        # =================================================================
        # 2. 解析相机内参与外参 (万无一失版)
        # =================================================================
        # 直接使用输入图像的真实分辨率，避免底层传回错乱
        H, W = rgb_img.shape[:2]

        # 兼容 Habitat 不同版本 (有的是方法，有的是属性)
        try:
            spec = sensor.specification() if callable(sensor.specification) else sensor.specification
            hfov_rad = np.deg2rad(float(spec.hfov))
        except Exception as e:
            # 如果极端情况拿不到，默认 90 度视场角
            hfov_rad = np.deg2rad(90.0)

        fx = (W / 2.0) / np.tan(hfov_rad / 2.0)
        fy = fx
        cx = W / 2.0
        cy = H / 2.0

        # 获取相机的 World-to-Camera 矩阵
        T_world_cam = np.array(sensor.render_camera.camera_matrix)

        def project_3d_to_2d(p_world):
            # 齐次坐标转换
            p_cam = T_world_cam @ np.array([p_world[0], p_world[1], p_world[2], 1.0])
            z_depth = -p_cam[2]
            if z_depth < 0.1: return None  # 在相机背后

            u = fx * (p_cam[0] / z_depth) + cx
            v = cy - (fy * (p_cam[1] / z_depth))
            return int(u), int(v)

        # =================================================================
        # 3. 绘制人类与机器人 (带高度偏移补偿)
        # =================================================================
        # 人类高度补偿 1.65米，投影在头顶
        human_head_pos = np.array(human_pos) + np.array([0.0, 1.65, 0.0])
        human_uv = project_3d_to_2d(human_head_pos)

        if human_uv and 0 <= human_uv[0] <= W and 0 <= human_uv[1] <= H:
            cv2.circle(annotated_img, human_uv, radius=12, color=(255, 50, 50), thickness=-1)
            cv2.circle(annotated_img, human_uv, radius=12, color=(255, 255, 255), thickness=2)
            text_pos = (human_uv[0] + 15, human_uv[1] + 5)
            cv2.putText(annotated_img, "Human", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(annotated_img, "Human", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)

        # 机器人高度补偿 0.5米，投影在背部
        robot_back_pos = np.array(robot_pos) + np.array([0.0, 0.5, 0.0])
        robot_uv = project_3d_to_2d(robot_back_pos)

        if robot_uv and 0 <= robot_uv[0] <= W and 0 <= robot_uv[1] <= H:
            cv2.circle(annotated_img, robot_uv, radius=12, color=(50, 255, 50), thickness=-1)
            cv2.circle(annotated_img, robot_uv, radius=12, color=(255, 255, 255), thickness=2)
            text_pos = (robot_uv[0] + 15, robot_uv[1] + 5)
            cv2.putText(annotated_img, "Robot", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(annotated_img, "Robot", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

    except Exception as e:
        import traceback
        print(f"[God View Annotator] Error: {e}")
        traceback.print_exc()  # 如果还有报错，打印出具体哪一行出错

    return annotated_img

def main():
    obs_sensors = ["agent_0_third_rgb",
                   "agent_0_god_rgb",
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
    with open_dict(config):
        agent0_cfg = config.habitat.simulator.agents.agent_0
        template_sensor = None
        template_sensor = agent0_cfg.sim_sensors.third_rgb_sensor
        print(f"[Config] Human Follower Cam set")
        god_sensor_config = template_sensor.copy()
        god_sensor_config.uuid = "god_rgb"
        god_sensor_config.position = [0.0, 15.0, 0.0]
        god_sensor_config.orientation = [-1.57, 0.0, 0.0]
        agent0_cfg.sim_sensors["god_rgb"] = god_sensor_config
        print(f"[Config] Global View Cam Set")
    config = setup_config(config, 0)
    with open_dict(config):
        config.habitat.environment.max_episode_steps = 500000
        config.habitat.environment.max_episode_seconds = 50000000
    dataset = habitat.make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)
    print("Initializing EnvironmentInterface...")
    env_interface = setup_env(config)
    sim = env_interface.sim
    human_agent = sim.agents_mgr[0]
    cameraman = SpotCameraman(sim, robot_agent_uid=1, human_agent_uid=0)
    print("Initializing Planner...")
    gt_plan_config = OmegaConf.create({"type": "GroundTruthPlanner"})
    planner = GroundTruthPlanner(gt_plan_config, env_interface)
    planner.reset()
    # llm_planner_conf = config.evaluation.planner
    # planner = instantiate(llm_planner_conf)
    # planner = planner(env_interface=env_interface)
    agent_config = config.evaluation.agents
    planner.agents = init_agents(agent_config, env_interface)

    total_episodes = len(dataset.episodes)
    print(f"Total episodes to process: {total_episodes}")
    max_num_steps = 400000
    ego_key = "agent_1_articulated_agent_jaw_rgb"
    third_key = "agent_0_third_rgb"
    global_key = "agent_0_god_rgb"
    # ego_raw_save_dir = "final_raw_videos/ego_raw_videos"
    # ego_ann_raw_save_dir = "final_raw_videos/ego_ann_raw_videos"
    # third_raw_save_dir = "final_raw_videos/third_raw_videos"
    # third_ann_raw_save_dir = "final_raw_videos/third_ann_raw_videos"
    # global_save_dir = "final_raw_videos/global_videos"
    # metadata_save_dir = "final_raw_videos/metadata"
    ego_raw_save_dir = "final_raw_videos/ego_raw_videos"
    ego_ann_raw_save_dir = "final_raw_videos/ego_ann_raw_videos"
    third_raw_save_dir = "final_raw_videos/third_raw_videos"
    third_ann_raw_save_dir = "final_raw_videos/third_ann_raw_videos"
    global_save_dir = "final_raw_videos/global_videos"
    global_ann_save_dir = "final_raw_videos/global_ann_videos"
    metadata_save_dir = "final_raw_videos/metadata"
    os.makedirs(metadata_save_dir, exist_ok=True)
    os.makedirs(global_ann_save_dir, exist_ok=True)
    ERROR_EPISODES = {}

    # for i in range(2):
    #     # for i in range(2):
    for i in range(total_episodes):  # Modified to run full loop
        current_episode = env_interface.env.env.env._env.current_episode
        current_episode_id = current_episode.episode_id

        # [MODIFIED] Only process specific episodes if needed (for test), otherwise run all
        # To run specific test mini, uncomment below
        # if current_episode_id not in ["0", "353", "357", "358", "832", "308"]:
        #     env_interface.reset_environment()
        #     continue

        current_instruction = current_episode.instruction
        current_scene_id = current_episode.scene_id
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
        print(f"Mapped {len(name_to_handle_map) - count_of_objects} receptacles")

        subtask_recorder = ActionBasedSubtaskRecorder(env_interface, prop_list, sim, handle_to_name_map)

        human_agent_sim = sim.agents_mgr[0]
        human_articulated_agent = human_agent_sim.articulated_agent
        # 检查一下resolution和hfov是否和传感器一致
        third_annotator = ProjectionAnnotator(sim, "agent_0_third_rgb")
        ego_annotator = ProjectionAnnotator(sim, "agent_1_articulated_agent_jaw_rgb")
        human_sensor_rig = env_interface.sim.agents[0]
        visibility_checker = FurnitureVisibilityChecker(sim, "agent_1_articulated_agent_jaw_rgb")

        # [MODIFIED] Initialize Streaming Writers
        writer_ego = init_video_writer(f"{ego_raw_save_dir}/{current_episode_id}.mp4")
        writer_ego_ann = init_video_writer(f"{ego_ann_raw_save_dir}/{current_episode_id}.mp4")
        writer_3rd = init_video_writer(f"{third_raw_save_dir}/{current_episode_id}.mp4")
        writer_3rd_ann = init_video_writer(f"{third_ann_raw_save_dir}/{current_episode_id}.mp4")
        writer_global = init_video_writer(f"{global_save_dir}/{current_episode_id}.mp4")
        writer_global_ann = init_video_writer(f"{global_ann_save_dir}/{current_episode_id}.mp4")

        task_done = False
        cstep = 0
        cameraman.reset()
        planner.reset()
        #--------修改god_camera位置---------
        # pathfinder = env_interface.sim.pathfinder
        # lower_bound, upper_bound = pathfinder.get_bounds()
        # center_x = (lower_bound[0]+upper_bound[0]) / 2.0
        # center_z = (lower_bound[1]+upper_bound[1]) / 2.0
        # scene_width = upper_bound[0]-lower_bound[0]
        # scene_length = upper_bound[1]-lower_bound[1]
        # max_dim = max(scene_width, scene_length)
        # optimal_height = max((max_dim / 2.0)*1.1, 12.0)
        # global_cam_pos = mn.Vector3(center_x, upper_bound[1]+optimal_height, center_z)
        # global_cam_rot = mn.Quaternion.rotation(mn.Rad(-math.pi/2), mn.Vector3(1.0, 0.0, 0.0))
        # target_global_transform = mn.Matrix4.from_(global_cam_rot.to_matrix(), global_cam_pos)
        # agent_node = env_interface.sim.agents[0].scene_node
        # agent_global_transform_inv = agent_node.absolute_transformation().inverted()
        # target_local_transform = agent_global_transform_inv @ target_global_transform
        # god_sensor = env_interface.sim.agents[0]._sensors.get("agent_0_god_rgb")
        # if god_sensor is not None:
        #     # 这一步是关键：直接修改底层 Magnum Node 的矩阵
        #     god_sensor.node.transformation = target_local_transform
        #     print("Change success!")
        # else:
        #     print("Can't find god_rgb")
        pathfinder = env_interface.sim.pathfinder
        lower_bound, upper_bound = pathfinder.get_bounds()
        
        center_x = (lower_bound[0] + upper_bound[0]) / 2.0
        center_z = (lower_bound[2] + upper_bound[2]) / 2.0
        
        scene_width = upper_bound[0] - lower_bound[0]
        scene_length = upper_bound[2] - lower_bound[2]
        max_dim = max(scene_width, scene_length)
        
        # 增加一点留白系数，并提高保底高度，确保视野更广阔
        optimal_height = max(max_dim * 0.6, 15.0) 
        
        # 获取相机的目标世界绝对坐标和旋转
        global_cam_pos = mn.Vector3(center_x, upper_bound[1] + optimal_height, center_z)
        global_cam_rot = mn.Quaternion.rotation(mn.Rad(-math.pi/2), mn.Vector3(1.0, 0.0, 0.0))
        target_global_transform = mn.Matrix4.from_(global_cam_rot.to_matrix(), global_cam_pos)
        
        # 【核心修正】：不要用 agent_node，而是获取相机的真实父节点
        god_sensor = env_interface.sim.agents[0]._sensors.get("agent_0_god_rgb")
        if god_sensor is not None:
            # 1. 拿到相机真实的直属上级节点
            parent_node = god_sensor.node.parent
            # 2. 获取该父节点在世界空间中的逆矩阵
            parent_global_transform_inv = parent_node.absolute_transformation().inverted()
            
            # 3. 计算出完美剔除所有层级偏移的局部矩阵
            target_local_transform = parent_global_transform_inv @ target_global_transform
            
            # 4. 覆写
            god_sensor.node.transformation = target_local_transform
            
            print(f"[God View] 成功接管！绝对坐标: X={center_x:.2f}, Y={global_cam_pos.y:.2f}, Z={center_z:.2f}")
        else:
            print("[God View] 严重警告：找不到 agent_0_god_rgb 传感器！")
        #--------------end-----------------
        observations = env_interface.get_observations()
        print(f"Spot Start Observing: {cameraman.start_step}")
        episode_metadata = []

        try:
            while not task_done and cstep < max_num_steps:
                low_level_actions, planner_info, task_done = planner.get_next_action(
                    current_instruction, observations, env_interface.world_graph
                )
                # if planner.current_action_tuple and "SET_CLEAN" in str(planner.current_action_tuple[1]):
                #     target_handle = planner.current_action_tuple[1].split(":", 1)[1]
                #     obj_node = env_interface.full_world_graph.get_node_from_sim_handle(target_handle)
                #     if obj_node:
                #         obj_node.properties["states"]["is_clean"] = True
                #     obj_states = sim.object_state_machine.objects_with_states.get(target_handle)
                #     found_state = False
                #     if obj_states:
                #         for state_instance in obj_states:
                #             if state_instance.name == "is_clean":
                #                 state_instance.value = True
                #                 found_state = True
                #                 break
                #     if not found_state:
                #         print(f"[Internal Patch Error] Object {target_handle} does not have is_clean")
                #     print(f"[Internal Patch] Object {target_handle} status forced to CLEAN.")
                #     agent = planner.agents[0]
                #     wait_tool = agent.tools.get("Wait")
                #     skill = wait_tool.skill
                #     skill.steps_elapsed += (skill.step_threshold - 50)
                #     print(f"Forced steps_elapsed max.")
                if planner.current_action_tuple and "SET_CLEAN" in str(planner.current_action_tuple[1]):
                    target_handle = planner.current_action_tuple[1].split(":", 1)[1]
                    agent = planner.agents[0]
                    wait_tool = agent.tools.get("Wait")
                    skill = wait_tool.skill

                    # 核心修正：只有在刚开始执行的第 1 步，才进行“快进”和“状态修改”
                    # 这样能避免第 2 步到第 50 步期间重复累加 steps_elapsed 导致瞬间结束
                    if skill.steps_elapsed[0].item() == 1:
                        obj_node = env_interface.full_world_graph.get_node_from_sim_handle(target_handle)
                        if obj_node:
                            obj_node.properties["states"]["is_clean"] = True
                        obj_states = sim.object_state_machine.objects_with_states.get(target_handle)
                        found_state = False
                        if obj_states:
                            for state_instance in obj_states:
                                if state_instance.name == "is_clean":
                                    state_instance.value = True
                                    found_state = True
                                    break
                        if not found_state:
                            print(f"[Internal Patch Error] Object {target_handle} does not have is_clean")

                        print(f"[Internal Patch] Object {target_handle} status forced to CLEAN.")

                        # 使用你的原版思路：将已用步数直接快进到距离结束只剩 50 步
                        skill.steps_elapsed += (skill.step_threshold - 50)
                        print(f"[Internal Patch] Fast-forwarded steps_elapsed. Human will now wait for 50 steps.")

                if planner.current_action_tuple and "SET_FILL" in str(planner.current_action_tuple[1]):
                    target_handle = planner.current_action_tuple[1].split(":", 1)[1]
                    agent = planner.agents[0]
                    wait_tool = agent.tools.get("Wait")
                    skill = wait_tool.skill

                    # 核心修正：只有在刚开始执行的第 1 步，才进行“快进”和“状态修改”
                    # 这样能避免第 2 步到第 50 步期间重复累加 steps_elapsed 导致瞬间结束
                    if skill.steps_elapsed[0].item() == 1:
                        obj_node = env_interface.full_world_graph.get_node_from_sim_handle(target_handle)
                        if obj_node:
                            obj_node.properties["states"]["is_filled"] = True
                        obj_states = sim.object_state_machine.objects_with_states.get(target_handle)
                        found_state = False
                        if obj_states:
                            for state_instance in obj_states:
                                if state_instance.name == "is_filled":
                                    state_instance.value = True
                                    found_state = True
                                    break
                        if not found_state:
                            print(f"[Internal Patch Error] Object {target_handle} does not have is_filled")

                        print(f"[Internal Patch] Object {target_handle} status forced to FILLED.")

                        # 使用你的原版思路：将已用步数直接快进到距离结束只剩 50 步
                        skill.steps_elapsed += (skill.step_threshold - 50)
                        print(f"[Internal Patch] Fast-forwarded steps_elapsed. Human will now wait for 50 steps.")

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
                    high_level_details = get_high_level_action_details(human_tool_instance)
                    if high_level_action == "Wait" and planner.current_action_tuple:
                        if planner.current_action_tuple[0] == "Wait":
                            high_level_details["args"] = planner.current_action_tuple[1]
                    atomic_details = get_atomic_action_details(human_tool_instance)
                    subtask_recorder.step(cstep, high_level_details)
                    held_obj_name = None
                    is_holding = human_agent_sim.grasp_mgr.is_grasped
                    if is_holding:
                        held_obj_name = f"obj_{human_agent_sim.grasp_mgr.snap_idx}"
                    world_snapshot = get_world_state_snapshot(env_interface)

                    robot_state = sim.get_agent_state(1)
                    human_state = sim.get_agent_state(0)
                    # 判断人类是否穿模
                    human_pos = sim.get_agent_state(0).position
                    pathfinder = env_interface.sim.pathfinder
                    snapped_pos = pathfinder.snap_point(human_pos)
                    dist_to_obs = pathfinder.distance_to_closest_obstacle(snapped_pos)
                    is_navigable = pathfinder.is_navigable(human_pos)

                    # import magnum as mn
                    all_furnitures = env_interface.full_world_graph.get_all_furnitures()
                    furniture_info_list = []
                    human_pos_np = np.array(human_pos)
                    h_w, h_x, h_y, h_z = human_state.rotation.components
                    human_rot_mn = mn.Quaternion(mn.Vector3(h_x, h_y, h_z), h_w)
                    human_forward = human_rot_mn.transform_vector(mn.Vector3(0, 0, -1))
                    human_forward_2d = np.array([human_forward.x, 0.0, human_forward.z])
                    if np.linalg.norm(human_forward_2d)>0:
                        human_forward_2d = human_forward_2d/np.linalg.norm(human_forward_2d)
                    # robot_sensor_state = robot_state.sensor_states.get("agent_1_articulated_agent_jaw_rgb")
                    # robot_cam_pos = np.array(robot_state.position)
                    # r_w, r_x, r_y, r_z = robot_state.rotation.components
                    # robot_rot_mn = mn.Quaternion(mn.Vector3(r_x, r_y, r_z), r_w)
                    # robot_forward = robot_rot_mn.transform_vector(mn.Vector3(0, 0, -1))
                    # robot_forward_3d = np.array([robot_forward.x, robot_forward.y, robot_forward.z])
                    # if np.linalg.norm(robot_forward_3d)>0:
                    #     robot_forward_3d = robot_forward_3d/np.linalg.norm(robot_forward_3d)
                    visible_furniture_handles = visibility_checker.get_visible_furniture_handles(env_interface.full_world_graph)
                    for fur in all_furnitures:
                        fur_pos = fur.properties.get("translation", None)
                        if fur_pos is None:
                            continue
                        fur_pos_np = np.array(fur_pos)
                        fur_handle = getattr(fur, "sim_handle", None)or fur.properties.get("handle", "")
                        human_to_fur = human_pos_np - np.array(fur_pos)
                        human_to_fur_dist_array = human_to_fur
                        human_to_fur_dist_array[1] = 0.0
                        dist = float(np.linalg.norm(human_to_fur_dist_array))
                        has_faucet = "components" in fur.properties and "faucet" in fur.properties["components"]
                        is_human_facing = False
                        dot_human = 0
                        if 0.1< dist < 5.0:
                            human_to_fur_dir = human_to_fur_dist_array/dist
                            dot_human = np.dot(human_forward_2d, human_to_fur_dir)
                            if dot_human>0.707:
                                is_human_facing = True
                        # is_visible_to_robot = False
                        # robot_to_fur = fur_pos_np - robot_cam_pos
                        # dist_to_robot = float(np.linalg.norm(robot_to_fur))
                        # if 0.1<dist_to_robot<8.0:
                        #     robot_to_fur_dir = robot_to_fur / dist_to_robot
                        #     dot_robot = np.dot(robot_forward_3d, robot_to_fur_dir)
                        #     if dot_robot > 0.707:
                        #         is_visible_to_robot = True
                        is_visible_to_robot = (fur_handle in visible_furniture_handles)

                        furniture_info_list.append({
                            "name": fur.name,
                            "sim_handle": getattr(fur, "sim_handle", None) or fur.properties.get("handle", ""),
                            "position": list(fur_pos),
                            "distance": dist,
                            "has_faucet": has_faucet,
                            "is_human_facing": is_human_facing,
                            "dot_human": dot_human,
                            "is_visible_to_robot": is_visible_to_robot,
                            # "dot_robot": dot_robot
                        })
                    frame_record = {
                        "step": cstep,
                        "time_sec": cstep / RECORDING_FPS,
                        "robot_active": cameraman.is_active,
                        "human_agent": {
                            "action": high_level_action,  # 之前为human_tool
                            "atomic_action": atomic_details["action"],
                            "action_args": atomic_details["args"],
                            "action_target": atomic_details["target"],
                            "is_holding": is_holding,
                            "held_obj": held_obj_name,
                            "pos": list(human_state.position),
                            "rot": list(human_state.rotation.components),
                            "dist_to_obs": float(dist_to_obs),
                            "is_navigable": bool(is_navigable)
                        },
                        "robot_agent": {
                            "pos": list(robot_state.position),
                            "rot": list(robot_state.rotation.components)
                        },
                        "furnitures": furniture_info_list,
                        "world_objects": world_snapshot
                    }
                    episode_metadata.append(frame_record)

                    observations, reward, done, info = env_interface.step(low_level_actions)

                    third_rgb = observations[third_key][:, :, :3].astype(np.uint8)
                    third_depth = observations.get("agent_0_third_depth", None)
                    if third_depth is not None and len(third_depth.shape) == 3:
                        third_depth = third_depth[:, :, 0]
                    ann_third = third_annotator.annotate_frame(third_rgb, env_interface.full_world_graph)
                    ego_rgb = observations[ego_key][:, :, :3].astype(np.uint8)
                    ego_depth = observations.get("agent_1_articulated_agent_jaw_depth", None)
                    if ego_depth is not None and len(ego_depth.shape) == 3:
                        ego_depth = ego_depth[:, :, 0]
                    ann_ego = ego_annotator.annotate_frame(ego_rgb, env_interface.full_world_graph)
                    global_rgb = observations[global_key][:, :, :3].astype(np.uint8)
                    ann_global = annotate_agents_in_god_view(
                        sim=sim,
                        rgb_img=global_rgb,
                        human_pos=human_state.position,
                        robot_pos=robot_state.position
                    )

                    # [MODIFIED] Stream Write Frames
                    append_frame(writer_ego, observations[ego_key].astype(np.uint8))
                    append_frame(writer_ego_ann, ann_ego)
                    append_frame(writer_3rd, observations[third_key].astype(np.uint8))
                    append_frame(writer_3rd_ann, ann_third)
                    append_frame(writer_global, observations[global_key].astype(np.uint8))
                    append_frame(writer_global_ann, ann_global)

                    cstep += 1

                    # Periodic manual cleanup if needed (Optional)
                    # if cstep % 1000 == 0:
                    #     gc.collect()

        finally:
            # [MODIFIED] Ensure writers are closed even if crash/break
            close_writer(writer_ego)
            close_writer(writer_ego_ann)
            close_writer(writer_3rd)
            close_writer(writer_3rd_ann)
            close_writer(writer_global)
            close_writer(writer_global_ann)

        if len(planner.ERROR) > 0:
            ERROR_EPISODES[current_episode_id] = planner.ERROR

        # episode_metadata_record["steps"] = episode_metadata
        subtask_recorder.finish_episode(cstep)
        episode_metadata_record = {
            "episode_id": current_episode_id,
            "scene_id": current_scene_id,
            "instruction": current_instruction,
            "recording": {
                "fps": RECORDING_FPS,
                "frame_count": len(episode_metadata),
                "frame_index_semantics": (
                    "steps[i] and frame i are emitted in the same loop iteration"
                ),
                "capture_phase": {
                    "metadata": "before env_interface.step(low_level_actions)",
                    "video_frame": "observation returned by env_interface.step(low_level_actions)",
                },
                "views": {
                    "ego_raw": f"{ego_raw_save_dir}/{current_episode_id}.mp4",
                    "ego_ann": f"{ego_ann_raw_save_dir}/{current_episode_id}.mp4",
                    "third_raw": f"{third_raw_save_dir}/{current_episode_id}.mp4",
                    "third_ann": f"{third_ann_raw_save_dir}/{current_episode_id}.mp4",
                    "global": f"{global_save_dir}/{current_episode_id}.mp4",
                    "global_ann": f"{global_ann_save_dir}/{current_episode_id}.mp4",
                },
            },
            "object_and_receptacle_handle_mapping": name_to_handle_map,
            "subtasks": subtask_recorder.completed_subtasks,
            "evaluation_propositions": prop_list,
            "steps": episode_metadata
        }
        meta_path = os.path.join(metadata_save_dir, f"{current_episode_id}.json")
        with open(meta_path, "w") as f:
            json.dump(episode_metadata_record, f, indent=2)
        print(f"Saved metadata to {meta_path}")
        # [MODIFIED] Removed save_video calls as videos are already saved streamingly
        print(f"Episode {current_episode_id} End cstep: {cstep}")
        print(f"Episode {current_episode_id} End task_done: {task_done}")
        print(f"----- Saved Videos of Episode {current_episode_id} -----")
        if i < total_episodes - 1:
            env_interface.reset_environment()
            cameraman.reset()
            sim = env_interface.sim
            planner.reset()
    print(
        f"TOTAL ERROR EPISODES COUNT: {len(ERROR_EPISODES.keys())}\nERROR EPISODES:\n{ERROR_EPISODES.keys()}\n ERROR INFO:\n{ERROR_EPISODES}")


if __name__ == "__main__":
    main()
