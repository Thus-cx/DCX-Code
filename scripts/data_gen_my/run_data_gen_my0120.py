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

    # 2. 解包 MotorSkillTool
    # Partnr 的工具通常被 MotorSkillTool 包裹，真实的技能逻辑在 .skill 属性中
    real_skill = getattr(tool_instance, "skill", tool_instance)

    # 3. 处理 OracleRearrangeSkill (字典结构 + Enum索引)
    # 特征：skills 是 dict，active_skill 是 Enum
    if hasattr(real_skill, "skills") and isinstance(real_skill.skills, dict) and hasattr(real_skill, "active_skill"):
        active_enum = real_skill.active_skill
        # 确保枚举值在字典中 (防止初始化时状态不一致)
        if active_enum in real_skill.skills:
            sub_skill = real_skill.skills[active_enum]
            return get_atomic_action_name(sub_skill)

    # 4. 处理 CompoundSkill (列表结构 + Int索引)
    # 特征：skills 是 list，active_skill 是 int
    if hasattr(real_skill, "skills") and isinstance(real_skill.skills, list) and hasattr(real_skill, "active_skill"):
        idx = real_skill.active_skill
        # 确保索引越界检查
        if isinstance(idx, int) and 0 <= idx < len(real_skill.skills):
            sub_skill = real_skill.skills[idx]
            return get_atomic_action_name(sub_skill)

    # 5. 基础情况：到达叶子节点 (原子技能)
    # 优先尝试获取配置中的名称 (通常更易读，如 "Nav", "Pick")
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
        
        # 1. [新增] 提取当前层级的完整参数
        # PARTNR 的技能通常在这里存储 {'object':..., 'receptacle':...} 或类似字符串
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
        current_episode_id = env_interface.env.env.env._env.current_episode.episode_id
        current_instruction = env_interface.env.env.env._env.current_episode.instruction
        print(f"----- Now processing Episode {current_episode_id} -----")
        print(f"The Instruction is '{current_instruction}'.")
        #----record metadata map-----
        name_to_handle_map = {}
        wg = env_interface.full_world_graph
        all_graph_objects = wg.get_all_objects()
        for obj in all_graph_objects:
            semantic_name = obj.name
            handle = getattr(obj, "sim_handle", None)
            if not handle and hasattr(obj, "properties"):
                handle = obj.properties.get("handle")
            if handle:
                name_to_handle_map[semantic_name] = handle
        print(f"Mapped {len(name_to_handle_map)} objects")
        #--------end-----------------
        
        human_agent_sim = sim.agents_mgr[0]
        spot_robot_sim = sim.agents_mgr[1]
        # ------DEBUG-----
        human_agent = human_agent_sim.articulated_agent
        print(f"Human agent type: {human_agent.__class__.__name__}")
        video_ego_frames = []
        video_3rd_frames = []
        task_done = False
        cstep = 0
        cameraman.reset()
        planner.reset()
        observations = env_interface.get_observations()
        print(f"Spot Start Observing: {cameraman.start_step}")
        episode_metadata = []
        episode_metadata_record = {
            "instruction": current_instruction,
            "object_handle_mapping": name_to_handle_map,
            "steps": None
        }
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
        episode_metadata_record["steps"] = episode_metadata
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