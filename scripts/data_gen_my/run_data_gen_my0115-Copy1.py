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
from omegaconf import OmegaConf, open_dict

# from habitat_llm.tests.test_planner import DATASET_OVERRIDES

# Habitat & DCX 核心导入
import habitat
# 只需要导入 dataset 以触发注册，不需要手动干预
import habitat_llm.agent.env.dataset
from habitat_llm.agent.env import EnvironmentInterface, register_sensors, register_actions, register_measures
from habitat_llm.utils import fix_config, setup_config
from habitat_llm.utils.sim import init_agents
from habitat_llm.agent.env.actions import find_action_range
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
    # "habitat.simulator.agents.agent_0.articulated_agent_type=KinematicHumanoid",
    # "habitat.simulator.agents.agent_0.articulated_agent_urdf=data/humanoids/humanoid_data/female_0/female_0.urdf",
    # "habitat.simulator.agents.agent_0.motion_data_path=data/humanoids/humanoid_data/female_0/female_0_motion_data_smplx.pkl",
    # "habitat.simulator.agents.agent_1.articulated_agent_type=SpotRobot",
    # "habitat.simulator.agents.agent_1.articulated_agent_urdf=data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf",
    ##-------habitat.gym.obs_keys部分可以后续再看一下怎么修改，涉及传感器定义----------------
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

    # for i in range(total_episodes):
    for i in range(5):
        current_episode_id = env_interface.env.env.env._env.current_episode.episode_id
        current_instruction = env_interface.env.env.env._env.current_episode.instruction
        print(f"----- Now processing Episode {current_episode_id} -----")
        print(f"The Instruction is '{current_instruction}'.")
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
        print(f"Start Observation Time: {cameraman.start_step}")
        while not task_done and cstep < max_num_steps:
            # print(f"Planner type: {type(planner)}")
            low_level_actions, planner_info, task_done = planner.get_next_action(
                current_instruction, observations, env_interface.world_graph
            )

            # print(f"Above low_level_actions keys: {low_level_actions.keys()}")
            # print(f"Above low_level_actions values: {low_level_actions.values()}")
            if "responses" in planner_info and any(planner_info["responses"].values()):
                print(f"\n[CRITICAL DEBUG] Agent Responses: {planner_info['responses']}\n")
            if low_level_actions:
                # print("action space: ", env_interface.orig_action_space)
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
                observations, reward, done, info = env_interface.step(low_level_actions)
                # cstep += 1
                # print(f"observation_keys: {observations.keys()}")
                # print(f"type of observations[]: {type(observations[ego_key])}")
                video_ego_frames.append(observations[ego_key].astype(np.uint8))
                video_3rd_frames.append(observations[third_key].astype(np.uint8))
                cstep += 1
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