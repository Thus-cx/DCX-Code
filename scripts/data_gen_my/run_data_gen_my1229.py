import sys
import os
import cv2
import numpy as np
import logging
from hydra import compose, initialize
from omegaconf import OmegaConf, open_dict

# 1. 路径设置
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from spot_follower_my import SpotFollower

# Habitat & DCX 核心导入
import habitat
# 只需要导入 dataset 以触发注册，不需要手动干预
import habitat_llm.agent.env.dataset
from habitat_llm.agent.env import EnvironmentInterface
from habitat_llm.agent.env import sensors 
from habitat_llm.planner.centralized_llm_planner import CentralizedLLMPlanner

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    # --- 2. 初始化配置 (Standard Hydra Way) ---
    config_rel_path = "../../habitat_llm/conf" 
    # overrides = ["+habitat.task.actions.agent_1_arm_action.num_joints=7"]
    with initialize(config_path=config_rel_path, version_base=None):
        cfg = compose(config_name="examples/data_gen_config_my1229")
    cfg.habitat.simulator.agents_order = ["agent_0","agent_1"]
    # 注册传感器
    sensors.register_sensors(cfg)
    
    # 双重保险：确保 Dataset 类型正确 (防止 BaseEpisode 问题)
    # 这虽然是手动干预，但属于 Hydra 在 Notebook/Script 环境下的已知局限，保留它是安全的
    with open_dict(cfg):
        if "habitat" in cfg:
            cfg.habitat.dataset.type = "CollaborationDataset-v0"
        if "agent_1_arm_action" in cfg.habitat.task.actions:
            cfg.habitat.task.actions.agent_1_arm_action.num_joints = 7

    print("Configuration loaded.")

    # --- 3. 标准化加载 Dataset ---
    print(f"Loading dataset: {cfg.habitat.dataset.data_path}")
    dataset = habitat.make_dataset(
        id_dataset=cfg.habitat.dataset.type,
        config=cfg.habitat.dataset
    )

    # --- 4. 标准化初始化环境 ---
    # 此时 Config 中已经包含了 agents.agent_0 的定义，EnvironmentInterface 会自动处理好一切
    print("Initializing EnvironmentInterface...")
    env_interface = EnvironmentInterface(cfg, dataset=dataset, init_wg=True, init_env=True)
    sim = env_interface.sim
    
    # --- 5. 标准化初始化 Planner ---
    # 此时 env_interface.agents 中已经是的高级 Agent 对象，Planner 能够直接识别
    print("Initializing Planner...")
    planner = CentralizedLLMPlanner(cfg.planner, env_interface)
    
    # 验证一下 (Optional Debug)
    if hasattr(planner, '_agents') and planner._agents:
        print(f"Planner initialized successfully with agents: {[a.uid for a in planner._agents]}")
    else:
        raise RuntimeError("Planner failed to bind agents from Config! Check YAML 'planner.agents' and 'defaults'.")

    # --- 6. 数据生成循环 ---
    total_episodes = len(dataset.episodes)
    print(f"Total episodes to process: {total_episodes}")

    for i in range(total_episodes):
        try:
            observations = env_interface.reset_environment()
        except StopIteration:
            break
            
        episode_id = env_interface.env.env.env._env.current_episode.episode_id
        print(f"--- Starting Episode ID: {episode_id} ---")
        
        # 获取 Instruction
        instruction_text = env_interface.env.env.env._env.current_episode.instruction

        print(f"Instruction: {instruction_text}")
        
        # 获取图谱
        world_graph = env_interface.world_graph
        
        # Reset Planner (现在这是一个合法的操作)
        planner.reset()
        
        # Spot Follower (依然是脚本控制)
        human_agent_sim = sim.agents_mgr[0]
        spot_agent_sim = sim.agents_mgr[1]
        spot_follower = SpotFollower(spot_agent_sim, human_agent_sim)
        
        video_ego_frames = []
        video_3rd_frames = []
        
        done = False
        step_count = 0
        max_steps = 200
        
        while not done and step_count < max_steps:
            # --- A. 规划 (Planner) ---
            # 此时 world_graph['agent_0'] 必然存在，因为 EnvironmentInterface 自动注册了它
            try:
                high_level_action = planner.get_next_action(
                    instruction_text, 
                    observations, 
                    world_graph
                )
            except Exception as e:
                print(f"Planner error: {e}")
                high_level_action = {"action": "wait"}

            # --- B. Spot (小脑) ---
            spot_vel = spot_follower.get_action()
            
            # --- C. 执行 ---
            step_action = {}
            if high_level_action:
                step_action["agent_0"] = high_level_action
            
            step_action["agent_1"] = {
                "action": "agent_1_base_velocity", 
                "action_args": {
                    "linear_velocity": float(spot_vel[0]), 
                    "angular_velocity": float(spot_vel[1])
                }
            }

            try:
                observations = env_interface.step(step_action)
                step_count += 1
            except Exception as e:
                print(f"Step failed: {e}")
                break
            
            # --- D. 记录 ---
            ego_key = "agent_1_jaw_rgb"
            third_key = "agent_0_third_person_rgb"
            
            if ego_key in observations:
                video_ego_frames.append(observations[ego_key].cpu().numpy().astype(np.uint8))
            if third_key in observations:
                video_3rd_frames.append(observations[third_key].cpu().numpy().astype(np.uint8))

            if observations.get('task_is_done', False):
                done = True

        save_dir = f"output_dataset/episode_{episode_id}"
        save_video(video_ego_frames, f"{save_dir}/robot_ego.mp4")
        save_video(video_3rd_frames, f"{save_dir}/third_person.mp4")

if __name__ == "__main__":
    main()