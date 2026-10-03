import sys
import os
import cv2
import numpy as np
import logging
from hydra import compose, initialize
from omegaconf import OmegaConf, open_dict

# --- 1. 路径与导入设置 ---
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入本地 SpotFollower
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from spot_follower_my import SpotFollower

# Habitat & DCX 核心导入
import habitat
# 必须显式导入 dataset 以触发 Registry
import habitat_llm.agent.env.dataset
from habitat_llm.agent.env import EnvironmentInterface
from habitat_llm.agent.env import sensors 
from habitat_llm.agent.agent import Agent
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
    # --- 2. 初始化配置 ---
    config_rel_path = "../../habitat_llm/conf" 
    
    with initialize(config_path=config_rel_path, version_base=None):
        cfg = compose(config_name="examples/data_gen_config_my")
    
    # 注入配置
    with open_dict(cfg):
        # 强制指明 Planner 控制 agent_0，即使这里失效，我们在下面的 Python 代码中也会强制修正
        if "agents" not in cfg.planner or not cfg.planner.agents:
            cfg.planner.agents = ["agent_0"]
        
        # 确保数据集类型
        if "habitat" in cfg:
            cfg.habitat.dataset.type = "CollaborationDataset-v0"
            cfg.habitat.simulator.agents_order = ["agent_0", "agent_1"]

    # 注册传感器
    sensors.register_sensors(cfg)
    print("Configuration loaded.")

    # --- 3. 显式加载数据集 ---
    print(f"Loading dataset: {cfg.habitat.dataset.data_path}")
    try:
        dataset = habitat.make_dataset(
            id_dataset=cfg.habitat.dataset.type,
            config=cfg.habitat.dataset
        )
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        raise e

    # --- 4. 初始化环境 ---
    print("Initializing EnvironmentInterface...")
    try:
        # 这里会创建 High-Level Agents (habitat_llm.agent.Agent)
        env_interface = EnvironmentInterface(cfg, dataset=dataset, init_wg=True, init_env=True)
    except Exception as e:
        print(f"Failed to create EnvironmentInterface: {e}")
        raise e

    sim = env_interface.sim
    
    # --- 5. 初始化 Planner ---
    print("Initializing Planner...")
    planner = CentralizedLLMPlanner(cfg.planner, env_interface)
    
    # ==============================================================================
    # [核心修复]：强制进行“换脑手术” (Force Agent Binding)
    # 解决 IndexError 和 AttributeError: 'ArticulatedAgentData' has no attribute 'reset'
    # ==============================================================================
    print("DEBUG: Force fixing Planner agents binding...")
    
    # # 1. 找到正确的高层 Agent (Agent 0 / Human)
    # target_high_level_agent = None
    
    # # env_interface.agents 存储的是 habitat_llm.agent.Agent (正确对象)
    # # sim.agents_mgr 存储的是 ArticulatedAgentData (错误对象)
    # if hasattr(env_interface, 'agents'):
    #     for ag in env_interface.agents:
    #         # 检查 uid 或 name，确保找到 agent_0
    #         uid = str(getattr(ag, 'uid', ''))
    #         name = str(getattr(ag, 'name', ''))
    #         if uid == "agent_0" or name == "agent_0":
    #             target_high_level_agent = ag
    #             break
        
    #     # 如果没找到名字匹配的，但列表不为空，默认取第一个（通常就是 Human）
    #     if target_high_level_agent is None and len(env_interface.agents) > 0:
    #         print("Warning: Could not match 'agent_0' by name. Defaulting to first high-level agent.")
    #         target_high_level_agent = env_interface.agents[0]
            
    # if target_high_level_agent is None:
    #     raise RuntimeError("CRITICAL ERROR: No High-Level Agents found in EnvironmentInterface! Cannot bind Planner.")
    low_level_human = sim.agents_mgr[0]
    need_repair = False
    if not hasattr(planner, "_agents") or not planner._agents:
        print("Planner has No agents.")
        need_repair = True
    else:
        first_agent = planner._agents[0]
        if not hasattr(first_agent, "reset"):
            print(f"Planner agent is Low-Level {type(first_agent)} Agent")
            need_repair = True
    if need_repair:
        print(">>> STARTING AGENT REPAIR <<<")
        agent_config = OmegaConf.create({
            "name": "agent_0",
            "uid": "agent_0",
            "description": "A humanoid agent",
            "articulated_agent_type": "KinematicHumanoid",
            "articulated_agent_urdf": "data/humanoids/humanoid_data/female_0/female_0.urdf",
            "motion_data_path": "data/humanoids/humanoid_data/female_0/female_0_motion_data_smplx.pkl",
            "skills": {
                # 定义一些基础 Skill 防止报错，即使我们不真的用
                "wait": {
                    "type": "WaitSkill",
                    "config": {}
                },
                "pick": {
                    "type": "PickSkill", 
                    "config": {}
                }
            },
            "tools": {} # 工具集
        })
        high_level_human = Agent("agent_0", agent_config)
        high_level_human.articulated_agent = low_level_human
        planner._agents = [high_level_human]
        print(">>> REPAIR SUCCESSFUL <<<")
    # # 2. 强制覆盖 Planner 内部列表
    # # 不管它之前初始化了什么（哪怕是错误的底层对象），直接覆盖
    # planner._agents = [target_high_level_agent]
    
    # print(f"SUCCESS: Planner bound to High-Level Agent: {target_high_level_agent} (Type: {type(target_high_level_agent)})")
    # ==============================================================================

    # --- 6. 数据生成循环 ---
    total_episodes = len(dataset.episodes)
    print(f"Total episodes to process: {total_episodes}")

    for i in range(total_episodes):
        try:
            observations = env_interface.reset_environment()
        except StopIteration:
            break
            
        current_episode = env_interface.env.current_episode()
        episode_id = current_episode.episode_id
        print(f"--- Starting Episode ID: {episode_id} ---")
        
        # 获取 Instruction
        instruction_text = env_interface.env.env.env._env.current_episode.instruction

        print(f"Instruction: {instruction_text}")
        
        world_graph = env_interface.world_graph
        
        # [关键测试] 此时调用 reset 应该不会报错了
        if hasattr(planner, "reset"):
            print("Resetting planner...")
            planner.reset()
        
        # 初始化 Spot Follower (Spot 不需要高层逻辑，使用底层物理对象即可)
        human_agent_sim = sim.agents_mgr[0] # ArticulatedAgentData
        spot_agent_sim = sim.agents_mgr[1]  # ArticulatedAgentData
        spot_follower = SpotFollower(spot_agent_sim, human_agent_sim)
        
        video_ego_frames = []
        video_3rd_frames = []
        
        done = False
        step_count = 0
        max_steps = 200
        
        while not done and step_count < max_steps:
            # --- A. 规划 (Planner) ---
            high_level_action = None
            try:
                # 获取 LLM 动作
                high_level_action = planner.get_next_action(
                    instruction_text, 
                    observations, 
                    world_graph
                )
            except Exception as e:
                print(f"Planner error: {e}")
                import traceback
                traceback.print_exc()
                break

            # --- B. Spot 跟随 (小脑) ---
            spot_vel = spot_follower.get_action()
            
            # --- C. 执行 (Step) ---
            step_action = {}
            
            # 1. 填入 Human 动作
            if high_level_action:
                # 检查 action 格式，如果是 tuple (skill_name, args)
                if isinstance(high_level_action, tuple):
                    # 这是一个简化的处理，实际 PARTNR 可能会更复杂
                    # 但通常 HighLevelAction 可以直接传给 env_interface
                    pass
                step_action["agent_0"] = high_level_action
            
            # 2. 填入 Spot 动作
            step_action["agent_1"] = {
                "action": "agent_1_base_velocity", 
                "action_args": {
                    "linear_velocity": float(spot_vel[0]), 
                    "angular_velocity": float(spot_vel[1])
                }
            }

            try:
                # 尝试多智能体步进
                observations = env_interface.step(step_action)
                step_count += 1
            except Exception as e:
                # 如果多智能体 step 失败，回退到仅执行 Human
                print(f"Multi-agent step failed: {e}. Falling back to Human-only.")
                try:
                    observations = env_interface.step(high_level_action)
                    step_count += 1
                except:
                    print("Step failed completely.")
                    break
            
            # --- D. 记录 ---
            ego_key = "agent_1_jaw_rgb"
            third_key = "agent_0_third_person_rgb"
            
            if ego_key in observations:
                frame = observations[ego_key]
                if hasattr(frame, 'cpu'): frame = frame.cpu().numpy()
                video_ego_frames.append(frame.astype(np.uint8))
                
            if third_key in observations:
                frame = observations[third_key]
                if hasattr(frame, 'cpu'): frame = frame.cpu().numpy()
                video_3rd_frames.append(frame.astype(np.uint8))

            if observations.get('task_is_done', False):
                done = True

        save_dir = f"output_dataset/episode_{episode_id}"
        save_video(video_ego_frames, f"{save_dir}/robot_ego.mp4")
        save_video(video_3rd_frames, f"{save_dir}/third_person.mp4")

if __name__ == "__main__":
    main()