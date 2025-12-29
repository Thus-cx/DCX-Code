import os
import sys
import json
import numpy as np
import cv2
import magnum as mn
from omegaconf import OmegaConf
from hydra import compose, initialize

# PARTNR imports
from habitat_llm.agent.env.environment_interface import EnvironmentInterface
from habitat_llm.planner.centralized_llm_planner import CentralizedLLMPlanner
# Habitat imports
from habitat.articulated_agent_controllers import HumanoidRearrangeController
# 【关键修正】直接使用基类配置，完全控制参数
from habitat.config.default_structured_configs import HabitatSimRGBSensorConfig

# -----------------------------------------------------------------------------
# 1. 辅助类：Spot 跟随控制
# -----------------------------------------------------------------------------
class SpotFollower:
    def __init__(self, sim, spot_agent_uuid, human_agent_uuid, keep_dist=1.8):
        self.sim = sim
        self.spot_uuid = spot_agent_uuid
        self.human_uuid = human_agent_uuid
        self.keep_dist = keep_dist

    def get_action(self, human_id, spot_id):
        # 安全性检查
        if human_id >= len(self.sim.agents_mgr) or spot_id >= len(self.sim.agents_mgr):
            return np.zeros(3), 0.0
        try:
            human_agent = self.sim.get_agent_data(human_id).articulated_agent
            spot_agent = self.sim.get_agent_data(spot_id).articulated_agent
        except Exception:
            return np.zeros(3), 0.0

        human_pos = np.array(human_agent.base_pos)
        spot_pos = np.array(spot_agent.base_pos)
        
        vec_h_to_s = spot_pos - human_pos
        dist = np.linalg.norm(vec_h_to_s)
        
        if dist < 0.1: 
            vec_h_to_s = np.array([1.0, 0.0, 0.0])
        else:
            vec_h_to_s = vec_h_to_s / dist
            
        target_pos = human_pos + vec_h_to_s * self.keep_dist
        
        if self.sim.pathfinder.is_loaded:
            target_pos = self.sim.pathfinder.snap_point(target_pos)

        look_dir = human_pos - target_pos
        yaw = np.arctan2(look_dir[0], look_dir[2]) 
        
        return target_pos, yaw

# -----------------------------------------------------------------------------
# 2. 核心类：数据集生成器
# -----------------------------------------------------------------------------
class DatasetGenerator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.output_dir = "output_dataset_videos"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("Initializing EnvironmentInterface...")
        self.env_interface = EnvironmentInterface(cfg)
        self.env = self.env_interface 
        self.sim = self.env_interface.sim
        
        self.human_id, self.spot_id = self._identify_agents()
        print(f"Agents Identified -> Human ID: {self.human_id}, Spot ID: {self.spot_id}")

        print("Initializing LLM Planner...")
        self.planner = CentralizedLLMPlanner(cfg.planner, self.env_interface)
        
        self._init_humanoid_controller()
        self.spot_follower = SpotFollower(self.sim, self.spot_id, self.human_id)

    def _identify_agents(self):
        human_id = 0
        spot_id = 1
        try:
            for i in range(len(self.sim.agents_mgr)):
                agent_data = self.sim.get_agent_data(i)
                urdf_path = str(agent_data.articulated_agent.urdf_path).lower()
                if "human" in urdf_path or "female" in urdf_path:
                    human_id = i
                elif "spot" in urdf_path or "robot" in urdf_path:
                    spot_id = i
        except Exception: pass
        return human_id, spot_id

    def _init_humanoid_controller(self):
        # 多路径寻找动作文件
        candidates = [
            "data/humanoids/humanoid_data/walk_motion.pkl",
            "../data/humanoids/humanoid_data/walk_motion.pkl",
            os.path.abspath("data/humanoids/humanoid_data/walk_motion.pkl")
        ]
        motion_path = None
        for p in candidates:
            if os.path.exists(p):
                motion_path = p
                break
        
        if motion_path is None:
            print("WARNING: Motion path not found. Human animation will default to slide.")
            self.human_controller = None
        else:
            print(f"Loading Humanoid Controller from {motion_path}")
            try:
                self.human_controller = HumanoidRearrangeController(motion_path)
            except Exception as e:
                print(f"Failed to init HumanoidController: {e}")
                self.human_controller = None

    def run_episode(self, episode_id=0):
        print(f"--- Running Episode {episode_id} ---")
        
        # Reset 环境并获取初始观测
        try:
            self.env_interface.reset()
            observations = self.env_interface.get_observations()
        except KeyError as e:
            print(f"CRITICAL ERROR during reset: {e}")
            print("Trying to debug available keys...")
            # 尝试直接从 sim 获取原始观测，绕过 gym wrapper
            raw_obs = self.sim.get_sensor_observations()
            print(f"Available Sim Keys: {list(raw_obs.keys())}")
            raise e

        done = False
        step_count = 0
        
        # 视频录制
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vid_spot = cv2.VideoWriter(f"{self.output_dir}/ep{episode_id}_spot.mp4", fourcc, 10, (512, 512))
        vid_third = cv2.VideoWriter(f"{self.output_dir}/ep{episode_id}_third.mp4", fourcc, 10, (512, 512))

        # 记录 Trace
        trace_data = {"episode_id": episode_id, "steps": []}
        
        try:
            prev_human_pos = np.array(self.sim.get_agent_data(self.human_id).articulated_agent.base_pos)
        except Exception:
            prev_human_pos = np.zeros(3)

        while not done and step_count < 150:
            # 1. LLM 规划
            try:
                action_dict = self.planner.step(observations)
            except Exception as e:
                print(f"Planner Error: {e}")
                break

            # 2. 物理执行
            observations, reward, done, info = self.env_interface.step(action_dict)
            
            # 3. 视觉修正
            self._apply_visual_corrections(prev_human_pos)
            
            # 4. 渲染
            final_obs = self.sim.get_sensor_observations()
            
            # 智能查找图像 Key (防止 uuid 命名不一致)
            spot_img = None
            third_img = None
            
            # 遍历所有观测，通过名称包含关系查找
            for k, v in final_obs.items():
                if "jaw_rgb" in k and isinstance(v, np.ndarray):
                    spot_img = v
                if "third_rgb" in k and isinstance(v, np.ndarray):
                    third_img = v
            
            if spot_img is not None: vid_spot.write(spot_img[:, :, :3][..., ::-1])
            if third_img is not None: vid_third.write(third_img[:, :, :3][..., ::-1])

            # 保存 CoT
            cot = ""
            if hasattr(self.planner, "last_response"):
                cot = self.planner.last_response
            elif hasattr(self.planner, "llm") and hasattr(self.planner.llm, "last_output"):
                cot = self.planner.llm.last_output

            trace_data["steps"].append({
                "step": step_count,
                "action": str(action_dict),
                "cot": cot,
                "human_pos": [float(x) for x in prev_human_pos]
            })
            
            try:
                prev_human_pos = np.array(self.sim.get_agent_data(self.human_id).articulated_agent.base_pos)
            except: pass
            
            step_count += 1
            if step_count % 10 == 0: print(f"Step {step_count}...")

        vid_spot.release()
        vid_third.release()
        with open(f"{self.output_dir}/ep{episode_id}_trace.json", "w") as f:
            json.dump(trace_data, f, indent=2)
        print("Episode Finished.")

    def _apply_visual_corrections(self, prev_human_pos):
        if self.human_controller:
            try:
                human_agent = self.sim.get_agent_data(self.human_id).articulated_agent
                curr_pos = np.array(human_agent.base_pos)
                disp = np.linalg.norm(curr_pos - prev_human_pos)
                if disp > 0.001:
                    self.human_controller.translate_and_rotate_with_gait(disp, 0)
                    human_agent.joint_positions = self.human_controller.get_pose()
                    human_agent.update()
            except Exception: pass
        try:
            target_pos, target_yaw = self.spot_follower.get_action(self.human_id, self.spot_id)
            spot_agent = self.sim.get_agent_data(self.spot_id).articulated_agent
            spot_agent.base_pos = target_pos
            spot_agent.base_rot = target_yaw
            spot_agent.update()
        except Exception: pass

# -----------------------------------------------------------------------------
# 3. 主逻辑：配置补丁
# -----------------------------------------------------------------------------
def main():
    with initialize(config_path="habitat_llm/conf", version_base=None):
        print("Loading Configs...")
        # 1. 加载环境
        cfg_env = compose(
            config_name="examples/planner_multi_agent_demo_config",
            overrides=[
                "habitat.dataset.data_path=data/datasets/partnr_episodes/v0_0/val_mini.json.gz",
            ]
        )
        # 2. 加载规划器
        cfg_planner = compose(
            config_name="planner/llm_zero_shot_react_planner",
            overrides=[
                "+planner.llm.model_name=/root/autodl-tmp/LLM-Research/Meta-Llama-3-8B-Instruct",
                "+planner.llm.quantization_config.load_in_8bit=True"
            ]
        )

    # 3. 融合配置
    OmegaConf.set_struct(cfg_env, False)
    cfg_env.planner = cfg_planner.planner if "planner" in cfg_planner else cfg_planner

    # 4. 动态添加传感器 - 【使用基类 + 显式 UUID】
    
    # --- Agent 0: Human (Third Person Camera) ---
    # 定义第三人称相机，必须显式设置 uuid，确保与 GymWrapper 预期一致
    # 注意：字典的 Key 最好与 uuid 保持某种一致，但内部 uuid 才是决定性因素
    third_cam_uuid = "third_rgb"
    
    third_cam = HabitatSimRGBSensorConfig()
    third_cam.type = "HabitatSimRGBSensor" # 冗余设置以防万一
    third_cam.uuid = third_cam_uuid        # 【关键】设置 UUID
    third_cam.height = 512
    third_cam.width = 512
    third_cam.position = [-2.0, 1.5, 0.0]
    third_cam.orientation = [-0.4, 0.0, 0.0]
    
    # 挂载到配置中
    if "sim_sensors" not in cfg_env.habitat.simulator.agents.agent_0:
        cfg_env.habitat.simulator.agents.agent_0.sim_sensors = {}
    
    # 使用 uuid 作为 key，避免混淆
    cfg_env.habitat.simulator.agents.agent_0.sim_sensors[third_cam_uuid] = third_cam

    # --- Agent 1: Spot (Jaw Camera) ---
    jaw_cam_uuid = "jaw_rgb"
    
    jaw_cam = HabitatSimRGBSensorConfig()
    jaw_cam.type = "HabitatSimRGBSensor"
    jaw_cam.uuid = jaw_cam_uuid            # 【关键】设置 UUID
    jaw_cam.height = 512
    jaw_cam.width = 512
    
    if "sim_sensors" not in cfg_env.habitat.simulator.agents.agent_1:
        cfg_env.habitat.simulator.agents.agent_1.sim_sensors = {}
        
    cfg_env.habitat.simulator.agents.agent_1.sim_sensors[jaw_cam_uuid] = jaw_cam

    # 5. 强制代理顺序
    cfg_env.habitat.simulator.agents_order = ["agent_0", "agent_1"]

    print("Configuration Patched. Sensors registered manually.")
    
    # 6. 运行
    generator = DatasetGenerator(cfg_env)
    generator.run_episode(0)

if __name__ == "__main__":
    main()