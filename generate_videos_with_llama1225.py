#!/usr/bin/env python3
import os
import cv2
import json
import hydra
import numpy as np
import magnum as mn
import random
import torch
import importlib
import pkgutil
from omegaconf import DictConfig, OmegaConf, open_dict  # <--- 引入 open_dict

# PARTNR 核心
import habitat_llm
from habitat_llm.agent.env import EnvironmentInterface

# ==========================================
# 1. 核心修复：配置清洗与补全
# ==========================================
def sanitize_config(cfg):
    """
    全能配置修复函数：
    1. 清洗 Gym 传感器 (解决 KeyError)。
    2. 补全 Agent Order (解决 Env Init Error)。
    3. [NEW] 补全 Device (解决 ConfigAttributeError)。
    """
    print("Sanitizing Config...")
    
    # 1. 解锁顶层配置
    OmegaConf.set_struct(cfg, False)
    
    # 2. 修复缺失的 'device' 配置 (CRITICAL FIX)
    # HabitatConfig 结构体默认锁定了键值，必须用 open_dict 解锁才能添加新键
    if "habitat" in cfg:
        with open_dict(cfg.habitat):
            if "device" not in cfg.habitat:
                print("  Auto-fixing missing 'habitat.device' -> 'cuda'")
                cfg.habitat.device = "cuda"  # 强制指定为 cuda

    # 3. 清理 Gym Observation Keys (移除 RL 传感器)
    if "gym" in cfg.habitat and "obs_keys" in cfg.habitat.gym:
        with open_dict(cfg.habitat.gym):
            original_keys = list(cfg.habitat.gym.obs_keys)
            safe_keys = [k for k in original_keys if "rgb" in k or "depth" in k or "panoptic" in k]
            # 保留必要的机械臂/Head深度信息以防万一
            extra_safe = ["agent_0_articulated_agent_arm_depth", "agent_1_head_depth"]
            for esk in extra_safe:
                if esk in original_keys: safe_keys.append(esk)
            
            print(f"  Sanitized Obs Keys: {safe_keys}")
            cfg.habitat.gym.obs_keys = safe_keys
        
    # 4. 确保 Simulator Agent Order 正确
    if "simulator" in cfg.habitat:
        with open_dict(cfg.habitat.simulator):
            if "agents_order" not in cfg.habitat.simulator:
                print("  Auto-fixing agents_order...")
                cfg.habitat.simulator.agents_order = ["agent_0", "agent_1"]

    return cfg

# ==========================================
# 2. 强制注册模块 (防止 Class Not Found)
# ==========================================
def force_register_partnr_modules():
    packages = ["habitat_llm.sensors", "habitat_llm.task.sensors", "habitat_llm.conf"]
    for p in packages:
        try:
            mod = importlib.import_module(p)
            if hasattr(mod, "__path__"):
                for _, name, _ in pkgutil.walk_packages(mod.__path__, mod.__name__ + "."):
                    try: importlib.import_module(name)
                    except: pass
        except: pass

force_register_partnr_modules()

# ==========================================
# 3. 辅助模块
# ==========================================
class DynamicRobotCameraman:
    def __init__(self, robot_agent_idx=1, human_agent_idx=0):
        self.robot_idx = robot_agent_idx
        self.human_idx = human_agent_idx
        self.target_pos = None
        self.update_timer = 0
        
    def get_nav_target(self, sim):
        try:
            # 尝试获取人类位置
            human_pos = sim.get_agent_data(self.human_idx).articulated_agent.base_pos
        except: return None

        if self.update_timer <= 0 or self.target_pos is None:
            angle = random.uniform(0, 2 * np.pi)
            dist = random.uniform(1.8, 3.0)
            offset = mn.Vector3(np.cos(angle) * dist, 0, np.sin(angle) * dist)
            raw_target = human_pos + offset
            if sim.pathfinder.is_loaded:
                self.target_pos = sim.pathfinder.snap_point(raw_target)
            else:
                self.target_pos = raw_target
            self.update_timer = 30 
        self.update_timer -= 1
        return self.target_pos

class DatasetRecorder:
    def __init__(self, output_dir, episode_id, fps=10):
        self.output_dir = output_dir
        self.episode_id = episode_id
        self.video_writer = None
        self.annotations = []
        self.fps = fps
        os.makedirs(output_dir, exist_ok=True)
        
    def record_frame(self, obs, info):
        img = None
        candidates = [k for k in obs.keys() if "agent_1" in k and "rgb" in k]
        # 尝试 jaw 或 head
        priority = [k for k in candidates if "jaw" in k]
        if not priority: priority = [k for k in candidates if "head" in k]
        target_key = priority[0] if priority else (candidates[0] if candidates else None)
        
        if target_key and target_key in obs:
            img = obs[target_key]
            if isinstance(img, torch.Tensor): img = img.cpu().numpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            if self.video_writer is None:
                h, w = img.shape[:2]
                path = os.path.join(self.output_dir, f"ep_{self.episode_id}_robot_view.mp4")
                self.video_writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w, h))
            
            self.video_writer.write(img)
        
        self.annotations.append(info)

    def close(self):
        if self.video_writer: self.video_writer.release()
        json_path = os.path.join(self.output_dir, f"ep_{self.episode_id}_labels.json")
        with open(json_path, "w") as f: json.dump(self.annotations, f, indent=2)

# ==========================================
# 4. 主程序
# ==========================================
@hydra.main(version_base=None, config_path="habitat_llm/conf", config_name="custom_video_gen")
def main(cfg: DictConfig):
    # 1. 执行配置修复
    cfg = sanitize_config(cfg)
    
    # 2. 设置 Llama 路径
    llama_path = "/root/autodl-tmp/partnr-planner-main/Meta-Llama-3-8B-Instruct"
    
    if "planner" in cfg:
        # 使用 open_dict 解锁 planner 配置
        with open_dict(cfg.planner):
            if "plan_config" in cfg.planner:
                cfg.planner.plan_config.llm.generation_params.engine = llama_path
            elif "llm" in cfg.planner:
                cfg.planner.llm.generation_params.engine = llama_path

    print("Initializing Environment...")
    try:
        env = EnvironmentInterface(cfg.habitat)
    except Exception as e:
        print(f"ENV INIT ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Initializing Planner...")
    try:
        planner = hydra.utils.instantiate(cfg.planner, env_interface=env)
    except Exception as e:
        print(f"PLANNER INIT ERROR: {e}")
        env.close()
        return

    robot_cam = DynamicRobotCameraman(robot_agent_idx=1, human_agent_idx=0)
    num_episodes = 2 
    
    while env.number_of_episodes_run < num_episodes:
        obs = env.reset()
        planner.reset()
        
        ep_id = env.current_episode.episode_id
        instruction = env.current_episode.instruction
        print(f"\n[Episode {ep_id}] Task: {instruction}")
        
        recorder = DatasetRecorder("output_dataset_final_v4", ep_id)
        
        done = False
        step = 0
        last_thought = ""
        last_action = "Idle"
        
        while not done and step < 500:
            # A. Planner
            try:
                action_dict, info, done = planner.get_next_action(obs)
            except Exception as e:
                print(f"Planner Error: {e}")
                import traceback
                traceback.print_exc()
                break
                
            # CoT
            if "thought" in info and planner.agents:
                uid = planner.agents[0].uid
                t = info["thought"].get(uid, "")
                if t: last_thought = t
            
            # Action
            if "high_level_actions" in info and planner.agents:
                uid = planner.agents[0].uid
                hl = info["high_level_actions"].get(uid, None)
                if hl: last_action = f"{hl[0]}({hl[1]})"

            # B. Robot
            tgt = robot_cam.get_nav_target(env.sim)
            if tgt is not None:
                if "agent_1_oracle_nav_action" in env.action_space:
                    action_dict["agent_1_oracle_nav_action"] = tgt
            
            # C. Step
            obs, reward, done, info = env.step(action_dict)
            
            # D. Record
            data = {
                "frame": step,
                "task": instruction,
                "cot": last_thought,
                "action": last_action,
                "subtask_guess": last_thought.split('.')[0] if last_thought else ""
            }
            recorder.record_frame(obs, data)
            
            if step % 20 == 0:
                print(f"Step {step}: {last_action}")
            step += 1
            
        recorder.close()
        print(f"Episode {ep_id} Done.")

    env.close()

if __name__ == "__main__":
    main()