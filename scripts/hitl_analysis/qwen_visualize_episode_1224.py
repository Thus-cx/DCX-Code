#!/usr/bin/env python3
"""
PARTNR/Habitat 可视化生成器 - 最终修复版 (Fix Reset Matrix)
修复核心：
1. [Bug修复] 修正 reset() 调用，恢复使用 mn.Matrix4.from_() 构建矩阵，解决 Swizzle 报错。
2. [关键修复] 优先加载 'walking_motion_processed_smplx.pkl'，解决无动作问题。
3. [体验优化] 5倍帧插值 + 轨迹平滑，解决瞬移和抖动。
"""

import os
import cv2
import numpy as np
import math
from typing import List, Dict, Any, Deque
import argparse
from multiprocessing import Pool
from collections import deque
import habitat_sim

# 引入 Habitat 官方走路控制器
from habitat.articulated_agent_controllers import HumanoidRearrangeController
import magnum as mn

# 引入原有加载逻辑
from hitl_episode import HITLSession, HITLEpisode, init_env

# ================= 配置区 =================
INTERP_STEPS = 5  # 插值倍数
DEBUG_MODE = False # 关闭调试打印以保持整洁
# =========================================

class TrajectorySmoother:
    """轨迹平滑器"""
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.pos_history: Deque[np.ndarray] = deque(maxlen=window_size)
        self.rot_history: Deque[np.ndarray] = deque(maxlen=window_size)

    def update(self, new_pos: np.ndarray, new_rot_quat: mn.Quaternion) -> (np.ndarray, mn.Quaternion):
        self.pos_history.append(new_pos)
        avg_pos = np.mean(np.array(self.pos_history), axis=0)

        q_np = np.array([new_rot_quat.vector.x, new_rot_quat.vector.y, new_rot_quat.vector.z, new_rot_quat.scalar])
        if len(self.rot_history) > 0:
            if np.dot(q_np, self.rot_history[-1]) < 0:
                q_np = -q_np
        
        self.rot_history.append(q_np)
        avg_q_np = np.mean(np.array(self.rot_history), axis=0)
        norm = np.linalg.norm(avg_q_np)
        if norm > 1e-6: avg_q_np /= norm
        else: avg_q_np = np.array([0, 0, 0, 1])
            
        avg_rot = mn.Quaternion(mn.Vector3(avg_q_np[0], avg_q_np[1], avg_q_np[2]), avg_q_np[3])
        return avg_pos, avg_rot

class HabitatHumanoidAnimator:
    """人类动画控制器"""
    def __init__(self, sim, agent_id=0):
        self.controller = None
        self.sim = sim
        self.agent_id = agent_id
        self.initialized = False
        self.prev_pos = None
        self.prev_rot = None

    def _init_controller(self):
        try:
            # 搜索路径 (优先 processed 文件)
            search_paths = [
                "data/humanoids/humanoid_data/walking_motion_processed_smplx.pkl",
                "data/humanoids/humanoid_data/female_0/walking_motion_processed_smplx.pkl",
                "data/hab3_bench_assets/humanoids/female_0/walking_motion_processed_smplx.pkl",
                "data/humanoids/humanoid_data/walk_motion.pkl"
            ]
            
            try:
                agent_name = self.sim.habitat_config.agents_order[self.agent_id]
                agent_config = self.sim.habitat_config.agents[agent_name]
                if hasattr(agent_config, "motion_data_path"):
                    search_paths.insert(0, agent_config.motion_data_path)
            except: pass

            motion_path = None
            for p in search_paths:
                if os.path.exists(p) or os.path.exists(os.path.abspath(p)):
                    motion_path = p if os.path.exists(p) else os.path.abspath(p)
                    break
            
            if motion_path:
                print(f"[Info] Animator Loaded: {motion_path}")
                self.controller = HumanoidRearrangeController(motion_path)
            else:
                print("[Warning] Motion file not found!")
            
            self.initialized = True
        except Exception as e:
            print(f"[Error] Animator Init: {e}")
            self.initialized = True

    def update_pose(self, agent_data, current_pos_np, current_rot_quat):
        if not self.initialized: self._init_controller()
        
        # 强制 Kinematic
        articulated_agent = agent_data.articulated_agent
        ao = articulated_agent.sim_obj if hasattr(articulated_agent, "sim_obj") else articulated_agent
        if ao.motion_type != habitat_sim.physics.MotionType.KINEMATIC:
            ao.motion_type = habitat_sim.physics.MotionType.KINEMATIC

        # 初始化/重置
        if self.prev_pos is None:
            self.prev_pos = current_pos_np
            self.prev_rot = current_rot_quat
            if self.controller:
                try:
                    # FIX: 必须传入 Matrix4，否则 Magnum 会报 Swizzle 错误
                    # 使用 from_ 而不是 from (Python关键字)
                    trans = mn.Matrix4.from_(current_rot_quat.to_matrix(), mn.Vector3(current_pos_np))
                    self.controller.reset(trans)
                except Exception as e: 
                    if DEBUG_MODE: print(f"Reset Error: {e}")
            return

        if self.controller is None: return

        # 计算位移
        diff_vec = current_pos_np - self.prev_pos
        lin_dist = float(np.linalg.norm(diff_vec)) # 确保是 float
        
        curr_yaw = self._get_yaw(current_rot_quat)
        prev_yaw = self._get_yaw(self.prev_rot)
        ang_dist = float(curr_yaw - prev_yaw)
        if ang_dist > np.pi: ang_dist -= 2*np.pi
        if ang_dist < -np.pi: ang_dist += 2*np.pi
        
        try:
            # 只要有移动就更新
            if lin_dist > 1e-5 or abs(ang_dist) > 1e-5:
                self.controller.translate_and_rotate_with_gait(lin_dist, ang_dist)

            new_pose = self.controller.get_pose()
            if len(new_pose) == len(ao.joint_positions):
                ao.joint_positions = new_pose
                
        except Exception as e:
            if DEBUG_MODE: print(f"Anim Error: {e}")

        self.prev_pos = current_pos_np
        self.prev_rot = current_rot_quat

    def _get_yaw(self, quat_mn):
        fwd = quat_mn.transform_vector(mn.Vector3(0, 0, -1))
        return math.atan2(fwd.x, -fwd.z)

class RobotViewVideoGenerator:
    def __init__(self):
        self.robot_agent_id = 1
        self.human_agent_id = 0
        self.human_animator = None
        self.human_smoother = TrajectorySmoother(window_size=10)
        self.robot_smoother = TrajectorySmoother(window_size=10)

    def process_single_episode(self, hitl_episode, dataset_file, output_dir, frame_rate) -> bool:
        return self._process_logic(hitl_episode, dataset_file, output_dir, frame_rate)

    def _process_logic(self, hitl_episode, dataset_file, output_dir, frame_rate):
        episode_id = hitl_episode.episode_info["episode_id"]
        print(f"Processing episode {episode_id}")

        try:
            env = self._initialize_with_hitl_method(hitl_episode, dataset_file)
            if env is None: return False
            
            # Reset
            self.human_animator = HabitatHumanoidAnimator(env.sim, agent_id=self.human_agent_id)
            self.human_smoother = TrajectorySmoother(window_size=10)
            self.robot_smoother = TrajectorySmoother(window_size=10)

            frames = [f for f in hitl_episode.hitl_data["frames"] if f and len(f) > 0]
            if not frames:
                env.close()
                return False
                
            out_robot = os.path.join(output_dir, "robot_view")
            out_human = os.path.join(output_dir, "human_third_view")
            os.makedirs(out_robot, exist_ok=True)
            os.makedirs(out_human, exist_ok=True)
            
            vid_robot = os.path.join(out_robot, f"robot_view_{episode_id}.mp4")
            vid_human = os.path.join(out_human, f"human_view_{episode_id}.mp4")
            
            success = self._generate_video(env, frames, vid_robot, vid_human, frame_rate)
            env.close()
            return success

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _initialize_with_hitl_method(self, hitl_episode, dataset_file):
        try:
            from hitl_episode import init_env
            episode_id = hitl_episode.episode_info["episode_id"]
            cfg_dict = hitl_episode.hitl_data["session"]["config"]
            env = init_env(None, episode_ids=[episode_id], cfg_dict=cfg_dict)
            if env is None: return None
            
            matches = [ep for ep in env._dataset.episodes if ep.episode_id == episode_id]
            if not matches: return None

            env.current_episode = matches[0]
            env.reset()
            return env
        except: return None

    def _apply_spot_standing_pose(self, agent_data):
        articulated_agent = agent_data.articulated_agent
        ao = articulated_agent.sim_obj if hasattr(articulated_agent, "sim_obj") else articulated_agent
        if ao.motion_type != habitat_sim.physics.MotionType.KINEMATIC:
            ao.motion_type = habitat_sim.physics.MotionType.KINEMATIC
            
        joints = list(ao.joint_positions)
        if len(joints) >= 12:
            joints[0:12] = [0.0, 0.7, -1.5] * 4 
            for i in range(12, len(joints)): joints[i] = 0.0
            ao.joint_positions = joints

    def _generate_video(self, env, frames: List[Dict], path_robot: str, path_human: str, fps: int) -> bool:
        obs = env.sim.get_sensor_observations()
        
        h, w = 256, 256
        if 'agent_1_articulated_agent_jaw_rgb' in obs:
            h, w = obs['agent_1_articulated_agent_jaw_rgb'].shape[:2]
        h1, w1 = 256, 256
        if 'agent_0_third_rgb' in obs:
            h1, w1 = obs['agent_0_third_rgb'].shape[:2]
            
        wr_robot = cv2.VideoWriter(path_robot, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        wr_human = cv2.VideoWriter(path_human, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w1, h1))

        if not wr_robot.isOpened() or not wr_human.isOpened(): return False

        cnt = 0
        
        for i in range(len(frames) - 1):
            f_curr = frames[i]
            f_next = frames[i+1]
            
            try:
                as_curr = f_curr.get("agent_states", [])
                as_next = f_next.get("agent_states", [])
                if len(as_curr) < 2 or len(as_next) < 2: continue

                h_pos1 = np.array(as_curr[0]["position"], dtype=np.float32)
                h_rot1 = mn.Quaternion(mn.Vector3(as_curr[0]["rotation"][:3]), as_curr[0]["rotation"][3])
                h_pos2 = np.array(as_next[0]["position"], dtype=np.float32)
                h_rot2 = mn.Quaternion(mn.Vector3(as_next[0]["rotation"][:3]), as_next[0]["rotation"][3])
                
                r_pos1 = np.array(as_curr[1]["position"], dtype=np.float32)
                r_rot1 = mn.Quaternion(mn.Vector3(as_curr[1]["rotation"][:3]), as_curr[1]["rotation"][3])
                r_pos2 = np.array(as_next[1]["position"], dtype=np.float32)
                r_rot2 = mn.Quaternion(mn.Vector3(as_next[1]["rotation"][:3]), as_next[1]["rotation"][3])
            except: continue 

            for step in range(INTERP_STEPS):
                alpha = step / float(INTERP_STEPS)
                
                # Human
                h_pos_int = h_pos1 * (1 - alpha) + h_pos2 * alpha
                h_rot_int = mn.math.slerp(h_rot1, h_rot2, alpha)
                h_pos_s, h_rot_s = self.human_smoother.update(h_pos_int, h_rot_int)
                
                # Robot
                r_pos_int = r_pos1 * (1 - alpha) + r_pos2 * alpha
                r_rot_int = mn.math.slerp(r_rot1, r_rot2, alpha)
                r_pos_s, r_rot_s = self.robot_smoother.update(r_pos_int, r_rot_int)
                
                # Update Human (Animation)
                ag0 = env.sim.get_agent_data(0)
                ag0.articulated_agent.base_pos = h_pos_s
                ag0.articulated_agent.base_rot = self.human_animator._get_yaw(h_rot_s)
                self.human_animator.update_pose(ag0, h_pos_s, h_rot_s)
                
                # Update Robot
                ag1 = env.sim.get_agent_data(1)
                ag1.articulated_agent.base_pos = r_pos_s
                fwd = r_rot_s.transform_vector(mn.Vector3(0, 0, -1))
                ag1.articulated_agent.base_rot = math.atan2(fwd.x, -fwd.z)
                self._apply_spot_standing_pose(ag1)
                
                # Objects
                if step == 0 and "object_states" in f_curr:
                    self._set_object_states(env.sim, f_curr["object_states"])

                # Render
                env.sim.step(-1)
                obs_new = env.sim.get_sensor_observations()
                
                rgb_r = obs_new.get("agent_1_articulated_agent_jaw_rgb")
                rgb_h = obs_new.get("agent_0_third_rgb")
                
                if rgb_r is not None and rgb_h is not None:
                    wr_robot.write(cv2.cvtColor(rgb_r, cv2.COLOR_RGB2BGR))
                    wr_human.write(cv2.cvtColor(rgb_h, cv2.COLOR_RGB2BGR))
                    cnt += 1

        wr_robot.release()
        wr_human.release()
        print(f"Generated video with {cnt} frames.")
        return cnt > 0

    def _set_object_states(self, sim, object_states):
        rom = sim.get_rigid_object_manager()
        if not hasattr(self, "_cached_obj_map"):
            self._cached_obj_map = {h: h for h in rom.get_object_handles()}
            
        for obj in object_states:
            handle = obj["object_handle"]
            robj = None
            if handle in self._cached_obj_map:
                robj = rom.get_object_by_handle(handle)
            else:
                prefix = handle.split(":")[0]
                for existing_h in self._cached_obj_map:
                    if existing_h.startswith(prefix):
                        self._cached_obj_map[handle] = existing_h
                        robj = rom.get_object_by_handle(existing_h)
                        break
            
            if robj is not None:
                if hasattr(robj, "motion_type"):
                     robj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
                robj.translation = np.array(obj["position"], dtype=np.float32)
                x, y, z, w = obj["rotation"]
                robj.rotation = mn.Quaternion(mn.Vector3(x, y, z), w)

def global_process_wrapper(args):
    return RobotViewVideoGenerator().process_single_episode(*args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-path", required=True)
    parser.add_argument("--dataset-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frame-rate", type=int, default=30)
    parser.add_argument("--num-processes", type=int, default=10)
    args = parser.parse_args()

    file_list = [
        os.path.join(args.episodes_path, f)
        for f in os.listdir(args.episodes_path)
        if f.endswith('.json.gz') and f != "session.json.gz"
    ]
    session = HITLSession(file_list, multi=True, hitl_data_file=args.dataset_file)
    tasks = [(ep, args.dataset_file, args.output_dir, args.frame_rate) for ep in session.episodes]

    with Pool(args.num_processes) as pool:
        pool.map(global_process_wrapper, tasks)

if __name__ == "__main__":
    main()