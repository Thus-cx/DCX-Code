#!/usr/bin/env python3
"""
Generate robot-view videos using init_env() and safe sensor injection.
"""

import os
import cv2
import numpy as np
import math
from typing import List, Dict, Any
import argparse
from multiprocessing import Pool
import habitat_sim
import torch

# Your existing data loader
from hitl_episode import HITLSession, HITLEpisode, init_env

# 导入导航技能
try:
    from habitat_llm.tools.motor_skills.nav.oracle_point_nav_skill import OraclePointNavSkill
    NAV_SKILL_AVAILABLE = True
except ImportError:
    NAV_SKILL_AVAILABLE = False
    print("导航技能模块不可用，将使用直接设置状态的方法")

# 导入OmegaConf用于配置对象
try:
    from omegaconf import OmegaConf
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False
    print("OmegaConf不可用，将尝试使用简单配置对象")


def process_single_episode(args):
    episode_info, dataset_file, output_dir, frame_rate, use_navigation = args
    try:
        generator = RobotViewVideoGenerator()
        success = generator.process_single_episode(
            episode_info, dataset_file, output_dir, frame_rate, use_navigation
        )
        return success
    except Exception as e:
        print(f"Error processing {episode_info.episode_info['episode_id']}: {e}")
        import traceback
        traceback.print_exc()
        return False


class RobotViewVideoGenerator:
    def __init__(self):
        self.robot_agent_id = 1
        self.human_agent_id = 0
        
        # 配置参数（使用字典格式）
        self.nav_config_dict = {
            "dist_thresh": 0.3,
            "turn_thresh": 0.2,
            "forward_velocity": 0.5,
            "turn_velocity": 1.0,
            "sim_freq": 120,
            "enable_backing_up": True,
            "face_object": True,
            "teleport": False
        }
        
        # 将字典转换为合适的配置对象
        self.nav_config = self._create_config_object(self.nav_config_dict)
        
        self.nav_skill = None

    def _create_config_object(self, config_dict):
        """将字典转换为配置对象"""
        if OMEGACONF_AVAILABLE:
            # 使用OmegaConf创建配置对象
            return OmegaConf.create(config_dict)
        else:
            # 创建一个简单的对象，支持属性访问
            class SimpleConfig:
                def __init__(self, config_dict):
                    for key, value in config_dict.items():
                        setattr(self, key, value)
            
            return SimpleConfig(config_dict)

    def generate_videos_from_episodes(
            self,
            episodes_path: str,
            dataset_file: str,
            output_dir: str,
            frame_rate: int = 30,
            num_processes: int = 10,
            use_navigation: bool = True  # 新增参数：是否使用导航
    ):
        """Your original multi-process framework"""
        print(f"Starting video generation (使用导航: {use_navigation})...")
        os.makedirs(output_dir, exist_ok=True)

        file_list = [
            os.path.join(episodes_path, f)
            for f in os.listdir(episodes_path)
            if f.endswith('.json.gz') and f != "session.json.gz"
        ]

        session = HITLSession(file_list, multi=True, hitl_data_file=dataset_file)
        tasks = [(ep, dataset_file, output_dir, frame_rate, use_navigation) for ep in session.episodes]

        with Pool(num_processes) as pool:
            results = pool.map(process_single_episode, tasks)

        successful_count = sum(results)
        print(f"Success: {successful_count}/{len(tasks)}")

    def process_single_episode(
            self,
            hitl_episode: HITLEpisode,
            dataset_file: str,
            output_dir: str,
            frame_rate: int,
            use_navigation: bool = True  # 新增参数
    ) -> bool:
        episode_id = hitl_episode.episode_info["episode_id"]
        print(f"Processing episode {episode_id} (使用导航: {use_navigation})")

        try:
            # Use your working init_env
            env = self._initialize_with_hitl_method(hitl_episode, dataset_file)
            if env is None:
                return False

            valid_frames = [f for f in hitl_episode.hitl_data["frames"] if f and len(f) > 0]
            if not valid_frames:
                env.close()
                return False
            
            output_dir_robot_view = os.path.join(output_dir, "robot_view")
            output_dir_human_third_view = os.path.join(output_dir, "human_third_view")
            os.makedirs(output_dir_robot_view, exist_ok=True)
            os.makedirs(output_dir_human_third_view, exist_ok=True)
            
            video_path = os.path.join(output_dir_robot_view, f"robot_view_{episode_id}.mp4")
            video_path1 = os.path.join(output_dir_human_third_view, f"human_view_{episode_id}.mp4")
            
            if use_navigation and NAV_SKILL_AVAILABLE:
                success = self._generate_video_with_navigation(env, hitl_episode, video_path, video_path1, frame_rate)
            else:
                success = self._generate_video_direct(env, valid_frames, video_path, video_path1, frame_rate)

            env.close()
            return success

        except Exception as e:
            print(f"Error in episode {episode_id}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _generate_video_with_navigation(self, env, hitl_episode, video_path: str, video_path1: str, frame_rate: int) -> bool:
        """使用导航技能生成视频"""
        try:
            # 1. 提取人类轨迹
            trajectory = self._extract_human_trajectory(hitl_episode)
            if len(trajectory) < 2:
                print(f"轨迹点数不足: {len(trajectory)}，回退到直接设置状态")
                valid_frames = [f for f in hitl_episode.hitl_data["frames"] if f and len(f) > 0]
                return self._generate_video_direct(env, valid_frames, video_path, video_path1, frame_rate)
            
            # 2. 采样关键点
            nav_points = self._sample_trajectory_points(trajectory)
            print(f"原始轨迹点: {len(trajectory)}, 采样后: {len(nav_points)}")
            
            # 3. 创建导航技能
            nav_skill = self._create_nav_skill(env)
            if nav_skill is None:
                print("导航技能创建失败，回退到直接设置状态")
                valid_frames = [f for f in hitl_episode.hitl_data["frames"] if f and len(f) > 0]
                return self._generate_video_direct(env, valid_frames, video_path, video_path1, frame_rate)
            
            # 4. 获取传感器信息
            obs = env.sim.get_sensor_observations()
            
            robot_sensor = "agent_1_articulated_agent_jaw_rgb"
            human_sensor = "agent_0_third_rgb"
            
            if robot_sensor not in obs or human_sensor not in obs:
                print(f"传感器缺失: robot={robot_sensor in obs}, human={human_sensor in obs}")
                return False
            
            # 获取视频尺寸
            rgb_sample = obs[robot_sensor]
            rgb1_sample = obs[human_sensor]
            if len(rgb_sample.shape) != 3 or len(rgb1_sample.shape) != 3:
                print(f"传感器数据格式异常: robot_shape={rgb_sample.shape}, human_shape={rgb1_sample.shape}")
                return False
            
            h, w = rgb_sample.shape[:2]
            h1, w1 = rgb1_sample.shape[:2]
            
            # 5. 创建视频写入器
            writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (w, h))
            writer1 = cv2.VideoWriter(video_path1, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (w1, h1))
            
            if not writer.isOpened() or not writer1.isOpened():
                print("无法创建视频写入器")
                return False
            
            # 6. 执行导航并录制
            frames_recorded = 0
            
            for i, nav_target in enumerate(nav_points):
                print(f"导航到点 {i+1}/{len(nav_points)}: {nav_target}")
                
                try:
                    # 重置导航技能
                    # nav_skill.reset(batch_idxs=[0])
                    
                    # 设置目标
                    nav_skill.set_target(nav_target, env)
                    
                    # 导航直到到达或超时
                    step_count = 0
                    max_steps_per_target = 500
                    
                    while step_count < max_steps_per_target:
                        # 检查是否到达
                        try:
                            is_done = nav_skill._is_skill_done(
                                observations=None,
                                rnn_hidden_states=None,
                                prev_actions=None,
                                masks=torch.ones(1),
                                batch_idx=0
                            )
                            if is_done:
                                break
                        except Exception as e:
                            print(f"检查技能完成状态时出错: {e}")
                            break
                        
                        try:
                            # 执行导航
                            
                            # action_tensor = torch.zeros((1, env.action_space.shape[0]))
                            action, _ = nav_skill._internal_act(
                                observations=None,
                                rnn_hidden_states=None,
                                prev_actions=None,
                                masks=torch.ones(1),
                                cur_batch_idx=0,
                                deterministic=False
                            )
                            
                            # 执行环境步骤
                            env.step(action[0].numpy())
                        except Exception as e:
                            print(f"执行导航步骤时出错: {e}")
                            import traceback
                            traceback.print_exc()
                            break
                        
                        # 获取传感器数据
                        obs = env.sim.get_sensor_observations()
                        
                        # 录制视频帧
                        if robot_sensor in obs and human_sensor in obs:
                            rgb = obs[robot_sensor]
                            rgb1 = obs[human_sensor]
                            
                            if isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8:
                                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                                writer.write(bgr)
                            
                            if isinstance(rgb1, np.ndarray) and rgb1.dtype == np.uint8:
                                bgr1 = cv2.cvtColor(rgb1, cv2.COLOR_RGB2BGR)
                                writer1.write(bgr1)
                            
                            frames_recorded += 1
                        
                        step_count += 1
                        
                except Exception as e:
                    import traceback
                    print(f"导航到目标 {nav_target} 时出错: {e}")
                    traceback.print_exc()
                    continue
            
            writer.release()
            writer1.release()
            
            print(f"使用导航技能录制了 {frames_recorded} 帧")
            return frames_recorded > 0
            
        except Exception as e:
            print(f"导航生成失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _generate_video_direct(self, env, frames: List[Dict], video_path: str, video_path1: str, frame_rate: int) -> bool:
        """原始方法：直接设置状态生成视频"""
        obs = env.sim.get_sensor_observations()
        
        robot_sensor = "agent_1_articulated_agent_jaw_rgb"
        human_sensor = "agent_0_third_rgb"
        
        if robot_sensor not in obs or human_sensor not in obs:
            print(f"传感器缺失: robot={robot_sensor in obs}, human={human_sensor in obs}")
            return False

        # rgb = obs["agent_1_articulated_agent_jaw_rgb"]
        rgb1 = obs['agent_0_third_rgb']
        rgb = obs['agent_1_articulated_agent_jaw_rgb']
        
        if len(rgb.shape) != 3 or len(rgb1.shape) != 3:
            print(f"传感器数据格式异常: robot_shape={rgb.shape}, human_shape={rgb1.shape}")
            return False
            
        h, w = rgb.shape[:2]
        h1, w1 = rgb1.shape[:2]
        
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (w, h))  # robot_view
        writer1 = cv2.VideoWriter(video_path1, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate,
                                  (w1, h1))  # human_third_view
        
        if not writer.isOpened():
            print(f"Cannot create video writer: {video_path}")
            return False

        if not writer1.isOpened():
            print(f"Cannot create video writer: {video_path1}")
            return False

        successful_frames = 0
        successful_frames1 = 0
        for fr in frames:
            try:
                self._set_environment_state(env, fr)
                obs = env.sim.get_sensor_observations()
                rgb = obs["agent_1_articulated_agent_jaw_rgb"]
                rgb1 = obs["agent_0_third_rgb"]

                if isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8:
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    writer.write(bgr)
                    successful_frames += 1
                if isinstance(rgb1, np.ndarray) and rgb1.dtype == np.uint8:
                    bgr1 = cv2.cvtColor(rgb1, cv2.COLOR_RGB2BGR)
                    writer1.write(bgr1)
                    successful_frames1 += 1
            except Exception as e:
                import traceback
                print(f"Frame error: {e}")
                traceback.print_exc()
                continue

        writer.release()
        print(f"Found {successful_frames} frames from robot view. Save video successfully in {video_path}!")
        writer1.release()
        print(f"Fount {successful_frames1} frames from human third view. Save video successfully in {video_path1}!")
        return successful_frames > 0 and successful_frames1 > 0

    def _extract_human_trajectory(self, hitl_episode: HITLEpisode) -> List[Dict]:
        """从HITL数据中提取人类代理轨迹"""
        trajectory = []
        
        valid_frames = [f for f in hitl_episode.hitl_data["frames"] if f and len(f) > 0]
        
        for frame in valid_frames:
            if "agent_states" in frame and len(frame["agent_states"]) > self.human_agent_id:
                agent_state = frame["agent_states"][self.human_agent_id]
                
                # 提取位置和旋转
                pos = agent_state["position"]  # [x, y, z]
                rot = agent_state["rotation"]  # [x, y, z, w] 四元数
                
                # 转换为yaw角
                yaw = self._quaternion_to_yaw(rot)
                
                trajectory.append({
                    "position": pos,
                    "rotation": rot,
                    "yaw": yaw,
                    "timestamp": frame.get("t", 0)
                })
        
        return trajectory
    
    def _quaternion_to_yaw(self, q: List[float]) -> float:
        """四元数转换为yaw角（弧度）"""
        x, y, z, w = q
        # 计算偏航角
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw
    
    def _sample_trajectory_points(self, trajectory: List[Dict], 
                                 min_distance: float = 0.5,
                                 max_points: int = 30) -> List[str]:
        """采样关键轨迹点，减少导航目标数量"""
        if len(trajectory) <= 2:
            # 如果轨迹点很少，直接使用
            return [f"{p['position'][0]},{p['position'][2]},{p['yaw']}" for p in trajectory]
        
        sampled_points = []
        
        # 总是包括起点
        start_point = trajectory[0]
        sampled_points.append(start_point)
        last_point = start_point
        
        # 采样中间点
        for i in range(1, len(trajectory) - 1):
            current_point = trajectory[i]
            
            # 计算距离
            pos1 = current_point["position"]
            pos2 = last_point["position"]
            distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[2] - pos2[2])**2)
            
            # 如果距离足够远，作为新的关键点
            if distance >= min_distance:
                sampled_points.append(current_point)
                last_point = current_point
        
        # 总是包括终点
        end_point = trajectory[-1]
        if len(sampled_points) == 0 or sampled_points[-1] != end_point:
            sampled_points.append(end_point)
        
        # 限制最大点数
        if len(sampled_points) > max_points:
            # 均匀采样
            indices = np.linspace(0, len(sampled_points)-1, max_points, dtype=int)
            sampled_points = [sampled_points[i] for i in indices]
        
        # 转换为导航技能需要的格式: "x,z,yaw"
        nav_points = [f"{p['position'][0]},{p['position'][2]},{p['yaw']}" for p in sampled_points]
        
        return nav_points
    
    def _create_nav_skill(self, env):
        """创建人类代理的导航技能"""
        try:
            # 创建导航技能
            observation_space = env.observation_space
            action_space = env.action_space
            batch_size = 1
            print("action_space: ", action_space)
            # 使用已转换的配置对象
            nav_skill = OraclePointNavSkill(
                config=self.nav_config,
                observation_space=observation_space,
                action_space=action_space,
                batch_size=batch_size,
                env=env,
                agent_uid=self.human_agent_id
            )
            
            print(f"成功创建导航技能 for agent_{self.human_agent_id}")
            return nav_skill
            
        except Exception as e:
            print(f"创建导航技能失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _initialize_with_hitl_method(self, hitl_episode: HITLEpisode, dataset_file: str):
        try:
            from hitl_episode import init_env

            episode_id = hitl_episode.episode_info["episode_id"]
            cfg_dict = hitl_episode.hitl_data["session"]["config"]

            env = init_env(None, episode_ids=[episode_id], cfg_dict=cfg_dict)
            if env is None:
                return None

            matching_episodes = [ep for ep in env._dataset.episodes if ep.episode_id == episode_id]
            if not matching_episodes:
                print(f"Episode {episode_id} not found in dataset")
                return None

            env.current_episode = matching_episodes[0]
            env.reset()

            print(f"Initialized environment with {len(env.sim.agents)} agents")
            return env

        except Exception as e:
            print(f"Failed to initialize env: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _inject_robot_sensor(self, sim):
        """Dynamically add agent_1_articulated_agent_jaw_rgb sensor after simulation setup"""
        try:
            from habitat_sim.sensor import SensorSpec
            import magnum as mn

            # Get agent_1's articulated agent data
            agent_data0 = sim.get_agent_data(0)
            agent_data = sim.get_agent_data(1)
            # print("agent_data0: ", agent_data0)
            # print("agent_data: ", agent_data)
            sensor_suite = agent_data.controlled_articulated_agent_sensors

            # Define new sensor
            sensor_spec = SensorSpec()
            sensor_spec.uuid = "agent_1_articulated_agent_jaw_rgb"
            sensor_spec.resolution = [256, 256]
            sensor_spec.position = mn.Vector3(0.0, 0.8, 0.0)
            sensor_spec.orientation = mn.Vector3(0.0, 0.0, 0.0)
            sensor_spec.hfov = 90.0
            sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensor_spec.gpu2gpu_transfer = False

            # Add sensor
            sensor_suite.add_sensor(sensor_spec)
            print("Injected sensor: agent_1_articulated_agent_jaw_rgb")

        except Exception as e:
            print(f"Failed to inject sensor: {e}")
            import traceback
            traceback.print_exc()

    def _set_environment_state(self, env, frame_data: Dict):
        sim = env.sim
        agent_states = frame_data.get("agent_states", [])

        if len(agent_states) > 0:
            pos = np.array(agent_states[0]["position"], dtype=np.float32)
            yaw_rad = self._extract_yaw(agent_states[0]["rotation"])
            agent_data = sim.get_agent_data(0)
            agent_data.articulated_agent.base_pos = pos
            agent_data.articulated_agent.base_rot = yaw_rad

        if len(agent_states) > 1:
            pos = np.array(agent_states[1]["position"], dtype=np.float32)
            yaw_rad = self._extract_yaw(agent_states[1]["rotation"])
            agent_data = sim.get_agent_data(1)
            agent_data.articulated_agent.base_pos = pos
            agent_data.articulated_agent.base_rot = yaw_rad

        # 触发传感器更新
        sim.step(-1)

        if "object_states" in frame_data:
            self._set_object_states(sim, frame_data["object_states"])

    def _extract_yaw(self, rot_xyzw):
        x, y, z, w = rot_xyzw
        R = np.array([
            [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w)],
            [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2)]
        ])
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def _set_object_states(self, sim, object_states: List[Dict]):
        rom = sim.get_rigid_object_manager()
        handle_map = {}
        for obj in object_states:
            handle = obj["object_handle"]
            if handle not in handle_map:
                prefix = handle.split(":")[0]
                for eh in rom.get_object_handles():
                    if eh.startswith(prefix):
                        handle_map[handle] = eh
                        break
            if handle in handle_map:
                robj = rom.get_object_by_handle(handle_map[handle])
                if robj is not None:
                    robj.translation = np.array(obj["position"], dtype=np.float32)
                    x, y, z, w = obj["rotation"]
                    # 根据你的数据格式调整顺序
                    # 如果数据是 [x, y, z, w] 格式
                    import magnum as mn
                    robj.rotation = mn.Quaternion((x, y, z), w)  # 注意：magnum 通常是 w,x,y,z 顺序


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-path", required=True)
    parser.add_argument("--dataset-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frame-rate", type=int, default=30)
    parser.add_argument("--num-processes", type=int, default=10)
    parser.add_argument("--use-navigation", action="store_true", help="使用导航技能生成更自然的运动")
    
    args = parser.parse_args()

    generator = RobotViewVideoGenerator()
    generator.generate_videos_from_episodes(
        episodes_path=args.episodes_path,
        dataset_file=args.dataset_file,
        output_dir=args.output_dir,
        frame_rate=args.frame_rate,
        num_processes=args.num_processes,
        use_navigation=args.use_navigation
    )


if __name__ == "__main__":
    main()