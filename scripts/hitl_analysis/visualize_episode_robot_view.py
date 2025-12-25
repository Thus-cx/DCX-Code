import os
import cv2
import numpy as np
from typing import List, Dict, Any
import argparse
from multiprocessing import Pool
from hitl_episode import HITLSession, HITLEpisode, divide_list
# from box import Box
from omegaconf import OmegaConf

# 尝试导入PARTNR的环境接口
try:
    from habitat_llm.agent.env.environment_interface import EnvironmentInterface

    HAS_ENV_INTERFACE = True
except ImportError:
    HAS_ENV_INTERFACE = False
    print("警告: 无法导入EnvironmentInterface，将使用备用方法")


def process_single_episode(args):
    """处理单个episode的worker函数"""
    episode_info, dataset_file, output_dir, frame_rate, use_env_interface = args

    try:
        generator = RobotViewVideoGenerator()
        success = generator.process_single_episode(
            episode_info, dataset_file, output_dir, frame_rate, use_env_interface
        )
        return success
    except Exception as e:
        print(f"处理episode时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


class RobotViewVideoGenerator:
    def __init__(self):
        self.robot_agent_id = "agent_1"  # agent_1是机器人
        self.human_agent_id = "agent_0"  # agent_0是人类

    def generate_videos_from_episodes(
            self,
            episodes_path: str,
            dataset_file: str,
            output_dir: str,
            frame_rate: int = 30,
            num_processes: int = 10,
            use_env_interface: bool = True
    ):
        """从多个episodes生成机器人视角视频（多进程版本）"""
        print("开始多线程处理")
        os.makedirs(output_dir, exist_ok=True)

        # 加载session
        file_list = [
            os.path.join(episodes_path, f)
            for f in os.listdir(episodes_path)
            if f.endswith('.json.gz') and f != "session.json.gz"
        ]

        if not file_list:
            print(f"在 {episodes_path} 中没有找到 .json.gz 文件")
            return

        session = HITLSession(file_list, multi=True, hitl_data_file=dataset_file)

        print(f"找到 {len(session.episodes)} 个episodes，使用 {num_processes} 个进程处理")
        print(f"使用环境接口: {use_env_interface}")

        # 准备参数
        tasks = [
            (episode, dataset_file, output_dir, frame_rate, use_env_interface)
            for episode in session.episodes
        ]

        # 使用进程池处理
        with Pool(num_processes) as pool:
            results = pool.map(process_single_episode, tasks)

        successful_count = sum(results)
        print(f"成功处理 {successful_count}/{len(session.episodes)} 个episodes")

    def process_single_episode(
            self,
            hitl_episode: HITLEpisode,
            dataset_file: str,
            output_dir: str,
            frame_rate: int,
            use_env_interface: bool = True
    ) -> bool:
        """处理单个episode，返回是否成功"""
        print("单线程处理")
        episode_id = hitl_episode.episode_info["episode_id"]
        print(f"处理 episode: {episode_id}")

        try:
            # 初始化环境
            if use_env_interface and HAS_ENV_INTERFACE:
                env = self._initialize_with_env_interface(hitl_episode, dataset_file)
            else:
                env = self._initialize_with_hitl_method(hitl_episode, dataset_file)

            if env is None:
                return False

            # 获取有效帧
            valid_frames = [f for f in hitl_episode.hitl_data["frames"] if f and len(f) > 0]

            if not valid_frames:
                print(f"Episode {episode_id} 没有有效帧")
                env.close()
                return False

            # 生成视频
            video_path = os.path.join(output_dir, f"robot_view_{episode_id}.mp4")
            success = self._generate_video(env, valid_frames, video_path, frame_rate)

            env.close()

            if success:
                print(f"完成: {video_path}")
                return True
            else:
                print(f"生成视频失败: {episode_id}")
                return False

        except Exception as e:
            print(f"处理episode {episode_id} 时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _initialize_with_env_interface(self, hitl_episode: HITLEpisode, dataset_file: str):
        """使用EnvironmentInterface初始化环境（推荐方法）"""
        def fix_multi_agent_config(cfg):
            """修复 agents_order 和 agents 为 list 类型"""
            from omegaconf import ListConfig, DictConfig

            # 修复 agents_order: {"0": "agent_0", "1": "agent_1"} → ["agent_0", "agent_1"]
            if isinstance(cfg.habitat.simulator.agents_order, (dict, DictConfig)):
                order_items = sorted(cfg.habitat.simulator.agents_order.items(), key=lambda x: int(x[0]))
                cfg.habitat.simulator.agents_order = ListConfig([item[1] for item in order_items])

            # 修复 agents: {"agent_0": {...}, "agent_1": {...}} → [{...}, {...}]
            if isinstance(cfg.habitat.simulator.agents, (dict, DictConfig)):
                agent_list = []
                for agent_name in cfg.habitat.simulator.agents_order:
                    if agent_name in cfg.habitat.simulator.agents:
                        agent_list.append(cfg.habitat.simulator.agents[agent_name])
                cfg.habitat.simulator.agents = ListConfig(agent_list)

            return cfg

        
        try:
            from habitat_llm.agent.env.environment_interface import EnvironmentInterface
            from habitat_llm.agent.env import sensors

            episode_id = hitl_episode.episode_info["episode_id"]
            scene_id = hitl_episode.hitl_data["episode"]["scene_id"]
            cfg_dict = hitl_episode.hitl_data["session"]["config"]
            # cfg_box = Box(cfg_dict, box_dots=True)
            cfg = OmegaConf.create(cfg_dict)
            cfg = fix_multi_agent_config(cfg) # 补充修复
            # print("dataset_file:", dataset_file)
            sensors.register_sensors(cfg)
            cfg.habitat.dataset.data_path = dataset_file
            cfg.habitat.dataset.content_scenes = [scene_id]
            # cfg.habitat.simulator.agents_order = ["agent_0","agent_1"]
            # print(type(cfg.habitat.simulator.agents))
            # print("cfg.habitat.simulator.agents:", cfg.habitat.simulator.agents)
            # print("cfg.habitat.dataset:", cfg.habitat.dataset)
            
            # print("agents_order type:", type(cfg.habitat.simulator.agents_order))
            # print("agents_order value:", cfg.habitat.simulator.agents_order)
            # print("agents type:", type(cfg.habitat.simulator.agents))

            print("After fix - agents_order:", cfg.habitat.simulator.agents_order)
            print("After fix - agents type:", type(cfg.habitat.simulator.agents))
            print("After fix - agents length:", len(cfg.habitat.simulator.agents))

            # 使用EnvironmentInterface，它应该能正确处理多代理
            env_if = EnvironmentInterface(cfg, dataset=None, init_wg=True, init_env=True)
            # print("cfg_box.habitat.dataset")
            env = env_if.env

            # 设置当前episode
            matching_episodes = [
                ep for ep in env._dataset.episodes
                if ep.episode_id == episode_id
            ]

            if not matching_episodes:
                print(f"在数据集中找不到episode {episode_id}")
                return None

            env.current_episode = matching_episodes[0]

            # 重置环境以应用episode
            env.reset()

            print(f"使用EnvironmentInterface初始化环境，代理数量: {len(env.sim.agents)}")

            return env

        except Exception as e:
            import traceback
            print(f"使用EnvironmentInterface初始化失败: {e}")
            # traceback.print_exc()
            return self._initialize_with_hitl_method(hitl_episode, dataset_file)

    def _initialize_with_hitl_method(self, hitl_episode: HITLEpisode, dataset_file: str):
        """使用HITL方法初始化环境（备用方法）"""

        try:
            from hitl_episode import init_env

            episode_id = hitl_episode.episode_info["episode_id"]
            cfg_dict = hitl_episode.hitl_data["session"]["config"]

            # 使用HITL的init_env函数
            env = init_env(None, episode_ids=[episode_id], cfg_dict=cfg_dict)

            # 设置当前episode
            matching_episodes = [
                ep for ep in env._dataset.episodes
                if ep.episode_id == episode_id
            ]

            if not matching_episodes:
                print(f"在数据集中找不到episode {episode_id}")
                return None

            env.current_episode = matching_episodes[0]

            # 初始化所有代理
            print("env.sim.agents:", env.sim.agents)
            print(f"初始化 {len(env.sim.agents)} 个代理")
            for agent_id in range(len(env.sim.agents)):
                env.sim.initialize_agent(agent_id)
                print(f"初始化代理 {agent_id}")

            env.reset()

            print(f"使用HITL方法初始化环境，代理数量: {len(env.sim.agents)}")

            return env

        except Exception as e:
            import traceback
            print(f"使用HITL方法初始化失败: {e}")
            traceback.print_exc()
            return None

    def _generate_video(
            self,
            env,
            frames: List[Dict],
            video_path: str,
            frame_rate: int
    ) -> bool:
        """生成视频文件，返回是否成功"""

        # 测试获取一帧来确定视频尺寸
        test_frame = self._get_robot_observation(env, frames[0])
        if test_frame is None:
            print("无法获取机器人视角观察")
            observations = env.sim.get_sensor_observations()
            print("可用的传感器键:", list(observations.keys()))
            return False

        height, width = test_frame.shape[:2]

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))

        if not video_writer.isOpened():
            print(f"无法创建视频文件: {video_path}")
            return False

        print(f"生成视频: {video_path}, 尺寸: {width}x{height}, 帧数: {len(frames)}")

        # 逐帧处理
        successful_frames = 0
        for frame_idx, frame_data in enumerate(frames):
            try:
                # 设置环境状态
                self._set_environment_state(env, frame_data)

                # 获取机器人视角
                frame = self._get_robot_observation(env, frame_data)

                if frame is not None:
                    # 转换为BGR
                    frame_bgr = self._convert_frame_to_bgr(frame)
                    if frame_bgr is not None:
                        video_writer.write(frame_bgr)
                        successful_frames += 1

                if frame_idx % 100 == 0:
                    print(f"进度: {frame_idx}/{len(frames)}")

            except Exception as e:
                print(f"处理帧 {frame_idx} 时出错: {e}")
                continue

        video_writer.release()

        success_ratio = successful_frames / len(frames)
        print(f"视频生成完成: {successful_frames}/{len(frames)} 帧成功 ({success_ratio:.1%})")

        return successful_frames > 0

    def _convert_frame_to_bgr(self, frame):
        """将帧转换为BGR格式"""
        try:
            if len(frame.shape) == 3:
                if frame.shape[2] == 3:  # RGB
                    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                elif frame.shape[2] == 4:  # RGBA
                    return cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR)
            elif len(frame.shape) == 2:  # 单通道
                return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            print(f"帧转换出错: {e}")
        return None

    def _set_environment_state(self, env, frame_data: Dict):
        """设置环境状态到指定帧"""

        # 设置代理状态
        agent_states = frame_data.get("agent_states", [])

        # 检查代理数量是否匹配
        if len(agent_states) != len(env.sim.agents):
            print(f"警告: 帧数据中有 {len(agent_states)} 个代理状态，但环境中只有 {len(env.sim.agents)} 个代理")
            # 只设置存在的代理状态
            agent_states = agent_states[:len(env.sim.agents)]
            

        for agent_idx, agent_state in enumerate(agent_states):
            position = np.array(agent_state["position"], dtype=np.float32)
            rotation = np.array(agent_state["rotation"], dtype=np.float32)

            # 创建代理状态对象
            agent = env.sim.agents[agent_idx]

            # 使用 Habitat 的 API
            from habitat_sim.agent import AgentState
            new_state = AgentState()
            new_state.position = position
            new_state.rotation = rotation

            # 设置代理状态
            agent.set_state(new_state)

        # 设置物体状态
        self._set_object_states(env, frame_data.get("object_states", []))

    def _set_object_states(self, env, object_states: List[Dict]):
        """设置物体状态"""

        rigid_obj_mgr = env.sim.get_rigid_object_manager()

        for obj_state in object_states:
            obj_handle = obj_state["object_handle"]
            position = np.array(obj_state["position"], dtype=np.float32)
            rotation = np.array(obj_state["rotation"], dtype=np.float32)

            try:
                obj = rigid_obj_mgr.get_object_by_handle(obj_handle)
                if obj is not None:
                    obj.translation = position
                    # 设置旋转
                    if len(rotation) == 4:  # 四元数
                        obj.rotation = rotation
            except Exception as e:
                # 忽略不存在的物体
                continue

    def _get_robot_observation(self, env, frame_data: Dict):
        """获取机器人视角的观察"""

        try:
            # 获取当前所有观察
            observations = env.sim.get_sensor_observations()

            # 尝试机器人代理的传感器名称
            possible_sensor_names = [
                f"{self.robot_agent_id}_head_rgb",  # 最可能的名称
                f"{self.robot_agent_id}_rgb",
                f"{self.robot_agent_id}_color_sensor",
                f"{self.robot_agent_id}_camera",
                "robot_rgb",
                "robot_color_sensor"
            ]
            print("Available sensors:", list(env.sim._sensor_suite.sensors.keys()))

            for sensor_name in possible_sensor_names:
                if sensor_name in observations:
                    obs = observations[sensor_name]
                    if obs is not None and hasattr(obs, 'shape') and len(obs.shape) >= 2:
                        print(f"使用机器人传感器: {sensor_name}")
                        return obs

            # 如果没找到机器人传感器，尝试任何RGB传感器
            for key, obs in observations.items():
                if any(x in key for x in ['rgb', 'color']) and obs is not None:
                    if hasattr(obs, 'shape') and len(obs.shape) >= 2:
                        print(f"使用通用RGB传感器: {key}")
                        return obs

            return None

        except Exception as e:
            print(f"获取观察时出错: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description="从四足机器人视角生成人类活动视频数据集")
    parser.add_argument(
        "--episodes-path",
        type=str,
        required=True,
        help="HITL episode数据路径"
    )
    parser.add_argument(
        "--dataset-file",
        type=str,
        required=True,
        help="环境配置文件路径"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/hitl_data/p5_multi_val/processed/best/robot_view_videos",
        help="输出视频目录"
    )
    parser.add_argument(
        "--frame-rate",
        type=int,
        default=30,
        help="输出视频帧率"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=10,
        help="进程数"
    )
    parser.add_argument(
        "--use-env-interface",
        action="store_true",
        help="使用EnvironmentInterface初始化环境（推荐）"
    )

    args = parser.parse_args()

    generator = RobotViewVideoGenerator()
    generator.generate_videos_from_episodes(
        episodes_path=args.episodes_path,
        dataset_file=args.dataset_file,
        output_dir=args.output_dir,
        frame_rate=args.frame_rate,
        num_processes=args.num_processes,
        use_env_interface=args.use_env_interface
    )


if __name__ == "__main__":
    main()