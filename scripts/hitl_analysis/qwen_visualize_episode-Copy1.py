#!/usr/bin/env python3
"""
Generate robot-view videos using init_env() and safe sensor injection.
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Any
import argparse
from multiprocessing import Pool
import habitat_sim

# Your existing data loader
from hitl_episode import HITLSession, HITLEpisode, init_env


def process_single_episode(args):
    episode_info, dataset_file, output_dir, frame_rate = args
    try:
        generator = RobotViewVideoGenerator()
        success = generator.process_single_episode(
            episode_info, dataset_file, output_dir, frame_rate
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

    def generate_videos_from_episodes(
            self,
            episodes_path: str,
            dataset_file: str,
            output_dir: str,
            frame_rate: int = 30,
            num_processes: int = 10
    ):
        """Your original multi-process framework"""
        print("Starting video generation...")
        os.makedirs(output_dir, exist_ok=True)

        file_list = [
            os.path.join(episodes_path, f)
            for f in os.listdir(episodes_path)
            if f.endswith('.json.gz') and f != "session.json.gz"
        ]

        session = HITLSession(file_list, multi=True, hitl_data_file=dataset_file)
        tasks = [(ep, dataset_file, output_dir, frame_rate) for ep in session.episodes]

        with Pool(num_processes) as pool:
            results = pool.map(process_single_episode, tasks)

        successful_count = sum(results)
        print(f"Success: {successful_count}/{len(tasks)}")

    def process_single_episode(
            self,
            hitl_episode: HITLEpisode,
            dataset_file: str,
            output_dir: str,
            frame_rate: int
    ) -> bool:
        episode_id = hitl_episode.episode_info["episode_id"]
        print(f"Processing episode {episode_id}")

        try:
            # Use your working init_env
            env = self._initialize_with_hitl_method(hitl_episode, dataset_file)
            if env is None:
                return False

            # Inject robot jaw RGB sensor
            # self._inject_robot_sensor(env.sim)

            valid_frames = [f for f in hitl_episode.hitl_data["frames"] if f and len(f) > 0]
            if not valid_frames:
                env.close()
                return False
            output_dir_robot_view = os.path.join(output_dir, "robot_view")
            output_dir_human_third_view = os.path.join(output_dir, "human_third_view")
            video_path = os.path.join(output_dir_robot_view, f"robot_view_{episode_id}.mp4")
            video_path1 = os.path.join(output_dir_human_third_view, f"human_view_{episode_id}.mp4")
            success = self._generate_video(env, valid_frames, video_path, video_path1, frame_rate)

            env.close()
            return success

        except Exception as e:
            print(f"Error in episode {episode_id}: {e}")
            import traceback
            traceback.print_exc()
            return False

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

    def _generate_video(self, env, frames: List[Dict], video_path: str, video_path1: str, frame_rate: int) -> bool:
        # print("get_sensor_observations: ")
        # print(env.sim.get_sensor_observations())
        obs = env.sim.get_sensor_observations()
        print("Available sensors:", list(obs.keys()))
        print("Type of agent_1_articulated_agent_jaw_rgb:", type(obs.get("agent_1_articulated_agent_jaw_rgb", None)))
        print("Shape:", obs.get("agent_1_articulated_agent_jaw_rgb", np.zeros(0)).shape)
        print("Max pixel value:", obs.get("agent_1_articulated_agent_jaw_rgb", np.zeros(0)).max())
        if "agent_1_articulated_agent_jaw_rgb" not in obs:
            print("Robot jaw sensor not found:", list(obs.keys()))
            return False

        # rgb = obs["agent_1_articulated_agent_jaw_rgb"]
        rgb1 = obs['agent_0_third_rgb']
        rgb = obs['agent_1_articulated_agent_jaw_rgb']
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

    def _set_environment_state(self, env, frame_data: Dict):
        sim = env.sim
        agent_states = frame_data.get("agent_states", [])

        if len(agent_states) > 0:
            pos = np.array(agent_states[0]["position"], dtype=np.float32)
            yaw_rad = self._extract_yaw(agent_states[0]["rotation"])
            agent_data = sim.get_agent_data(0)
            agent_data.articulated_agent.base_pos = pos
            agent_data.articulated_agent.base_rot = yaw_rad
            # agent_data.agent_sensors.update()
            # print("agent_data: ", agent_data)
            # agent_data.controlled_articulated_agent_sensors.update()

        if len(agent_states) > 1:
            pos = np.array(agent_states[1]["position"], dtype=np.float32)
            yaw_rad = self._extract_yaw(agent_states[1]["rotation"])
            agent_data = sim.get_agent_data(1)
            agent_data.articulated_agent.base_pos = pos
            agent_data.articulated_agent.base_rot = yaw_rad
            # print("agent_data1: ", agent_data)
            # if agent_data.agent_sensors is not None:
            #     agent_data.agent_sensors.update()
            # agent_data.controlled_articulated_agent_sensors.update()

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
    args = parser.parse_args()

    generator = RobotViewVideoGenerator()
    generator.generate_videos_from_episodes(
        episodes_path=args.episodes_path,
        dataset_file=args.dataset_file,
        output_dir=args.output_dir,
        frame_rate=args.frame_rate,
        num_processes=args.num_processes
    )


if __name__ == "__main__":
    main()
