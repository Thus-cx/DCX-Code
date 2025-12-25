#!/usr/bin/env python3
"""
Native Multi-Agent Simulator with Full Debug Mode
- Stops on error and prints full traceback
- Verifies file paths before sim creation
- Ensures agent state consistency
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Any
import argparse
import gzip
import json
import traceback
from pathlib import Path
from multiprocessing import Pool


def load_json_any(fp):
    """Load gzipped or plain JSON file with full debug output."""
    print(f"Loading data from: {fp}")
    if not os.path.exists(fp):
        raise FileNotFoundError(f"File does not exist: {fp}")

    try:
        if fp.endswith(".gz"):
            with gzip.open(fp, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)

        print(f"✓ Successfully loaded: {fp}")
        return data

    except Exception as e:
        print(f"[ERROR] Failed to load {fp}")
        traceback.print_exc()
        raise


def get_episode_scene_id(dataset_json, episode_id):
    """Extract scene_id from val dataset; stops if not found."""
    print(f"Searching for episode_id={episode_id} in dataset")
    for ep in dataset_json["episodes"]:
        if str(ep["episode_id"]) == str(episode_id):
            full_path = ep["scene_id"]
            print(f"Raw scene_id: {full_path}")
            scene_id = full_path.split("/")[-1].replace(".basis.glb", "")
            print(f"Resolved scene_id: {scene_id}")
            return scene_id
    raise KeyError(f"Episode {episode_id} not found in dataset")


def build_native_multi_agent_sim(scene_id: str):
    """Build native multi-agent simulator with correct SimulatorConfiguration usage"""
    try:
        import habitat_sim

        scene_path = f"data/fpss/stages/{scene_id}.glb"
        print(f"Expected scene file path: {scene_path}")
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"Scene file not found: {scene_path}")

        # --- Step 1: Create empty config and assign fields one by one ---
        sim_cfg = habitat_sim.SimulatorConfiguration()

        # ✅ These are valid fields in v0.3.3
        sim_cfg.scene_id = scene_path
        sim_cfg.enable_physics = True
        sim_cfg.gpu_device_id = 0

        # ⚠️ create_renderer 是合法字段，必须显式启用
        #   否则不会创建 renderer，导致视觉传感器失败
        sim_cfg.create_renderer = True  # ← 关键！让系统知道需要渲染器

        physics_config = "data/default.phys_scene_config.json"
        if os.path.exists(physics_config):
            sim_cfg.physics_config_file = physics_config
        else:
            print("⚠️ Physics config not found, using default.")

        # --- Agent 0: Human ---
        agent_cfg_0 = habitat_sim.AgentConfiguration()
        agent_cfg_0.radius = 0.3
        agent_cfg_0.height = 1.5

        human_rgb = habitat_sim.SensorSpec()
        human_rgb.uuid = "agent_0_head_rgb"
        human_rgb.resolution = [256, 256]
        human_rgb.position = [0.0, 1.25, 0.0]
        human_rgb.orientation = [0.0, 0.0, 0.0]
        human_rgb.hfov = 90.0
        human_rgb.sensor_type = habitat_sim.SensorType.COLOR
        human_rgb.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        human_rgb.gpu2gpu_transfer = False
        agent_cfg_0.sensor_specifications = [human_rgb]

        # --- Agent 1: Robot ---
        agent_cfg_1 = habitat_sim.AgentConfiguration()
        agent_cfg_1.radius = 0.4
        agent_cfg_1.height = 0.9

        robot_rgb = habitat_sim.SensorSpec()
        robot_rgb.uuid = "agent_1_head_rgb"
        robot_rgb.resolution = [256, 256]
        robot_rgb.position = [0.0, 0.8, 0.0]
        robot_rgb.orientation = [0.0, 0.0, 0.0]
        robot_rgb.hfov = 90.0
        robot_rgb.sensor_type = habitat_sim.SensorType.COLOR
        robot_rgb.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        robot_rgb.gpu2gpu_transfer = False
        agent_cfg_1.sensor_specifications = [robot_rgb]

        # ✅ List of agents
        agents = [agent_cfg_0, agent_cfg_1]

        # Build backend configuration
        backend_cfg = habitat_sim.Configuration(sim_cfg, agents)
        sim = habitat_sim.Simulator(backend_cfg)

        print(f"✅ Native multi-agent sim created: {len(sim.agents)} agents")
        return sim

    except Exception as e:
        print(f"[FATAL] Failed to build simulator for scene '{scene_id}'")
        traceback.print_exc()
        raise




def set_agent_state(sim, agent_id: int, pos, rot_xyzw):
    """Set agent pose with full error trace."""
    try:
        import habitat_sim
        from habitat_sim.utils.common import quat_from_coeffs

        agent = sim.get_agent(agent_id)
        state = agent.get_state()

        state.position = np.array(pos, dtype=np.float32)

        x, y, z, w = rot_xyzw
        q_wxyz = np.array([w, x, y, z], dtype=np.float32)
        state.rotation = quat_from_coeffs(q_wxyz)

        agent.set_state(state, reset_sensors=True)
        # print(f"✓ Set agent {agent_id} at {pos}, rotation applied")

    except Exception as e:
        print(f"[ERROR] Failed to set agent {agent_id} state:")
        traceback.print_exc()
        raise


def set_object_states(sim, object_states):
    """Set rigid object poses with full error tracing."""
    try:
        rom = sim.get_rigid_object_manager()
        existing_handles = {
            h.split("/")[-1]: h for h in rom.get_existing_object_handles()
        }

        for obj in object_states:
            handle = obj["object_handle"]
            base_handle = handle.split(":")[0]

            full_handle = None
            if base_handle in existing_handles:
                full_handle = existing_handles[base_handle]
            else:
                for eh in existing_handles:
                    if eh.startswith(base_handle):
                        full_handle = eh
                        break
            if not full_handle:
                print(f"⚠️ Object handle not found: {handle}")
                continue

            robj = rom.get_object_by_handle(full_handle)
            if robj is None:
                print(f"⚠️ Cannot get object by handle: {full_handle}")
                continue

            # Set translation
            pos = np.array(obj["position"], dtype=np.float32)
            robj.translation = pos

            # Set rotation
            x, y, z, w = obj["rotation"]
            from habitat_sim.utils.common import quat_from_coeffs
            q_wxyz = np.array([w, x, y, z], dtype=np.float32)
            robj.rotation = quat_from_coeffs(q_wxyz)

            # print(f"✓ Updated object: {handle} → {pos}")

    except Exception as e:
        print("[ERROR] Failed to set object states:")
        traceback.print_exc()
        raise


def process_single_episode(args):
    """Worker function that preserves full exception stack."""
    try:
        episode_file, dataset_file, output_dir, frame_rate = args
        generator = RobotViewVideoGenerator()
        success = generator.process_single_episode(
            episode_file, dataset_file, output_dir, frame_rate
        )
        return success
    except Exception as e:
        print(f"[CRITICAL] Unhandled error in worker process:")
        traceback.print_exc()
        raise  # Let it crash so you see the real issue


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
        """Multi-process entry point with early validation."""
        print("=" * 60)
        print("🚀 STARTING NATIVE MULTI-AGENT VIDEO GENERATION")
        print("=" * 60)

        if not os.path.exists(episodes_path):
            raise FileNotFoundError(f"Episodes path does not exist: {episodes_path}")
        if not os.path.exists(dataset_file):
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

        os.makedirs(output_dir, exist_ok=True)

        episode_files = sorted([
            str(p) for p in Path(episodes_path).iterdir()
            if p.name[0].isdigit() and p.suffix in [".json", ".gz"]
        ])

        if not episode_files:
            raise RuntimeError(f"No valid episode files in {episodes_path}")

        tasks = [(f, dataset_file, output_dir, frame_rate) for f in episode_files]

        with Pool(num_processes) as pool:
            results = pool.map(process_single_episode, tasks)

        successful_count = sum(results)
        print(f"\n🎉 Completed processing: {successful_count}/{len(tasks)} episodes\n")


    def process_single_episode(
        self,
        episode_file: str,
        dataset_file: str,
        output_dir: str,
        frame_rate: int
    ) -> bool:
        """Process one episode with full debug info."""
        print(f"\n🔍 Processing episode file: {episode_file}")
        epi = load_json_any(episode_file)
        episode_id = epi["episode"]["episode_id"]
        print(f"📌 Episode ID: {episode_id}")

        dataset_json = load_json_any(dataset_file)
        scene_id = get_episode_scene_id(dataset_json, episode_id)

        # Build sim
        sim = build_native_multi_agent_sim(scene_id)
        if sim is None:
            raise RuntimeError("Failed to create simulator.")

        frames = [f for f in epi["frames"] if f and len(f) > 0]
        if not frames:
            raise ValueError(f"No valid frames in episode {episode_id}")

        video_path = os.path.join(output_dir, f"robot_view_{episode_id}.mp4")
        writer = None

        try:
            for frame_idx, fr in enumerate(frames):
                print(f"  → Frame {frame_idx}/{len(frames)-1}", end='\r')

                # Validate agent states
                agent_states = fr.get("agent_states", [])
                if len(agent_states) != 2:
                    raise ValueError(
                        f"Expected 2 agent_states, got {len(agent_states)} "
                        f"in episode {episode_id}, frame {frame_idx}"
                    )

                # Set human
                set_agent_state(sim, 0, agent_states[0]["position"], agent_states[0]["rotation"])
                # Set robot
                set_agent_state(sim, 1, agent_states[1]["position"], agent_states[1]["rotation"])

                # Set objects
                if "object_states" in fr:
                    set_object_states(sim, fr["object_states"])

                # Get observation
                obs = sim.get_sensor_observations(agent_ids=[1])
                rgb = obs[1]["agent_1_head_rgb"]

                if writer is None:
                    h, w = rgb.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(video_path, fourcc, frame_rate, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError(f"Cannot create video writer: {video_path}")

                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                writer.write(bgr)

            print()

        except Exception as e:
            print(f"\n💥 ERROR in episode {episode_id}, frame {frame_idx}:")
            traceback.print_exc()
            raise

        finally:
            if writer is not None:
                writer.release()
                print(f"🎬 Saved: {video_path}")
            sim.close()
            print(f"🧹 Closed simulator for episode {episode_id}")

        return True


def main():
    parser = argparse.ArgumentParser(description="Generate robot-view videos via native habitat-sim")
    parser.add_argument("--episodes-path", required=True)
    parser.add_argument("--dataset-file", required=True)
    # parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-dir", type=str, default='data/hitl_data/p5_multi_val/processed/best/habitat_sim_robot_view_videos')
    parser.add_argument("--frame-rate", type=int, default=30)
    parser.add_argument("--num-processes", type=int, default=1)
    args = parser.parse_args()

    try:
        generator = RobotViewVideoGenerator()
        generator.generate_videos_from_episodes(
            episodes_path=args.episodes_path,
            dataset_file=args.dataset_file,
            output_dir=args.output_dir,
            frame_rate=args.frame_rate,
            num_processes=args.num_processes
        )
    except Exception as e:
        print("\n🛑 FATAL ERROR — Execution stopped.\n")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
