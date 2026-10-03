import json
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

# 目标常见物体列表（你可以根据需要增删）
TARGET_OBJECTS = ['apple']


def quaternion_to_forward_vector(quat):
    # 假设 Habitat 默认格式，具体视你的数据是 [w,x,y,z] 还是 [x,y,z,w]
    # 这里假设将四元数转换为朝向向量 (基于 Y轴向上, -Z 轴向前的假设)
    # scipy Rotation 默认接收 [x, y, z, w]
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # 如果你的 JSON 是 [w,x,y,z]
    # 如果你的 JSON 原本就是 [x,y,z,w]，请直接用 R.from_quat(quat)
    forward_vector = r.apply([0, 0, -1])
    return np.array([forward_vector[0], forward_vector[2]])  # 只取 2D 平面 (X, Z)


def angle_between_quaternions(q1, q2):
    # 计算两个四元数之间的角位移（弧度）
    r1 = R.from_quat([q1[1], q1[2], q1[3], q1[0]])
    r2 = R.from_quat([q2[1], q2[2], q2[3], q2[0]])
    diff = r1.inv() * r2
    return diff.magnitude()


def find_best_video_clips(json_dir):
    best_clips = []

    for filename in os.listdir(json_dir):
        if not filename.endswith('.json'):
            continue

        with open(os.path.join(json_dir, filename), 'r') as f:
            data = json.load(f)

        episode_id = data.get("episode_id", "unknown")
        steps_data = {step["step"]: step for step in data["steps"]}

        for subtask in data["subtasks"]:
            # 1. 检查是否包含日常目标物体
            obj_name_raw = subtask["object_names"][0]  # e.g., "apple_1"
            is_target_obj = any(target in obj_name_raw.lower() for target in TARGET_OBJECTS)

            if not is_target_obj:
                continue

            start_step = subtask["start_step"]
            end_step = subtask["end_step"]

            occlusion_frames = 0
            jitter_frames = 0
            total_transit_frames = 0

            # 遍历这个 subtask 的所有帧，计算物理指标
            for s in range(start_step, end_step):
                if s not in steps_data or (s + 1) not in steps_data:
                    continue

                total_transit_frames += 1
                current_step = steps_data[s]
                next_step = steps_data[s + 1]

                # 获取 2D 坐标 (X, Z)
                h_pos = np.array([current_step["human_agent"]["pos"][0], current_step["human_agent"]["pos"][2]])
                r_pos = np.array([current_step["robot_agent"]["pos"][0], current_step["robot_agent"]["pos"][2]])

                dt = np.linalg.norm(r_pos - h_pos)
                if dt == 0: dt = 0.001

                # --- 检查 Torso Occlusion (< -0.3) ---
                v_h = quaternion_to_forward_vector(current_step["human_agent"]["rot"])
                relative_pos_norm = (r_pos - h_pos) / dt
                dot_product = np.dot(v_h, relative_pos_norm)
                if dot_product < -0.3:
                    occlusion_frames += 1

                # --- 检查 Jitter (> 0.8 rad/s) ---
                time_diff = next_step["time_sec"] - current_step["time_sec"]
                if time_diff > 0:
                    angular_dist = angle_between_quaternions(current_step["robot_agent"]["rot"],
                                                             next_step["robot_agent"]["rot"])
                    angular_vel = angular_dist / time_diff
                    if angular_vel > 0.8:
                        jitter_frames += 1

            # 评估该片段的“极品程度”
            if total_transit_frames > 0:
                occ_ratio = occlusion_frames / total_transit_frames
                jitter_ratio = jitter_frames / total_transit_frames

                # 如果遮挡比例 > 40% 且 抖动比例 > 10%，说明是一个非常有代表性的 Hard 样本
                if occ_ratio > 0.4 and jitter_ratio > 0.1:
                    best_clips.append({
                        "episode": episode_id,
                        "object": obj_name_raw,
                        "start_frame": start_step,
                        "end_frame": end_step,
                        "occ_ratio": round(occ_ratio, 2),
                        "jitter_ratio": round(jitter_ratio, 2)
                    })

    # 按遮挡严重程度和抖动程度排序
    best_clips.sort(key=lambda x: (x["occ_ratio"] + x["jitter_ratio"]), reverse=True)
    return best_clips

# 执行筛选
top_clips = find_best_video_clips("/root/autodl-tmp/partnr-planner-main/final_raw_videos/metadata")
for clip in top_clips[:10]:
   print(clip)