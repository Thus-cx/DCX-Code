import json
import os
import numpy as np
import math
import random
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.spatial.transform import Rotation as R

# ================= 配置区域 =================
# 输出目录 (可在 main 中覆盖)
OUTPUT_DIR = "processed_dataset_rovi"
OUTPUT_RES = (480, 480)
FPS = 10.0

# 协议参数
SAFE_MARGIN_FRAMES = 50  # 0.5s (Arrival前的安全缓冲)
CONTEXT_FRAMES = 20  # 1.0s (Accumulative起始上下文)
WINDOW_FRAMES = 60  # 3.0s (滑动窗口长度)
MIN_TASK_DURATION = 6.0 # 最小任务时长(秒)


# ================= 辅助函数 =================

def get_scipy_quat(habitat_quat):
    # w,x,y,z -> x,y,z,w
    w, x, y, z = habitat_quat
    return [x, y, z, w]


def calculate_metrics_for_slice(robot_steps, human_steps, steps_data):
    """
    计算特定切片范围内的指标
    """
    if not robot_steps or len(robot_steps) < 2:
        return {"view_loss_ratio": 0.0, "proximity_ratio": 0.0, "shake_ratio": 0.0, "avg_view_loss": 0.0}

    dists = []
    view_losses = 0
    ang_vels = []
    view_loss_values = []  # 用于计算平均值

    for i in range(len(robot_steps)):
        r_step = robot_steps[i]
        h_step = human_steps[i]

        # 1. 距离
        r_pos = np.array(r_step['pos'])
        target = np.array(h_step['pos'])
        vec_2d = np.array([target[0] - r_pos[0], 0, target[2] - r_pos[2]])
        dist = np.linalg.norm(vec_2d)
        dists.append(dist)

        # 2. 角速度 (Shake)
        if i > 0:
            dt = steps_data[i]['time_sec'] - steps_data[i - 1]['time_sec']
            if dt > 1e-4:
                q_curr = R.from_quat(get_scipy_quat(r_step['rot']))
                q_prev = R.from_quat(get_scipy_quat(robot_steps[i - 1]['rot']))
                q_diff = q_prev.inv() * q_curr
                ang_vels.append(q_diff.magnitude() / dt)
            else:
                ang_vels.append(0.0)
        else:
            ang_vels.append(0.0)

        # 3. View Loss
        rot = R.from_quat(get_scipy_quat(r_step['rot']))
        forward_3d = rot.apply([0, 0, -1])
        forward_2d = np.array([forward_3d[0], 0, forward_3d[2]])
        norm_f = np.linalg.norm(forward_2d)
        forward_2d = forward_2d / norm_f if norm_f > 1e-6 else np.array([0, 0, 1])

        vec_norm = vec_2d / dist if dist > 0.05 else np.zeros(3)

        dot = 0.0
        if dist > 0.05:
            dot = np.dot(forward_2d, vec_norm)  # 1.0 = 正对

        # 简单定义：dot < 0.5 (60度以外) 视为丢失
        # View Loss Value: 0 (Good) -> 1 (Bad)
        vl_val = 1.0 - max(0.0, dot)
        view_loss_values.append(vl_val)

        if dot <= 0.5:
            view_losses += 1

    count = len(dists)
    if count == 0: return {}

    return {
        "view_loss_ratio": view_losses / count,
        "avg_view_loss": float(np.mean(view_loss_values)) if view_loss_values else 0.0,
        "proximity_ratio": sum(1 for d in dists if d < 0.8) / count,  # 0.8m 内视为过近
        "shake_ratio": sum(1 for v in ang_vels if abs(v) > 0.8) / len(ang_vels),  # 抖动阈值
        "min_dist": float(np.min(dists)),
        "avg_dist": float(np.mean(dists)),
        "max_ang_vel": float(np.max(ang_vels))
    }


def get_difficulty_bucket(metrics):
    """根据指标判断难度等级"""
    vl = metrics.get('view_loss_ratio', 0.0)
    shake = metrics.get('shake_ratio', 0.0)
    prox = metrics.get('proximity_ratio', 0.0)

    if vl > 0.6 or shake > 0.1 or prox > 0.7:
        return "Hard"
    elif vl > 0.3 or shake > 0.05 or prox > 0.4:
        return "Medium"
    else:
        return "Easy"


def process_episode(video_path, metadata_file, output_base_dir):
    # 创建输出目录结构
    view_names = ['ego_raw', 'ego_ann', 'third_raw', 'third_ann', 'global']
    raw_dirs = {k: os.path.join(output_base_dir, k) for k in view_names}
    for d in raw_dirs.values(): os.makedirs(d, exist_ok=True)

    with open(metadata_file, 'r') as f:
        data = json.load(f)

    ep_id = data.get('episode_id', 'unknown')
    instruction = data.get('instruction', 'unknown')
    all_steps = data['steps']
    total_frames = len(all_steps)

    # 确定 Robot Active 时间点
    robot_active_step = 0
    for step_idx in range(len(all_steps)):
        if all_steps[step_idx]["robot_active"]:
            robot_active_step = step_idx
            break

    # 视频加载 (Lazy Loading)
    # 注意：这里假设视频文件存在。如果文件缺失可能会报错，建议加 try-except
    try:
        vid_files = {
            'ego_raw': VideoFileClip(os.path.join(video_path, 'ego_raw_videos', f"{ep_id}.mp4")),
            'ego_ann': VideoFileClip(os.path.join(video_path, 'ego_ann_raw_videos', f"{ep_id}.mp4")),
            'third_raw': VideoFileClip(os.path.join(video_path, 'third_raw_videos', f"{ep_id}.mp4")),
            'third_ann': VideoFileClip(os.path.join(video_path, 'third_ann_raw_videos', f"{ep_id}.mp4")),
            'global': VideoFileClip(os.path.join(video_path, 'global_videos', f"{ep_id}.mp4"))
        }
        # Resize if needed
        for k in vid_files:
            if vid_files[k].h > 480: vid_files[k] = vid_files[k].resized(height=480)
    except Exception as e:
        print(f"  [Error] Failed to load videos for {ep_id}: {e}")
        return []

    processed_records = []

    # 遍历子任务
    for i, task in enumerate(data.get('subtasks', [])):
        # 任务本身的范围
        t_start = max(task['start_step'], robot_active_step)
        t_end = min(task['end_step'], total_frames)

        if t_end - t_start < FPS: continue

        task_steps = all_steps[t_start:t_end]

        # --- 1. 定位 Pick Step (拿起物体的瞬间) ---
        pick_step_rel = -1
        target_obj = task.get('object_names', [''])[0]  # 获取目标物体名称

        for idx, step in enumerate(task_steps):
            # 检查人类是否拿着东西
            # 注意：is_holding 有时是 bool, 有时是 list/dict, 视具体 json 而定
            # 这里沿用你之前代码的假设: step['human_agent'].get('is_holding')
            if step['human_agent'].get('is_holding', False):
                pick_step_rel = idx
                break

        if pick_step_rel == -1:
            # print(f"  [Skip] Task {i}: No pick action detected.")
            continue

        pick_step_abs = t_start + pick_step_rel

        # --- 2. 定位 Arrival Step (到达终点前) ---
        # 沿用原逻辑：根据距离判断
        final_pos = np.array(all_steps[t_end - 1]['human_agent']['pos'])
        arrival_step_abs = -1

        for idx in range(pick_step_abs, t_end):
            curr_pos = np.array(all_steps[idx]['human_agent']['pos'])
            dist = np.linalg.norm(curr_pos[[0, 2]] - final_pos[[0, 2]])
            if dist < 2.0:
                arrival_step_abs = idx
                break

        # 如果没检测到，默认设为任务结束前一点点
        if arrival_step_abs == -1:
            arrival_step_abs = max(pick_step_abs + 1, t_end - 10)

        # 核心参数
        transport_duration_frames = arrival_step_abs - pick_step_abs
        if transport_duration_frames < (MIN_TASK_DURATION * FPS):
            continue

        # ==========================================
        # 生成切片计划 (Slice Plan)
        # ==========================================
        slice_plan = []

        # 策略 A: 累积观测 (Accumulative)
        # 起点: Pick 前 1.0s
        accum_start = max(t_start, pick_step_abs - CONTEXT_FRAMES)

        for ratio in [0.2, 0.4, 0.6, 0.8]:
            cut_len = int(transport_duration_frames * ratio)
            accum_end = pick_step_abs + cut_len

            # 安全红线
            hard_limit = arrival_step_abs - SAFE_MARGIN_FRAMES
            if accum_end > hard_limit: accum_end = hard_limit

            if (accum_end - accum_start) >= FPS:  # 至少1秒
                slice_plan.append({
                    "type": "accumulative",
                    "param": f"p{int(ratio * 100)}",
                    "start": accum_start,
                    "end": accum_end,
                    "ratio": ratio
                })

        # 策略 B: 滑动窗口 (Sliding Window)
        win_len = WINDOW_FRAMES  # 30 frames

        # B1. Init (含头)
        w_init_start = max(t_start, pick_step_abs - int(0.5 * FPS))
        w_init_end = w_init_start + win_len
        if w_init_end < (arrival_step_abs - SAFE_MARGIN_FRAMES):
            slice_plan.append({
                "type": "window",
                "param": "init",
                "start": w_init_start,
                "end": w_init_end
            })

        # B2. Transit (盲区)
        t_min = pick_step_abs + int(transport_duration_frames * 0.3)
        t_max = pick_step_abs + int(transport_duration_frames * 0.6)
        if (t_max - t_min) > FPS:
            w_trans_start = random.randint(t_min, t_max)
            w_trans_end = w_trans_start + win_len
            if w_trans_end < (arrival_step_abs - SAFE_MARGIN_FRAMES):
                slice_plan.append({
                    "type": "window",
                    "param": "transit",
                    "start": w_trans_start,
                    "end": w_trans_end
                })

        # B3. Converge (趋近)
        w_conv_end = arrival_step_abs - SAFE_MARGIN_FRAMES
        w_conv_start = w_conv_end - win_len
        if w_conv_start > (pick_step_abs + FPS):  # 必须在 Pick 之后
            slice_plan.append({
                "type": "window",
                "param": "converge",
                "start": w_conv_start,
                "end": w_conv_end
            })

        # ==========================================
        # 执行切片
        # ==========================================
        for plan in slice_plan:
            s, e = plan['start'], plan['end']
            stype, sparam = plan['type'], plan['param']

            # 1. 计算该片段的 Metrics & Difficulty
            slice_metrics = calculate_metrics_for_slice(
                [s['robot_agent'] for s in all_steps[s:e]],
                [s['human_agent'] for s in all_steps[s:e]],
                all_steps[s:e]
            )
            difficulty = get_difficulty_bucket(slice_metrics)

            # 2. 生成文件名
            # e.g. 0_t0_accum_p20_ego_raw.mp4
            file_base = f"{ep_id}_t{i}_{stype}_{sparam}"

            # 3. 切割所有视角
            view_paths = {}
            for k, v in vid_files.items():
                out_name = f"{file_base}_{k}.mp4"
                out_path = os.path.join(raw_dirs[k], out_name)

                # 使用 subclip 并写入
                # 注意：MoviePy 的 subclip 接受的是秒 (float)
                v.subclipped(s / FPS, e / FPS).write_videofile(
                    out_path, codec='libx264', logger=None, audio=False
                )
                view_paths[f"{k}_path"] = out_path

            # 4. 记录索引
            record = {
                "episode_id": ep_id,
                "instruction": instruction,
                "task_index": i,
                "subtask": task['description_processed'],
                "gt_intent": "Unknown",  # 需要你确认 intent 字段在哪里，或者后续解析
                "gt_target": task['receptacle_names'][0] if task['receptacle_names'] else "Unknown",
                "observation_type": stype,
                "slice_param": sparam,
                "difficulty": difficulty,
                "duration": (e - s) / FPS,
                "metrics": slice_metrics
            }
            if "ratio" in plan: record["progress_ratio"] = plan["ratio"]

            # 合并视频路径
            record.update(view_paths)

            processed_records.append(record)
            print(f"  [Slice] {stype}-{sparam} ({difficulty}): {file_base}")

    # 关闭视频句柄
    for v in vid_files.values(): v.close()

    return processed_records


if __name__ == "__main__":
    # 请根据实际路径修改
    VIDEO_DIR_ROOT = "test_my_videos_2/"
    META_DIR_ROOT = "test_my_videos_2/metadata"
    SAVE_DIR_ROOT = "test_my_videos_2/final_benchmark_rovi"

    all_info = []

    # 遍历元数据文件
    if os.path.exists(META_DIR_ROOT):
        json_files = sorted([f for f in os.listdir(META_DIR_ROOT) if f.endswith(".json")])
        print(f"Found {len(json_files)} metadata files.")

        for f in json_files:
            # 示例：只处理特定文件，或者移除此 if 处理所有
            # if f.startswith("3"):
            try:
                print(f"Processing {f}...")
                video_input_dir = VIDEO_DIR_ROOT  # 假设视频都在这个根目录下
                records = process_episode(video_input_dir, os.path.join(META_DIR_ROOT, f), SAVE_DIR_ROOT)
                all_info.extend(records)
            except Exception as e:
                print(f"Error processing {f}: {e}")
                import traceback

                traceback.print_exc()
    else:
        print(f"Metadata directory not found: {META_DIR_ROOT}")

    # 保存总索引
    os.makedirs(SAVE_DIR_ROOT, exist_ok=True)
    with open(os.path.join(SAVE_DIR_ROOT, "rovi_benchmark_index.json"), "w") as f:
        json.dump(all_info, f, indent=2)

    print(f"Done! Index saved to {os.path.join(SAVE_DIR_ROOT, 'rovi_benchmark_index.json')}")