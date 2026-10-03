import json
import os
import numpy as np
import subprocess
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# ================= 配置区域 =================
FPS = 10.0
MIN_TASK_DURATION = 6.0  # 最小任务时长(秒)

# 请根据实际路径修改
VIDEO_DIR_ROOT = "final_raw_videos/"
META_DIR_ROOT = "final_raw_videos/metadata"
SAVE_DIR_ROOT = "final_raw_videos/train_dataset_full"

# ================= 辅助函数 (你原本的逻辑) =================
def get_scipy_quat(habitat_quat):
    w, x, y, z = habitat_quat
    return [x, y, z, w]

def calculate_metrics_for_slice(robot_steps, human_steps, steps_data):
    if not robot_steps or len(robot_steps) < 2:
        return {
            "view_loss_ratio": 0.0, "occlusion_ratio": 0.0,
            "proximity_ratio": 0.0, "shake_ratio": 0.0, "avg_dist": 0.0
        }

    dists, ang_vels = [], []
    view_losses, occlusion_frames = 0, 0

    for i in range(len(robot_steps)):
        r_step = robot_steps[i]
        h_step = human_steps[i]

        r_pos = np.array(r_step['pos'])
        h_pos = np.array(h_step['pos'])
        vec_2d = np.array([h_pos[0] - r_pos[0], 0, h_pos[2] - r_pos[2]])
        dist = np.linalg.norm(vec_2d)
        dists.append(dist)

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

        vec_norm = vec_2d / dist if dist > 0.05 else np.zeros(3)

        # View Loss
        r_rot = R.from_quat(get_scipy_quat(r_step['rot']))
        r_forward_3d = r_rot.apply([0, 0, -1])
        r_forward_2d = np.array([r_forward_3d[0], 0, r_forward_3d[2]])
        r_norm = np.linalg.norm(r_forward_2d)
        r_forward_2d = r_forward_2d / r_norm if r_norm > 1e-6 else np.array([0, 0, 1])

        r_dot = np.dot(r_forward_2d, vec_norm) if dist > 0.05 else 0.0
        if r_dot <= 0.707:
            view_losses += 1

        # Occlusion
        h_rot = R.from_quat(get_scipy_quat(h_step['rot']))
        h_forward_3d = h_rot.apply([0, 0, -1])
        h_forward_2d = np.array([h_forward_3d[0], 0, h_forward_3d[2]])
        h_norm = np.linalg.norm(h_forward_2d)
        h_forward_2d = h_forward_2d / h_norm if h_norm > 1e-6 else np.array([0, 0, 1])

        vec_human_to_robot = -vec_norm
        h_dot = np.dot(h_forward_2d, vec_human_to_robot) if dist > 0.05 else 0.0

        if h_dot < -0.3:
            occlusion_frames += 1

    count = len(dists)
    if count == 0: return {}

    return {
        "view_loss_ratio": view_losses / count,
        "occlusion_ratio": occlusion_frames / count,
        "proximity_ratio": sum(1 for d in dists if d < 0.75) / count,
        "shake_ratio": sum(1 for v in ang_vels if abs(v) > 0.8) / len(ang_vels),
        "avg_dist": float(np.mean(dists))
    }

def get_difficulty_bucket(metrics):
    vl = metrics.get('view_loss_ratio', 0.0)
    occ = metrics.get('occlusion_ratio', 0.0)
    prox = metrics.get('proximity_ratio', 0.0)
    shake = metrics.get('shake_ratio', 0.0)

    if occ > 0.60 or vl > 0.50 or prox > 0.60: return "Hard"
    elif occ < 0.25 and vl < 0.15 and prox < 0.20 and shake < 0.05: return "Easy"
    else: return "Medium"

# ================= 核心处理函数 (FFmpeg加速版) =================
def process_episode(video_path, metadata_file, output_base_dir):
    view_names = ['ego_raw', 'ego_ann', 'third_raw', 'third_ann', 'global', 'global_ann']
    raw_dirs = {k: os.path.join(output_base_dir, k) for k in view_names}
    for d in raw_dirs.values(): os.makedirs(d, exist_ok=True)

    with open(metadata_file, 'r') as f:
        data = json.load(f)

    ep_id = data.get('episode_id', 'unknown')
    instruction = data.get('instruction', 'unknown')
    all_steps = data['steps']
    total_frames = len(all_steps)

    robot_active_step = next((i for i, step in enumerate(all_steps) if step["robot_active"]), 0)

    processed_records = []

    for i, task in enumerate(data.get('subtasks', [])):
        t_start = max(task['start_step'], robot_active_step)
        t_end = min(task['end_step'], total_frames)

        if t_end - t_start < FPS: continue
        task_steps = all_steps[t_start:t_end]

        pick_step_rel = next((idx for idx, step in enumerate(task_steps) if step['human_agent'].get('is_holding', False)), -1)
        if pick_step_rel == -1: continue
        pick_step_abs = t_start + pick_step_rel

        final_pos = np.array(all_steps[t_end - 1]['human_agent']['pos'])
        arrival_step_abs = -1
        for idx in range(pick_step_abs, t_end):
            curr_pos = np.array(all_steps[idx]['human_agent']['pos'])
            dist = np.linalg.norm(curr_pos[[0, 2]] - final_pos[[0, 2]])
            if dist < 2.0:
                arrival_step_abs = idx
                break

        if arrival_step_abs == -1:
            arrival_step_abs = max(pick_step_abs + 1, t_end - 10)

        transport_duration_frames = arrival_step_abs - pick_step_abs
        if transport_duration_frames < (MIN_TASK_DURATION * FPS): continue

        slice_plan = []
        L = transport_duration_frames
        L_30, L_60, L_90, L_15 = int(L * 0.3), int(L * 0.6), int(L * 0.9), int(L * 0.15)
        T_start, T_end = pick_step_abs, arrival_step_abs

        slice_plan.extend([
            {"type": "accumulative", "param": "transit_30", "start": T_start, "end": T_start + L_30, "ratio": 0.3},
            {"type": "accumulative", "param": "transit_60", "start": T_start, "end": T_start + L_60, "ratio": 0.6},
            {"type": "accumulative", "param": "transit_90", "start": T_start, "end": T_start + L_90, "ratio": 0.9}
        ])

        pre_start = max(t_start, T_start - L_15)
        pre_end = min(pre_start + L_30, total_frames)
        slice_plan.append({"type": "fixed_len", "param": "pre_transit", "start": pre_start, "end": pre_end})
        slice_plan.append({"type": "fixed_len", "param": "transit_30_60", "start": T_start + L_30, "end": T_start + L_60})
        slice_plan.append({"type": "fixed_len", "param": "transit_60_90", "start": T_start + L_60, "end": T_start + L_90})
        
        post_end = t_end
        post_start = max(t_start, post_end - L_30)
        slice_plan.append({"type": "fixed_len", "param": "post_transit", "start": post_start, "end": post_end})

        for plan in slice_plan:
            s, e = int(plan['start']), int(plan['end'])
            stype, sparam = plan['type'], plan['param']

            slice_metrics = calculate_metrics_for_slice(
                [step['robot_agent'] for step in all_steps[s:e]],
                [step['human_agent'] for step in all_steps[s:e]],
                all_steps[s:e]
            )
            difficulty = get_difficulty_bucket(slice_metrics)

            file_base = f"{ep_id}_t{i}_{stype}_{sparam}"
            view_paths = {}

            start_sec = s / FPS
            duration_sec = (e - s) / FPS

            for view_name in view_names:
                if view_name == 'ego_raw': in_folder = 'ego_raw_videos'
                elif view_name == 'ego_ann': in_folder = 'ego_ann_raw_videos'
                elif view_name == 'third_raw': in_folder = 'third_raw_videos'
                elif view_name == 'third_ann': in_folder = 'third_ann_raw_videos'
                elif view_name == 'global': in_folder = 'global_videos'
                elif view_name == 'global_ann': in_folder = 'global_ann_videos'
                
                in_path = os.path.join(video_path, in_folder, f"{ep_id}.mp4")
                out_path = os.path.join(raw_dirs[view_name], f"{file_base}_{view_name}.mp4")

                if not os.path.exists(in_path): continue

                ffmpeg_cmd = [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-ss", str(start_sec), "-i", in_path, "-t", str(duration_sec),
                    "-vf", "scale=-2:'min(480,ih)'", "-c:v", "libx264",
                    "-preset", "ultrafast", "-an", out_path
                ]
                
                try:
                    subprocess.run(ffmpeg_cmd, check=True)
                    view_paths[f"{view_name}_path"] = out_path
                except subprocess.CalledProcessError:
                    pass # 忽略极个别损坏帧导致的报错

            record = {
                "episode_id": ep_id,
                "instruction": instruction,
                "task_index": i,
                "subtask": task.get('description_processed', ''),
                "gt_intent": "Unknown",
                "gt_target": task['receptacle_names'][0] if task.get('receptacle_names') else "Unknown",
                "observation_type": stype,
                "slice_param": sparam,
                "difficulty": difficulty,
                "duration": duration_sec,
                "metrics": slice_metrics
            }
            if "ratio" in plan: record["progress_ratio"] = plan["ratio"]
            record.update(view_paths)
            processed_records.append(record)

    return processed_records

# ================= 多进程启动 =================
def worker(f):
    try:
        return True, process_episode(VIDEO_DIR_ROOT, os.path.join(META_DIR_ROOT, f), SAVE_DIR_ROOT), None
    except Exception as e:
        import traceback
        return False, [], f"处理 {f} 失败:\n{traceback.format_exc()}"

if __name__ == "__main__":
    logging.basicConfig(filename="data_gen_errors.log", filemode="w", level=logging.ERROR)
    all_info = []

    if os.path.exists(META_DIR_ROOT):
        json_files = sorted([f for f in os.listdir(META_DIR_ROOT) if f.endswith(".json")])
        
        # 核心保护机制：最大开启 4 个进程，防止磁盘 I/O 锁死
        max_workers = 4 
        print(f"🚀 启动 FFmpeg 加速切片，开启 {max_workers} 个并行进程保护 I/O...")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(worker, f): f for f in json_files}
            
            with tqdm(total=len(json_files), desc="🎬 视频切片处理中", unit="ep") as pbar:
                for future in as_completed(futures):
                    success, records, err_msg = future.result()
                    if success:
                        all_info.extend(records)
                    else:
                        logging.error(err_msg)
                    pbar.update(1)

        os.makedirs(SAVE_DIR_ROOT, exist_ok=True)
        out_file = os.path.join(SAVE_DIR_ROOT, "benchmark_index.json")
        with open(out_file, "w") as f:
            json.dump(all_info, f, indent=2)
        print(f"\n🎉 完美收工！总切片索引已保存至: {out_file}")
    else:
        print(f"❌ 找不到元数据目录: {META_DIR_ROOT}")