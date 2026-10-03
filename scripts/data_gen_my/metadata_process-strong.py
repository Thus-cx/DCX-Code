import json
import os
import numpy as np
import math
import random
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.spatial.transform import Rotation as R

# ================= 配置区域 =================
OUTPUT_DIR = "test_my_videos_2/processed_dataset2"
OUTPUT_RES = (480, 480)
MIN_CLIP_DURATION = 3.0  # 最小视频长度(秒)
FPS = 10.0

# ================= 辅助函数 =================

def get_scipy_quat(habitat_quat):
    # 修复了四元数顺序问题: w,x,y,z -> x,y,z,w
    w, x, y, z = habitat_quat
    return [x, y, z, w]

def calculate_metrics_and_velocities(robot_steps, human_steps, steps_data):
    if not robot_steps or len(robot_steps) < 2:
        return {"view_loss_ratio": 0.0, "proximity_ratio": 0.0, "shake_ratio": 0.0}

    dists = []
    view_losses = 0
    ang_vels = []
    
    for i in range(len(robot_steps)):
        r_step = robot_steps[i]
        h_step = human_steps[i]
        
        # 1. 距离
        r_pos = np.array(r_step['pos'])
        target = np.array(h_step['pos'])
        vec_2d = np.array([target[0] - r_pos[0], 0, target[2] - r_pos[2]])
        dist = np.linalg.norm(vec_2d)
        dists.append(dist)
        
        # 2. 角速度
        if i > 0:
            dt = steps_data[i]['time_sec'] - steps_data[i-1]['time_sec']
            if dt > 1e-4:
                q_curr = R.from_quat(get_scipy_quat(r_step['rot']))
                q_prev = R.from_quat(get_scipy_quat(robot_steps[i-1]['rot']))
                q_diff = q_prev.inv() * q_curr
                ang_vels.append(q_diff.magnitude() / dt)
            else:
                ang_vels.append(0.0)
        else:
            ang_vels.append(0.0)

        # 3. View Loss (2D 投影版)
        rot = R.from_quat(get_scipy_quat(r_step['rot']))
        forward_3d = rot.apply([0, 0, -1]) 
        forward_2d = np.array([forward_3d[0], 0, forward_3d[2]])
        norm_f = np.linalg.norm(forward_2d)
        forward_2d = forward_2d / norm_f if norm_f > 1e-6 else np.array([0, 0, 1])
        
        vec_norm = vec_2d / dist if dist > 0.05 else np.zeros(3)
        
        is_visible = False
        if dist > 0.05: 
            dot = np.dot(forward_2d, vec_norm)
            if dot > 0.5: 
                is_visible = True
        else:
            is_visible = True 
            
        if not is_visible:
            view_losses += 1

    count = len(dists)
    return {
        "view_loss_ratio": view_losses / count if count > 0 else 0.0,
        "proximity_ratio": sum(1 for d in dists if d < 0.8) / count if count > 0 else 0.0,
        "shake_ratio": sum(1 for v in ang_vels if abs(v) > 0.8) / len(ang_vels) if ang_vels else 0.0,
        "min_dist": float(np.min(dists)) if dists else 0.0,
        "avg_dist": float(np.mean(dists)) if dists else 0.0,
        "max_ang_vel": float(np.max(ang_vels)) if ang_vels else 0.0
    }

def process_episode(video_path, metadata_file, output_base_dir):
    raw_dirs = {k: os.path.join(output_base_dir, k) for k in ['ego_raw', 'ego_ann', 'third_raw', 'third_ann', 'global']}
    for d in raw_dirs.values(): os.makedirs(d, exist_ok=True)
    
    with open(metadata_file, 'r') as f:
        data = json.load(f)
        
    ep_id = data.get('episode_id', 'unknown')
    all_steps = data['steps']
    total_frames = len(all_steps)
    for step_idx in range(len(all_steps)):
        step = all_steps[step_idx]
        if step["robot_active"]:
            robot_active_step = step_idx
            break
    
    # 视频加载 (Lazy Loading)
    vid_files = {
        'ego_raw': VideoFileClip(os.path.join(video_path, 'ego_raw_videos', f"{ep_id}.mp4")),
        'ego_ann': VideoFileClip(os.path.join(video_path, 'ego_ann_raw_videos', f"{ep_id}.mp4")),
        'third_raw': VideoFileClip(os.path.join(video_path, 'third_raw_videos', f"{ep_id}.mp4")),
        'third_ann': VideoFileClip(os.path.join(video_path, 'third_ann_raw_videos', f"{ep_id}.mp4")),
        'global': VideoFileClip(os.path.join(video_path, 'global_videos', f"{ep_id}.mp4"))
    }
    # Resize
    for k in vid_files:
        if vid_files[k].h > 480: vid_files[k] = vid_files[k].resized(height=480)

    processed_records = []

    for i, task in enumerate(data.get('subtasks', [])):
        t_start = max(task['start_step'], robot_active_step)
        t_end = task['end_step']
        if t_end > total_frames: t_end = total_frames
        
        task_slice = all_steps[t_start:t_end]
        if not task_slice: continue
        
        # --- 1. 定位 Pick (拿起瞬间) ---
        pick_step = -1
        for idx, step in enumerate(task_slice):
            if step['human_agent'].get('is_holding', False):
                pick_step = t_start + idx
                break
        
        if pick_step == -1: 
            # print(f"  [Skip] Task {i}: No object picked up.")
            continue
            
        # --- 2. 定位 Arrival (到达终点附近) ---
        # 宽松阈值: 2.0米
        final_pos = np.array(all_steps[t_end-1]['human_agent']['pos'])
        arrival_step = -1
        
        # 从 Pick 之后开始找
        for idx in range(pick_step, t_end):
            curr_pos = np.array(all_steps[idx]['human_agent']['pos'])
            dist = np.linalg.norm(curr_pos[[0,2]] - final_pos[[0,2]])
            if dist < 2.0: # 宽松一点，防止 Verification 没素材
                arrival_step = idx
                break
        
        # 如果一直没检测到 (始终在远处? 或者一直在终点?)
        if arrival_step == -1:
             arrival_step = t_end - 20 # 强行切最后2秒

        # ==========================================
        # 3. 灵活切分
        # ==========================================
        
        # --- Clip 1: Prediction (Transport) ---
        # 起点: Pick 往前推 1~3秒 (10~30帧), 但不能早于 t_start
        pred_offset_start = random.randint(10, 30)
        pred_start = max(t_start, pick_step - pred_offset_start)
        
        # 终点: Arrival 往前推 0.5~1.5秒 (5~15帧), 制造悬念
        # 必须保证终点 > 起点
        pred_offset_end = random.randint(5, 15)
        pred_end = max(pred_start + 20, arrival_step - pred_offset_end) 
        
        pred_dur = (pred_end - pred_start) / FPS
        
        if pred_dur >= MIN_CLIP_DURATION:
            clip_name = f"{ep_id}_task{i}_prediction.mp4"
            metrics = calculate_metrics_and_velocities(
                [s['robot_agent'] for s in all_steps[pred_start:pred_end]], 
                [s['human_agent'] for s in all_steps[pred_start:pred_end]],
                all_steps[pred_start:pred_end]
            )
            
            for k, v in vid_files.items():
                v.subclipped(pred_start/FPS, pred_end/FPS).write_videofile(
                    os.path.join(raw_dirs[k], clip_name), codec='libx264', logger=None
                )
                
            processed_records.append({
                "episode_id": ep_id,
                "phase": "prediction",
                "gt_task": task['description_processed'],
                "metrics": metrics,
                "duration": pred_dur,
                "ego_ann_video_path": os.path.join(raw_dirs['ego_ann'], clip_name),
                "ego_raw_video_path": os.path.join(raw_dirs['ego_raw'], clip_name),
                "third_ann_video_path": os.path.join(raw_dirs['third_ann'], clip_name),
                "third_raw_video_path": os.path.join(raw_dirs['third_raw'], clip_name),
                "global_video_path": os.path.join(raw_dirs['global'], clip_name),
            })
            print(f"  [Gen] Prediction Clip: {clip_name} ({pred_dur:.1f}s)")

        # --- Clip 2: Verification (Execution) ---
        # 起点: Arrival 往前推 0~1秒 (0~10帧), 多给点环境上下文
        ver_offset_start = random.randint(0, 10)
        ver_start = max(pick_step, arrival_step - ver_offset_start) # 不能早于 pick
        
        ver_dur = (t_end - ver_start) / FPS
        
        if ver_dur >= MIN_CLIP_DURATION:
            clip_name = f"{ep_id}_task{i}_verification.mp4"
            metrics = calculate_metrics_and_velocities(
                [s['robot_agent'] for s in all_steps[ver_start:t_end]], 
                [s['human_agent'] for s in all_steps[ver_start:t_end]],
                all_steps[ver_start:t_end]
            )
            
            for k, v in vid_files.items():
                v.subclipped(ver_start/FPS, t_end/FPS).write_videofile(
                    os.path.join(raw_dirs[k], clip_name), codec='libx264', logger=None
                )
                
            processed_records.append({
                "episode_id": ep_id,
                "phase": "verification",
                "gt_task": task['description_processed'],
                "metrics": metrics,
                "duration": ver_dur,
                "ego_ann_video_path": os.path.join(raw_dirs['ego_ann'], clip_name),
                "ego_raw_video_path": os.path.join(raw_dirs['ego_raw'], clip_name),
                "third_ann_video_path": os.path.join(raw_dirs['third_ann'], clip_name),
                "third_raw_video_path": os.path.join(raw_dirs['third_raw'], clip_name),
                "global_video_path": os.path.join(raw_dirs['global'], clip_name),
            })
            print(f"  [Gen] Verification Clip: {clip_name} ({ver_dur:.1f}s)")

    for v in vid_files.values(): v.close()
    return processed_records

if __name__ == "__main__":
    VIDEO_DIR = "test_my_videos_2/"
    META_DIR = "test_my_videos_2/metadata"
    SAVE_DIR = "test_my_videos_2/final_benchmark_clips_2"
    
    all_info = []
    # 简单遍历
    for f in os.listdir(META_DIR):
        if f.endswith(".json") and f.startswith("3"): 
            try:
                print(f"Processing {f}...")
                all_info.extend(process_episode(VIDEO_DIR, os.path.join(META_DIR, f), SAVE_DIR))
            except Exception as e:
                print(f"Error: {e}")
                
    with open(os.path.join(SAVE_DIR, "processed_clips_index.json"), "w") as f:
        json.dump(all_info, f, indent=2)