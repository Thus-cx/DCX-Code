import json
import os
import numpy as np
import math
import random
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.spatial.transform import Rotation as R
import subprocess

# ================= 配置区域 =================
# 输出目录 (可在 main 中覆盖)
OUTPUT_DIR = "processed_dataset"
OUTPUT_RES = (480, 480)
FPS = 10.0

# 协议参数
SAFE_MARGIN_FRAMES = 50  # 0.5s (Arrival前的安全缓冲)
CONTEXT_FRAMES = 20  # 1.0s (Accumulative起始上下文)
WINDOW_FRAMES = 60  # 3.0s (滑动窗口长度)
MIN_TASK_DURATION = 6.0  # 最小任务时长(秒)


# ================= 辅助函数 =================

def get_scipy_quat(habitat_quat):
    # w,x,y,z -> x,y,z,w
    w, x, y, z = habitat_quat
    return [x, y, z, w]


# def calculate_metrics_for_slice(robot_steps, human_steps, steps_data):
#     """
#     计算特定切片范围内的指标
#     """
#     if not robot_steps or len(robot_steps) < 2:
#         return {"view_loss_ratio": 0.0, "proximity_ratio": 0.0, "shake_ratio": 0.0, "avg_view_loss": 0.0}
#
#     dists = []
#     view_losses = 0
#     ang_vels = []
#     view_loss_values = []  # 用于计算平均值
#
#     for i in range(len(robot_steps)):
#         r_step = robot_steps[i]
#         h_step = human_steps[i]
#
#         # 1. 距离
#         r_pos = np.array(r_step['pos'])
#         target = np.array(h_step['pos'])
#         vec_2d = np.array([target[0] - r_pos[0], 0, target[2] - r_pos[2]])
#         dist = np.linalg.norm(vec_2d)
#         dists.append(dist)
#
#         # 2. 角速度 (Shake)
#         if i > 0:
#             dt = steps_data[i]['time_sec'] - steps_data[i - 1]['time_sec']
#             if dt > 1e-4:
#                 q_curr = R.from_quat(get_scipy_quat(r_step['rot']))
#                 q_prev = R.from_quat(get_scipy_quat(robot_steps[i - 1]['rot']))
#                 q_diff = q_prev.inv() * q_curr
#                 ang_vels.append(q_diff.magnitude() / dt)
#             else:
#                 ang_vels.append(0.0)
#         else:
#             ang_vels.append(0.0)
#
#         # 3. View Loss
#         rot = R.from_quat(get_scipy_quat(r_step['rot']))
#         forward_3d = rot.apply([0, 0, -1])
#         forward_2d = np.array([forward_3d[0], 0, forward_3d[2]])
#         norm_f = np.linalg.norm(forward_2d)
#         forward_2d = forward_2d / norm_f if norm_f > 1e-6 else np.array([0, 0, 1])
#
#         vec_norm = vec_2d / dist if dist > 0.05 else np.zeros(3)
#
#         dot = 0.0
#         if dist > 0.05:
#             dot = np.dot(forward_2d, vec_norm)  # 1.0 = 正对
#
#         # 简单定义：dot < 0.5 (60度以外) 视为丢失
#         # View Loss Value: 0 (Good) -> 1 (Bad)
#         vl_val = 1.0 - max(0.0, dot)
#         view_loss_values.append(vl_val)
#
#         if dot <= 0.5:
#             view_losses += 1
#
#     count = len(dists)
#     if count == 0: return {}
#
#     return {
#         "view_loss_ratio": view_losses / count,
#         "avg_view_loss": float(np.mean(view_loss_values)) if view_loss_values else 0.0,
#         "proximity_ratio": sum(1 for d in dists if d < 0.8) / count,  # 0.8m 内视为过近
#         "shake_ratio": sum(1 for v in ang_vels if abs(v) > 0.8) / len(ang_vels),  # 抖动阈值
#         "min_dist": float(np.min(dists)),
#         "avg_dist": float(np.mean(dists)),
#         "max_ang_vel": float(np.max(ang_vels))
#     }
def calculate_metrics_for_slice(robot_steps, human_steps, steps_data):
    if not robot_steps or len(robot_steps) < 2:
        return {
            "view_loss_ratio": 0.0, "occlusion_ratio": 0.0,
            "proximity_ratio": 0.0, "shake_ratio": 0.0, "avg_dist": 0.0
        }

    dists = []
    view_losses = 0
    occlusion_frames = 0
    ang_vels = []

    for i in range(len(robot_steps)):
        r_step = robot_steps[i]
        h_step = human_steps[i]

        # 1. 距离
        r_pos = np.array(r_step['pos'])
        h_pos = np.array(h_step['pos'])
        vec_2d = np.array([h_pos[0] - r_pos[0], 0, h_pos[2] - r_pos[2]])
        dist = np.linalg.norm(vec_2d)
        dists.append(dist)

        # 2. 视角抖动 (Shake)
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

        # 3. 视野丢失 (View Loss) - 放宽要求，偏离中心点超过 45度才算丢失
        r_rot = R.from_quat(get_scipy_quat(r_step['rot']))
        r_forward_3d = r_rot.apply([0, 0, -1])
        r_forward_2d = np.array([r_forward_3d[0], 0, r_forward_3d[2]])
        r_norm = np.linalg.norm(r_forward_2d)
        r_forward_2d = r_forward_2d / r_norm if r_norm > 1e-6 else np.array([0, 0, 1])

        r_dot = np.dot(r_forward_2d, vec_norm) if dist > 0.05 else 0.0
        if r_dot <= 0.707:  # cos(45度) = 0.707，人类移出了画面的核心区
            view_losses += 1

        # 4. 躯干遮挡 (Torso Occlusion) - 严格定义为"后背遮挡"
        h_rot = R.from_quat(get_scipy_quat(h_step['rot']))
        h_forward_3d = h_rot.apply([0, 0, -1])
        h_forward_2d = np.array([h_forward_3d[0], 0, h_forward_3d[2]])
        h_norm = np.linalg.norm(h_forward_2d)
        h_forward_2d = h_forward_2d / h_norm if h_norm > 1e-6 else np.array([0, 0, 1])

        vec_human_to_robot = -vec_norm
        # dot = 1 (正前), dot = 0 (正侧), dot = -1 (正后)
        h_dot = np.dot(h_forward_2d, vec_human_to_robot) if dist > 0.05 else 0.0

        # 【修正核心】：只有当 dot < -0.3（即机器人处于人类后方的 145度扇区内）才算严重躯干遮挡
        # 完美的侧面 (dot 约为 0) 和斜前方不再被计入遮挡！
        if h_dot < -0.3:
            occlusion_frames += 1

    count = len(dists)
    if count == 0: return {}

    return {
        "view_loss_ratio": view_losses / count,
        "occlusion_ratio": occlusion_frames / count,
        "proximity_ratio": sum(1 for d in dists if d < 0.75) / count,  # 低于 0.75m 算贴脸
        "shake_ratio": sum(1 for v in ang_vels if abs(v) > 0.8) / len(ang_vels),
        "avg_dist": float(np.mean(dists))
    }

# def get_difficulty_bucket(metrics):
#     """根据指标判断难度等级"""
#     vl = metrics.get('view_loss_ratio', 0.0)
#     shake = metrics.get('shake_ratio', 0.0)
#     prox = metrics.get('proximity_ratio', 0.0)
#
#     if vl > 0.6 or shake > 0.1 or prox > 0.7:
#         return "Hard"
#     elif vl > 0.3 or shake > 0.05 or prox > 0.4:
#         return "Medium"
#     else:
#         return "Easy"
def get_difficulty_bucket(metrics):
    """
    采用“剥洋葱”式的健康度排查法则，确保合理的难度分布
    """
    vl = metrics.get('view_loss_ratio', 0.0)
    occ = metrics.get('occlusion_ratio', 0.0)
    prox = metrics.get('proximity_ratio', 0.0)
    shake = metrics.get('shake_ratio', 0.0)

    # 【Hard 模式】定义：彻底的视觉灾难
    # 大部分时间看背影(>60%)，或者大半时间人飞出核心画面(>50%)，或者严重贴脸大腿(>60%)
    if occ > 0.60 or vl > 0.50 or prox > 0.60:
        return "Hard"

    # 【Easy 模式】定义：教科书般的良好跟拍
    # 遮挡很少（大概率是侧面跟拍），人稳定在视野内，距离保持完美
    elif occ < 0.25 and vl < 0.15 and prox < 0.20 and shake < 0.05:
        return "Easy"

    # 【Medium 模式】定义：真实协作的中间态
    # 有一定的遮挡（斜后方），或者短暂的跟丢，或者偶然的贴脸
    else:
        return "Medium"


# def process_episode(video_path, metadata_file, output_base_dir):
#     # 创建输出目录结构
#     # view_names = ['ego_raw', 'ego_ann', 'third_raw', 'third_ann', 'global', 'global_ann']
#     view_names = ['ego_raw', 'ego_ann', 'third_raw', 'third_ann', 'global']
#     raw_dirs = {k: os.path.join(output_base_dir, k) for k in view_names}
#     for d in raw_dirs.values(): os.makedirs(d, exist_ok=True)

#     with open(metadata_file, 'r') as f:
#         data = json.load(f)

#     ep_id = data.get('episode_id', 'unknown')
#     instruction = data.get('instruction', 'unknown')
#     all_steps = data['steps']
#     total_frames = len(all_steps)

#     # 确定 Robot Active 时间点
#     robot_active_step = 0
#     for step_idx in range(len(all_steps)):
#         if all_steps[step_idx]["robot_active"]:
#             robot_active_step = step_idx
#             break

#     # 视频加载 (Lazy Loading)
#     try:
#         vid_files = {
#             'ego_raw': VideoFileClip(os.path.join(video_path, 'ego_raw_videos', f"{ep_id}.mp4")),
#             'ego_ann': VideoFileClip(os.path.join(video_path, 'ego_ann_raw_videos', f"{ep_id}.mp4")),
#             'third_raw': VideoFileClip(os.path.join(video_path, 'third_raw_videos', f"{ep_id}.mp4")),
#             'third_ann': VideoFileClip(os.path.join(video_path, 'third_ann_raw_videos', f"{ep_id}.mp4")),
#             'global': VideoFileClip(os.path.join(video_path, 'global_videos', f"{ep_id}.mp4")),
#             # 'global_ann': VideoFileClip(os.path.join(video_path, 'global_ann_videos', f"{ep_id}.mp4"))
#         }
#         # Resize if needed
#         for k in vid_files:
#             if vid_files[k].h > 480: vid_files[k] = vid_files[k].resized(height=480)
#     except Exception as e:
#         print(f"  [Error] Failed to load videos for {ep_id}: {e}")
#         return []

#     processed_records = []

#     # 遍历子任务
#     for i, task in enumerate(data.get('subtasks', [])):
#         # 任务本身的范围
#         t_start = max(task['start_step'], robot_active_step)
#         t_end = min(task['end_step'], total_frames)

#         if t_end - t_start < FPS: continue

#         task_steps = all_steps[t_start:t_end]

#         # --- 1. 定位 Pick Step (拿起物体的瞬间) ---
#         pick_step_rel = -1
#         target_obj = task.get('object_names', [''])[0]  # 获取目标物体名称

#         for idx, step in enumerate(task_steps):
#             if step['human_agent'].get('is_holding', False):
#                 pick_step_rel = idx
#                 break

#         if pick_step_rel == -1:
#             continue

#         pick_step_abs = t_start + pick_step_rel

#         # --- 2. 定位 Arrival Step (到达终点前) ---
#         final_pos = np.array(all_steps[t_end - 1]['human_agent']['pos'])
#         arrival_step_abs = -1

#         for idx in range(pick_step_abs, t_end):
#             curr_pos = np.array(all_steps[idx]['human_agent']['pos'])
#             dist = np.linalg.norm(curr_pos[[0, 2]] - final_pos[[0, 2]])
#             if dist < 2.0:
#                 arrival_step_abs = idx
#                 break

#         if arrival_step_abs == -1:
#             arrival_step_abs = max(pick_step_abs + 1, t_end - 10)

#         # 核心参数
#         transport_duration_frames = arrival_step_abs - pick_step_abs
#         if transport_duration_frames < (MIN_TASK_DURATION * FPS):
#             continue

#         # ==========================================
#         # 生成切片计划 (Slice Plan) - 新策略
#         # ==========================================
#         slice_plan = []

#         L = transport_duration_frames
#         L_30 = int(L * 0.3)
#         L_60 = int(L * 0.6)
#         L_90 = int(L * 0.9)
#         L_15 = int(L * 0.15)

#         T_start = pick_step_abs
#         T_end = arrival_step_abs

#         # 策略 1: 视频长度影响对比 (累积长度 30%, 60%, 90%)
#         slice_plan.extend([
#             {
#                 "type": "accumulative",
#                 "param": "transit_30",
#                 "start": T_start,
#                 "end": T_start + L_30,
#                 "ratio": 0.3
#             },
#             {
#                 "type": "accumulative",
#                 "param": "transit_60",
#                 "start": T_start,
#                 "end": T_start + L_60,
#                 "ratio": 0.6
#             },
#             {
#                 "type": "accumulative",
#                 "param": "transit_90",
#                 "start": T_start,
#                 "end": T_start + L_90,
#                 "ratio": 0.9
#             }
#         ])

#         # 策略 2: 等长不同阶段对比 (统一保持 transit 的 30% 长度)

#         # [片段A] transit 前: 从 transit 前 15% 到 transit 开启后的 15%
#         pre_start = T_start - L_15
#         if pre_start < t_start:  # 如果前面不够15%，就从起点开始往后截足30%
#             pre_start = t_start
#         pre_end = pre_start + L_30
#         pre_end = min(pre_end, total_frames)  # 防止越界

#         slice_plan.append({
#             "type": "fixed_len",
#             "param": "pre_transit",
#             "start": pre_start,
#             "end": pre_end
#         })

#         # [片段B] transit阶段 30%-60%
#         slice_plan.append({
#             "type": "fixed_len",
#             "param": "transit_30_60",
#             "start": T_start + L_30,
#             "end": T_start + L_60
#         })

#         # [片段C] transit阶段 60%-90%
#         slice_plan.append({
#             "type": "fixed_len",
#             "param": "transit_60_90",
#             "start": T_start + L_60,
#             "end": T_start + L_90
#         })

#         # [片段D] transit后半部分到最后交互结束
#         post_end = t_end  # 到子任务最后交互结束
#         post_start = post_end - L_30
#         if post_start < t_start:  # 极低概率的越界保护
#             post_start = t_start

#         slice_plan.append({
#             "type": "fixed_len",
#             "param": "post_transit",
#             "start": post_start,
#             "end": post_end
#         })

#         # ==========================================
#         # 执行切片
#         # ==========================================
#         for plan in slice_plan:
#             s, e = int(plan['start']), int(plan['end'])
#             stype, sparam = plan['type'], plan['param']

#             # 1. 计算该片段的 Metrics & Difficulty
#             slice_metrics = calculate_metrics_for_slice(
#                 [step['robot_agent'] for step in all_steps[s:e]],
#                 [step['human_agent'] for step in all_steps[s:e]],
#                 all_steps[s:e]
#             )
#             difficulty = get_difficulty_bucket(slice_metrics)

#             # 2. 生成文件名
#             file_base = f"{ep_id}_t{i}_{stype}_{sparam}"

#             # 3. 切割所有视角
#             view_paths = {}
#             for k, v in vid_files.items():
#                 out_name = f"{file_base}_{k}.mp4"
#                 out_path = os.path.join(raw_dirs[k], out_name)

#                 # 使用 subclip 并写入 (以秒为单位)
#                 # v.subclipped(s / FPS, e / FPS).write_videofile(
#                 #     out_path, codec='libx264', logger=None, audio=False
#                 # )
#                 # 使用 subclip 并写入 (以秒为单位)，加入加速参数
#                 v.subclipped(s / FPS, e / FPS).write_videofile(
#                     out_path, 
#                     codec='libx264', 
#                     logger=None, 
#                     audio=False,
#                     preset='ultrafast',  # 极大提升编码速度
#                     threads=2            # 分配单任务线程
#                 )
#                 view_paths[f"{k}_path"] = out_path

#             # 4. 记录索引
#             record = {
#                 "episode_id": ep_id,
#                 "instruction": instruction,
#                 "task_index": i,
#                 "subtask": task['description_processed'],
#                 "gt_intent": "Unknown",  # 需要你确认 intent 字段在哪里，或者后续解析
#                 "gt_target": task['receptacle_names'][0] if task['receptacle_names'] else "Unknown",
#                 "observation_type": stype,
#                 "slice_param": sparam,
#                 "difficulty": difficulty,
#                 "duration": (e - s) / FPS,
#                 "metrics": slice_metrics
#             }
#             if "ratio" in plan: record["progress_ratio"] = plan["ratio"]

#             # 合并视频路径
#             record.update(view_paths)

#             processed_records.append(record)
#             # print(f"  [Slice] {stype}-{sparam} ({difficulty}): {file_base}")

#     # 关闭视频句柄
#     for v in vid_files.values(): v.close()

#     return processed_records

#-------修改后的process_episode-------
def process_episode(video_path, metadata_file, output_base_dir):
    # 创建输出目录结构
    view_names = ['ego_raw', 'ego_ann', 'third_raw', 'third_ann', 'global', 'global_ann']
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

    # [移除] 之前在这里的 MoviePy Lazy Loading 代码被彻底移除了！
    
    processed_records = []

    # 遍历子任务
    for i, task in enumerate(data.get('subtasks', [])):
        t_start = max(task['start_step'], robot_active_step)
        t_end = min(task['end_step'], total_frames)

        if t_end - t_start < FPS: continue
        task_steps = all_steps[t_start:t_end]

        # --- 1. 定位 Pick Step ---
        pick_step_rel = -1
        target_obj = task.get('object_names', [''])[0] 
        for idx, step in enumerate(task_steps):
            if step['human_agent'].get('is_holding', False):
                pick_step_rel = idx
                break

        if pick_step_rel == -1: continue
        pick_step_abs = t_start + pick_step_rel

        # --- 2. 定位 Arrival Step ---
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

        # ==========================================
        # 生成切片计划 (Slice Plan)
        # ==========================================
        slice_plan = []
        L = transport_duration_frames
        L_30, L_60, L_90, L_15 = int(L * 0.3), int(L * 0.6), int(L * 0.9), int(L * 0.15)
        T_start, T_end = pick_step_abs, arrival_step_abs

        # 策略 1: 累积长度
        slice_plan.extend([
            {"type": "accumulative", "param": "transit_30", "start": T_start, "end": T_start + L_30, "ratio": 0.3},
            {"type": "accumulative", "param": "transit_60", "start": T_start, "end": T_start + L_60, "ratio": 0.6},
            {"type": "accumulative", "param": "transit_90", "start": T_start, "end": T_start + L_90, "ratio": 0.9}
        ])

        # 策略 2: 等长分段
        pre_start = max(t_start, T_start - L_15)
        pre_end = min(pre_start + L_30, total_frames)
        slice_plan.append({"type": "fixed_len", "param": "pre_transit", "start": pre_start, "end": pre_end})
        slice_plan.append({"type": "fixed_len", "param": "transit_30_60", "start": T_start + L_30, "end": T_start + L_60})
        slice_plan.append({"type": "fixed_len", "param": "transit_60_90", "start": T_start + L_60, "end": T_start + L_90})
        
        post_end = t_end
        post_start = max(t_start, post_end - L_30)
        slice_plan.append({"type": "fixed_len", "param": "post_transit", "start": post_start, "end": post_end})

        # ==========================================
        # 执行底层 FFmpeg 切片 (最核心的修改)
        # ==========================================
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

            # 计算起始秒数和持续秒数
            start_sec = s / FPS
            duration_sec = (e - s) / FPS

            for view_name in view_names:
                # 确定原视频路径和新视频路径
                # 注意：假设你的原始视频存放在对应名称的文件夹中 (需与你的文件结构一致)
                in_folder = view_name + "_videos" if "raw" not in view_name else view_name.replace("raw", "raw_videos")
                # 纠正目录名映射 (ego_raw -> ego_raw_videos, ego_ann -> ego_ann_raw_videos 等)
                if view_name == 'ego_raw': in_folder = 'ego_raw_videos'
                elif view_name == 'ego_ann': in_folder = 'ego_ann_raw_videos'
                elif view_name == 'third_raw': in_folder = 'third_raw_videos'
                elif view_name == 'third_ann': in_folder = 'third_ann_raw_videos'
                elif view_name == 'global': in_folder = 'global_videos'
                elif view_name == 'global_ann': in_folder = 'global_ann_videos'
                
                in_path = os.path.join(video_path, in_folder, f"{ep_id}.mp4")
                out_name = f"{file_base}_{view_name}.mp4"
                out_path = os.path.join(raw_dirs[view_name], out_name)

                if not os.path.exists(in_path):
                    continue  # 原视频不存在则跳过

                # 【神级优化】：直接组装 FFmpeg 命令
                # -ss 放在 -i 前面代表快速寻址，极大提升切片速度
                # -t 限制持续时间
                # -vf "scale=-2:'min(480,ih)'" 意思是：如果原高度大于480则缩放到480，否则保持原样 (-2自动保证宽度是双数)
                ffmpeg_cmd = [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-ss", str(start_sec),
                    "-i", in_path,
                    "-t", str(duration_sec),
                    "-vf", "scale=-2:'min(480,ih)'",
                    "-c:v", "libx264",
                    "-preset", "ultrafast",  # 追求极致速度
                    "-an",  # 无音频
                    out_path
                ]
                
                try:
                    subprocess.run(ffmpeg_cmd, check=True)
                    view_paths[f"{view_name}_path"] = out_path
                except subprocess.CalledProcessError as err:
                    print(f"  [FFmpeg Error] 处理 {in_path} 失败!")

            record = {
                "episode_id": ep_id,
                "instruction": instruction,
                "task_index": i,
                "subtask": task['description_processed'],
                "gt_intent": "Unknown",
                "gt_target": task['receptacle_names'][0] if task['receptacle_names'] else "Unknown",
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


if __name__ == "__main__":
    # # 请根据实际路径修改
    # VIDEO_DIR_ROOT = "final_raw_videos/"
    # META_DIR_ROOT = "final_raw_videos/metadata"
    # SAVE_DIR_ROOT = "final_raw_videos/train_dataset"

    # all_info = []

    # # 遍历元数据文件
    # if os.path.exists(META_DIR_ROOT):
    #     json_files = sorted([f for f in os.listdir(META_DIR_ROOT) if f.endswith(".json")])
    #     print(f"Found {len(json_files)} metadata files.")

    #     for f in json_files:
    #         try:
    #             print(f"Processing {f}...")
    #             video_input_dir = VIDEO_DIR_ROOT
    #             records = process_episode(video_input_dir, os.path.join(META_DIR_ROOT, f), SAVE_DIR_ROOT)
    #             all_info.extend(records)
    #         except Exception as e:
    #             print(f"Error processing {f}: {e}")
    #             import traceback

    #             traceback.print_exc()
    # else:
    #     print(f"Metadata directory not found: {META_DIR_ROOT}")

    # # 保存总索引
    # os.makedirs(SAVE_DIR_ROOT, exist_ok=True)
    # with open(os.path.join(SAVE_DIR_ROOT, "benchmark_index.json"), "w") as f:
    #     json.dump(all_info, f, indent=2)

    # print(f"Done! Index saved to {os.path.join(SAVE_DIR_ROOT, 'rovi_benchmark_index.json')}")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm
    import logging

    logging.basicConfig(filename="data_gen_errors.log", filemode="w", level=logging.ERROR)

    VIDEO_DIR_ROOT = "final_raw_videos/"
    META_DIR_ROOT = "final_raw_videos/metadata"
    SAVE_DIR_ROOT = "final_raw_videos/train_dataset"

    def worker(f):
        try:
            records = process_episode(VIDEO_DIR_ROOT, os.path.join(META_DIR_ROOT, f), SAVE_DIR_ROOT)
            return True, records, None
        except Exception as e:
            return False, [], str(e)

    all_info = []

    if os.path.exists(META_DIR_ROOT):
        json_files = sorted([f for f in os.listdir(META_DIR_ROOT) if f.endswith(".json")])
        
        # 【关键修改】：严格控制并发数，保护硬盘！
        # 即使你有 32 核，这里也建议设为 4 或 8。FFmpeg 自己会使用多线程。
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
                        tqdm.write("❌ 一个任务失败，已记录。")
                    pbar.update(1)

    os.makedirs(SAVE_DIR_ROOT, exist_ok=True)
    with open(os.path.join(SAVE_DIR_ROOT, "benchmark_index.json"), "w") as f:
        json.dump(all_info, f, indent=2)