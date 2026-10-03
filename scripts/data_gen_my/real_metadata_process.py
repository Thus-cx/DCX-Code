import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import logging

# ================= 配置区域 =================
TARGET_W = 480 
TARGET_H = 480 
FPS = 10.0  

# ================= 辅助函数 =================
def get_video_duration(video_path):
    """通过 ffprobe 获取视频精确时长(秒)"""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of",
        "default=noprint_wrappers=1:nokey=1", video_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        logging.error(f"无法读取视频时长: {video_path}, 错误: {e}")
        return 0.0

# ================= 核心处理函数 =================
def process_real_episode(video_path, metadata_file, output_base_dir):
    view_names = ['ego', 'third', 'global'] 
    
    raw_dirs = {k: os.path.join(output_base_dir, k) for k in view_names}
    for d in raw_dirs.values(): 
        os.makedirs(d, exist_ok=True)

    with open(metadata_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 【重要修改】强制转为字符串，确保处理 1.mp4 而不是 001.mp4 报错
    ep_id = str(data.get('episode_id', 'unknown'))
    
    reference_video = os.path.join(video_path, f"ego", f"{ep_id}.mp4")
    if not os.path.exists(reference_video):
        logging.error(f"找不到基准视频，跳过: {reference_video}")
        return []
    
    total_duration_sec = get_video_duration(reference_video)
    if total_duration_sec <= 0:
        return []

    processed_records = []

    # ==========================================
    # 【重要修改】移除 subtasks 循环，直接对整个视频计算切片秒数
    # ==========================================
    L = total_duration_sec
    T30 = L * 0.3
    T60 = L * 0.6
    T90 = L * 0.9

    slice_plan = [
        # 累积观测
        {"type": "accumulative", "param": "transit_30", "start_sec": 0, "end_sec": T30},
        {"type": "accumulative", "param": "transit_60", "start_sec": 0, "end_sec": T60},
        {"type": "accumulative", "param": "transit_90", "start_sec": 0, "end_sec": T90},
        # 等长分段
        {"type": "fixed_len", "param": "phase_early", "start_sec": 0, "end_sec": T30},
        {"type": "fixed_len", "param": "phase_mid", "start_sec": T30, "end_sec": T60},
        {"type": "fixed_len", "param": "phase_late", "start_sec": T60, "end_sec": L}
    ]

    # ==========================================
    # 执行 FFmpeg 切片
    # ==========================================
    for plan in slice_plan:
        s_sec = plan['start_sec']
        e_sec = plan['end_sec']
        stype = plan['type']
        sparam = plan['param']
        duration_sec = e_sec - s_sec

        if duration_sec <= 0.1: 
            continue

        # 去掉了 t{i} 任务索引，因为一个视频就是一个任务
        file_base = f"{ep_id}_{stype}_{sparam}"
        view_paths = {}

        for view_name in view_names:
            in_folder = f"{view_name}"
            in_path = os.path.join(video_path, in_folder, f"{ep_id}.mp4")
            out_name = f"{file_base}_{view_name}.mp4"
            out_path = os.path.join(raw_dirs[view_name], out_name)

            if not os.path.exists(in_path): 
                continue  

            vf_filter = f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=decrease,pad={TARGET_W}:{TARGET_H}:(ow-iw)/2:(oh-ih)/2"

            ffmpeg_cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{s_sec:.3f}",
                "-i", in_path,
                "-t", f"{duration_sec:.3f}",
                "-vf", vf_filter,
                "-r", str(FPS),  
                "-c:v", "libx264",
                "-preset", "ultrafast", 
                "-an",  
                out_path
            ]
            
            try:
                subprocess.run(ffmpeg_cmd, check=True)
                view_paths[f"{view_name}_path"] = out_path
            except subprocess.CalledProcessError as err:
                logging.error(f"[FFmpeg Error] 处理 {in_path} 失败: {err}")

        # 【重要修改】按照极简三元组格式组装记录
        record = {
            "episode_id": ep_id,
            "area": data.get("area", "Unknown"),
            "is_held": data.get("is_held", "Unknown"),
            "furniture": data.get("furniture", "Unknown"),
            "affordance": data.get("affordance", "Unknown"),
            "object": data.get("object", "Unknown"),
            "gt_intent": data.get("intent", []),
            "observation_type": stype,
            "slice_param": sparam,
            "duration": round(duration_sec, 2)
        }
        
        record.update(view_paths)
        processed_records.append(record)

    return processed_records

# ================= 多进程调度与启动 =================
def worker(f_name):
    try:
        records = process_real_episode(VIDEO_DIR_ROOT, os.path.join(META_DIR_ROOT, f_name), SAVE_DIR_ROOT)
        return True, records, None
    except Exception as e:
        import traceback
        return False, [], f"{f_name} 处理异常: {str(e)}\n{traceback.format_exc()}"

if __name__ == "__main__":
    VIDEO_DIR_ROOT = "real_raw_videos/"
    META_DIR_ROOT = "real_raw_videos/metadata"
    SAVE_DIR_ROOT = "real_raw_videos/train_dataset"

    logging.basicConfig(
        filename="real_data_gen_errors.log", 
        filemode="w", 
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    all_info = []

    if not os.path.exists(META_DIR_ROOT):
        print(f"❌ 找不到元数据目录: {META_DIR_ROOT}")
    else:
        # 这里顺便做了个优化：按文件名中的数字大小排序，而不是字符串规则 (防止 10.json 排在 2.json 前面)
        json_files = sorted(
            [f for f in os.listdir(META_DIR_ROOT) if f.endswith(".json")],
            key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x
        )
        
        max_workers = 4 
        print(f"🚀 启动极简版 FFmpeg 切片，检测到 {len(json_files)} 个剧集...")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(worker, f): f for f in json_files}
            
            with tqdm(total=len(json_files), desc="🎬 处理中", unit="ep") as pbar:
                for future in as_completed(futures):
                    success, records, err_msg = future.result()
                    if success:
                        all_info.extend(records)
                    else:
                        logging.error(err_msg)
                    pbar.update(1)

        os.makedirs(SAVE_DIR_ROOT, exist_ok=True)
        index_path = os.path.join(SAVE_DIR_ROOT, "real_benchmark_index.json")
        with open(index_path, "w", encoding='utf-8') as f:
            json.dump(all_info, f, indent=2, ensure_ascii=False)
            
        print(f"✅ 处理完成！完整索引已保存至: {index_path}")