import json
import os
import random
import shutil

# ================= 配置区域 =================
# 你跑完模型后生成的原始 JSON 结果文件
INPUT_RESULT_FILE = "final_results/qwen_results_ego_raw_nframes_8.json"

# 输出的根目录
OUTPUT_DIR = "human_eval_task"
VIDEO_DIR = os.path.join(OUTPUT_DIR, "videos")
SUBSET_JSON_FILE = os.path.join(OUTPUT_DIR, "human_eval_ground_truth.json")
ZIP_FILE_NAME = "human_eval_samples" # 最终会生成 human_eval_samples.zip

NUM_SAMPLES = 200
RANDOM_SEED = 42 # 固定随机种子，保证每次抽出来的 200 个一样

# ================= 主程序 =================

def main():
    print(f"Loading data from {INPUT_RESULT_FILE}...")
    with open(INPUT_RESULT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 过滤掉视频路径不存在的无效数据
    valid_data = []
    for item in data:
        video_path = item.get("video_paths", {}).get("ego_raw")
        if video_path:
            # 【新增修复逻辑】：纠正 JSON 中错误的文件夹路径
            if "test_dataset/" in video_path:
                video_path = video_path.replace("test_dataset/", "test_dataset_strong/")
                # 必须更新回字典中，保证后续 shutil.copy2 读到的是正确路径
                item["video_paths"]["ego_raw"] = video_path
                
            if os.path.exists(video_path):
                valid_data.append(item)

    if len(valid_data) < NUM_SAMPLES:
        print(f"Error: Not enough valid videos. Found {len(valid_data)}, need {NUM_SAMPLES}.")
        return

    # 设置随机种子并抽取 200 个样本
    random.seed(RANDOM_SEED)
    sampled_items = random.sample(valid_data, NUM_SAMPLES)

    # 创建输出目录
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR) # 如果存在老文件夹，先清空
    os.makedirs(VIDEO_DIR, exist_ok=True)

    print(f"Successfully sampled {NUM_SAMPLES} clips. Copying videos...")

    human_eval_records = []

    for index, item in enumerate(sampled_items):
        original_video_path = item["video_paths"]["ego_raw"]
        
        # 【关键保护机制】：重命名视频文件，如 "sample_001.mp4"
        # 绝对不能用原始的 question_id，否则人类可能从文件名里的 transit_90 猜到进度！
        # 这里改用了 03d 格式，因为样本数是 200，保证对齐效果更好 (sample_001.mp4)
        new_video_name = f"sample_{index + 1:03d}.mp4"
        new_video_path = os.path.join(VIDEO_DIR, new_video_name)
        
        # 复制文件
        shutil.copy2(original_video_path, new_video_path)

        # 记录映射关系和 GT，方便你后期自动算分
        record = {
            "human_task_id": new_video_name,
            "original_question_id": item["question_id"],
            "difficulty": item["difficulty"],
            "slice_param": item["slice_param"],
            "instruction": item["instruction"], # 真实的文字提示（绝对不能给人类看！）
            "ground_truth": item["ground_truth"],
            "model_pred": item["parsed_json"] # 顺便记录当时大模型是怎么答的
        }
        human_eval_records.append(record)

    # 保存 Ground Truth JSON
    with open(SUBSET_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(human_eval_records, f, indent=4)
    print(f"Ground truth subset saved to {SUBSET_JSON_FILE}")

    # 打包成 ZIP 文件
    print(f"Zipping the videos into {ZIP_FILE_NAME}.zip ...")
    shutil.make_archive(ZIP_FILE_NAME, 'zip', OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("✅ Human Evaluation Task Preparation Complete!")
    print("="*80)
    print(f"1. Send this to your human evaluators: {ZIP_FILE_NAME}.zip (Contains only renamed videos)")
    print(f"2. Keep this for yourself: {SUBSET_JSON_FILE} (Contains answers and mapping)")
    print("="*80)

if __name__ == "__main__":
    main()