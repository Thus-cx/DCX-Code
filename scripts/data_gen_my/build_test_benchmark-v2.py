import json
import os
import random
import re
from tqdm import tqdm

# ================= 配置区域 =================
# 输入：上一步生成的索引文件
INPUT_FILE = "test_my_videos_2/final_benchmark_rovi/rovi_benchmark_index.json"
# 输出：最终考卷
OUTPUT_FILE = "test_my_videos_2/final_benchmark_rovi/rovi_final_exam.json"

# ================= 1. 语义常识库 (保留你原有的精华) =================

# 家具意图映射表 (用于自动推断 GT Intent)
REC_INTENT_MAP = {
    # --- Storage (收纳/储藏) ---
    "cabinet": "Storage",
    "fridge": "Storage",
    "refrigerator": "Storage",
    "shelf": "Storage",
    "shelves": "Storage",
    "wardrobe": "Storage",
    "chest_of_drawers": "Storage",
    "filing_cabinet": "Storage",
    "breadbin": "Storage",
    "box": "Storage",
    "canister": "Storage",
    "basket": "Storage",
    "dresser": "Storage",
    "cupboard": "Storage",
    "drawer": "Storage",

    # --- Placement (放置) ---
    "table": "Placement",
    "counter": "Placement",
    "desk": "Placement",
    "stand": "Placement",
    "nightstand": "Placement",
    "coffee_table": "Placement",
    "stool": "Placement",
    "bench": "Placement",
    "chair": "Placement",
    "sofa": "Placement",
    "couch": "Placement",
    "bed": "Placement",
    "floor": "Placement",
    "tv_stand": "Placement",
    "rack": "Placement",
    "tray": "Placement",

    # --- Maintenance (清洁/维护) ---
    "sink": "Maintenance",
    "bathtub": "Maintenance",
    "shower": "Maintenance",
    "dishwasher": "Maintenance",
    "washer_dryer": "Maintenance",
    "washer": "Maintenance",
    "dryer": "Maintenance",

    # --- Disposal (丢弃) ---
    "trashcan": "Disposal",
    "toilet": "Disposal",
    "bin": "Disposal",
    "garbage": "Disposal",

    # --- Heating (特殊处理) ---
    "stove": "Placement",
    "oven": "Storage",
    "microwave": "Storage"
}

# 意图大类列表 (用于 System Prompt)
INTENT_CATEGORIES = [
    "Storage",  # putting away
    "Placement",  # staging for use
    "Maintenance",  # cleaning/filling
    "Disposal"  # throwing away
]

# 全局物体候选列表 (Context List)
# 基于你的 REC_INTENT_MAP keys 构建，作为模型的选项池
GLOBAL_OBJECT_CONTEXT = sorted(list(set([
    "armchair", "bar_stool", "bathtub", "bed", "bench",
    "cabinet", "chair", "chest_of_drawers", "coffee_table",
    "counter", "cupboard", "desk", "dining_table",
    "dishwasher", "dresser", "fridge", "lamp",
    "microwave", "nightstand", "ottoman", "oven",
    "shelf", "shower", "sink", "sofa",
    "stool", "stove", "table", "toilet",
    "towel_rack", "trashcan", "tv_stand", "vanity",
    "washer_dryer", "wardrobe", "box", "bin"
])))


# ================= 2. 核心逻辑函数 =================

def sanitize_target_name(text):
    """
    清洗 Habitat 的物体名 (e.g., 'table_10_0' -> 'table')
    """
    if not text: return "unknown"
    # 去除数字后缀 _0, _1 等
    text = re.sub(r'_\d+', '', text)
    # 转换为小写
    text = text.lower()
    return text.strip()


def match_target_to_context(raw_target):
    """
    将原始 target 映射到全局列表中的类别
    """
    clean_name = sanitize_target_name(raw_target)

    # 1. 精确匹配
    if clean_name in GLOBAL_OBJECT_CONTEXT:
        return clean_name

    # 2. 模糊匹配 (最长前缀匹配)
    best_match = None
    max_len = 0

    for cand in GLOBAL_OBJECT_CONTEXT:
        # e.g. "kitchen_counter" contains "counter"
        if cand in clean_name or clean_name in cand:
            if len(cand) > max_len:
                max_len = len(cand)
                best_match = cand

    return best_match if best_match else "other"


def infer_intent(target_category):
    """
    根据目标物体类别推断意图 (利用 REC_INTENT_MAP)
    """
    return REC_INTENT_MAP.get(target_category, "Placement")  # 默认 Placement


def generate_prompts(context_list):
    """
    生成符合 Benchmark 要求的 System Prompt 和 User Prompt
    """
    # System Prompt: 定义任务和输出格式
    system_prompt = f"""You are an embodied robot assistant. You will watch a fragment of a human activity video.
Your goal is to anticipate the human's future intent and the final target object.

**Context:**
The environment contains the following interactable objects:
{json.dumps(context_list)}

**Task:**
Analyze the user's trajectory, body orientation, and the object they are holding.
Output a SINGLE JSON object with the following keys:
1. "reasoning": A brief explanation (max 20 words) of the visual evidence.
2. "intent_category": Choose ONE from {json.dumps(INTENT_CATEGORIES)}.
3. "predicted_target": Select the Top-3 most likely target categories from the Context list.
4. "confidence": Choose from ["Low", "Medium", "High"].

**Output Format:**
Return ONLY the JSON object. Do not include markdown formatting or explanations.
"""

    # User Prompt: 简单引导
    user_prompt = "Based on the video fragment, predict the user's intent."

    return system_prompt, user_prompt


# ================= 3. 主程序 =================

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Index file not found at {INPUT_FILE}")
        return

    print(f"Loading index from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        clips = json.load(f)

    benchmark_dataset = []
    skipped_count = 0

    print(f"Processing {len(clips)} clips...")

    for clip in tqdm(clips):
        # 1. 基础校验
        if 'gt_target' not in clip:
            skipped_count += 1
            continue

        # 2. GT 处理
        raw_target = clip['gt_target']
        target_category = match_target_to_context(raw_target)
        subtask = clip['subtask']

        # 自动推断 Intent (如果元数据里没有或为 Unknown)
        gt_intent = clip.get('gt_intent', 'Unknown')
        if gt_intent == 'Unknown' or not gt_intent:
            if subtask.startswith("Clean") or subtask.startswith("Fill") or subtask.startswith("Pour"):
                gt_intent = "Maintenance"
            gt_intent = infer_intent(target_category)

        # 3. 构造考题
        sys_prompt, usr_prompt = generate_prompts(GLOBAL_OBJECT_CONTEXT)

        # 构造 Entry
        entry = {
            "question_id": f"{clip['episode_id']}_t{clip['task_index']}_{clip['observation_type']}_{clip['slice_param']}",

            # 核心元数据
            "episode_id": clip['episode_id'],
            "task_index": clip['task_index'],
            "instruction": clip['instruction'],

            # 评估维度
            "observation_type": clip['observation_type'],  # accumulative / window
            "slice_param": clip['slice_param'],  # p20 / init ...
            "difficulty": clip.get('difficulty', 'Unknown'),
            "duration": clip.get('duration', 0.0),

            # 视觉输入路径 (保留 metadata_process-strong.py 生成的所有路径)
            "video_paths": {
                "ego_raw": clip.get('ego_raw_path'),
                "third_raw": clip.get('third_raw_path'),
                "global": clip.get('global_path'),
                "ego_ann": clip.get('ego_ann_path')
            },

            # 考题内容
            "system_prompt": sys_prompt,
            "user_prompt": usr_prompt,
            "candidate_list": GLOBAL_OBJECT_CONTEXT,

            # Ground Truth (用于自动判分)
            "ground_truth": {
                "intent_category": gt_intent,
                "target_object_raw": raw_target,
                "target_category": target_category,
                # 辅助信息
                "valid_targets": [target_category]
            }
        }

        benchmark_dataset.append(entry)

    # 保存结果
    print(f"Generated {len(benchmark_dataset)} exam questions. (Skipped {skipped_count} invalid clips)")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_FILE) if os.path.dirname(OUTPUT_FILE) else '.', exist_ok=True)

    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(benchmark_dataset, f, indent=2)

    print("Done! Benchmark building complete.")


if __name__ == "__main__":
    main()