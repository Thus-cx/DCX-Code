import json
import random
import os
import re

# 尝试保留原本的wandb引用，防止报错，如果不需要可以注释掉
try:
    from wandb.apis.importers.internals.internal import ROOT_DIR
except ImportError:
    ROOT_DIR = "."

# ================= 配置 =================
INPUT_ROOT_DIR = "test_my_videos_2/final_benchmark_clips_2"
INPUT_FILE = os.path.join(INPUT_ROOT_DIR, "processed_clips_index.json")
OUTPUT_FILE = os.path.join(INPUT_ROOT_DIR, "final_test_benchmark_third_ann.json")
# ================= 1. 语义常识库 (保留你原本的映射表) =================

# 家具意图映射表 (Furniture Affordance)
# 我们将利用这个表来推断 Ground Truth，但在选项中不再显示具体的家具名
REC_INTENT_MAP = {
    # --- Storage (收纳/储藏) ---
    "cabinet": "Storage (putting away)",
    "fridge": "Storage (putting away)",
    "refrigerator": "Storage (putting away)",
    "shelf": "Storage (putting away)",
    "shelves": "Storage (putting away)",
    "wardrobe": "Storage (putting away)",
    "chest_of_drawers": "Storage (putting away)",
    "filing_cabinet": "Storage (putting away)",
    "breadbin": "Storage (putting away)",
    "box": "Storage (putting away)",
    "canister": "Storage (putting away)",
    "basket": "Storage (putting away)",

    # --- Placement (放置/陈列/使用) ---
    "table": "Placement (staging for use)",
    "counter": "Placement (staging for use)",
    "desk": "Placement (staging for use)",
    "stand": "Placement (staging for use)",
    "nightstand": "Placement (staging for use)",
    "coffee_table": "Placement (staging for use)",
    "stool": "Placement (staging for use)",
    "bench": "Placement (staging for use)",
    "chair": "Placement (staging for use)",
    "sofa": "Placement (staging for use)",
    "couch": "Placement (staging for use)",
    "bed": "Placement (staging for use)",
    "floor": "Placement (staging for use)",
    "tv_stand": "Placement (staging for use)",
    "rack": "Placement (staging for use)",
    "tray": "Placement (staging for use)",
    "stove": "Placement (staging for use)", # 修正：stove在不使用时通常视为放置台面或准备区

    # --- Maintenance (清洁/维护/水源) ---
    "sink": "Maintenance (cleaning/filling)",
    "bathtub": "Maintenance (cleaning/filling)",
    "shower": "Maintenance (cleaning/filling)",
    "dishwasher": "Maintenance (cleaning/filling)",
    "washer_dryer": "Maintenance (cleaning/filling)",

    # --- Disposal (丢弃) ---
    "trashcan": "Disposal (throwing away)",
    "toilet": "Disposal (throwing away)",

    # --- Appliances acting as Storage ---
    "oven": "Storage (putting away)",
    "microwave": "Storage (putting away)"
}

# 定义四个标准化的意图选项（不包含物体信息）
PURE_INTENT_OPTIONS = {
    "Storage": "Storage (putting an object away into a container or furniture)",
    "Placement": "Placement (placing an object on a surface for use, display, or resting)",
    "Maintenance": "Maintenance (cleaning, washing, or interacting with water/appliances)",
    "Disposal": "Disposal (throwing an object away into trash or similar)"
}

# 辅助映射：将 REC_INTENT_MAP 中的详细描述映射到四大类 Key
INTENT_GROUP_MAPPING = {
    "Storage (putting away)": "Storage",
    "Placement (staging for use)": "Placement",
    "Maintenance (cleaning/filling)": "Maintenance",
    "Disposal (throwing away)": "Disposal"
}

# ================= 2. 核心逻辑函数 =================

def sanitize_text(text):
    # 去除 ID 后缀 (e.g., table_1 -> table)
    text = re.sub(r'_\d+', '', text)
    # 归一化一些特殊命名
    text = text.replace("floor_kitchen", "floor").replace("floor_living", "floor")
    text = text.replace("kitchen_island", "counter")
    text = text.replace("coffee_table", "table")
    return text

def parse_task_info(task_str):
    """
    解析原始任务指令
    """
    clean_task = sanitize_text(task_str)
    words = clean_task.split()
    verb = words[0]

    # 提取 Object 和 Target
    prep_indices = [i for i, w in enumerate(words) if w in ["to", "in", "on", "at"]]

    if prep_indices:
        split_idx = prep_indices[0]
        obj = " ".join(words[1:split_idx])
        target = " ".join(words[split_idx + 1:])
    else:
        obj = " ".join(words[1:])
        target = "unknown"

    return verb, obj, target

def get_target_category_raw(target_name):
    """
    根据目标家具名称，从 REC_INTENT_MAP 获取原始详细意图
    """
    target_lower = target_name.lower()

    # 1. 优先精确匹配
    for k, v in REC_INTENT_MAP.items():
        if k == target_lower:
            return v

    # 2. 模糊匹配 (e.g., "kitchen_cabinet" 包含 "cabinet")
    sorted_keys = sorted(REC_INTENT_MAP.keys(), key=len, reverse=True)
    for k in sorted_keys:
        if k in target_lower:
            return REC_INTENT_MAP[k]

    # 默认兜底
    return "Placement (staging for use)"

def get_ground_truth_intent_category(verb, target_name):
    """
    获取 GT 对应的 PURE_INTENT_OPTIONS 的 Key (Storage, Placement, etc.)
    """
    # 1. 动词强覆盖 (Clean/Fill/Pour/Wash 必然是 Maintenance)
    if verb in ["Clean", "Fill", "Pour", "Wash"]:
        return "Maintenance"
    
    # 2. 目的地推断
    raw_intent_desc = get_target_category_raw(target_name)
    
    # 3. 映射到四大类
    return INTENT_GROUP_MAPPING.get(raw_intent_desc, "Placement")

def generate_pure_intent_options(verb, target_name):
    """
    [修改核心] 生成纯意图选项：
    1. 不再使用 confusing targets 和物体名称
    2. 只有固定的4个类别
    3. 选项内容不包含物体（避免文本泄露）
    """
    # 1. 获取正确的意图 Key (例如 "Storage")
    gt_key = get_ground_truth_intent_category(verb, target_name)
    
    # 2. 获取对应的完整描述文本
    gt_option_text = PURE_INTENT_OPTIONS[gt_key]
    
    # 3. 收集所有可能的选项文本
    all_keys = list(PURE_INTENT_OPTIONS.keys())
    options_text_list = []
    
    # 先把正确的放进去
    options_text_list.append(gt_option_text)
    
    # 把剩下的放进去
    for key in all_keys:
        desc = PURE_INTENT_OPTIONS[key]
        if desc != gt_option_text:
            options_text_list.append(desc)
    
    # 4. 打乱选项顺序
    random.shuffle(options_text_list)
    
    # 5. 找到正确答案的索引 (A, B, C, D)
    correct_idx = options_text_list.index(gt_option_text)
    correct_char = chr(65 + correct_idx)
    
    return {
        "A": options_text_list[0],
        "B": options_text_list[1],
        "C": options_text_list[2],
        "D": options_text_list[3]
    }, correct_char, gt_key  # 返回 gt_key (如 "Storage") 方便后续统计

# ================= 3. 主流程 =================

def main():
    print(f"Loading data from {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, 'r') as f:
        clips = json.load(f)
        
    benchmark_data = []
    
    for clip in clips:
        if clip.get('duration', 0) < 1.0: continue
        
        # Parse GT
        verb, obj, target = parse_task_info(clip['gt_task'])
        
        # [修改] 使用新的生成函数
        options, answer, gt_category = generate_pure_intent_options(verb, target)
        
        # Bucket Logic (保留你原有的难度分级逻辑)
        metrics = clip.get('metrics', {})
        vl = metrics.get('view_loss_ratio', 0.0)
        pr = metrics.get('proximity_ratio', 0.0)
        sr = metrics.get('shake_ratio', 0.0)
        
        if vl > 0.20 or pr > 0.20 or sr > 0.30: 
            bucket = "Bucket_C_Hard"
        elif vl > 0.10 or pr > 0.10 or sr > 0.15: 
            bucket = "Bucket_B_Medium"
        else: 
            bucket = "Bucket_A_Easy"

        # [修改] Prompt 去除物体信息
        question = (
            "The robot is observing a human performing a daily task.\n"
            "Task: Based ONLY on the visual observation (human trajectory, body language, and environment context), "
            "predict the high-level intent category of the human.\n"
            "Note: Do not rely on object detection alone, focus on the navigation destination and interaction pattern.\n"
            "Select the most appropriate category."
        )

        benchmark_data.append({
            "video_path": clip.get('third_ann_video_path'),
            "third_person_video_path": clip.get('third_raw_video_path'), # 确保你有这个字段
            "bucket": bucket,
            "phase": clip['phase'],
            "question": question,
            "options": options,
            "correct_option": answer,
            "gt_content": gt_category, # 这里记录简短的意图类别 (e.g. "Storage")
            "raw_gt_task": clip['gt_task'], # 保留原始任务做对照
            "metrics": metrics
        })
        
    print(f"Generated {len(benchmark_data)} Pure Intent Prediction questions.")
    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(benchmark_data, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    main()