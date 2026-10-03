import json
import os
from tqdm import tqdm

# ================= 配置区域 =================
# 输入：刚才 FFmpeg 脚本跑出来的真实视频索引文件
INPUT_FILE = "real_raw_videos/train_dataset/real_benchmark_index.json"
# 输出：大模型直接读取的最终考卷
OUTPUT_FILE = "real_raw_videos/train_dataset/real_final_exam.json"

# ================= 1. 语义常识库 (已补充真实场景词汇) =================

# 补充了 washing_machine, faucet, trash_can (统一命名风格)
REC_INTENT_MAP = {
    # --- Storage (收纳/储藏) ---
    "cabinet": "Storage", "fridge": "Storage", "refrigerator": "Storage",
    "shelf": "Storage", "wardrobe": "Storage", "box": "Storage",
    "drawer": "Storage",
    # --- Placement (放置) ---
    "dining_table": "Placement", "table": "Placement", "desk": "Placement",
    "nightstand": "Placement", "coffee_table": "Placement", "chair": "Placement",
    "sofa": "Placement", "bed": "Placement", "floor": "Placement", "rack": "Placement",
    # --- Maintenance (清洁/维护) ---
    "sink": "Maintenance", "bathtub": "Maintenance", "dishwasher": "Maintenance",
    "washer_dryer": "Maintenance", "washing_machine": "Maintenance", "faucet": "Maintenance",
    # --- Disposal (丢弃) ---
    "trashcan": "Disposal", "trash_can": "Disposal", "toilet": "Disposal", "bin": "Disposal"
}

# 补充了 pillow, blanket, clothes, computer
object_category_map = {
    "apple": "kitchenware", "bottle": "kitchenware", "box": "miscellaneous",
    "cup": "kitchenware", "kettle": "kitchenware", "knife": "kitchenware",
    "plate": "kitchenware", "spoon": "kitchenware", "bowl": "kitchenware",
    # --- 新增真实场景物品 ---
    "pillow": "household_items",
    "blanket": "household_items",
    "clothes": "clothing",
    "computer": "electronics",
    "laptop": "electronics"
}

OBJECTS = list(object_category_map.keys()) + ["Unknown"]

INTENT_CATEGORIES = [
    "Storage", "Placement", "Maintenance", "Disposal", "Unknown"
]

# 补充了 laundry_room 匹配你的 JSON
AREAS = [
    'Unknown', 'bedroom', 'hallway', 'kitchen', 'living_room',
    'bathroom', 'dining_room', 'garage', 'laundryroom', 'laundry_room',
    'office'
]

FURNITURES = list(REC_INTENT_MAP.keys()) + ["Unknown"]

# ================= 2. Prompt 构造器 =================

def generate_prompts():
    """生成阶梯式诊断 (Step-by-Step Diagnostic) 的 Prompt (完全对齐仿真)"""
    system_prompt = f"""You are an embodied AI assistant analyzing a human-robot collaboration video from a real-world perspective.
Due to the camera angle, the human's torso might cause physical occlusion.

**Task:**
Perform a step-by-step diagnostic reasoning to anticipate the human's final intent.
Output a SINGLE JSON object exactly matching this structure, answering the 6 progressive questions:

{{
  "Q1_target_area": "String. Choose ONE functional area from this list that the human is approaching: {json.dumps(AREAS)}.",
  "Q2_is_holding_object": "Boolean. True if the human appears to be holding an object, False otherwise.",
  "Q3_held_object_category": "String. If Q2 is True, identify the object. If severely obscured, output 'Unknown'. If not, choose from: {json.dumps(OBJECTS)}.",
  "Q4_target_furniture": "String. Choose ONE furniture from this list that the human is targeting: {json.dumps(FURNITURES)}.",
  "Q5_furniture_affordance": "String. Choose the primary affordance of the target furniture from: {json.dumps(INTENT_CATEGORIES)}.",
  "Q6_final_intent_triplet": ["Action", "Object", "Target_Furniture"]
}}

**Notes:**
- For Q6 Action, choose from: {json.dumps(INTENT_CATEGORIES)}.
- For Q6 Object, choose from: {json.dumps(OBJECTS)}.
- For Q6 Target, choose from {json.dumps(FURNITURES)}.
- Output ONLY valid JSON.
"""
    user_prompt = "Analyze the video slice and output the required 6-step JSON diagnostic reasoning."
    return system_prompt, user_prompt

# ================= 3. 主程序 =================

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到索引文件: {INPUT_FILE}")
        return

    print(f"📥 正在加载切片索引: {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        clips = json.load(f)

    benchmark_dataset = []
    sys_prompt, usr_prompt = generate_prompts()

    print(f"⚙️ 正在生成考卷，共 {len(clips)} 个切片...")

    for clip in tqdm(clips):
        # 【核心逻辑】：再也不需要复杂的匹配和推导了，直接拿来用！
        gt_area = clip.get("area", "Unknown")
        gt_furniture = clip.get("furniture", "Unknown")
        gt_object = clip.get("object", "Unknown")
        gt_intent = clip.get("affordance", "Unknown").capitalize() # 统一首字母大写对齐选项
        gt_triplet = clip.get("gt_intent", [])

        # 防止大小写或拼写导致不匹配选项池
        if gt_intent not in INTENT_CATEGORIES:
             gt_intent = REC_INTENT_MAP.get(gt_furniture, "Placement")

        # 构造独立考题 Entry
        entry = {
            "question_id": f"{clip['episode_id']}_{clip['observation_type']}_{clip['slice_param']}",

            # 核心元数据
            "episode_id": clip['episode_id'],
            "observation_type": clip['observation_type'],
            "slice_param": clip['slice_param'],
            "duration": clip.get('duration', 0.0),

            # 视觉输入路径 (提取三个视角的真实路径)
            "video_paths": {
                "ego": clip.get('ego_path'),
                "third": clip.get('third_path'),
                "global": clip.get('global_path')
            },

            # 考题内容
            "system_prompt": sys_prompt,
            "user_prompt": usr_prompt,

            # Ground Truth (用于自动判分脚本比对)
            "ground_truth": {
                "target_area": gt_area,
                "is_holding_object": True,  # 你的 JSON 中 is_held 全部为 true
                "held_object_category": gt_object,
                "target_furniture": gt_furniture,
                "furniture_affordance": gt_intent,
                "final_intent_triplet": gt_triplet
            }
        }

        benchmark_dataset.append(entry)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(benchmark_dataset, f, indent=2, ensure_ascii=False)

    print(f"✅ 生成完毕！成功打包了 {len(benchmark_dataset)} 道测试题。")
    print(f"📄 考卷已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()