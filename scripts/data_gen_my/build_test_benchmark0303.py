import json
import os
import random
import re
from tqdm import tqdm

# ================= 配置区域 =================
# 输入：上一步生成的索引文件
INPUT_FILE = "test_dataset_fetch_robot_08/fetch_08_final_test/benchmark_index.json"
# 输出：最终考卷
OUTPUT_FILE = "test_dataset_fetch_robot_08/fetch_08_final_test/final_exam.json"

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
    "armchair": "Placement",
    "bar_stool": "Placement",
    "dining_table": "Placement",
    "ottoman": "Placement",
    "vanity": "Placement",
    "towel_rack": "Placement",
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

object_category_map = {
    "action_figure": "toys",
    "android_figure": "toys",
    "apple": "kitchenware",
    "backpack": "clothing",
    "baseballbat": "sports_equipment",
    "basket": "household_items",
    "basketball": "sports_equipment",
    "bath_towel": "household_items",
    "battery_charger": "electronics",
    "board_game": "toys",
    "book": "stationery",
    "bottle": "kitchenware",
    "bowl": "kitchenware",
    "box": "miscellaneous",
    "bread": "kitchenware",
    "bundt_pan": "kitchenware",
    "butter_dish": "kitchenware",
    "c-clamp": "tools",
    "cake_pan": "kitchenware",
    "can": "kitchenware",
    "can_opener": "kitchenware",
    "candle": "household_items",
    "candle_holder": "household_items",
    "candy_bar": "food",
    "canister": "kitchenware",
    "carrying_case": "miscellaneous",
    "casserole": "kitchenware",
    "cellphone": "electronics",
    "clock": "household_items",
    "credit_card": "stationery",
    "cup": "kitchenware",
    "cushion": "furniture",
    "doll": "toys",
    "dumbbell": "sports_equipment",
    "egg": "food",
    "electric_kettle": "kitchenware",
    "electronic_cable": "electronics",
    "file_sorter": "stationery",
    "folder": "stationery",
    "fork": "kitchenware",
    "gaming_console": "electronics",
    "glass": "kitchenware",
    "hammer": "tools",
    "hand_towel": "household_items",
    "handbag": "clothing",
    "hard_drive": "electronics",
    "hat": "clothing",
    "helmet": "sports_equipment",
    "jug": "kitchenware",
    "kettle": "kitchenware",
    "keychain": "miscellaneous",
    "knife": "kitchenware",
    "ladle": "kitchenware",
    "lamp": "household_items",
    "laptop": "electronics",
    "laptop_cover": "electronics",
    "laptop_stand": "electronics",
    "lunch_box": "kitchenware",
    "milk_frother_cup": "kitchenware",
    "monitor_stand": "electronics",
    "mouse_pad": "electronics",
    "multiport_hub": "electronics",
    "pan": "kitchenware",
    "pen": "stationery",
    "pencil_case": "stationery",
    "phone_stand": "electronics",
    "picture_frame": "household_items",
    "pitcher": "kitchenware",
    "plant_container": "household_items",
    "plant_saucer": "household_items",
    "plate": "kitchenware",
    "potato": "food",
    "ramekin": "kitchenware",
    "scissors": "stationery",
    "screwdriver": "tools",
    "shoe": "clothing",
    "soap_dish": "household_items",
    "soap_dispenser": "household_items",
    "spatula": "kitchenware",
    "spectacles": "stationery",
    "spicemill": "kitchenware",
    "sponge": "household_items",
    "spoon": "kitchenware",
    "spray_bottle": "household_items",
    "squeezer": "kitchenware",
    "statue": "household_items",
    "stuffed toy": "toys",
    "stuffed_toy": "toys",
    "sushi_mat": "kitchenware",
    "tape": "stationery",
    "teapot": "kitchenware",
    "tennis_racquet": "sports_equipment",
    "tomato": "food",
    "toy_airplane": "toys",
    "toy_animal": "toys",
    "toy_bee": "toys",
    "toy_cactus": "toys",
    "toy_construction_set": "toys",
    "toy_fire_truck": "toys",
    "toy_food": "toys",
    "toy_fruits": "toys",
    "toy_pineapple": "toys",
    "toy_swing": "toys",
    "toy_vehicle": "toys",
    "tray": "kitchenware",
    "vase": "household_items",
    "watch": "accessories",
}

OBJECTS = list(object_category_map.keys()) + ["Unknown"]

# 意图大类列表 (用于 System Prompt)
INTENT_CATEGORIES = [
    "Storage",  # putting away
    "Placement",  # staging for use
    "Maintenance",  # cleaning/filling
    "Disposal",  # throwing away
    "Unknown"
]


# 全局物体候选列表 (Context List)
# 基于你的 REC_INTENT_MAP keys 构建，作为模型的选项池
# GLOBAL_OBJECT_CONTEXT = sorted(list(set([
#     "armchair", "bar_stool", "bathtub", "bed", "bench",
#     "cabinet", "chair", "chest_of_drawers", "coffee_table",
#     "counter", "cupboard", "desk", "dining_table",
#     "dishwasher", "dresser", "fridge", "lamp",
#     "microwave", "nightstand", "ottoman", "oven",
#     "shelf", "shower", "sink", "sofa",
#     "stool", "stove", "table", "toilet",
#     "towel_rack", "trashcan", "tv_stand", "vanity",
#     "washer_dryer", "wardrobe", "box", "bin"
# ])))

AREAS = ['Unknown','bedroom', 'hallway', 'kitchen', 'living_room',
         'bathroom', 'dining_room', 'garage', 'laundryroom',
         'entryway', 'closet', 'office', 'other_room', 'toilet',
         'outdoor', 'porch', 'workout', 'tv', 'lounge', 'utilityroom']

FURNITURES = list(REC_INTENT_MAP.keys()) + ["Unknown"]

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
    if clean_name in FURNITURES:
        return clean_name

    # 2. 模糊匹配 (最长前缀匹配)
    best_match = None
    max_len = 0

    for cand in FURNITURES:
        # e.g. "kitchen_counter" contains "counter"
        if cand in clean_name or clean_name in cand:
            if len(cand) > max_len:
                max_len = len(cand)
                best_match = cand

    return best_match if best_match else "Unknown"


def infer_intent(target_category):
    """
    根据目标物体类别推断意图 (利用 REC_INTENT_MAP)
    """
    return REC_INTENT_MAP.get(target_category, "Placement")  # 默认 Placement


def generate_prompts(furniture_list):
    """生成阶梯式诊断 (Step-by-Step Diagnostic) 的 Prompt"""
    system_prompt = f"""You are an embodied AI assistant analyzing a human-robot collaboration video from a quadruped robot's egocentric perspective.
Due to the low-angle view, the human's torso might cause severe physical occlusion.

**Task:**
Perform a step-by-step diagnostic reasoning to anticipate the human's final intent.
Output a SINGLE JSON object exactly matching this structure, answering the 6 progressive questions:

{{
  "Q1_target_area": "String. Choose ONE functional area from this list that the human is approaching: {json.dumps(AREAS)}.",
  "Q2_is_holding_object": "Boolean. True if the human appears to be holding an object, False otherwise.",
  "Q3_held_object_category": "String. If Q2 is True, identify the object. If severely obscured by the torso, output 'Unknown'. If not, choose from: {json.dumps(OBJECTS)}.",
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
        print(f"Error: Index file not found at {INPUT_FILE}")
        return

    print(f"Loading index from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        clips = json.load(f)

    benchmark_dataset = []
    skipped_count = 0
    metadat_file_path = "test_dataset_fetch_robot_08/metadata"

    print(f"Processing {len(clips)} clips...")

    for clip in tqdm(clips):
        # 1. 基础校验
        if 'gt_target' not in clip:
            skipped_count += 1
            continue

        # 2. GT 处理
        episode_id = clip["episode_id"]
        subtask = clip['subtask']
        metadata = os.path.join(metadat_file_path, f"{episode_id}.json")
        with open(metadata, 'rt', encoding='utf-8') as f:
            data = json.load(f)
            subtasks = data["subtasks"]
            steps = data["steps"]
            for gt_subtask in subtasks:
                if subtask == gt_subtask["description_processed"]:
                    t_end = int(gt_subtask["end_step"])
                    if t_end < len(steps):
                        end_step = steps[t_end]
                    else:
                        t_end = t_end - 1
                        end_step = steps[t_end]
                    furnitures = end_step["furnitures"]
                    distances = {}
                    for furniture in furnitures:
                        if furniture["name"].startswith("floor"):
                            distances[furniture["name"]] = furniture["distance"]
                    world_objects = end_step["world_objects"]
                    object_name = subtask.split()[1]
                    if "room" in world_objects[object_name].keys():
                        object_area = world_objects[object_name]["room"].rsplit("_", 1)[0]
                        gt_area = object_area
                    else:
                        min_area = min(distances, key=distances.get)
                        gt_area = min_area.replace("floor_", "", 1).rsplit("_", 1)[0]

        raw_target = clip['gt_target']
        target_category = match_target_to_context(raw_target)
        gt_furniture = target_category

        object_raw = subtask.split()[1]
        gt_object = object_raw.rsplit("_", 1)[0]

        # 自动推断 Intent (如果元数据里没有或为 Unknown)
        gt_intent = clip.get('gt_intent', 'Unknown')
        if gt_intent == 'Unknown' or not gt_intent:
            if subtask.startswith("Clean") or subtask.startswith("Fill") or subtask.startswith("Pour"):
                gt_intent = "Maintenance"
            gt_intent = infer_intent(target_category)
        gt_triplet = [gt_intent, gt_object, gt_furniture]
        # 3. 构造考题
        sys_prompt, usr_prompt = generate_prompts(FURNITURES)

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
                "ego_ann": clip.get('ego_ann_path'),
                "third_ann": clip.get('third_ann_path'),
                "global_ann": clip.get('global_ann_path')
            },

            # 考题内容
            "system_prompt": sys_prompt,
            "user_prompt": usr_prompt,
            # "candidate_list": GLOBAL_OBJECT_CONTEXT,

            # Ground Truth (用于自动判分)
            "ground_truth": {
                "target_area": gt_area,
                "is_holding_object": True,
                "held_object_category": gt_object,
                "target_furniture": gt_furniture,
                "furniture_affordance": gt_intent,
                "final_intent_triplet": gt_triplet
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