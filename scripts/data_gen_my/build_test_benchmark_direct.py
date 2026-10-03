import json
import os
import random
import re
from tqdm import tqdm

try:
    from .clip_metadata_reference import resolve_metadata_path
except ImportError:  # Support direct execution from scripts/data_gen_my
    from clip_metadata_reference import resolve_metadata_path

# ================= 配置区域 =================
# 输入：上一步生成的索引文件
INPUT_FILE = "test_dataset/final_benchmark_0302/benchmark_index.json"
# 输出：最终考卷 (修改为 direct 版本)
OUTPUT_FILE = "test_dataset/final_benchmark_0302/final_exam_direct.json"

# ================= 1. 语义常识库 =================

# 家具意图映射表 (用于自动推断 GT Intent)
REC_INTENT_MAP = {
    # --- Storage (收纳/储藏) ---
    "cabinet": "Storage", "fridge": "Storage", "refrigerator": "Storage",
    "shelf": "Storage", "shelves": "Storage", "wardrobe": "Storage",
    "chest_of_drawers": "Storage", "filing_cabinet": "Storage",
    "breadbin": "Storage", "box": "Storage", "canister": "Storage",
    "basket": "Storage", "dresser": "Storage", "cupboard": "Storage",
    "drawer": "Storage",

    # --- Placement (放置) ---
    "armchair": "Placement", "bar_stool": "Placement", "dining_table": "Placement",
    "ottoman": "Placement", "vanity": "Placement", "towel_rack": "Placement",
    "table": "Placement", "counter": "Placement", "desk": "Placement",
    "stand": "Placement", "nightstand": "Placement", "coffee_table": "Placement",
    "stool": "Placement", "bench": "Placement", "chair": "Placement",
    "sofa": "Placement", "couch": "Placement", "bed": "Placement",
    "floor": "Placement", "tv_stand": "Placement", "rack": "Placement",
    "tray": "Placement",

    # --- Maintenance (清洁/维护) ---
    "sink": "Maintenance", "bathtub": "Maintenance", "shower": "Maintenance",
    "dishwasher": "Maintenance", "washer_dryer": "Maintenance",
    "washer": "Maintenance", "dryer": "Maintenance",

    # --- Disposal (丢弃) ---
    "trashcan": "Disposal", "toilet": "Disposal", "bin": "Disposal", "garbage": "Disposal",

    # --- Heating (特殊处理) ---
    "stove": "Placement", "oven": "Storage", "microwave": "Storage"
}

object_category_map = {
    "action_figure": "toys", "android_figure": "toys", "apple": "kitchenware",
    "backpack": "clothing", "baseballbat": "sports_equipment", "basket": "household_items",
    "basketball": "sports_equipment", "bath_towel": "household_items", "battery_charger": "electronics",
    "board_game": "toys", "book": "stationery", "bottle": "kitchenware",
    "bowl": "kitchenware", "box": "miscellaneous", "bread": "kitchenware",
    "bundt_pan": "kitchenware", "butter_dish": "kitchenware", "c-clamp": "tools",
    "cake_pan": "kitchenware", "can": "kitchenware", "can_opener": "kitchenware",
    "candle": "household_items", "candle_holder": "household_items", "candy_bar": "food",
    "canister": "kitchenware", "carrying_case": "miscellaneous", "casserole": "kitchenware",
    "cellphone": "electronics", "clock": "household_items", "credit_card": "stationery",
    "cup": "kitchenware", "cushion": "furniture", "doll": "toys",
    "dumbbell": "sports_equipment", "egg": "food", "electric_kettle": "kitchenware",
    "electronic_cable": "electronics", "file_sorter": "stationery", "folder": "stationery",
    "fork": "kitchenware", "gaming_console": "electronics", "glass": "kitchenware",
    "hammer": "tools", "hand_towel": "household_items", "handbag": "clothing",
    "hard_drive": "electronics", "hat": "clothing", "helmet": "sports_equipment",
    "jug": "kitchenware", "kettle": "kitchenware", "keychain": "miscellaneous",
    "knife": "kitchenware", "ladle": "kitchenware", "lamp": "household_items",
    "laptop": "electronics", "laptop_cover": "electronics", "laptop_stand": "electronics",
    "lunch_box": "kitchenware", "milk_frother_cup": "kitchenware", "monitor_stand": "electronics",
    "mouse_pad": "electronics", "multiport_hub": "electronics", "pan": "kitchenware",
    "pen": "stationery", "pencil_case": "stationery", "phone_stand": "electronics",
    "picture_frame": "household_items", "pitcher": "kitchenware", "plant_container": "household_items",
    "plant_saucer": "household_items", "plate": "kitchenware", "potato": "food",
    "ramekin": "kitchenware", "scissors": "stationery", "screwdriver": "tools",
    "shoe": "clothing", "soap_dish": "household_items", "soap_dispenser": "household_items",
    "spatula": "kitchenware", "spectacles": "stationery", "spicemill": "kitchenware",
    "sponge": "household_items", "spoon": "kitchenware", "spray_bottle": "household_items",
    "squeezer": "kitchenware", "statue": "household_items", "stuffed toy": "toys",
    "stuffed_toy": "toys", "sushi_mat": "kitchenware", "tape": "stationery",
    "teapot": "kitchenware", "tennis_racquet": "sports_equipment", "tomato": "food",
    "toy_airplane": "toys", "toy_animal": "toys", "toy_bee": "toys",
    "toy_cactus": "toys", "toy_construction_set": "toys", "toy_fire_truck": "toys",
    "toy_food": "toys", "toy_fruits": "toys", "toy_pineapple": "toys",
    "toy_swing": "toys", "toy_vehicle": "toys", "tray": "kitchenware",
    "vase": "household_items", "watch": "accessories",
}

OBJECTS = list(object_category_map.keys()) + ["Unknown"]

INTENT_CATEGORIES = [
    "Storage", "Placement", "Maintenance", "Disposal", "Unknown"
]

AREAS = ['Unknown', 'bedroom', 'hallway', 'kitchen', 'living_room',
         'bathroom', 'dining_room', 'garage', 'laundryroom',
         'entryway', 'closet', 'office', 'other_room', 'toilet',
         'outdoor', 'porch', 'workout', 'tv', 'lounge', 'utilityroom']

FURNITURES = list(REC_INTENT_MAP.keys()) + ["Unknown"]


# ================= 2. 核心逻辑函数 =================

def sanitize_target_name(text):
    if not text: return "unknown"
    text = re.sub(r'_\d+', '', text)
    text = text.lower()
    return text.strip()


def match_target_to_context(raw_target):
    clean_name = sanitize_target_name(raw_target)
    if clean_name in FURNITURES:
        return clean_name

    best_match = None
    max_len = 0
    for cand in FURNITURES:
        if cand in clean_name or clean_name in cand:
            if len(cand) > max_len:
                max_len = len(cand)
                best_match = cand

    return best_match if best_match else "Unknown"


def infer_intent(target_category):
    return REC_INTENT_MAP.get(target_category, "Placement")


# ================= 修改点：专为 Direct 推断设计的 Prompt =================
def generate_prompts_direct():
    """生成直接意图推断 (Direct Baseline) 的 Prompt"""
    system_prompt = f"""You are an embodied AI assistant analyzing a human-robot collaboration video from a quadruped robot's egocentric perspective.
Due to the low-angle view, the human's torso might cause severe physical occlusion.

**Task:**
Directly anticipate the human's final intent without intermediate reasoning. 
Output a SINGLE JSON object exactly matching this structure:

{{
  "final_intent_triplet": ["Action", "Object", "Target_Furniture"]
}}

**Notes:**
- For Action, choose from: {json.dumps(INTENT_CATEGORIES)}.
- For Object, choose from: {json.dumps(OBJECTS)}. If the object is severely obscured by the torso, output 'Unknown'.
- For Target_Furniture, choose from: {json.dumps(FURNITURES)}.
- Output ONLY valid JSON, do not include any other text or markdown.
"""
    user_prompt = "Analyze the video slice and directly output the final intent triplet in JSON format."
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
    metadat_file_path = "test_dataset/metadata"

    print(f"Processing {len(clips)} clips for DIRECT inference...")

    for clip in tqdm(clips):
        if 'gt_target' not in clip:
            skipped_count += 1
            continue

        episode_id = clip["episode_id"]
        subtask = clip['subtask']
        metadata_ref = clip.get("metadata_ref")
        if metadata_ref:
            metadata = resolve_metadata_path(
                metadata_ref,
                metadata_root=metadat_file_path,
            )
        else:
            # Backward compatibility with benchmark indexes generated before v1.
            metadata = os.path.join(metadat_file_path, f"{episode_id}.json")

        gt_area = "Unknown"  # 默认值，防崩溃
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

                    try:
                        object_name = subtask.split()[1]
                        if "room" in world_objects.get(object_name, {}):
                            object_area = world_objects[object_name]["room"].rsplit("_", 1)[0]
                            gt_area = object_area
                        elif distances:
                            min_area = min(distances, key=distances.get)
                            gt_area = min_area.replace("floor_", "", 1).rsplit("_", 1)[0]
                    except IndexError:
                        pass  # 处理 subtask 文本异常的情况

        raw_target = clip['gt_target']
        target_category = match_target_to_context(raw_target)
        gt_furniture = target_category

        try:
            object_raw = subtask.split()[1]
            gt_object = object_raw.rsplit("_", 1)[0]
        except IndexError:
            gt_object = "Unknown"

        gt_intent = clip.get('gt_intent', 'Unknown')
        if gt_intent == 'Unknown' or not gt_intent:
            if subtask.startswith("Clean") or subtask.startswith("Fill") or subtask.startswith("Pour"):
                gt_intent = "Maintenance"
            gt_intent = infer_intent(target_category)

        gt_triplet = [gt_intent, gt_object, gt_furniture]

        # === 获取 Direct 版本的 Prompt ===
        sys_prompt, usr_prompt = generate_prompts_direct()

        entry = {
            "question_id": f"{clip['episode_id']}_t{clip['task_index']}_{clip['observation_type']}_{clip['slice_param']}_direct",
            "episode_id": clip['episode_id'],
            "task_index": clip['task_index'],
            "instruction": clip['instruction'],
            "observation_type": clip['observation_type'],
            "slice_param": clip['slice_param'],
            "difficulty": clip.get('difficulty', 'Unknown'),
            "duration": clip.get('duration', 0.0),
            "metadata_ref": metadata_ref,

            "video_paths": {
                "ego_raw": clip.get('ego_raw_path'),
                "third_raw": clip.get('third_raw_path'),
                "global": clip.get('global_path'),
                "ego_ann": clip.get('ego_ann_path'),
                "third_ann": clip.get('third_ann_path'),
                "global_ann": clip.get('global_ann_path')
            },

            "system_prompt": sys_prompt,
            "user_prompt": usr_prompt,

            # 保留完整的 Ground Truth 结构，方便统一脚本读取
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

    print(f"Generated {len(benchmark_dataset)} direct exam questions. (Skipped {skipped_count} invalid clips)")
    os.makedirs(os.path.dirname(OUTPUT_FILE) if os.path.dirname(OUTPUT_FILE) else '.', exist_ok=True)

    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(benchmark_dataset, f, indent=2)

    print("Done! Direct Benchmark building complete.")


if __name__ == "__main__":
    main()
