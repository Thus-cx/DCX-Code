import json
import random
import os
import re

from wandb.apis.importers.internals.internal import ROOT_DIR

# ================= 配置 =================
INPUT_ROOT_DIR = "test_my_videos/final_benchmark_clips_2"
INPUT_FILE = os.path.join(INPUT_ROOT_DIR, "processed_clips_index.json")
OUTPUT_FILE = os.path.join(INPUT_ROOT_DIR, "final_test_benchmark_v13_third.json")

# ================= 1. 语义常识库 (基于 semantic_constants.py) =================

# 家具意图映射表 (Furniture Affordance)
# 逻辑：根据家具的功能定义意图
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

    # --- Maintenance (清洁/维护/水源) ---
    "sink": "Maintenance (cleaning/filling)",
    "bathtub": "Maintenance (cleaning/filling)",
    "shower": "Maintenance (cleaning/filling)",
    "dishwasher": "Maintenance (cleaning/filling)",
    "washer_dryer": "Maintenance (cleaning/filling)",

    # --- Disposal (丢弃) ---
    "trashcan": "Disposal (throwing away)",
    "toilet": "Disposal (throwing away)",

    # --- Heating (作为放置台面处理，不强行叫Cooking) ---
    "stove": "Placement (staging for use)",
    "oven": "Storage (putting away)",  # 烤箱不开火时常作为储物空间
    "microwave": "Storage (putting away)"
}

OBJECT_CATEGORY_MAP = {
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

# 物体类别 (辅助判断，用于增强 Prompt，可选)
OBJ_TYPE_MAP = {
    "food": ["apple", "bread", "egg", "potato", "tomato", "candy_bar", "fruit", "vegetable"],
    "kitchenware": ["kettle", "pan", "pot", "bowl", "plate", "cup", "mug", "glass", "bottle", "jug", "pitcher", "spoon",
                    "fork", "knife", "spatula", "ladle"],
    "tool": ["hammer", "screwdriver", "c-clamp", "tape", "scissors", "sponge", "soap"],
    "media": ["laptop", "mouse", "keyboard", "monitor", "phone", "book", "remote"],
    "comfort": ["cushion", "pillow", "toy", "doll", "plant", "vase"]
}

CONFUSION_GROUPS =[
    {"table", "counter", "desk", "stand", "coffe_table"},
    {"cabinet", "chest_of_drawer", "drawer", "wardrobe", "shelf", "cupboard", "filing_cabinet", "table", "counter"},
    {"sofa", "bed", "chair", "couch", "bench", "stool"},
    {"sink", "cabinet", "counter", "table", "shower", "bathtub"},
    {"fridge", "cabinet"},
    {"microwave", "oven", "cabinet"},
]


# 类别合理去处 (用于 Context-Aware Distractors)
PLAUSIBLE_TARGETS = {
    "kitchenware": ["table", "counter", "sink", "cabinet", "fridge", "dishwasher", "stove", "trashcan"],
    "food": ["table", "counter", "fridge", "cabinet", "microwave", "sink", "trashcan"],
    "clothing": ["wardrobe", "chest_of_drawers", "bed", "sofa", "washer_dryer", "cabinet", "floor"],
    "toys": ["chest_of_drawers", "shelves", "floor", "bed", "table", "box"],
    "electronics": ["table", "desk", "shelf", "cabinet", "counter", "bed", "sofa"],
    "stationery": ["desk", "table", "shelf", "cabinet", "drawer"],
    "household_items": ["table", "counter", "shelf", "cabinet", "sink", "toilet"],
    "tools": ["cabinet", "drawer", "table", "shelf"],
    "miscellaneous": ["table", "shelf", "cabinet", "drawer", "counter"],
    "furniture": ["sofa", "bed", "floor"],
    "sports_equipment": ["floor", "wardrobe", "shelf"]
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
    # 典型格式: Rearrange X to Y, Clean X in Y
    prep_indices = [i for i, w in enumerate(words) if w in ["to", "in", "on", "at"]]

    if prep_indices:
        split_idx = prep_indices[0]
        obj = " ".join(words[1:split_idx])
        target = " ".join(words[split_idx + 1:])
    else:
        # Fallback
        obj = " ".join(words[1:])
        target = "unknown"

    return verb, obj, target


def get_target_category(target_name):
    """
    模糊匹配家具类别
    """
    target_lower = target_name.lower()

    # 优先精确匹配
    for k, v in REC_INTENT_MAP.items():
        if k == target_lower:
            return v

    # 其次模糊匹配 (e.g., "kitchen_cabinet" 包含 "cabinet")
    # 优先匹配长词 (avoid 'table' matching 'vegetable' - though unlikely in furniture list)
    sorted_keys = sorted(REC_INTENT_MAP.keys(), key=len, reverse=True)
    for k in sorted_keys:
        if k in target_lower:
            return REC_INTENT_MAP[k]

    # 默认兜底
    return "Placement (staging for use)"


def get_pragmatic_intent(verb, target_name):
    """
    核心推断逻辑：根据动词和目的地推断意图
    """
    # 1. 强动词覆盖 (Clean/Fill/Pour 必然是 Maintenance)
    if verb in ["Clean", "Fill", "Pour", "Wash"]:
        return "Maintenance (cleaning or filling the object)"

    # 2. 弱动词 (Rearrange/Place/Pick) 依赖目的地
    return get_target_category(target_name)


def get_obj_category(obj_name):
    # 处理复合词 e.g., "blue_bottle" -> "bottle"
    # 或者直接查表
    obj_clean = obj_name.lower().replace(" ", "_")

    # 1. 完整匹配
    if obj_clean in OBJECT_CATEGORY_MAP:
        return OBJECT_CATEGORY_MAP[obj_clean]

    # 2. 后缀匹配 (e.g. "toy_airplane" -> "toys" via "airplane"? No, direct map has toy_airplane)
    # 尝试分割匹配
    parts = obj_clean.split('_')
    for p in reversed(parts):
        if p in OBJECT_CATEGORY_MAP:
            return OBJECT_CATEGORY_MAP[p]

    # 3. 关键字匹配
    for k, v in OBJECT_CATEGORY_MAP.items():
        if k in obj_clean:
            return v

    return "miscellaneous"


def get_confusing_targets(real_target):
    target_lower = real_target.lower()
    candidates = set()

    for group in CONFUSION_GROUPS:
        # 检查 target 是否属于某个混淆组
        # 使用模糊匹配: if 'table' in 'coffee_table'
        is_in_group = False
        for member in group:
            if member in target_lower:
                is_in_group = True
                break

        if is_in_group:
            # 加入组内其他成员
            for member in group:
                if member not in target_lower:
                    candidates.add(member)

    return list(candidates)

def generate_hardcore_options(gt_verb, gt_target, gt_obj):
    """
    终极硬核模式：所有选项物体相同，只变地点和意图。
    """
    real_intent = get_pragmatic_intent(gt_verb, gt_target)
    gt_option_str = f"Holding **{gt_obj}**, interacting or intend to interact with **{gt_target}** => **{real_intent}**"

    options = [gt_option_str]
    # 获取物体类别
    obj_cat = get_obj_category(gt_obj)

    # --- 策略 1: 混淆目标 (Confusing Targets) ---
    # 优先级最高。生成视觉相似但目标名不同的选项。
    # 意图可能相同 (Placement vs Placement) 也可能不同 (Table vs Sink -> Placement vs Maintenance)

    confusing_targets = get_confusing_targets(gt_target)
    random.shuffle(confusing_targets)

    for ct in confusing_targets:
        if len(options) >= 4: break

        # 获取该混淆目标的意图
        ct_intent = get_target_category(ct)

        # 构造选项
        # 必须确保 target 文字不同 (Visual Distinction)
        cand = f"Holding **{gt_obj}**, interacting or intend to interact with **{ct}** => **{ct_intent}**"

        if cand not in options:
            options.append(cand)

    # --- 策略 2: 类别合理去处 (Plausible Context Targets) ---
    # 如果混淆组不够，使用该物体类别常去的地方
    # 这测试逻辑推理和更广泛的视觉识别

    plausible = PLAUSIBLE_TARGETS.get(obj_cat, PLAUSIBLE_TARGETS["miscellaneous"])[:]
    random.shuffle(plausible)

    for pt in plausible:
        if len(options) >= 4: break
        if pt in gt_target.lower(): continue  # 跳过 GT

        pt_intent = get_target_category(pt)
        cand = f"Holding **{gt_obj}**, interacting or intend to interact with **{pt}** => **{pt_intent}**"

        # 避免重复
        is_duplicate = False
        for opt in options:
            # 简单查重
            if f"**{pt}**" in opt:
                is_duplicate = True
                break

        if not is_duplicate and cand not in options:
            options.append(cand)

    # --- 策略 3: 兜底 (Fallbacks) ---
    while len(options) < 4:
        # 随机选通用家具
        fallbacks = ["table", "shelf", "trashcan", "cabinet"]
        random.shuffle(fallbacks)
        for fb in fallbacks:
            if fb not in gt_target.lower():
                fb_intent = get_target_category(fb)
                cand = f"Holding **{gt_obj}**, interacting or intend to interact with **{fb}** => **{fb_intent}**"
                if cand not in options:
                    options.append(cand)
                    break
        # 实在不行就 break，不要死循环
        if len(options) < 4:
            # 强行加个 Handover 或者其他的
            cand = f"Holding **{gt_obj}**, interacting or intend to interact with **another person** => **Handover**"
            if cand not in options:
                options.append(cand)
            else:
                break

    random.shuffle(options)
    idx = options.index(gt_option_str)
    return {"A": options[0], "B": options[1], "C": options[2], "D": options[3]}, chr(65 + idx), gt_option_str

# ================= 4. 主流程 =================


def main():
    print(f"Loading data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        clips = json.load(f)
        
    benchmark_data = []
    
    for clip in clips:
        if clip['duration'] < 1.0: continue
        
        # Parse GT
        verb, obj, target = parse_task_info(clip['gt_task'])
        
        # Generate Options
        options, answer, gt_desc = generate_hardcore_options(verb, target, obj)
        
        # Bucket Logic
        metrics = clip['metrics']
        vl = metrics.get('view_loss_ratio', 0.0)
        pr = metrics.get('proximity_ratio', 0.0)
        sr = metrics.get('shake_ratio', 0.0)
        
        # Strict thresholds for Hard bucket
        if vl > 0.20 or pr > 0.20 or sr > 0.30: 
            bucket = "Bucket_C_Hard"
        elif vl > 0.10 or pr > 0.10 or sr > 0.15: 
            bucket = "Bucket_B_Medium"
        else: 
            bucket = "Bucket_A_Easy"

        # Prompt
        question = (
            "The robot is observing a human performing a task.\n"
            "Task: Analyze the video to identify the **Object**, the **Target Furniture**, and the **High-level Intent**.\n"
            "Select the option that correctly matches all three elements based on your visual observation.\n"
            # "Note: The object is constant in all options, so focus on identifying the **Target Furniture**."
        )

        benchmark_data.append({
            "video_path": clip.get('ego_raw_video_path'),
            "bucket": bucket,
            "phase": clip['phase'],
            "question": question,
            "options": options,
            "correct_option": answer,
            "gt_content": gt_desc,
            "raw_gt": clip['gt_task'],
            "metrics": clip['metrics']
        })
        
    print(f"Generated {len(benchmark_data)} Context-Aware Hardcore questions.")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(benchmark_data, f, indent=2)

if __name__ == "__main__":
    main()