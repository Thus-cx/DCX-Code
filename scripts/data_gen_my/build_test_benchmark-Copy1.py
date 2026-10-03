import json
import random
import os
import re

from wandb.apis.importers.internals.internal import ROOT_DIR

# ================= 配置 =================
INPUT_ROOT_DIR = "test_my_videos/final_benchmark_clips_2"
INPUT_FILE = os.path.join(INPUT_ROOT_DIR, "processed_clips_index.json")
OUTPUT_FILE = os.path.join(INPUT_ROOT_DIR, "final_test_benchmark_three_composite.json")

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

# 物体类别 (辅助判断，用于增强 Prompt，可选)
OBJ_TYPE_MAP = {
    "food": ["apple", "bread", "egg", "potato", "tomato", "candy_bar", "fruit", "vegetable"],
    "kitchenware": ["kettle", "pan", "pot", "bowl", "plate", "cup", "mug", "glass", "bottle", "jug", "pitcher", "spoon",
                    "fork", "knife", "spatula", "ladle"],
    "tool": ["hammer", "screwdriver", "c-clamp", "tape", "scissors", "sponge", "soap"],
    "media": ["laptop", "mouse", "keyboard", "monitor", "phone", "book", "remote"],
    "comfort": ["cushion", "pillow", "toy", "doll", "plant", "vase"]
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


def generate_adversarial_options(gt_verb, gt_target, obj_name):
    """
    生成对抗性选项：针对特定物体，强制包含容易混淆的意图。
    """
    gt_intent = get_pragmatic_intent(gt_verb, gt_target)
    
    # 1. 定义针对特定物体的“诱导性干扰项” (Lures)
    # 这些是模型仅凭物体本身（不看环境）最容易猜的意图
    lures = []
    obj_lower = obj_name.lower()
    
    if any(x in obj_lower for x in ['kettle', 'pan', 'pot', 'knife', 'spoon', 'plate', 'bowl', 'mug']):
        # 厨具/餐具：最容易混淆的是 "清洗" vs "放置" vs "储藏"
        lures = [
            "Maintenance (cleaning or filling)", 
            "Placement (staging for use)", 
            "Storage (putting away)"
        ]
    elif any(x in obj_lower for x in ['apple', 'bread', 'fruit', 'food']):
        # 食物：最容易混淆的是 "储藏" vs "放置(吃)" vs "清洗"
        lures = [
            "Storage (putting away)", 
            "Placement (staging for use)", 
            "Maintenance (washing food)"
        ]
    elif any(x in obj_lower for x in ['book', 'laptop', 'phone']):
        # 电子产品/书：最容易混淆 "放置(使用)" vs "储藏"
        lures = [
            "Placement (staging for use)", 
            "Storage (putting away)"
        ]
    
    # 2. 构建选项池
    options = [gt_intent]
    
    # 将诱导项加入（排除掉 GT 本身）
    for lure in lures:
        # 只取 intent 的前缀进行比对 (Storage vs Storage...)
        if lure.split('(')[0] != gt_intent.split('(')[0]:
            if lure not in options:
                options.append(lure)
                
    # 3. 如果还没凑够 4 个，再从通用池里补
    ALL_INTENTS = [
        "Storage (putting away)",
        "Placement (staging for use)",
        "Maintenance (cleaning or filling)",
        "Disposal (throwing away)"
    ]
    random.shuffle(ALL_INTENTS)
    for intent in ALL_INTENTS:
        if len(options) >= 4: break
        if intent.split('(')[0] != gt_intent.split('(')[0] and intent not in options:
            options.append(intent)
            
    random.shuffle(options)
    idx = options.index(gt_intent)
    return {"A": options[0], "B": options[1], "C": options[2], "D": options[3]}, chr(65+idx), gt_intent

def generate_composite_options(gt_verb, gt_target, gt_obj):
    """
    生成 [Object] + [Target] -> [Intent] 的复合选项
    """
    real_intent = get_pragmatic_intent(gt_verb, gt_target)
    
    # 构造 GT 选项字符串
    # 格式: Holding [Obj], moving to [Target] -> [Intent]
    gt_option_str = f"Holding **{gt_obj}**, interacting with **{gt_target}** => **{real_intent}**"
    
    options = [gt_option_str]
    
    # --- 生成干扰项策略 ---
    
    # 策略 1: 【物体错误】地点对，意图对，但物体看错了
    # 针对 Placement 任务，造一个拿错东西的 Placement
    fake_obj_list = ["apple", "book", "laptop", "cushion"] 
    fake_obj = random.choice([o for o in fake_obj_list if o not in gt_obj])
    opt_1 = f"Holding **{fake_obj}**, interacting or intend to interact with **{gt_target}** => **{real_intent}**"
    if opt_1 not in options: options.append(opt_1)
    
    # 策略 2: 【地点错误】物体对，地点错 -> 导致意图错 (最强的逻辑干扰)
    # 如果 GT 是 Table (Placement)，造一个 Sink (Maintenance)
    # 如果 GT 是 Sink (Maintenance)，造一个 Table (Placement)
    
    if "Placement" in real_intent:
        # 造一个 Maintenance 的假象
        fake_target = "sink"
        fake_intent = "Maintenance (cleaning or filling)"
    elif "Maintenance" in real_intent:
        # 造一个 Placement 的假象
        fake_target = "table"
        fake_intent = "Placement (staging for use)"
    elif "Storage" in real_intent:
        # 造一个 Placement 的假象
        fake_target = "counter"
        fake_intent = "Placement (staging for use)"
    else:
        fake_target = "cabinet"
        fake_intent = "Storage (putting away)"
        
    opt_2 = f"Holding **{gt_obj}**, interacting with **{fake_target}** => **{fake_intent}**"
    if opt_2 not in options: options.append(opt_2)
    
    # 策略 3: 【完全错误】物体对，地点对，但意图强行说错 (逻辑测试)
    # 比如：去 Table 说是 Storage (虽然少见但逻辑不通)
    # 或者造一个完全不相关的
    fake_intent_logic = "Disposal (throwing away)" if "Disposal" not in real_intent else "Placement"
    opt_3 = f"Holding **{gt_obj}**, interacting with **{gt_target}** => **{fake_intent_logic}**"
    
    # 如果凑不够4个，随机补
    if len(options) < 4:
        options.append(opt_3)
        
    # 打乱
    random.shuffle(options)
    
    # 找答案
    idx = options.index(gt_option_str)
    return {"A": options[0], "B": options[1], "C": options[2], "D": options[3]}, chr(65+idx), gt_option_str

def generate_hardcore_options(gt_verb, gt_target, gt_obj):
    """
    终极硬核模式：所有选项物体相同，只变地点和意图。
    """
    real_intent = get_pragmatic_intent(gt_verb, gt_target)
    gt_option_str = f"Holding **{gt_obj}**, interacting or intend to interact with **{gt_target}** => **{real_intent}**"
    
    options = [gt_option_str]
    
    # 1. 定义针对该物体的合理去处 (Plausible Destinations)
    # 必须是逻辑上成立的，不能瞎编
    plausible_targets = []
    
    # 厨具类
    if any(x in gt_obj for x in ['kettle', 'pan', 'pot', 'knife', 'spoon', 'plate', 'mug']):
        plausible_targets = [
            ("sink", "Maintenance (cleaning or filling)"),
            ("cabinet", "Storage (putting away)"),
            ("table", "Placement (staging for use)"),
            ("stove", "Placement (staging for use)") # 或者叫 Heating
        ]
    # 食物类
    elif any(x in gt_obj for x in ['apple', 'bread', 'fruit']):
        plausible_targets = [
            ("fridge", "Storage (putting away)"),
            ("table", "Placement (staging for use)"),
            ("sink", "Maintenance (washing food)"),
            ("trashcan", "Disposal (throwing away)")
        ]
    # 通用/杂物
    else:
        plausible_targets = [
            ("table", "Placement (staging for use)"),
            ("cabinet", "Storage (putting away)"),
            ("sofa", "Placement (staging for use)"),
            ("shelf", "Storage (putting away)")
        ]
        
    # 2. 生成干扰项
    for fake_target, fake_intent in plausible_targets:
        # 排除掉 GT 那个意图/地点
        # 只要意图大类不同，或者地点明显不同，就可以作为干扰项
        if fake_target not in gt_target and fake_intent.split('(')[0] != real_intent.split('(')[0]:
            opt = f"Holding **{gt_obj}**, interacting with **{fake_target}** => **{fake_intent}**"
            if opt not in options:
                options.append(opt)
                
    # 3. 如果还不够 4 个，强行补一个 "Handover" 或者其他通用合理的
    while len(options) < 4:
        fake_opt = f"Holding **{gt_obj}**, interacting with **another person** => **Handover**"
        if fake_opt not in options: options.append(fake_opt)
        else:
             # 再不够就补一个 Disposal
             fake_opt2 = f"Holding **{gt_obj}**, interacting with **trashcan** => **Disposal (throwing away)**"
             if fake_opt2 not in options: options.append(fake_opt2)
             else: break # 实在没有就算了

    random.shuffle(options)
    idx = options.index(gt_option_str)
    return {"A": options[0], "B": options[1], "C": options[2], "D": options[3]}, chr(65+idx), gt_option_str

# ================= 3. 主程序 =================

def main():
    print(f"Loading data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        clips = json.load(f)

    benchmark_data = []
    skipped_count = 0

    print(f"Processing {len(clips)} clips...")

    for clip in clips:
        # 1. 基础过滤
        if clip['duration'] < 1.0:
            skipped_count += 1
            continue

        # 2. 解析任务
        verb, obj, target = parse_task_info(clip['gt_task'])

        # 3. 生成考题
        # options, answer, gt_desc = generate_adversarial_options(verb, target, obj)
        options, answer, gt_desc = generate_hardcore_options(verb, target, obj)

        # 4. 难度分桶 (保持你的逻辑)
        metrics = clip['metrics']
        vl = metrics.get('view_loss_ratio', 0.0)
        pr = metrics.get('proximity_ratio', 0.0)
        sr = metrics.get('shake_ratio', 0.0)

        if vl > 0.15 or pr > 0.25 or sr > 0.20:
            bucket = "Bucket_C_Hard"
        elif vl > 0.05 or pr > 0.10 or sr > 0.10:
            bucket = "Bucket_B_Medium"
        else:
            bucket = "Bucket_A_Easy"

        # 5. 构建 Prompt
        # 这里不需要区分 Phase，因为无论是 Approach 还是 Interaction，
        # 核心都是预测/确认"这个物体要去哪里/在哪里干什么"

        # 增强版 Prompt：引导模型关注物体类型和家具功能
        question = (
            # "The robot is observing a human performing a task in an indoor environment.\n"
            # "Visual Evidence:\n"
            # "**Object** (Green Box): Is the human holding or manipulating any objects? What is the human holding or manipulating?\n"
            # "**Target Location**: What furniture is the human interacting with or walking towards?\n\n"
            # # "2. **Target Location** (Blue Box): What furniture is the human interacting with or walking towards?\n\n"
            # "Task: Infer the high-level **Goal Category** of this action.\n"
            # # "Reasoning Guide:\n"
            # # "- If the target is a container (cabinet/fridge), the goal is likely **Storage**.\n"
            # # "- If the target is a surface (table/counter), the goal is likely **Placement**.\n"
            # # "- If the target is a water source (sink), the goal is likely **Maintenance**.\n"
            # # "- If the target is a trashcan, the goal is likely **Disposal**.\n\n"
            # "Based on your visual observation, choose the correct category."
            "The robot is observing a human performing a task.\n"
            "Task: Analyze the video to identify the **Object**, the **Target Furniture**, and the **High-level Intent**.\n"
            "Select the option that correctly matches all three elements based on your visual observation."
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

    print(f"Generated {len(benchmark_data)} questions. (Skipped {skipped_count} short clips)")
    print(f"Saved to {OUTPUT_FILE}")

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(benchmark_data, f, indent=2)


if __name__ == "__main__":
    main()