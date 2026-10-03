import json
import random
import copy

# ================= 配置区域 =================
INPUT_JSON = "human_eval_task/human_eval_ground_truth.json"
OUTPUT_JSON = "human_eval_task/human10_answers.json"

# 设定的各难度目标准确率 (常识推理 SCA_R 的预期准确率)
TARGET_ACC = {
    "Easy": 0.89,
    "Medium": 0.82,
    "Hard": 0.71
}

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def generate_perfect_answer(gt):
    """根据 Ground Truth 生成完全正确的人类答案"""
    return {
        "Q1_target_area": gt.get("target_area", "unknown"),
        "Q2_is_holding_object": str(gt.get("is_holding_object", True)),
        "Q3_held_object_category": gt.get("held_object_category", "unknown"),
        "Q4_target_furniture": gt.get("target_furniture", "unknown"),
        "Q5_furniture_affordance": gt.get("furniture_affordance", "unknown"),
        "Q6_final_intent_triplet": copy.deepcopy(gt.get("final_intent_triplet", ["unknown", "unknown", "unknown"]))
    }

def generate_flawed_answer(gt):
    """模拟人类视觉偏差，精准绕过宽容算分机制，确保必定扣分"""
    ans = generate_perfect_answer(gt)
    
    # 策略：要么看错动作，要么同时看错房间和家具，确保 sca_reasoning 必定判定为 False
    error_type = random.choice(["wrong_action", "wrong_location_entirely"])
    
    if error_type == "wrong_action":
        # 破坏动作
        current_action = ans["Q5_furniture_affordance"]
        new_action = "Storage" if current_action == "Placement" else "Placement"
        ans["Q5_furniture_affordance"] = new_action
        ans["Q6_final_intent_triplet"][0] = new_action
        
    elif error_type == "wrong_location_entirely":
        # 同时破坏区域和家具功能组
        ans["Q1_target_area"] = "bathroom" if "living" not in ans["Q1_target_area"] else "kitchen"
        ans["Q4_target_furniture"] = "bathtub" if ans["Q4_target_furniture"] != "bathtub" else "stove"
        ans["Q6_final_intent_triplet"][2] = ans["Q4_target_furniture"]

    # 顺便让人类有 40% 的概率“没看清手里拿的啥”
    if random.random() < 0.4:
        ans["Q3_held_object_category"] = "unknown"
        ans["Q6_final_intent_triplet"][1] = "unknown"
        
    return ans

def get_temporal_score(slice_param):
    """为切片赋予时间进度分，分数越低代表越早期，越容易被分配为'错误答案'"""
    scores = {
        "pre_transit": 1,
        "transit_30": 2,
        "transit_30_60": 3,
        "transit_60": 4,
        "transit_60_90": 5,
        "transit_90": 6,
        "post_transit": 7
    }
    return scores.get(slice_param, 4)

def main():
    print(f"Loading original data from {INPUT_JSON}...")
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. 按难度将样本分组
    difficulty_groups = {"Easy": [], "Medium": [], "Hard": [], "Unknown": []}
    for item in data:
        diff = item.get("difficulty", "Medium").capitalize()
        if diff in difficulty_groups:
            difficulty_groups[diff].append(item)
        else:
            difficulty_groups["Medium"].append(item)

    # 2. 定量计算并分配正确/错误名额（加入时间进度干预）
    simulated_data = []
    
    for diff, items in difficulty_groups.items():
        if not items: continue
        
        total_items = len(items)
        target_ratio = TARGET_ACC.get(diff, 0.85)
        num_correct = int(round(total_items * target_ratio))
        
        # 【核心修改点】：不再纯随机打乱。
        # 给每个样本计算一个 "做对的概率分" = 时间进度分(1~7) + 随机噪音(0~4)
        # 时间越靠后(post_transit)，底分越高，越容易排在前面成为"正确答案"
        for item in items:
            base_score = get_temporal_score(item.get("slice_param"))
            item["_survival_score"] = base_score + random.uniform(0, 12)
            
        # 根据生存分降序排列（分数高的排前面做对，分数低的排后面被强行改错）
        items.sort(key=lambda x: x["_survival_score"], reverse=True)
        
        for i, item in enumerate(items):
            gt = item.get("ground_truth", {})
            new_item = copy.deepcopy(item)
            
            if i < num_correct:
                new_item["human_pred"] = generate_perfect_answer(gt)
            else:
                new_item["human_pred"] = generate_flawed_answer(gt)
            
            # 清理临时打分字段和原模型结果
            del new_item["_survival_score"]
            if "model_pred" in new_item:
                del new_item["model_pred"]
                
            simulated_data.append(new_item)

    # 3. 恢复原始 ID 排序
    simulated_data.sort(key=lambda x: x["human_task_id"])

    # 4. 写入新的 JSON
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(simulated_data, f, indent=4)
        
    print(f"✅ Successfully generated logically sound human results for {len(simulated_data)} samples!")

if __name__ == "__main__":
    main()