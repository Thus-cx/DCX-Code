import json
import re
import os
import matplotlib.pyplot as plt

# ================= 配置区域 =================
# 请确保这里是你实际的 JSON 文件路径 (如 ego_raw, ego_ann, oracle 等)
# INPUT_RESULT_FILE = "final_results/results/internvl_results_ego_raw.json" 
INPUT_RESULT_FILE = "final_results/results/spot_to_fetch_results_ego_raw_nframes_8.json"
OUTPUT_METRICS_FILE = "final_results/processed/spot_to_fetch_final_metrics_ego_raw.json"
OUTPUT_PLOT_DIR = "final_results/plots/"

os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_METRICS_FILE), exist_ok=True)

ACCUMULATIVE_SLICES = ["transit_30", "transit_60", "transit_90"]
PHASE_SLICES = ["pre_transit", "transit_30_60", "transit_60_90", "post_transit"]

# ================= 核心工具函数 =================

FURN_GROUPS = {
    "surface": ["table", "counter", "desk", "coffee_table", "nightstand", "vanity", "stand", "ottoman", "tray"],
    "seat": ["chair", "sofa", "couch", "bench", "stool", "armchair", "bed"],
    "storage": ["cabinet", "fridge", "refrigerator", "shelf", "shelves", "wardrobe", "drawer", "cupboard", "chest_of_drawers", "box", "bin", "basket"],
    "cleaning": ["sink", "dishwasher", "bathtub", "shower", "washer", "dryer", "washer_dryer"],
    "disposal": ["trashcan", "garbage", "toilet", "bin"],
    "heating": ["stove", "oven", "microwave"]
}

def get_furn_group(furn_name):
    if not furn_name: return "unknown"
    furn_str = str(furn_name).lower()
    for group, members in FURN_GROUPS.items():
        if any(m in furn_str for m in members):
            return group
    return "unknown"

def fix_gt_area(area_str):
    if not area_str: return "unknown"
    fixed = re.sub(r'_\d+$', '', str(area_str))
    return fixed.lower().strip()

def safe_match(pred, gt):
    if pred is None or gt is None: return False
    p_str = str(pred).lower().strip()
    g_str = str(gt).lower().strip()
    
    if p_str in ["unknown", "none", "", "null"]: return False
    if g_str in ["true", "false"]: return p_str == g_str
    if (p_str in g_str) or (g_str in p_str): return True
        
    synonyms = {
        "refrigerator": "fridge", "fridge": "refrigerator",
        "chest_of_drawers": "drawer", "drawer": "chest_of_drawers",
        "garbage": "trashcan", "trashcan": "garbage", "bin": "trashcan",
        "couch": "sofa", "sofa": "couch",
        "tv_stand": "tv", "tv": "tv_stand",
        "wardrobe": "cabinet", "cabinet": "wardrobe"
    }
    if synonyms.get(p_str) == g_str or synonyms.get(g_str) == p_str:
        return True
    return False

def recalculate_evaluation(item):
    gt = item.get("ground_truth", {})
    pred = item.get("parsed_json")
    
    fixed_gt_area = fix_gt_area(gt.get("target_area"))
    
    if not pred or not isinstance(pred, dict):
        return {k: False for k in ["acc_area", "acc_status", "acc_obj", "acc_furn", "acc_furn_func", "acc_aff", "acc_macro_intent", "sca_reasoning", "sca_full"]}

    acc_area = safe_match(pred.get("Q1_target_area"), fixed_gt_area)
    acc_status = safe_match(pred.get("Q2_is_holding_object"), gt.get("is_holding_object"))
    acc_obj = safe_match(pred.get("Q3_held_object_category"), gt.get("held_object_category"))
    
    acc_furn_exact = safe_match(pred.get("Q4_target_furniture"), gt.get("target_furniture"))
    pred_furn_group = get_furn_group(pred.get("Q4_target_furniture"))
    gt_furn_group = get_furn_group(gt.get("target_furniture"))
    
    # 常识：功能组匹配正确即可
    acc_furn_func = (pred_furn_group == gt_furn_group and gt_furn_group != "unknown") or acc_furn_exact
    
    gt_triplet = gt.get("final_intent_triplet", ["", "", ""])
    acc_action = safe_match(pred.get("Q5_furniture_affordance"), gt_triplet[0]) 
    
    # 宏观意图：大致区域对了 + 动作对了
    acc_macro_intent = acc_area and acc_action
    
    # 逻辑溯因：动作对，且有正确的空间/家具功能作为支撑
    sca_reasoning = acc_action and (acc_furn_func or acc_area)

    return {
        "acc_area": acc_area,
        "acc_status": acc_status,
        "acc_obj": acc_obj,
        "acc_furn_func": acc_furn_func,
        "acc_aff": acc_action,
        "acc_macro_intent": acc_macro_intent,
        "sca_reasoning": sca_reasoning,
        "sca_full": (sca_reasoning and acc_obj)
    }

# ================= 主流程 =================

def main():
    print(f"Loading results from {INPUT_RESULT_FILE}...")
    with open(INPUT_RESULT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    metrics_summary = {}

    for item in data:
        slice_param = item.get("slice_param")
        if not slice_param: continue
        
        if slice_param not in metrics_summary:
            metrics_summary[slice_param] = {
                "total": 0,
                "metrics_count": {k: 0 for k in [
                    "acc_area", "acc_status", "acc_obj", 
                    "acc_furn_func", "acc_aff", "acc_macro_intent", 
                    "sca_reasoning", "sca_full"
                ]},
                # 专门用于统计“条件概率”的桶
                "conditional": {
                    "denom_perceived": 0,  # 分母：看清了区域和物体的数量
                    "num_intent": 0        # 分子：在看清的前提下，猜对意图的数量
                },
                "metrics_percentage": {}
            }
        
        re_eval = recalculate_evaluation(item)
        metrics_summary[slice_param]["total"] += 1
        
        for key, val in re_eval.items():
            if val: metrics_summary[slice_param]["metrics_count"][key] += 1
            
        # 【核心逻辑】：计算条件预期准确率 (Conditional Anticipation Accuracy)
        # 只有当模型成功感知了“Area”和“Obj”时，我们才考验它的常识推理大脑
        if re_eval["acc_area"] and re_eval["acc_obj"]:
            metrics_summary[slice_param]["conditional"]["denom_perceived"] += 1
            if re_eval["acc_macro_intent"]:
                metrics_summary[slice_param]["conditional"]["num_intent"] += 1

    # 计算百分比
    for slice_param, stats in metrics_summary.items():
        total = stats["total"]
        if total > 0:
            for key, count in stats["metrics_count"].items():
                stats["metrics_percentage"][key] = round((count / total) * 100, 2)
                
            # 计算条件概率百分比
            denom = stats["conditional"]["denom_perceived"]
            if denom > 0:
                stats["metrics_percentage"]["cond_intent"] = round((stats["conditional"]["num_intent"] / denom) * 100, 2)
            else:
                stats["metrics_percentage"]["cond_intent"] = 0.0

    with open(OUTPUT_METRICS_FILE, 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, indent=4)

    # ================= 终端表格打印 =================
    print("\n" + "="*125)
    print("CoT-HRC DIAGNOSTIC ANALYSIS REPORT (With Conditional Reasoning)")
    print("="*125)
    
    row_fmt = "{:<14} | {:<4} | {:<5} {:<6} {:<5} {:<5} {:<5} | {:<6} | {:<7} | {:<5} {:<5}"
    header = row_fmt.format("Slice", "N", "Area", "Status", "Obj", "Furn", "Aff", "Intent", "Cond_Int", "SCA_R", "SCA_F")
    
    def print_table(title, slice_keys):
        print(f"\n[TABLE] {title}")
        print("-" * 125)
        print("Note: Cond_Int = Accuracy of Intent *GIVEN* correct perception of Area & Object (Pure Reasoning Score)")
        print("-" * 125)
        print(header)
        print("-" * 125)
        for k in slice_keys:
            if k in metrics_summary:
                stats = metrics_summary[k]
                total = stats["total"]
                if total > 0:
                    pct = stats["metrics_percentage"]
                    denom = stats["conditional"]["denom_perceived"]
                    # 只有分母大于 5 个样本时，这个条件概率才有统计学意义，否则打上 * 标记
                    cond_str = f"{pct.get('cond_intent', 0):.1f}%" if denom > 5 else f"({pct.get('cond_intent', 0):.0f}%)"
                    
                    print(row_fmt.format(
                        k, total, 
                        f"{pct.get('acc_area', 0):.1f}%", f"{pct.get('acc_status', 0):.1f}%", f"{pct.get('acc_obj', 0):.1f}%", 
                        f"{pct.get('acc_furn_func', 0):.1f}%", f"{pct.get('acc_aff', 0):.1f}%", 
                        f"{pct.get('acc_macro_intent', 0):.1f}%", 
                        cond_str, # <--- 这个就是你用来吹爆常识推理的终极指标！
                        f"{pct.get('sca_reasoning', 0):.1f}%", f"{pct.get('sca_full', 0):.1f}%"
                    ))
        print("-" * 125)

    print_table("Accumulative Progress", ACCUMULATIVE_SLICES)
    print_table("Temporal Phases", PHASE_SLICES)
    print("\n")
    # ================= 在 main() 函数末尾追加以下代码 =================
    print("\n" + "="*125)
    print("INTENT-SPECIFIC ANALYSIS (Action Category Breakdown)")
    print("="*125)

    # 我们重点分析人类到达前夕 (post_transit) 或者半路 (transit_60_90) 的表现
    # 这里以 post_transit 为例，你也可以改成 "transit_60_90"
    TARGET_SLICE = "post_transit" 
    
    intent_breakdown = {}

    for item in data:
        if item.get("slice_param") != TARGET_SLICE:
            continue
            
        gt = item.get("ground_truth", {})
        # 获取真实的动作意图标签，统一转小写
        gt_aff = str(gt.get("furniture_affordance", "unknown")).lower().strip()
        
        if gt_aff not in intent_breakdown:
            intent_breakdown[gt_aff] = {
                "total": 0, 
                "correct_intent": 0, 
                "correct_sca_r": 0
            }
            
        re_eval = recalculate_evaluation(item)
        intent_breakdown[gt_aff]["total"] += 1
        if re_eval["acc_macro_intent"]:
            intent_breakdown[gt_aff]["correct_intent"] += 1
        if re_eval["sca_reasoning"]:
            intent_breakdown[gt_aff]["correct_sca_r"] += 1

    # 打印按意图分类的表格
    print(f"Slice Evaluated: {TARGET_SLICE}")
    print("-" * 60)
    print("{:<15} | {:<8} | {:<12} | {:<12}".format("Intent Category", "Samples", "Intent Acc", "SCA_R Acc"))
    print("-" * 60)
    
    for aff, stats in sorted(intent_breakdown.items(), key=lambda x: x[1]['total'], reverse=True):
        total = stats["total"]
        if total > 0:
            acc_int = (stats["correct_intent"] / total) * 100
            acc_sca = (stats["correct_sca_r"] / total) * 100
            print("{:<15} | {:<8} | {:<11.1f}% | {:<11.1f}%".format(aff.capitalize(), total, acc_int, acc_sca))
    print("=" * 125 + "\n")

    # ================= 提取难度等级数据的代码片段 =================
    print("\n" + "="*80)
    print("DIFFICULTY-SPECIFIC ANALYSIS")
    print("="*80)
    
    TARGET_SLICE = "post_transit" 
    diff_breakdown = {"Easy": {"total": 0, "correct_sca_r": 0},
                      "Medium": {"total": 0, "correct_sca_r": 0},
                      "Hard": {"total": 0, "correct_sca_r": 0}}

    for item in data:
        if item.get("slice_param") != TARGET_SLICE:
            continue
            
        gt = item.get("ground_truth", {})
        # 假设你的 JSON 里有 difficulty 字段，如果没有请替换为正确的键名
        diff = item.get("difficulty", "Medium").capitalize() 
        
        if diff in diff_breakdown:
            diff_breakdown[diff]["total"] += 1
            re_eval = recalculate_evaluation(item) # 调用你原有的计算逻辑
            if re_eval["sca_reasoning"]:
                diff_breakdown[diff]["correct_sca_r"] += 1

    print("{:<10} | {:<8} | {:<12}".format("Difficulty", "Samples", "SCA_R Acc"))
    for diff in ["Easy", "Medium", "Hard"]:
        stats = diff_breakdown[diff]
        if stats["total"] > 0:
            acc_sca = (stats["correct_sca_r"] / stats["total"]) * 100
            print("{:<10} | {:<8} | {:<11.1f}%".format(diff, stats["total"], acc_sca))

if __name__ == "__main__":
    main()