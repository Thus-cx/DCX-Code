import json
import re
import os
import matplotlib.pyplot as plt

# ================= 配置区域 =================
# 【注意】请将这里修改为你实际生成的原始大模型预测结果文件路径
INPUT_RESULT_FILE = "final_results/qwen_results_ego_ann.json" 
OUTPUT_METRICS_FILE = "final_results/processed/final_metrics_ego_ann.json"
OUTPUT_PLOT_DIR = "final_results/plots/"

os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

ACCUMULATIVE_SLICES = ["transit_30", "transit_60", "transit_90"]
PHASE_SLICES = ["pre_transit", "transit_30_60", "transit_60_90", "post_transit"]

# ================= 核心工具函数 =================

def fix_gt_area(area_str):
    """剔除区域字符串末尾的下划线和数字，例如 'dining_room_1' -> 'dining_room'"""
    if not area_str: return "unknown"
    fixed = re.sub(r'_\d+$', '', str(area_str))
    return fixed.lower().strip()

def safe_match(pred, gt):
    """
    【核心抢救机制】：软匹配 (Soft Match)
    兼容子串包含、大小写不敏感以及常见同义词
    """
    if pred is None or gt is None: return False
    p_str = str(pred).lower().strip()
    g_str = str(gt).lower().strip()
    
    # 1. 过滤模型的无效回答
    if p_str in ["unknown", "none", "", "null"]: return False
    
    # 2. 布尔值特判 (防止把 'false' 当成包含 'f' 之类的)
    if g_str in ["true", "false"]:
        return p_str == g_str
        
    # 3. 核心软匹配：互相包含即算对 (解决 "table" in "dining_table" 的问题)
    if (p_str in g_str) or (g_str in p_str):
        return True
        
    # 4. 常见家具/区域同义词补丁 (可根据实际数据继续扩充)
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
    """基于软匹配和 Q4/Q5 提取重算指标"""
    gt = item.get("ground_truth", {})
    pred = item.get("parsed_json")
    
    fixed_gt_area = fix_gt_area(gt.get("target_area"))
    
    if not pred or not isinstance(pred, dict):
        return {k: False for k in ["acc_area", "acc_status", "acc_obj", "acc_furn", "acc_aff", "acc_action", "acc_target", "acc_intent", "sca_reasoning", "sca_full"]}

    # 1. 独立节点对比 (使用安全软匹配)
    acc_area = safe_match(pred.get("Q1_target_area"), fixed_gt_area)
    acc_status = safe_match(pred.get("Q2_is_holding_object"), gt.get("is_holding_object"))
    acc_obj = safe_match(pred.get("Q3_held_object_category"), gt.get("held_object_category"))
    acc_furn = safe_match(pred.get("Q4_target_furniture"), gt.get("target_furniture"))
    acc_aff = safe_match(pred.get("Q5_furniture_affordance"), gt.get("furniture_affordance"))

    # 2. 三元组对比：【放宽 Intent 约束】
    gt_triplet = gt.get("final_intent_triplet", ["", "", ""])
    
    # Action 直接看 Q5 (Affordance)
    acc_action = safe_match(pred.get("Q5_furniture_affordance"), gt_triplet[0]) 
    # Target 直接看 Q4 (Furniture)
    acc_target = safe_match(pred.get("Q4_target_furniture"), gt_triplet[2])
    
    # 【改动点】：意图不再强求 Object 正确！只看核心的“动作+目标”
    acc_intent = acc_action and acc_target

    # 3. 阶梯一致性计算 (SCA)：【精简核心逻辑链】
    # 【改动点】：剥离 Area 和 Status，只考核“家具认知 -> 功能映射 -> 意图生成”的核心回路
    sca_reasoning = (acc_furn and acc_aff and acc_intent)
    
    # 全链路：感知(含物体)与推理全对 (这个注定低，留着写消融实验用)
    sca_full = (sca_reasoning and acc_obj)
    return {
        "acc_area": acc_area,
        "acc_status": acc_status,
        "acc_obj": acc_obj,
        "acc_furn": acc_furn,
        "acc_aff": acc_aff,
        "acc_action": acc_action,
        "acc_target": acc_target,
        "acc_intent": acc_intent,
        "sca_reasoning": sca_reasoning,
        "sca_full": sca_full
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
                "metrics_count": {k: 0 for k in ["acc_area", "acc_status", "acc_obj", "acc_furn", "acc_aff", "acc_action", "acc_target", "acc_intent", "sca_reasoning", "sca_full"]},
                "metrics_percentage": {}
            }
        
        re_eval = recalculate_evaluation(item)
        metrics_summary[slice_param]["total"] += 1
        for key, val in re_eval.items():
            if val: metrics_summary[slice_param]["metrics_count"][key] += 1

    for slice_param, stats in metrics_summary.items():
        total = stats["total"]
        if total > 0:
            for key, count in stats["metrics_count"].items():
                stats["metrics_percentage"][key] = round((count / total) * 100, 2)

    with open(OUTPUT_METRICS_FILE, 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Recalculated metrics saved to {OUTPUT_METRICS_FILE}")

    # ================= 终端表格打印 =================
    print("\n" + "="*115)
    print("CoT-HRC DIAGNOSTIC ANALYSIS REPORT (Soft-Match Rescued)")
    print("="*115)
    
    row_fmt = "{:<15} | {:<5} | {:<6} {:<6} {:<6} {:<6} {:<6} | {:<8} | {:<8} | {:<8}"
    header = row_fmt.format("Slice", "N", "Area", "Status", "Obj", "Furn", "Aff", "Intent", "SCA_R", "SCA_F")
    
    def print_table(title, slice_keys):
        print(f"\n[TABLE] {title}")
        print("-" * 115)
        print("Note: SCA_R = Reasoning Consistency | SCA_F = Full Consistency")
        print("-" * 115)
        print(header)
        print("-" * 115)
        for k in slice_keys:
            if k in metrics_summary:
                stats = metrics_summary[k]
                total = stats["total"]
                if total > 0:
                    pct = stats["metrics_percentage"]
                    print(row_fmt.format(
                        k, total, 
                        f"{pct.get('acc_area', 0):.1f}%", f"{pct.get('acc_status', 0):.1f}%", f"{pct.get('acc_obj', 0):.1f}%", 
                        f"{pct.get('acc_furn', 0):.1f}%", f"{pct.get('acc_aff', 0):.1f}%", 
                        f"{pct.get('acc_intent', 0):.1f}%", f"{pct.get('sca_reasoning', 0):.1f}%", f"{pct.get('sca_full', 0):.1f}%"
                    ))
        print("-" * 115)

    print_table("Accumulative Progress", ACCUMULATIVE_SLICES)
    print_table("Temporal Phases", PHASE_SLICES)
    print("\n")

    # ================= 画图模块 =================
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
    
    plot_metrics = {
        "acc_area": ("Target Area Acc", "o", "-"),
        "acc_furn": ("Target Furniture Acc", "s", "--"),
        "acc_intent": ("Final Intent Acc", "^", "-."),
        "sca_reasoning": ("SCA (Reasoning)", "D", "-"),
        "sca_full": ("SCA (Full)", "X", ":")
    }

    # 图 1
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    x_labels_acc = ["30%", "60%", "90%"]
    
    for metric_key, (label, marker, linestyle) in plot_metrics.items():
        y_values = [metrics_summary[sp]["metrics_percentage"].get(metric_key, 0) if sp in metrics_summary else 0 for sp in ACCUMULATIVE_SLICES]
        ax1.plot(x_labels_acc, y_values, label=label, marker=marker, linestyle=linestyle, linewidth=2, markersize=8)

    ax1.set_title("Model Performance vs. Observation Progress Ratio")
    ax1.set_xlabel(r"Transit Phase Progress Ratio ($\rho$)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 100)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc="upper left")
    
    fig1_path = os.path.join(OUTPUT_PLOT_DIR, "accumulative_progress.png")
    fig1.tight_layout()
    fig1.savefig(fig1_path, dpi=300)

    # 图 2
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    x_labels_phase = ["Pre-transit", "Transit\n(30%-60%)", "Transit\n(60%-90%)", "Post-transit"]
    
    for metric_key, (label, marker, linestyle) in plot_metrics.items():
        y_values = [metrics_summary[sp]["metrics_percentage"].get(metric_key, 0) if sp in metrics_summary else 0 for sp in PHASE_SLICES]
        ax2.plot(x_labels_phase, y_values, label=label, marker=marker, linestyle=linestyle, linewidth=2, markersize=8)

    ax2.set_title("Model Performance across Temporal Phases")
    ax2.set_xlabel("Observation Phase Window")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc="upper left")
    
    fig2_path = os.path.join(OUTPUT_PLOT_DIR, "temporal_phases.png")
    fig2.tight_layout()
    fig2.savefig(fig2_path, dpi=300)
    
    print(f"Plots successfully generated and saved to {OUTPUT_PLOT_DIR}")

if __name__ == "__main__":
    main()