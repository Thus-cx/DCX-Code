import json
import re
import os
import matplotlib.pyplot as plt

# ================= 配置区域 =================
# 你原始大模型测试跑出来的结果文件
INPUT_RESULT_FILE = "final_results/qwen_results_ego_raw.json" 
# 重新计算并汇总后的数值保存路径
OUTPUT_METRICS_FILE = "final_results/processed/final_metrics_qwen_ego_raw.json"
# 图表保存目录
OUTPUT_PLOT_DIR = "final_results/plots/"

os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

# 定义两组用于画图的切片参数
ACCUMULATIVE_SLICES = ["transit_30", "transit_60", "transit_90"]
PHASE_SLICES = ["pre_transit", "transit_30_60", "transit_60_90", "post_transit"]

# ================= 核心工具函数 =================

def fix_gt_area(area_str):
    """剔除区域字符串末尾的下划线和数字，例如 'dining_room_1' -> 'dining_room'"""
    if not area_str: return "unknown"
    # 使用正则匹配末尾的 _ 和数字并替换为空
    fixed = re.sub(r'_\d+$', '', str(area_str))
    return fixed.lower().strip()

def safe_match(pred, gt):
    """安全的字符串匹配函数，兼容 boolean 和 None"""
    if pred is None or gt is None: return False
    return str(pred).lower().strip() == str(gt).lower().strip()

def recalculate_evaluation(item):
    """基于修复后的 GT 重新计算所有指标"""
    gt = item.get("ground_truth", {})
    pred = item.get("parsed_json")
    
    # 1. 修复 GT Area
    fixed_gt_area = fix_gt_area(gt.get("target_area"))
    
    # 如果没有解析出合法的 JSON 字典，则全部算错
    if not pred or not isinstance(pred, dict):
        return {
            "acc_area": False, "acc_status": False, "acc_obj": False,
            "acc_furn": False, "acc_aff": False, "acc_action": False,
            "acc_target": False, "acc_intent": False, 
            "sca_reasoning": False, "sca_full": False
        }

    # 2. 独立节点对比
    acc_area = safe_match(pred.get("Q1_target_area"), fixed_gt_area)
    acc_status = safe_match(pred.get("Q2_is_holding_object"), gt.get("is_holding_object"))
    acc_obj = safe_match(pred.get("Q3_held_object_category"), gt.get("held_object_category"))
    acc_furn = safe_match(pred.get("Q4_target_furniture"), gt.get("target_furniture"))
    acc_aff = safe_match(pred.get("Q5_furniture_affordance"), gt.get("furniture_affordance"))

    # 3. 三元组对比 [Action, Object, Target]
    gt_triplet = gt.get("final_intent_triplet", ["", "", ""])
    p_triplet = pred.get("Q6_final_intent_triplet", [])
    
    if isinstance(p_triplet, list) and len(p_triplet) == 3:
        acc_action = safe_match(p_triplet[0], gt_triplet[0])
        acc_target = safe_match(p_triplet[2], gt_triplet[2])
        acc_intent = acc_action and safe_match(p_triplet[1], gt_triplet[1]) and acc_target
    else:
        acc_action, acc_target, acc_intent = False, False, False

    # 4. 阶梯一致性计算 (SCA)
    # 逻辑溯因：排除物体识别(acc_obj)，看其他推理逻辑是否全对
    sca_reasoning = (acc_area and acc_status and acc_furn and acc_aff and acc_action and acc_target)
    # 全链路：必须全部正确
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

    # 初始化统计字典
    metrics_summary = {}

    # 1. 遍历并重新计算
    for item in data:
        slice_param = item.get("slice_param")
        if not slice_param: continue
        
        # 初始化这个切片的统计桶
        if slice_param not in metrics_summary:
            metrics_summary[slice_param] = {
                "total": 0,
                "metrics_count": {
                    "acc_area": 0, "acc_status": 0, "acc_obj": 0,
                    "acc_furn": 0, "acc_aff": 0, "acc_action": 0,
                    "acc_target": 0, "acc_intent": 0, 
                    "sca_reasoning": 0, "sca_full": 0
                },
                "metrics_percentage": {}
            }
        
        # 重新评估
        re_eval = recalculate_evaluation(item)
        
        # 累加统计
        metrics_summary[slice_param]["total"] += 1
        for key, val in re_eval.items():
            if val: metrics_summary[slice_param]["metrics_count"][key] += 1

    # 2. 计算百分比
    for slice_param, stats in metrics_summary.items():
        total = stats["total"]
        if total > 0:
            for key, count in stats["metrics_count"].items():
                stats["metrics_percentage"][key] = round((count / total) * 100, 2)

    # 3. 将数值保存到文件 (论文里做表格可以直接查这个文件)
    with open(OUTPUT_METRICS_FILE, 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Recalculated metrics saved to {OUTPUT_METRICS_FILE}")

    # ================= 在终端打印分析报表 =================
    print("\n" + "="*115)
    print("CoT-HRC DIAGNOSTIC ANALYSIS REPORT")
    print("="*115)
    
    # 定义表头和格式
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

    # 打印累积时间和不同阶段的两张表
    print_table("Accumulative Progress", ACCUMULATIVE_SLICES)
    print_table("Temporal Phases", PHASE_SLICES)
    print("\n")

    # ================= 画图模块 =================
    
    # 画图全局设置 (学术风)
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
    
    # 定义我们需要画在图上的关键线 (线太多会乱，挑选最具代表性的核心指标)
    plot_metrics = {
        "acc_area": ("Target Area Acc", "o", "-"),
        "acc_furn": ("Target Furniture Acc", "s", "--"),
        "acc_intent": ("Final Intent Acc", "^", "-."),
        "sca_reasoning": ("SCA (Reasoning)", "D", "-"),
        "sca_full": ("SCA (Full)", "X", ":")
    }

    # 图 1: 累积时间比例 (Accumulative Time Ratio)
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    x_labels_acc = ["30%", "60%", "90%"]
    
    for metric_key, (label, marker, linestyle) in plot_metrics.items():
        y_values = []
        for sp in ACCUMULATIVE_SLICES:
            if sp in metrics_summary:
                y_values.append(metrics_summary[sp]["metrics_percentage"].get(metric_key, 0))
            else:
                y_values.append(0)
        ax1.plot(x_labels_acc, y_values, label=label, marker=marker, linestyle=linestyle, linewidth=2, markersize=8)

    ax1.set_title("Model Performance vs. Observation Progress Ratio")
    ax1.set_xlabel(r"Transit Phase Progress Ratio ($\rho$)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 100)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc="upper left")
    
    fig1_path = os.path.join(OUTPUT_PLOT_DIR, "accumulative_progress1.png")
    fig1.tight_layout()
    fig1.savefig(fig1_path, dpi=300)
    print(f"Plot 1 saved to {fig1_path}")

    # 图 2: 同时长不同阶段 (Fixed-length Phase Slicing)
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    x_labels_phase = ["Pre-transit", "Transit\n(30%-60%)", "Transit\n(60%-90%)", "Post-transit"]
    
    for metric_key, (label, marker, linestyle) in plot_metrics.items():
        y_values = []
        for sp in PHASE_SLICES:
            if sp in metrics_summary:
                y_values.append(metrics_summary[sp]["metrics_percentage"].get(metric_key, 0))
            else:
                y_values.append(0)
        ax2.plot(x_labels_phase, y_values, label=label, marker=marker, linestyle=linestyle, linewidth=2, markersize=8)

    ax2.set_title("Model Performance across Temporal Phases")
    ax2.set_xlabel("Observation Phase Window")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc="upper left")
    
    fig2_path = os.path.join(OUTPUT_PLOT_DIR, "temporal_phases1.png")
    fig2.tight_layout()
    fig2.savefig(fig2_path, dpi=300)
    print(f"Plot 2 saved to {fig2_path}")

if __name__ == "__main__":
    main()