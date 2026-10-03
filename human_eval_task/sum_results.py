import json
import os
import numpy as np

# ================= 配置区域 =================
# 填写你10个人的指标 JSON 路径列表
# 根据你上传的文件名，这里假设文件名为 human1_metrics_200_samples.json 等
FILE_LIST = [f"human_eval_task/human{i}_metrics_200_samples.json" for i in range(1, 11)]

# ================= 主程序 =================
def main():
    overall_acc_list = []
    diff_acc_list = {"Easy": [], "Medium": [], "Hard": []}
    
    valid_files_count = 0

    for file_path in FILE_LIST:
        if not os.path.exists(file_path):
            print(f"⚠️ Warning: File not found -> {file_path}")
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        valid_files_count += 1
        
        # 提取难度指标模块
        diff_metrics = data.get("difficulty_metrics_all", {})
        
        total_correct = 0
        total_samples = 0
        
        # 提取各个难度的准确率，并累加计算当前人类受试者的 Overall Accuracy
        for diff in ["Easy", "Medium", "Hard"]:
            stats = diff_metrics.get(diff, {})
            
            # 累加用于计算 overall
            total_correct += stats.get("correct_sca_r", 0)
            total_samples += stats.get("total", 0)
            
            # 直接提取已经算好的难度百分比
            diff_acc_list[diff].append(stats.get("acc_sca_r_percentage", 0.0))
            
        # 计算该人类受试者的全局准确率
        if total_samples > 0:
            overall_acc_list.append((total_correct / total_samples) * 100)
                
    if valid_files_count == 0:
        print("❌ No valid files found. Please check FILE_LIST paths.")
        return

    # ================= 计算均值和标准差 (Mean & Std) =================
    # ddof=0 表示总体标准差, ddof=1 表示样本标准差。在学术统计中，评测受试者通常用 ddof=1 (样本标准差)
    mean_overall = np.mean(overall_acc_list)
    std_overall = np.std(overall_acc_list, ddof=1) 
    
    mean_diff = {d: np.mean(diff_acc_list[d]) for d in diff_acc_list}
    std_diff = {d: np.std(diff_acc_list[d], ddof=1) for d in diff_acc_list}
    
    print("\n" + "="*80)
    print(f"📊 Statistical Results for {valid_files_count} Human Subjects")
    print("="*80)
    print(f"Overall Accuracy : {mean_overall:.1f}% ± {std_overall:.1f}%")
    print("-" * 80)
    for d in ["Easy", "Medium", "Hard"]:
        print(f"{d:<8} Accuracy : {mean_diff[d]:.1f}% ± {std_diff[d]:.1f}%")
    print("="*80)
    
    # 自动生成用于论文/Rebuttal的 LaTeX 文本
    rebuttal_text = (
        f"For \\textbf{{R2}}'s well-posedness concern: ten subjects performed strict blind evaluations "
        f"on 200 constrained egocentric videos, yielding an average intent accuracy of "
        f"{mean_overall:.1f}\\% $\\pm$ {std_overall:.1f}\\% "
        f"(\\textbf{{Easy: {mean_diff['Easy']:.1f}\\% $\\pm$ {std_diff['Easy']:.1f}\\%, "
        f"Medium: {mean_diff['Medium']:.1f}\\% $\\pm$ {std_diff['Medium']:.1f}\\%, "
        f"Hard: {mean_diff['Hard']:.1f}\\% $\\pm$ {std_diff['Hard']:.1f}\\%}})."
    )
    
    print("\n📝 LaTeX Snippet for your Paper/Rebuttal:\n")
    print(rebuttal_text)
    print("\n" + "="*80)

if __name__ == "__main__":
    main()