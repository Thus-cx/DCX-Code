import json
import random
from collections import defaultdict

# ================= 配置区域 =================
ORIGINAL_DATASET = "test_dataset_strong/final_benchmark_0302/final_exam.json"
SAMPLED_DATASET = "test_dataset_strong/final_benchmark_0302/final_exam_sampled_1200.json"
TARGET_SAMPLE_SIZE = 1200  # 你可以在 1000-1500 之间自行调整
RANDOM_SEED = 42           # 固定随机种子，保证每次抽样结果一致（为了科研严谨性）

def main():
    random.seed(RANDOM_SEED)
    
    print(f"正在读取原始数据集: {ORIGINAL_DATASET} ...")
    with open(ORIGINAL_DATASET, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    total_original = len(data)
    print(f"原始数据集大小: {total_original} 条")

    # 1. 根据 slice_param 和 difficulty 进行分组
    grouped_data = defaultdict(list)
    for item in data:
        slice_p = item.get('slice_param', 'unknown')
        diff = item.get('difficulty', 'unknown')
        # 组合键，例如 "transit_30_Hard"
        key = f"{slice_p}_{diff}"
        grouped_data[key].append(item)
        
    print(f"共划分为 {len(grouped_data)} 个数据层 (Strata)。")

    # 2. 等比例采样
    sampled_data = []
    
    for key, items in grouped_data.items():
        # 计算该组在原数据集中的占比
        proportion = len(items) / total_original
        # 计算该组在目标样本集里应该抽取的数量
        sample_size_for_group = int(round(proportion * TARGET_SAMPLE_SIZE))
        
        # 处理四舍五入可能导致的极小分组数量为 0 的情况，至少保底抽 1 个
        if sample_size_for_group == 0 and len(items) > 0:
            sample_size_for_group = 1
            
        # 防止计算数量超过实际该组数量
        sample_size_for_group = min(sample_size_for_group, len(items))
        
        # 随机抽取
        sampled_items = random.sample(items, sample_size_for_group)
        sampled_data.extend(sampled_items)
        
        print(f" - 分组 [{key:<20}]: 原有 {len(items):<4} 条 -> 抽取 {sample_size_for_group:<4} 条")

    # 3. 打乱最终数据的顺序 (可选，防止按类别聚集)
    random.shuffle(sampled_data)

    # 4. 保存为新的 JSON 文件
    print(f"\n抽样完成！实际抽取总数: {len(sampled_data)} 条 (由于四舍五入，可能与目标值有极小误差)")
    with open(SAMPLED_DATASET, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, indent=2, ensure_ascii=False)
        
    print(f"抽样后数据集已保存至: {SAMPLED_DATASET}")

if __name__ == "__main__":
    main()