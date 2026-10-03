import json
import random
from collections import defaultdict

def main():
    # ================= 配置路径 =================
    # 输入：你用 FFmpeg 切片生成的包含 13000+ 条数据的原始索引
    input_file = "final_raw_videos/train_dataset_full/benchmark_index.json"
    # 输出：精简后的黄金训练集索引
    output_file = "final_raw_videos/train_dataset_full/train_filtered_index.json"
    # ============================================

    print("⏳ 正在加载原始数据集...")
    with open(input_file, "r", encoding="utf-8") as f:
        all_records = json.load(f)

    print(f"📦 原始数据总量: {len(all_records)} 个切片")

    # 1. 按照 (episode_id, task_index) 进行分组
    task_groups = defaultdict(list)
    for record in all_records:
        # 严格校验：确保数据包含 ego_raw_path，这是我们唯一的指定视角
        if "ego_raw_path" not in record:
            continue
            
        key = (record["episode_id"], record["task_index"])
        task_groups[key].append(record)

    print(f"🔍 识别到独立子任务总数: {len(task_groups)} 个")

    filtered_records = []
    
    # 2. 核心抽取逻辑 (层级优先级抽样)
    for key, slices in task_groups.items():
        chosen_slice = None
        
        # [第一顺位] 寻找 transit_90 (前 90% 的累积视角，最利于意图推理)
        for s in slices:
            if s.get("slice_param") == "transit_90":
                chosen_slice = s
                break
                
        # [第二顺位] 如果没有 transit_90，尝试找 transit_60 或 transit_60_90
        if not chosen_slice:
            for s in slices:
                if s.get("slice_param") in ["transit_60", "transit_60_90"]:
                    chosen_slice = s
                    break
        
        # [第三顺位] 如果还是没有，说明视频很特殊，随机挑一个最长的
        if not chosen_slice and slices:
            # 按照 duration (时长) 排序，选最长的那个
            slices_sorted = sorted(slices, key=lambda x: x.get("duration", 0), reverse=True)
            chosen_slice = slices_sorted[0]
            
        if chosen_slice:
            filtered_records.append(chosen_slice)

    # 3. 保存精简后的纯净数据集
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(filtered_records, f, indent=2, ensure_ascii=False)

    print("\n" + "="*40)
    print("🎉 抽样完成！")
    print(f"📉 冗余数据削减率: {(1 - len(filtered_records)/len(all_records))*100:.1f}%")
    print(f"🎯 精简后的训练集大小: {len(filtered_records)} 个切片 (黄金数据)")
    print(f"💾 已保存至: {output_file}")
    print("="*40)

if __name__ == "__main__":
    # 设定随机种子，保证每次运行抽取的结果完全一致（可复现）
    random.seed(42)
    main()