import json
import gzip

processed_instruction_file = "/root/autodl-tmp/partnr-planner-main/data/datasets/partnr_episodes/v0_0/processed_instructions.json.gz"

# 1. 读取压缩文件的数据
with gzip.open(processed_instruction_file, 'rt', encoding='utf-8') as f:
    data = json.load(f)

# 将 data 转换为字典映射，大幅提升匹配速度 (O(1) 查找)
instruction_map = {episode["episode_id"]: episode["processed_intent"] for episode in data}

benchmark_file = "/root/autodl-tmp/partnr-planner-main/test_dataset/final_benchmark_0302/final_exam.json"

# 2. 以 'rt' (只读) 模式打开并读取 benchmark 数据
with open(benchmark_file, 'rt', encoding='utf-8') as f:
    benchmark_data = json.load(f)

# 3. 在内存中处理数据
for i in range(len(benchmark_data)):
    current_episode_id = benchmark_data[i]["episode_id"]
    # 如果在映射字典中找到了对应的 episode_id，则添加字段
    if current_episode_id in instruction_map:
        benchmark_data[i]["processed_instruction"] = instruction_map[current_episode_id]

# 4. 以 'wt' (写入) 模式打开文件，将更新后的数据写回
with open(benchmark_file, 'wt', encoding='utf-8') as f:
    # 使用 indent=4 让输出的 JSON 文件具有良好的可读性格式
    json.dump(benchmark_data, f, ensure_ascii=False, indent=4)

print("数据处理并保存完成！")