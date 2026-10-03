import json
from pathlib import Path
import gzip
import os  # 添加 os 模块以增强路径检查（可选，但推荐）


def check_videos(input_file):
    check = True
    sub_num_up = False
    try:
        # 增加文件存在性检查，防止文件不存在导致报错中断
        if not input_file.exists():
            print(f"Warning: File not found {input_file}")
            return False, False

        with open(input_file, 'r') as f:
            data = json.load(f)
        subtasks = data.get('subtasks', [])  # 使用 .get 防止键不存在
        gt_subtasks = data.get('evaluation_propositions', [])

        if len(subtasks) != len(gt_subtasks):
            check = False
            if len(subtasks) > len(gt_subtasks):
                sub_num_up = True
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        check = False

    return check, sub_num_up


def main():
    # 1. 将字符串转换为 Path 对象，这样才能使用 path 的方法
    input_path = Path("/root/autodl-tmp/partnr-planner-main/my_videos/metadata")
    original_dataset = "/root/autodl-tmp/partnr-planner-main/data/datasets/partnr_episodes/v0_0/val.json.gz"
    re_dataset = "/root/autodl-tmp/partnr-planner-main/data/datasets/partnr_episodes/v0_0/val_re.json.gz"

    wrong_episodes = []
    sub_num_up_episodes = []
    print("开始检查 episodes...")  # 添加打印，确认程序开始运行

    for i in range(1000):
        # 2. 修正路径拼接方式：Path对象可以直接用 / 拼接
        input_file = input_path / f"{i}.json"

        # 可选：打印进度，不然 1000 个文件可能以为卡死了
        if i % 100 == 0:
            print(f"Checking index {i}...")

        check, sub_num_up = check_videos(input_file)
        if not check:
            wrong_episodes.append(i)
            if sub_num_up:
                sub_num_up_episodes.append(i)

    print(f"{len(wrong_episodes)} episodes need to rerun: ", wrong_episodes)
    print(f"{len(sub_num_up_episodes)} episodes have more subtasks: ", sub_num_up_episodes)

    # 如果没有错误文件，就不需要写入新数据集，防止报错
    if not wrong_episodes:
        print("No wrong episodes found.")
        return

    # # 读取原始数据集并生成重跑集
    # try:
    #     with gzip.open(original_dataset, 'rt', encoding='utf-8') as f:
    #         data = json.load(f)
    #         episodes = data["episodes"]
    #         re_episodes = []
    #         for episode_id in wrong_episodes:
    #             # 注意：这里假设 episode_id 正好对应 list 的 index
    #             # 如果 dataset 里的 episodes 顺序不一定是 0-1000，这里可能需要根据 id 查找
    #             if episode_id < len(episodes):
    #                 re_episodes.append(episodes[episode_id])
    #             else:
    #                 print(f"Warning: Episode ID {episode_id} out of range")

    #     re_data = {
    #         "config": None,
    #         "episodes": re_episodes
    #     }

    #     with gzip.open(re_dataset, 'wt', encoding='utf-8') as f:
    #         json.dump(re_data, f, ensure_ascii=False, indent=2)
    #     print(f"Saved re-run dataset to {re_dataset}")

    # except FileNotFoundError:
    #     print(f"Original dataset not found at {original_dataset}")


# 3. 关键修复：程序入口
if __name__ == "__main__":
    main()