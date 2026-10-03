import os
import json

# ================= 配置区域 =================
# 替换为你的 metadata 文件夹的实际路径
META_DIR_ROOT = "final_raw_videos/metadata"
FPS = 10.0  # 根据你的仿真设定，步长/帧率为 10


def calculate_dataset_statistics(meta_dir):
    total_episodes = 0
    total_subtasks = 0
    total_subtask_frames = 0
    total_episode_frames = 0

    # 确保目录存在
    if not os.path.exists(meta_dir):
        print(f"找不到目录: {meta_dir}")
        return

    # 遍历所有的 json 文件
    for filename in os.listdir(meta_dir):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(meta_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

                total_episodes += 1

                # 1. 统计整个 Episode 的总帧数
                # metadata 中包含 "steps" 数组，其长度即为整个视频的物理总帧数
                steps = data.get("steps", [])
                total_episode_frames += len(steps)

                # 2. 统计子任务数量和子任务内部总帧数
                subtasks = data.get("subtasks", [])
                total_subtasks += len(subtasks)

                for task in subtasks:
                    start = task.get("start_step", 0)
                    end = task.get("end_step", 0)
                    total_subtask_frames += (end - start)

        except Exception as e:
            print(f"读取文件 {filename} 时出错: {e}")

    # ================= 换算为时间 (秒, 分钟, 小时) =================
    total_subtask_sec = total_subtask_frames / FPS
    total_episode_sec = total_episode_frames / FPS

    # ================= 打印学术统计结果 =================
    print("=" * 50)
    print("📊 数据集规模统计报告 (Dataset Statistics)")
    print("=" * 50)
    print(f"总计 Episode 视频数 : {total_episodes} 个")
    print(f"总计 子任务 (Subtasks) : {total_subtasks} 个")
    print("-" * 50)
    print(f"子任务有效总时长 : {total_subtask_sec:.2f} 秒")
    print(f"                  约 {total_subtask_sec / 60:.2f} 分钟")
    print(f"                  约 {total_subtask_sec / 3600:.2f} 小时")
    print("-" * 50)
    print(f"数据集物理总时长 : {total_episode_sec:.2f} 秒")
    print(f"                  约 {total_episode_sec / 60:.2f} 分钟")
    print(f"                  约 {total_episode_sec / 3600:.2f} 小时")
    print("=" * 50)


if __name__ == "__main__":
    calculate_dataset_statistics(META_DIR_ROOT)