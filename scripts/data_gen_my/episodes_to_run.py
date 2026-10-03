import os
import glob
import json
import gzip
from pathlib import Path
metadata_dir = "/root/autodl-tmp/partnr-planner-main/final_raw_videos/metadata"
json_files = glob.glob(os.path.join(metadata_dir, "*.json"))
completed = []
for f in json_files:
    basename = os.path.basename(f)
    try:
        episode_num = int(basename[:-5])
        completed.append(episode_num)
    except ValueError:
        print(f"忽略非数字文件名：{basename}")
print(f"已完成 episode：{sorted(completed)}")

val_new = "/root/autodl-tmp/partnr-planner-main/data/datasets/partnr_episodes/v0_0/val_new.json.gz"
with gzip.open(val_new, 'rt', encoding='utf-8') as f:
    val_new_data = json.load(f)
total_episodes = val_new_data["episodes"]
episode_to_run = []
test_episode_933 = []
for i in range(len(total_episodes)):
    current_episode = total_episodes[i]
    current_episode_id = current_episode["episode_id"]
    if int(current_episode_id) not in completed:
        episode_to_run.append(current_episode)
    if int(current_episode_id)==933:
        test_episode_933.append(current_episode)
val_part_2 = {
    "config": None,
    "episodes": episode_to_run
}
val_part_2_file = "/root/autodl-tmp/partnr-planner-main/data/datasets/partnr_episodes/v0_0/val_part2.json.gz"
with gzip.open(val_part_2_file, 'wt', encoding='utf-8') as f:
    json.dump(val_part_2, f, ensure_ascii=False, indent=2)

test_episode_933_data = {
    "config": None,
    "episodes": test_episode_933
}
test_episode_933_file = "/root/autodl-tmp/partnr-planner-main/data/datasets/partnr_episodes/v0_0/test_episode_933.json.gz"
with gzip.open(test_episode_933_file, 'wt', encoding='utf-8') as f:
    json.dump(test_episode_933_data, f, ensure_ascii=False, indent=2)
