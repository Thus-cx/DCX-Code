import json
import gzip
import random

val_new_file = "/root/autodl-tmp/partnr-planner-main/data/datasets/partnr_episodes/v0_0/val_new.json.gz"
with gzip.open(val_new_file, 'rt', encoding='utf-8') as f:
    data = json.load(f)
episodes = data["episodes"]
all_scene_ids = ["106878960_174887073",
                 "104348361_171513414",
                 "108736824_177263559",
                 "102344049",
                 "108736872_177263607",
                 "103997460_171030507",
                 "103997919_171031233",
                 "108736851_177263586"]
part_scene_ids = {"106366386_174226770": 28,
                  "102817140": 24,
                  "106366410_174226806": 23,
                  "102344529":10,
                  "107734176_176000019": 9,
                  "103997895_171031182": 9,
                  "102816756": 9,
                  "107733960_175999701": 7,
                  "104348010_171512832": 7,
                  "106878915_174887025": 7}
test_episodes = []
part_scene_episodes = {"106366386_174226770": [],
                  "102817140": [],
                  "106366410_174226806": [],
                  "102344529":[],
                  "107734176_176000019": [],
                  "103997895_171031182": [],
                  "102816756": [],
                  "107733960_175999701": [],
                  "104348010_171512832": [],
                  "106878915_174887025": []}
for i in range(len(episodes)):
    episode = episodes[i]
    scene_id = episode["scene_id"]
    if scene_id in all_scene_ids:
        test_episodes.append(episode)
    if scene_id in part_scene_ids.keys():
        part_scene_episodes[scene_id].append(episode)
for scene_id in part_scene_ids.keys():
    n = int(part_scene_ids[scene_id])
    random_list = random.sample(part_scene_episodes[scene_id], n)
    test_episodes = test_episodes + random_list
test_datasets_data = {
    "config": None,
    "episodes": test_episodes
}
final_test_dataset_file = "/root/autodl-tmp/partnr-planner-main/data/datasets/partnr_episodes/v0_0/final_test_datasets.json.gz"
with gzip.open(final_test_dataset_file, 'wt', encoding='utf-8') as f:
    json.dump(test_datasets_data, f, ensure_ascii=False, indent=2)
# 106878960_174887073:53
# 104348361_171513414:49
# 108736824_177263559:4
# 102344049:1
# 108736872_177263607:5
# 103997460_171030507:1
# 103997919_171031233:1
# 108736851_177263586:1
#
# "106366386_174226770": 28
# "102817140": 24
# "106366410_174226806": 23
# "102344529":10
# "107734176_176000019": 9
# "103997895_171031182": 9
# "102816756": 9
# "107733960_175999701": 7
# "104348010_171512832": 7
# "106878915_174887025": 7