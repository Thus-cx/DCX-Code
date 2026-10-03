import json
import gzip

def main():
    val_data_path = "/root/autodl-tmp/partnr-planner-main/data/datasets/partnr_episodes/v0_0/val_new.json.gz"
    with gzip.open(val_data_path, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    episodes = data["episodes"]
    scene_ids = []
    scene_id_to_episodes = {}
    scene_id_to_episodes_num = {}
    for i in range(len(episodes)):
        episode = episodes[i]
        scene_id = episode["scene_id"]
        if scene_id not in scene_ids:
            scene_ids.append(scene_id)
            scene_id_to_episodes[scene_id] = []
        scene_id_to_episodes[scene_id].append(episode["episode_id"])
    for scene_id in scene_ids:
        scene_id_to_episodes_num[scene_id] = len(scene_id_to_episodes[scene_id])
    print(f"Total scenes: {len(scene_ids)}")
    for id in scene_id_to_episodes.keys():
        print(f"{id}:{len(scene_id_to_episodes[id])}")
    # print(f"scene_id_to_episodes_num:{scene_id_to_episodes_num}")

if __name__ == "__main__":
    main()

