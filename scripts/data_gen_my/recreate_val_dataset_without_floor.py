import json
import gzip
from email.encoders import encode_noop

from starlette.config import environ


def main():
    val_0_path = "/root/autodl-tmp/partnr-planner-main/data/datasets/partnr_episodes/v0_0/val.json.gz"
    train_path = "/root/autodl-tmp/partnr-planner-main/data/datasets/partnr_episodes/v0_0/train_mini.json.gz"
    val_new_path = "/root/autodl-tmp/partnr-planner-main/data/datasets/partnr_episodes/v0_0/val_new.json.gz"
    new_episodes_data = []
    choice = [5, 6, 19, 65, 90, 99, 123, 129, 375, 380, 391, 393, 398]
    with gzip.open(val_0_path, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    with gzip.open(train_path, 'rt', encoding='utf-8') as f:
        data_choose = json.load(f)
    episodes_data = data["episodes"]
    episodes_data_choose = data_choose["episodes"]
    floor_count = 0
    floor_check = False
    for i in range(len(episodes_data)):
        floor_check = False
        propositions = episodes_data[i]["evaluation_propositions"]
        for prop in propositions:
            if prop["function_name"] == "is_on_floor":
                floor_count += 1
                floor_check = True
                break
        if floor_check:
            episode = episodes_data_choose[choice[floor_count-1]]
            episode["episode_id"] = episodes_data[i]["episode_id"]
        else:
            episode = episodes_data[i]
        new_episodes_data.append(episode)
    print(f"Total floor count: {floor_count}")
    print(f"New val data length: {len(new_episodes_data)}")

    val_new_data = {
        "config": None,
        "episodes": new_episodes_data
    }
    with gzip.open(val_new_path, 'wt', encoding='utf-8') as f:
        json.dump(val_new_data, f, ensure_ascii=False, indent=2)
    print(f"Saved new val data in {val_new_path}")

if __name__ == "__main__":
    main()