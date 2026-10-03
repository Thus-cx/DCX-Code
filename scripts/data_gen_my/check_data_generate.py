import json
import gzip

def main():
    clean_episodes = []
    fill_episodes = []
    powered_on_episodes = []
    powered_off_episodes = []
    in_room_episodes = []
    on_floor_episodes = []
    next_to_episodes = []
    val_datasets = "/root/autodl-tmp/partnr-planner-main/data/datasets/partnr_episodes/v0_0/val.json.gz"
    val_check_datasets = "/root/autodl-tmp/partnr-planner-main/data/datasets/partnr_episodes/v0_0/val_check.json.gz"
    with gzip.open(val_datasets, 'rt', encoding='utf-8') as f:
        data = json.load(f)
        episodes_data = data["episodes"]
        for i in range(len(episodes_data)):
            evaluation_propositions = episodes_data[i]["evaluation_propositions"]
            for prop in evaluation_propositions:
                if prop["function_name"] == "is_clean":
                    if episodes_data[i]["episode_id"] not in clean_episodes:
                        clean_episodes.append(episodes_data[i]["episode_id"])
                if prop["function_name"] == "is_filled":
                    if episodes_data[i]["episode_id"] not in fill_episodes:
                        fill_episodes.append(episodes_data[i]["episode_id"])
                if prop["function_name"] == "is_powered_on":
                    if episodes_data[i]["episode_id"] not in powered_on_episodes:
                        powered_on_episodes.append(episodes_data[i]["episode_id"])
                if prop["function_name"] == "is_powered_off":
                    if episodes_data[i]["episode_id"] not in powered_off_episodes:
                        powered_off_episodes.append(episodes_data[i]["episode_id"])
                if prop["function_name"] == "is_in_room":
                    if episodes_data[i]["episode_id"] not in in_room_episodes:
                        in_room_episodes.append(episodes_data[i]["episode_id"])
                if prop["function_name"] == "is_on_floor":
                    if episodes_data[i]["episode_id"] not in on_floor_episodes:
                        on_floor_episodes.append(episodes_data[i]["episode_id"])
                if prop["function_name"] == "is_next_to":
                    if episodes_data[i]["episode_id"] not in next_to_episodes:
                        next_to_episodes.append(episodes_data[i]["episode_id"])
        print(f"The clean episodes are:\n{clean_episodes}")
        print(f"The fill episodes are:\n{fill_episodes}")
        print(f"The power on episodes are:\n{powered_on_episodes}")
        print(f"The power off episodes are:\n{powered_off_episodes}")
        print(f"The in room episodes are:\n{in_room_episodes}")
        print(f"The on floor episodes are:\n{on_floor_episodes}")
        print(f"The next to episodes are:\n{next_to_episodes}")
        check_list = [22, 124, 353, 354, 357, 358, 405, 412, 516, 540, 607, 610, 770, 859, 862, 904, 919, 990]
        val_check_episodes = []
        for i in check_list:
            val_check_episodes.append(episodes_data[i])
        val_check_data = {
            "config": None,
            "episodes": val_check_episodes
        }
        with gzip.open(val_check_datasets, "wt", encoding='utf-8') as f:
            json.dump(val_check_data, f, ensure_ascii=False, indent=2)
        print("Saved check val datasets.")

        on_floor_list = []
        for i in on_floor_episodes:
            on_floor_list.append(episodes_data[int(i)])
        on_floor_data = {
            "config": None,
            "episodes": on_floor_list
        }
        on_floor_dataset = "/root/autodl-tmp/partnr-planner-main/data/datasets/partnr_episodes/v0_0/on_floor_dataset.json.gz"
        with gzip.open(on_floor_dataset, 'wt', encoding='utf-8') as f:
            json.dump(on_floor_data, f, ensure_ascii=False, indent=2)

        clean_list = []
        for i in clean_episodes:
            clean_list.append(episodes_data[int(i)])
        clean_data = {
            "config": None,
            "episodes": clean_list
        }
        clean_dataset = "/root/autodl-tmp/partnr-planner-main/data/datasets/partnr_episodes/v0_0/clean_dataset.json.gz"
        with gzip.open(clean_dataset, 'wt', encoding='utf-8') as f:
            json.dump(clean_data, f, ensure_ascii=False, indent=2)

        fill_list = []
        for i in fill_episodes:
            fill_list.append(episodes_data[int(i)])
        fill_data = {
            "config": None,
            "episodes": fill_list
        }
        fill_dataset = "/root/autodl-tmp/partnr-planner-main/data/datasets/partnr_episodes/v0_0/fill_dataset.json.gz"
        with gzip.open(fill_dataset, 'wt', encoding='utf-8') as f:
            json.dump(fill_data, f, ensure_ascii=False, indent=2)

        powered_list = []
        for i in powered_on_episodes:
            powered_list.append(episodes_data[int(i)])
        for i in powered_off_episodes:
            powered_list.append(episodes_data[int(i)])
        powered_data = {
            "config": None,
            "episodes": powered_list
        }
        powered_dataset = "/root/autodl-tmp/partnr-planner-main/data/datasets/partnr_episodes/v0_0/powered_dataset.json.gz"
        with gzip.open(powered_dataset, 'wt', encoding='utf-8') as f:
            json.dump(powered_data, f, ensure_ascii=False, indent=2)
        count_complex = 0
        for i in range(len(episodes_data)):
            if (str(i) in powered_on_episodes) or (str(i) in powered_off_episodes):
                count_complex += 1
        print(f"count complex: {count_complex}")

        open_list = []
        open_episodes = [39, 249]
        for i in open_episodes:
            open_list.append(episodes_data[int(i)])
        open_data = {
            "config": None,
            "episodes": open_list
        }
        open_dataset = "/root/autodl-tmp/partnr-planner-main/data/datasets/partnr_episodes/v0_0/open_dataset.json.gz"
        with gzip.open(open_dataset, 'wt', encoding='utf-8') as f:
            json.dump(open_data, f, ensure_ascii=False, indent=2)

        test_list = []
        test_episodes = [308, 358, 832]
        for i in test_episodes:
            test_list.append(episodes_data[int(i)])
        test_data = {
            "config": None,
            "episodes": test_list
        }
        test_dataset = "/root/autodl-tmp/partnr-planner-main/data/datasets/partnr_episodes/v0_0/test_dataset.json.gz"
        with gzip.open(test_dataset, 'wt', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
