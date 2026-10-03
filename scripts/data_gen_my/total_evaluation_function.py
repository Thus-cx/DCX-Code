import json
import gzip

def main():
    input_data_file = "/root/autodl-tmp/partnr-planner-main/data/datasets/partnr_episodes/v0_0/val_re.json.gz"
    with gzip.open(input_data_file, 'rt', encoding="utf-8") as f:
        data = json.load(f)
        episodes = data["episodes"]
        evaluation_propositions_func_name = []
        non_func_error = []
        for i in range(len(episodes)):
            episode_evaluation_propositions = episodes[i]["evaluation_propositions"]
            episode_evaluation_funcs = []
            for j in range(len(episode_evaluation_propositions)):
                if episode_evaluation_propositions[j]["function_name"] not in evaluation_propositions_func_name:
                    evaluation_propositions_func_name.append(episode_evaluation_propositions[j]["function_name"])
                if episode_evaluation_propositions[j]["function_name"] not in episode_evaluation_funcs:
                    episode_evaluation_funcs.append(episode_evaluation_propositions[j]["function_name"])
            if "is_in_room" not in episode_evaluation_funcs and "is_next_to" not in episode_evaluation_funcs and "is_on_floor" not in episode_evaluation_funcs:
                non_func_error.append(episodes[i]["episode_id"])
        print(f"Total {len(evaluation_propositions_func_name)} func_names: {evaluation_propositions_func_name}")
        print(f"Total {len(non_func_error)} has no func error: {non_func_error}")

if __name__ == "__main__":
    main()