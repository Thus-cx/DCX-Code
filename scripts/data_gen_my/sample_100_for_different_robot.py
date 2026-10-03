import os
import json
import random
from collections import defaultdict

def get_intent_category(function_name):
    """
    根据 evaluation_propositions 中的 function 名称映射到宏观意图
    (结合论文中的 Placement, Storage, Maintenance, Disposal)
    """
    func_lower = function_name.lower()
    if func_lower in ["is_on_top", "is_in_room"]:
        return "Placement"
    elif func_lower in ["is_inside"]:
        return "Storage"
    elif func_lower in ["is_clean", "is_filled", "is_powered_on", "is_powered_off"]:
        return "Maintenance"
    # 如果有垃圾桶相关的判定可以归为 Disposal，这里做个默认兜底
    else:
        return "Other"

def main():
    # 1. 严格设置随机种子，确保每次抽样的 100 个 ID 完全一致
    random.seed(42)
    
    metadata_dir = "test_dataset_strong/metadata"
    if not os.path.exists(metadata_dir):
        print(f"目录 {metadata_dir} 不存在，请检查路径。")
        return

    all_episodes = []
    
    # 2. 读取所有已完成的 metadata JSON 文件
    for filename in os.listdir(metadata_dir):
        if not filename.endswith(".json"):
            continue
            
        filepath = os.path.join(metadata_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                ep_id = data.get("episode_id")
                props = data.get("evaluation_propositions", [])
                
                # 获取该 episode 的主要任务类型（取第一个 proposition 作为代表）
                intent_type = "Unknown"
                if props and len(props) > 0:
                    intent_type = get_intent_category(props[0].get("function", ""))
                
                all_episodes.append({
                    "episode_id": ep_id,
                    "intent_type": intent_type
                })
        except Exception as e:
            print(f"读取文件 {filename} 失败: {e}")

    print(f"共读取到 {len(all_episodes)} 个 episode metadata。")

    # 3. 将 episode 按照意图类别进行分桶 (Bucket)
    buckets = defaultdict(list)
    for ep in all_episodes:
        key = ep['intent_type']
        buckets[key].append(ep['episode_id'])

    # 4. 分层抽样 (目标: 100个)
    sampled_episode_ids = []
    
    # 计算每个类别桶应该分配的配额
    num_buckets = len(buckets)
    if num_buckets == 0:
        print("未找到任何有效的任务分类。")
        return
        
    base_quota = 100 // num_buckets
    
    for intent, ep_list in buckets.items():
        # 打乱当前桶内的顺序
        random.shuffle(ep_list)
        
        # 抽取配额数量 (如果该桶内总数不足配额，则全取)
        quota = min(base_quota, len(ep_list))
        sampled_episode_ids.extend(ep_list[:quota])
        print(f"类别 '{intent}' 抽取了 {quota} 个样本。")

    # 5. 如果因为某些桶数量不足导致总数不够 100，从剩余未被抽中的样本中随机补齐
    remaining_needed = 100 - len(sampled_episode_ids)
    if remaining_needed > 0:
        all_remaining_ids = [
            ep['episode_id'] for ep in all_episodes 
            if ep['episode_id'] not in sampled_episode_ids
        ]
        # 再次打乱剩余池并抽取
        random.shuffle(all_remaining_ids)
        fill_samples = all_remaining_ids[:remaining_needed]
        sampled_episode_ids.extend(fill_samples)
        print(f"随机补齐了 {len(fill_samples)} 个样本。")

    # 6. 输出最终的 100 个 ID 列表，并格式化为 Python List 形式方便直接复制
    print(f"\n成功抽取 {len(sampled_episode_ids)} 个 episode_ids。请将以下列表复制到 planner_demo.py 中：\n")
    print(f"target_episodes = {sampled_episode_ids}")

if __name__ == "__main__":
    main()