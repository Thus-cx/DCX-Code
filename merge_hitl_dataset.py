import os
import gzip
import json

# 原始 HITL 数据的根目录
RAW_DIR = "data/hitl_data/p5_learn_val_v2/raw"

# 输出的聚合文件路径（用于 --dataset-file 参数）
OUTPUT_FILE = "data/hitl_data/p5_learn_val_v2/processed/p5_learn_val_v2_dataset.json.gz"

def collect_all_episodes():
    all_eps = []
    for session_folder in sorted(os.listdir(RAW_DIR)):
        folder_path = os.path.join(RAW_DIR, session_folder)
        if not os.path.isdir(folder_path):
            continue

        # 优先读取 session.json.gz，因为里面已经是 episodes 列表
        session_file = os.path.join(folder_path, "session.json.gz")
        if os.path.exists(session_file):
            try:
                with gzip.open(session_file, "rt", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "episodes" in data and isinstance(data["episodes"], list):
                    all_eps.extend(data["episodes"])
                    continue
            except Exception as e:
                print(f"[WARN] 读取失败 {session_file}: {e}")

        # 如果没有 session.json.gz，则从各单集文件提取 episode 元信息
        for fname in os.listdir(folder_path):
            if not fname.endswith(".json.gz"):
                continue
            if fname == "session.json.gz":
                continue
            full_path = os.path.join(folder_path, fname)
            try:
                with gzip.open(full_path, "rt", encoding="utf-8") as f:
                    data = json.load(f)
                # 提取 episode 元信息
                if isinstance(data, dict) and "episode" in data and isinstance(data["episode"], dict):
                    all_eps.append(data["episode"])
                elif isinstance(data, dict) and "episodes" in data:
                    all_eps.extend(data["episodes"])
                else:
                    print(f"[WARN] 未识别结构 {full_path}")
            except Exception as e:
                print(f"[WARN] 读取失败 {full_path}: {e}")
    return all_eps

if __name__ == "__main__":
    episodes = collect_all_episodes()
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with gzip.open(OUTPUT_FILE, "wt", encoding="utf-8") as fout:
        json.dump({"episodes": episodes}, fout)
    print(f"[OK] 写入 {OUTPUT_FILE}, 总 episodes 数量: {len(episodes)}")