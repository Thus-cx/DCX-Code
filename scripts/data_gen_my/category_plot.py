import os
import json
import matplotlib.pyplot as plt
import numpy as np

# ================= 配置区域 =================
META_DIR_ROOT = "final_raw_videos/metadata"
SAVE_DIR = "final_results"


def calculate_dataset_objects_and_furs(meta_dir):
    # ... (这部分提取逻辑保持不变) ...
    fur_dir = {}
    object_dir = {}
    if not os.path.exists(meta_dir): return {}, {}
    for filename in os.listdir(meta_dir):
        if not filename.endswith(".json"): continue
        filepath = os.path.join(meta_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                subtasks = data.get("subtasks", [])
                for subtask in subtasks:
                    subtask_name = subtask.get("description_processed", "")
                    object_name_raw, fur_name_raw = None, None
                    if subtask_name.startswith("Rearrange"):
                        parts = subtask_name.split()
                        if len(parts) >= 3:
                            object_name_raw = parts[1]
                            fur_name_raw = parts[-1]
                    elif subtask_name.startswith("Clean") or subtask_name.startswith("Fill"):
                        parts = subtask_name.split()
                        if len(parts) >= 2:
                            object_name_raw = parts[1]
                            fur_name_raw = "cabinet"

                    if object_name_raw and fur_name_raw:
                        object_name = object_name_raw.rsplit("_", 1)[0].capitalize()
                        fur_name = fur_name_raw.rsplit("_", 1)[0].capitalize()
                        object_dir[object_name] = object_dir.get(object_name, 0) + 1
                        fur_dir[fur_name] = fur_dir.get(fur_name, 0) + 1
        except Exception as e:
            pass
    return fur_dir, object_dir


def plot_individual_distribution(data_dict, save_path, x_label, is_object=False, top_k=30):
    """绘制单张干净的横向长尾图，强制限制高度和 Top-K"""
    if not data_dict: return
    sorted_dict = dict(sorted(data_dict.items(), key=lambda item: item[1], reverse=True))

    names = list(sorted_dict.keys())
    counts = list(sorted_dict.values())

    # 核心修改：截断展示 Top-K
    if len(names) > top_k:
        names = names[:top_k]
        counts = counts[:top_k]

    y_pos = np.arange(len(names))

    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 11,
        "xtick.labelsize": 10
    })

    # 核心修改：强制写死画布大小，确保两张图高度 100% 一致！
    fig, ax = plt.subplots(figsize=(6, 8))

    color = '#2c3e50' if is_object else '#c0392b'
    ax.barh(y_pos, counts, color=color, edgecolor='black', linewidth=0.5)

    ax.set_xlabel(x_label)
    ax.set_yticks(y_pos)

    # 因为固定了最多 30 个，字号可以放心设置大一点（8pt 或 9pt），绝对不会违规
    ax.set_yticklabels(names, fontsize=9)

    ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=1200, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    fur_dir, object_dir = calculate_dataset_objects_and_furs(META_DIR_ROOT)
    os.makedirs(SAVE_DIR, exist_ok=True)

    obj_path = os.path.join(SAVE_DIR, "dist_held_objects.pdf")
    fur_path = os.path.join(SAVE_DIR, "dist_target_furniture.pdf")

    # 对物体设置 Top-30 截断，对家具直接展示
    plot_individual_distribution(object_dir, obj_path, "Frequency (Number of Slices)", is_object=True, top_k=30)
    plot_individual_distribution(fur_dir, fur_path, "Frequency (Number of Slices)", is_object=False, top_k=30)

    print(f"🎉 统一高度的 PDF 已生成！")