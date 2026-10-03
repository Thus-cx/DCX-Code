import json
import os

# ================= 配置文件路径 =================
MAIN_FILE = "gpt-4o_results_oracle_dual_nframes_8.json"
PATCH_FILE = "gpt-4o_results_oracle_dual_nframes_8_part_570_951.json"
# 建议输出为一个新文件，确保原文件安全
OUTPUT_FILE = "gpt-4o_results_oracle_dual_nframes_8_merged.json"

def get_item_id(item):
    """
    提取样本的唯一标识。
    优先使用 'id'，如果没有则使用 'third_raw' 视频路径作为唯一标识。
    """
    item_id = item.get('question_id')
    if not item_id and 'video_paths' in item:
        item_id = item['video_paths'].get('third_raw')
    return item_id

def main():
    if not os.path.exists(MAIN_FILE) or not os.path.exists(PATCH_FILE):
        print("[!] 错误：请确保主文件和补丁文件都在当前目录下。")
        return

    print(f"[*] 正在读取主文件: {MAIN_FILE}")
    with open(MAIN_FILE, 'r', encoding='utf-8') as f:
        main_data = json.load(f)

    print(f"[*] 正在读取补丁文件 (570-951): {PATCH_FILE}")
    with open(PATCH_FILE, 'r', encoding='utf-8') as f:
        patch_data = json.load(f)

    # 1. 将补丁数据转化为字典，方便通过唯一 ID 快速 O(1) 查找
    patch_dict = {}
    for item in patch_data:
        item_id = get_item_id(item)
        if item_id:
            patch_dict[item_id] = item
        else:
            print("[!] 警告: 补丁文件中存在无法提取 ID 的异常数据。")

    # 2. 遍历主数据并执行精准替换
    replaced_count = 0
    main_ids = set()

    for i in range(len(main_data)):
        item_id = get_item_id(main_data[i])
        main_ids.add(item_id)

        # 如果主文件中的样本在补丁文件中有对应结果，则用补丁覆盖它
        if item_id in patch_dict:
            main_data[i] = patch_dict[item_id]
            replaced_count += 1

    # 3. 检查是否有补丁中的数据在主文件中完全丢失（缺失），如果有则追加到末尾
    appended_count = 0
    for item_id, item_data in patch_dict.items():
        if item_id not in main_ids:
            main_data.append(item_data)
            appended_count += 1

    # 4. 保存合并后的文件
    print(f"[*] 正在保存合并后的文件: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # indent=2 保持格式美观，与你原有的文件格式一致
        json.dump(main_data, f, indent=2, ensure_ascii=False)

    print("\n✅ 合并完成！")
    print(f" ➔ 成功替换了 {replaced_count} 条错误/空数据。")
    if appended_count > 0:
        print(f" ➔ 成功追加了 {appended_count} 条在主文件中完全缺失的数据。")
    print(f" ➔ 最终完整结果已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()