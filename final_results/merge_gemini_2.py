import json
import os

# ================= 配置文件路径 =================
PART1_FILE = "gemini-2.5-pro_results_dual_view_nframes_8_GPTGod_Part1.json"
PART2_FILE = "gemini-2.5-pro_results_dual_view_nframes_8_GPTGod_Part2.json"
OUTPUT_FILE = "gemini_dual_merged_full.json"

def main():
    # 检查文件是否存在
    if not os.path.exists(PART1_FILE) or not os.path.exists(PART2_FILE):
        print(f"[!] 错误：请确保 {PART1_FILE} 和 {PART2_FILE} 都在当前目录下。")
        return

    # 读取 Part 1
    print(f"[*] 正在读取 {PART1_FILE} ...")
    with open(PART1_FILE, 'r', encoding='utf-8') as f:
        part1_data = json.load(f)

    # 读取 Part 2
    print(f"[*] 正在读取 {PART2_FILE} ...")
    with open(PART2_FILE, 'r', encoding='utf-8') as f:
        part2_data = json.load(f)

    # 确保两者都是列表格式
    if not isinstance(part1_data, list) or not isinstance(part2_data, list):
        print("[!] 错误：JSON 文件的最外层应该是列表 (List) 格式。")
        return

    # 直接拼接两个列表
    merged_data = part1_data + part2_data

    # 打印统计信息
    print(f"[*] Part 1 包含 {len(part1_data)} 条数据")
    print(f"[*] Part 2 包含 {len(part2_data)} 条数据")
    print(f"[*] 合并后共计 {len(merged_data)} 条数据")

    # 保存合并后的文件
    print(f"[*] 正在保存合并后的文件至: {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)

    print("\n✅ 合并完成！")

if __name__ == "__main__":
    main()