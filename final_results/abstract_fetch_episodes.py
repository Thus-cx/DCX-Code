import json
import os

# 1. 目标 episode_id 列表 
# 使用集合 (set) 可以将查找时间复杂度从 O(N) 降到 O(1)，加快处理速度
TARGET_EPISODES = {
    '699', '409', '134', '693', '24', '721', '587', '561', '889', '87', 
    '778', '188', '601', '880', '936', '874', '607', '421', '453', '598', 
    '877', '431', '262', '545', '970', '729', '539', '614', '567', '867', 
    '865', '367', '770', '869', '994', '870', '866', '871', '694', '944', 
    '992', '740', '66', '757', '620', '569', '939', '548', '604', '531', 
    '481', '782', '426', '660', '475', '479', '796', '692', '712', '875', 
    '547', '862', '791', '542', '130', '489', '365', '931', '538', '149', 
    '527', '818', '933', '596', '927', '896', '775', '938', '845', '118', 
    '435', '250', '401', '390', '940', '19', '698', '572', '554', '571', 
    '945', '17', '85', '108', '36', '543', '861', '696', '433', '898'
}

# 2. 配置文件路径
INPUT_FILE = "final_results/results/qwen_results_ego_raw_nframes_8.json"
OUTPUT_FILE = "final_results/results/spot_to_fetch_results_ego_raw_nframes_8.json"

def main():
    # 检查输入文件是否存在
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误: 找不到输入文件 {INPUT_FILE}")
        return

    print(f"📂 正在读取原始全集文件: {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 解析 JSON 文件失败: {e}")
        return

    # 确保 data 是一个列表 (通常大模型评测结果会保存为包含多个字典的 list)
    if not isinstance(data, list):
        print(f"❌ 错误: 预期的 JSON 顶层结构是 list，但实际是 {type(data)}。请检查文件格式。")
        return

    print(f"✅ 成功读取，原始全集共有 {len(data)} 个评估片段 (slices)。")
    
    # 3. 过滤数据
    filtered_data = []
    missing_episodes = set(TARGET_EPISODES)
    
    for item in data:
        # 获取 episode_id，统一转换为字符串以避免类型不匹配（int vs str）
        ep_id = str(item.get('episode_id', ''))
        
        if ep_id in TARGET_EPISODES:
            filtered_data.append(item)
            # 如果在缺失集合中找到了，就移除它（用于最后检查是否有没提取到的）
            if ep_id in missing_episodes:
                missing_episodes.remove(ep_id)

    print(f"🎯 提取完成！从上述 {len(TARGET_EPISODES)} 个 episode 中共提取到了 {len(filtered_data)} 个评估片段。")
    
    if missing_episodes:
        print(f"⚠️ 注意: 以下 {len(missing_episodes)} 个 episode 在全集中未找到任何片段: {missing_episodes}")

    # 4. 保存为新文件
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    print(f"💾 正在保存子集文件至: {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # indent=4 使 JSON 格式化易读，ensure_ascii=False 保证中文字符串正常显示
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)
        
    print("✨ 处理完毕！")

if __name__ == "__main__":
    main()