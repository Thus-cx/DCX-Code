import os

def main():
    # ================= 配置区域 =================
    # 1. 路径设置
    test_meta_dir = "/root/autodl-tmp/partnr-planner-main/test_dataset_strong/metadata"
    final_data_dir = "/root/autodl-tmp/partnr-planner-main/final_raw_videos"
    
    # 2. 定义包含 json 和 mp4 的子文件夹
    subfolders_json = ["metadata"]
    # 如果你还有 global_ann 等文件夹，可以直接在这个列表里添加
    subfolders_mp4 = ["ego_raw_videos", "ego_ann_raw_videos", "third_raw_videos", "third_ann_raw_videos", "global_videos"]
    
    # 3. 安全模式开关（True: 只打印不删除；False: 真实执行删除）
    DRY_RUN = False 
    # ============================================

    # 第一步：获取所有测试集的 episode_id
    if not os.path.exists(test_meta_dir):
        print(f"❌ 找不到测试集目录: {test_meta_dir}")
        return

    test_episode_ids = set()
    for filename in os.listdir(test_meta_dir):
        if filename.endswith(".json"):
            # 提取文件名去掉 .json 的部分作为 episode_id
            ep_id = filename.replace(".json", "")
            test_episode_ids.add(ep_id)
            
    print(f"🔍 从测试集中解析出 {len(test_episode_ids)} 个 episode_id需要被删除。")

    # 第二步：在 final_dataset 中查找并删除对应的文件
    deleted_count = 0
    not_found_count = 0

    if DRY_RUN:
        print("\n⚠️ 当前为【安全模式 (DRY_RUN=True)】，只打印计划删除的文件，不会进行真实删除。")
        print("请确认以下文件列表，如果没问题，请将代码中的 DRY_RUN 改为 False 后重新运行。\n")
    else:
        print("\n🔥 当前为【真实删除模式 (DRY_RUN=False)】，正在清理文件...\n")

    for ep_id in test_episode_ids:
        # 待检查的文件路径列表
        files_to_check = []
        
        # 组装 JSON 文件路径
        for folder in subfolders_json:
            files_to_check.append(os.path.join(final_data_dir, folder, f"{ep_id}.json"))
            
        # 组装 MP4 文件路径
        for folder in subfolders_mp4:
            files_to_check.append(os.path.join(final_data_dir, folder, f"{ep_id}.mp4"))

        # 执行检查和删除
        for file_path in files_to_check:
            if os.path.exists(file_path):
                if DRY_RUN:
                    print(f"[待删除] {file_path}")
                else:
                    os.remove(file_path)
                    print(f"[已删除] {file_path}")
                deleted_count += 1
            else:
                # 记录一下哪些文件在 final 里没找到（可能是本来就没有，也可能是名字对不上）
                not_found_count += 1

    # 第三步：打印总结报告
    print("\n" + "="*40)
    print("📊 清理任务总结：")
    print(f"目标删除 episode 数量: {len(test_episode_ids)}")
    if DRY_RUN:
        print(f"预计将被删除的文件总数: {deleted_count}")
    else:
        print(f"✅ 成功删除的文件总数: {deleted_count}")
    print(f"在 final_dataset 中未找到的文件数: {not_found_count}")
    print("="*40)

if __name__ == "__main__":
    main()