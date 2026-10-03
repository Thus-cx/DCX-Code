import json
import os
import glob
import numpy as np

# --- 配置 ---
METADATA_DIR = "my_videos/metadata" 
OUTPUT_DIR = "my_videos/llm_intermediary_data"
VAL_DATASET_PATH = "/data/datasets/partnr_episodes/v0_0/val_mini.json" # 请确保此文件在当前目录下或修改路径
os.makedirs(OUTPUT_DIR, exist_ok=True)

class FactExtractor:
    def __init__(self, metadata_path, val_goals_map):
        with open(metadata_path, 'r') as f:
            self.raw_data = json.load(f)
        
        self.episode_id = str(self.raw_data.get("episode_id", "unknown"))
        self.instruction = self.raw_data.get("instruction", "None")
        self.steps = self.raw_data.get("steps", [])
        
        # 从预加载的 map 中获取当前 episode 的目标真值
        # 格式: {'Object_Handle': 'Receptacle_Handle'}
        self.gt_goals = val_goals_map.get(self.episode_id, {})

    def _safe_get_args(self, human_agent_data):
        """健壮地提取 args 字典"""
        args = human_agent_data.get('action_args')
        if args is None: return {}
        if isinstance(args, str):
            try: return eval(args)
            except: return {}
        if isinstance(args, dict): return args
        return {}

    def _resolve_semantic_name(self, handle, step_data):
        """
        [核心功能] ID 翻译器：将环境中的哈希 ID 转换为语义名称
        输入: "925f48efff..."
        输出: "candle_0" (如果能找到) 或 "item_925f" (兜底)
        """
        if handle in ["None", None, ""]: return "None"
        
        # 1. 尝试从 world_objects 中查找 (这是最准确的语义来源)
        # run_data_gen 生成的 world_objects key 通常是 "candle_0" 这种名字
        # 我们需要遍历 values 看看谁的 id/handle 匹配
        if 'world_objects' in step_data:
            for semantic_name, info in step_data['world_objects'].items():
                # 检查 info 里的字段 (根据你的 metadata 结构调整)
                # 假设 info 里有 'id' 或 'handle'，或者我们只能根据名字猜
                # 如果 metadata 没有存 handle，我们只能用 heuristic
                if semantic_name in handle: # 弱匹配：如果 handle 包含名字
                    return semantic_name
        
        # 2. 启发式清洗 (如果找不到映射)
        # 比如: "Ecoforms_Plant_Pot_GP9AAvocado_:0000" -> "Plant_Pot"
        clean = handle.split(':')[0] # 去掉 :0000
        clean = clean.split('_')[0] + "_" + clean.split('_')[1] # 取前两个词
        if len(clean) > 20: # 如果还是很长，可能是哈希
            return f"item_{handle[:4]}"
        return clean

    def _get_obj_state_description(self, step_data, obj_handle):
        """生成物体状态描述 (e.g., 'on_table_32')"""
        semantic_name = self._resolve_semantic_name(obj_handle, step_data)
        
        # 检查是否被抓着
        human = step_data['human_agent']
        if human.get('is_holding', False):
            # 检查抓的是不是这个物体
            held_obj = human.get('held_obj', '')
            if obj_handle in str(held_obj) or semantic_name in str(held_obj):
                return "held_by_human"

        # 检查位置 (Parent)
        if 'world_objects' in step_data:
            # 尝试找到对应的物体记录
            # 由于 key 是 semantic_name，我们需要模糊匹配
            matched_key = None
            for key in step_data['world_objects']:
                if key in semantic_name or semantic_name in key:
                    matched_key = key
                    break
            
            if matched_key:
                info = step_data['world_objects'][matched_key]
                parent = info.get('parent', 'unknown')
                return f"on_{parent}"
        
        return "location_unknown"

    def _get_successful_state(self, target_handle):
        """
        [关键] 从 val_mini.json 的真值中获取成功状态
        """
        if target_handle in self.gt_goals:
            target_recep_handle = self.gt_goals[target_handle]
            # 翻译 Receptacle ID
            # 这里我们没有 Receptacle 的语义库，只能做简单的清洗
            # 比如 "8d97...|receptacle_mesh..." -> "target_receptacle"
            # 或者我们让 LLM 知道这就是目标位置 ID
            return f"placed_on_{target_recep_handle[:6]}..." # 简略显示
        
        # 如果没在目标列表里，可能只是中间过程
        return "picked_up"

    def process(self):
        samples = []
        current_target_handle = None
        start_idx = 0
        
        for i, frame in enumerate(self.steps):
            human = frame['human_agent']
            args = self._safe_get_args(human)
            
            # 提取 Handle (哈希 ID)
            target_handle = args.get('object', args.get('name', None))
            if target_handle is None: target_handle = human.get('action_target')
            target_handle = str(target_handle) if target_handle else "None"
            
            # 切分逻辑
            is_switch = (target_handle != current_target_handle and target_handle != "None")
            is_end = (i == len(self.steps) - 1)
            
            if is_switch or is_end:
                if current_target_handle and current_target_handle != "None" and start_idx < i:
                    clip_frames = self.steps[start_idx:i]
                    active_frames = [f for f in clip_frames if f.get('robot_active', False)]
                    
                    if len(active_frames) > 5: # 放宽限制，捕捉短动作
                        
                        # 1. 翻译 ID -> 语义名
                        # 使用第一帧的数据来解析名字
                        semantic_name = self._resolve_semantic_name(current_target_handle, clip_frames[0])
                        
                        # 2. 动作序列
                        raw_actions = []
                        last_act = None
                        for f in active_frames:
                            act = f['human_agent'].get('atomic_action', 'Wait')
                            if act != last_act:
                                raw_actions.append(act)
                                last_act = act
                        
                        # 3. 状态序列 (带语义)
                        state_seq = []
                        # 采样: Start, End
                        for idx in [0, -1]:
                            state_desc = self._get_obj_state_description(active_frames[idx], current_target_handle)
                            state_seq.append(f"{semantic_name}_{state_desc}")
                            
                        # 4. 成功状态 (来自 val_mini 真值)
                        success_state = self._get_successful_state(current_target_handle)
                        
                        # 如果目标是 Place，把 ID 替换成语义描述会更好
                        # 但这里为了准确性，我们保留 ID 前缀，让 LLM 做匹配
                        
                        llm_input_data = {
                            "video_id": f"{self.episode_id}_{start_idx}",
                            "instruction": self.instruction,
                            "target_object_semantic": semantic_name,
                            "target_object_id": current_target_handle,
                            "human_action_sequence": raw_actions,
                            "observation_object_states": state_seq,
                            "successful_state_ground_truth": success_state,
                            "clip_range": [active_frames[0]['step'], active_frames[-1]['step']]
                        }
                        samples.append(llm_input_data)
                
                current_target_handle = target_handle
                start_idx = i
                
        return samples

def load_val_goals(val_path):
    """预加载 val_mini.json 中的目标映射"""
    goals_map = {} # {ep_id: {obj_handle: recep_handle}}
    try:
        with open(val_path, 'r') as f:
            data = json.load(f)
            if 'episodes' in data:
                for ep in data['episodes']:
                    ep_id = str(ep['episode_id'])
                    # name_to_receptacle 存储了 Object -> TargetReceptacle 的映射
                    if 'name_to_receptacle' in ep:
                        goals_map[ep_id] = ep['name_to_receptacle']
    except Exception as e:
        print(f"Warning: Could not load val dataset: {e}")
    return goals_map

def main():
    # 1. 加载真值目标
    print(f"Loading Ground Truth Goals from {VAL_DATASET_PATH}...")
    val_goals = load_val_goals(VAL_DATASET_PATH)
    print(f"Loaded goals for {len(val_goals)} episodes.")

    # 2. 处理 Metadata
    json_files = glob.glob(os.path.join(METADATA_DIR, "*.json"))
    all_prompts = []
    
    for f in json_files:
        try:
            extractor = FactExtractor(f, val_goals)
            all_prompts.extend(extractor.process())
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    # 3. 保存
    with open(os.path.join(OUTPUT_DIR, "llm_inputs.json"), 'w') as f:
        json.dump(all_prompts, f, indent=2)
    print(f"Generated {len(all_prompts)} items for LLM processing.")

if __name__ == "__main__":
    main()