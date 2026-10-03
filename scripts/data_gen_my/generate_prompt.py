import json
import os
import gzip

class PromptGenerator:
    def __init__(self, metadata_path, val_dataset_path):
        self.metadata = self._load_json(metadata_path)
        self.val_data = self._load_json(val_dataset_path)
        
        # 1. 建立 Handle 到 Semantic Name 的反向映射表
        # metadata 中存的是 Name -> Handle，我们需要 Handle -> Name 来翻译 val 数据
        self.name_to_handle = self.metadata.get("object_handle_mapping", {})
        self.handle_to_name = {v: k for k, v in self.name_to_handle.items()}
        
        # 2. 获取当前 Episode 的 ID (假设 metadata文件名就是 ID.json)
        self.episode_id = os.path.basename(metadata_path).split('.')[0]
        
        # 3. 从 val 数据集中找到当前 episode 的配置
        self.episode_config = self._find_episode_config(self.episode_id)

    def _load_json(self, path):
        if path.endswith('.gz'):
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                return json.load(f)
        else:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)

    def _find_episode_config(self, ep_id):
        """在 val_mini.json 中查找对应 episode_id 的配置"""
        if 'episodes' not in self.val_data:
            raise ValueError("Invalid val dataset format: 'episodes' key missing")
        
        for ep in self.val_data['episodes']:
            # 兼容字符串和数字类型的比较
            if str(ep['episode_id']) == str(ep_id):
                return ep
        print(f"Warning: Episode {ep_id} not found in validation dataset.")
        return None

    def translate_handle(self, handle_str):
        """将长哈希 handle 翻译为 candle_1 这种语义名"""
        # 有些 handle 可能带有 _:0000 后缀，需要处理
        # 你的 mapping 中存储的可能是带后缀也可能是不带的，这里尝试匹配
        if handle_str in self.handle_to_name:
            return self.handle_to_name[handle_str]
        
        # 尝试去掉 _:0000 后缀匹配 (Partnr 中常见的 handle 格式)
        base_handle = handle_str.split("_:")[0]
        # 这里需要遍历 mapping 查找，或者建立更鲁棒的 fuzzy matching
        # 简单起见，如果完全匹配找不到，就返回原始 hash 的前几位方便调试
        return self.handle_to_name.get(handle_str, f"UnknownObject({handle_str[:6]}...)")

    def parse_success_criteria(self):
        """解析 val_mini 中的 evaluation_conditions 并翻译为自然语言"""
        if not self.episode_config:
            return "No episode config found."

        # Partnr 的结构通常是 episodes -> tasks -> [task_name] -> evaluation_conditions
        # 或者直接在 episode info 里，这取决于具体版本，根据提供的 snippet 尝试解析
        # 假设结构: ep['tasks']['rearrange']['evaluation_conditions'] 或类似的
        # 如果找不到标准路径，我们尝试搜索 key
        
        conditions = []
        # 简单的路径尝试，根据你的 val_mini 结构调整
        task_info = self.episode_config.get('tasks', {}).get('rearrange', {})
        # 如果 tasks 结构不存在，可能直接在 info 或其他字段，这里假设标准 Partnr 结构
        # 注意：提供的 snippet 显示 val_mini.json 的结构比较扁平，
        # 但 evaluation_conditions 通常位于 task 定义中。
        # 如果你的 val_mini 是通过 standard habitat process 生成的：
        eval_conditions = []
        
        # 尝试直接查找 (根据你提供的 snippetFromBack)
        # snippet 看起来像是 predicates 的列表
        # 让我们假设我们能获取到那个 list
        if 'evaluation_conditions' in self.episode_config:
             eval_conditions = self.episode_config['evaluation_conditions']
        elif 'info' in self.episode_config and 'evaluation_conditions' in self.episode_config['info']:
             eval_conditions = self.episode_config['info']['evaluation_conditions']
        # 还有一种情况，如果是 rearrange 任务，可能在 targets 字段
        
        # --- 针对 Partnr 的特定解析逻辑 ---
        # 如果实在找不到，我们先用自然语言描述所有的 predicates
        desc_list = []
        
        # 模拟解析一个 predicate 列表 (你需要确认 json 里的确切位置)
        # 这里为了代码能跑，我写一个通用的解析器，假设我们拿到了那个 list
        # 实际使用时，如果打印出来是空的，说明路径不对
        
        # 临时：如果没有找到 conditions，我们尝试解析 instruction 作为兜底
        if not eval_conditions:
             return f"(Derived from Instruction) {self.metadata['instruction']}"

        for cond in eval_conditions:
            fn_name = cond.get('function_name')
            args = cond.get('args', {})
            
            if fn_name == 'is_on_top':
                obj_handles = args.get('object_handles', [])
                rec_handles = args.get('receptacle_handles', [])
                
                # 翻译
                objs = [self.translate_handle(h) for h in obj_handles]
                recs = [self.translate_handle(h) for h in rec_handles]
                
                desc_list.append(f"{' or '.join(objs)} should be on {' or '.join(recs)}")
            
            # 可以添加更多谓词解析，如 is_clean, is_filled 等
            elif fn_name == 'is_clean':
                obj_handles = args.get('object_handles', [])
                objs = [self.translate_handle(h) for h in obj_handles]
                desc_list.append(f"{' or '.join(objs)} should be clean")

        return "\n".join(desc_list) if desc_list else "Criteria parsing failed or empty."

    def get_action_sequence(self):
        """提取并压缩动作序列"""
        raw_steps = self.metadata['steps']
        compressed_seq = []
        last_action_desc = ""

        for step in raw_steps:
            human = step['human_agent']
            # 获取动作名
            act = human.get('atomic_action', 'Unknown')
            # 获取目标名 (如果是 None，尝试从 args 看，或者忽略)
            target = human.get('action_target', 'None')
            
            # 组合描述
            if target and target != "None":
                action_desc = f"{act}_to_{target}"
            else:
                action_desc = act
            
            # 简单去重：如果和上一个动作完全一样，则跳过
            if action_desc != last_action_desc:
                compressed_seq.append(action_desc)
                last_action_desc = action_desc
        
        return compressed_seq

    def get_final_object_states(self):
        """获取最后一帧的物体状态"""
        if not self.metadata['steps']:
            return "No steps recorded."
        
        final_frame = self.metadata['steps'][-1]
        world_objs = final_frame['world_objects']
        
        state_descs = []
        for name, info in world_objs.items():
            # 过滤掉无关物体：只保留跟人有关系的（拿着的）或者在特定家具上的
            # 这里可以根据你的需求调整过滤逻辑
            is_relevant = False
            
            # 条件1: 被人拿着
            if info.get('relation') == 'held_by_human_candidate' or info.get('parent') == 'agent_0':
                is_relevant = True
            
            # 条件2: 它的状态被改变了 (例如 is_clean 从 False 变 True，这里暂时只看静态)
            # 简单策略：记录所有不在 floor 上的物体，或者直接全量记录交给 LLM 筛选（如果物体不多）
            if info.get('parent') != 'floor' and info.get('parent') != 'None':
                is_relevant = True
            
            if is_relevant:
                parent = info.get('parent', 'unknown')
                states = [k for k, v in info.get('states', {}).items() if v]
                state_str = f"{name}: is on/in {parent}"
                if states:
                    state_str += f" ({', '.join(states)})"
                state_descs.append(state_str)
        
        return "\n".join(state_descs)

    def generate_full_prompt(self):
        instruction = self.metadata['instruction']
        # 注意：这里我们暂时用 Mapping 后的 Instruction 兜底，因为解析 evaluation_conditions 需要精确的 JSON 路径
        # 你运行一次后告诉我 val_mini 的具体结构，我可以帮你微调 parse_success_criteria
        success_criteria = self.parse_success_criteria() 
        action_seq = self.get_action_sequence()
        obj_states = self.get_final_object_states()
        
        prompt = f"""
# Context
You are an expert in understanding human-robot collaboration tasks.
The robot is observing a human performing a household task. Based on the observation, infer the human's current intent.

# Task Information
- Instruction: "{instruction}"
- Successful State Criteria: 
{success_criteria}

# Observation Data
- Observed Human Action Sequence: {json.dumps(action_seq, ensure_ascii=False)}
- Current Object States:
{obj_states}

# Reasoning Task
Analyze the gap between the "Current Object States" and the "Successful State Criteria". Combine this with the "Observed Human Action Sequence" to infer the specific sub-task the human is currently performing.

# Output Format (Chain of Thought)
Generate a response in the following format:
1. Observation Analysis: [Summarize actions and states]
2. Goal Gap Analysis: [Compare current vs success criteria]
3. Intent Inference: [Conclude the specific sub-task]
"""
        return prompt

# ================= 使用示例 =================
if __name__ == "__main__":
    # 请根据实际路径修改
    metadata_file = "my_videos/metadata/0.json" 
    val_file = "data/datasets/partnr_episodes/v0_0/val_mini.json.gz" # 或者解压后的 .json
    
    try:
        generator = PromptGenerator(metadata_file, val_dataset_path=val_file)
        prompt_text = generator.generate_full_prompt()
        
        print("------ Generated Prompt ------")
        print(prompt_text)
        print("------------------------------")
        
        # 可以选择保存到文件
        with open("prompt_output.txt", "w", encoding='utf-8') as f:
            f.write(prompt_text)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()