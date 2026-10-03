import json
import os
import glob
import numpy as np
import random

# --- 配置 ---
METADATA_DIR = "/root/autodl-tmp/partnr-planner-main/my_videos/metadata"
OUTPUT_DIR = "/root/autodl-tmp/partnr-planner-main/my_videos/annotations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class AnnotationGenerator:
    def __init__(self, episode_data, episode_id):
        self.episode_id = episode_id
        
        # [核心修复]：自动识别数据结构 (Dict vs List)
        if isinstance(episode_data, dict) and "steps" in episode_data:
            # 新格式：{"instruction": "...", "steps": [...]}
            self.data = episode_data["steps"]
            self.instruction = episode_data.get("instruction", "Unknown instruction")
        elif isinstance(episode_data, list) and len(episode_data) > 0:
            # 旧格式：[frame1, frame2, ...]
            self.data = episode_data
            # 旧格式指令通常在每一帧里都有
            self.instruction = episode_data[0].get("instruction", "Unknown instruction")
        else:
            print(f"Warning: Unrecognized data format for {episode_id}")
            self.data = []
            self.instruction = "Unknown instruction"

        self.robot_start_step = self._find_robot_start_step()
        
    def _find_robot_start_step(self):
        """找到机器人开始介入的第一帧"""
        for frame in self.data:
            if frame.get('robot_active', False):
                return frame['step']
        return None

    def _clean_action_name(self, raw_name):
        """清洗动作名称"""
        name = raw_name.replace("Oracle", "").replace("Skill", "").replace("Tool", "")
        if name == "Nav": return "Navigating"
        if name == "Pick": return "Picking"
        if name == "Place": return "Placing"
        if name == "FindObject": return "Looking for"
        if name == "Rearrange": return "Rearranging"
        return name

    def _clean_obj_name(self, raw_name):
        """清洗物体名称"""
        if raw_name in ["None", "Unknown", None, ""]:
            return None
        base_name = raw_name.split('_')[0] 
        return f"the {base_name} ({raw_name})"

    def _get_object_state_description(self, frame_idx, obj_unique_name):
        """获取物体状态"""
        if frame_idx >= len(self.data): return ""
        world_objs = self.data[frame_idx].get('world_objects', {})
        if obj_unique_name not in world_objs: return ""
        
        obj_data = world_objs[obj_unique_name]
        states = obj_data.get('states', {})
        
        desc_parts = []
        if states.get('is_clean', False): desc_parts.append("clean")
        elif 'is_clean' in states: desc_parts.append("dirty")
        
        if states.get('is_filled', False): desc_parts.append("filled")
        elif 'is_filled' in states: desc_parts.append("empty")
        
        if states.get('is_powered_on', False): desc_parts.append("powered on")
        
        state_str = ", ".join(desc_parts)
        if state_str:
            return f"The object is {state_str}."
        return ""

    def segment_atomic_actions(self):
        """基于动作流推断 Holding 状态的切分逻辑"""
        if self.robot_start_step is None:
            # 即使没有机器人活动，可能也需要切分，或者返回空取决于你的需求
            # 这里如果不返回数据，process_all 会生成空条目
            pass 
        
        segments = []
        current_segment = None
        
        # --- 状态追踪器 ---
        tracker_held_object = None 
        frame_holding_map = {} 
        
        # 1. 全局状态追踪 (扫描所有步骤)
        for i, frame in enumerate(self.data):
            human_agent = frame['human_agent']
            raw_action = human_agent.get('atomic_action', 'Idle')
            raw_target = human_agent.get('action_target', 'None')
            
            if "Pick" in raw_action and raw_target != "None":
                tracker_held_object = raw_target
            elif "Place" in raw_action:
                tracker_held_object = None
            
            frame_holding_map[i] = tracker_held_object

        # 2. 生成切片 (只处理机器人介入后的部分)
        start_idx = 0
        if self.robot_start_step is not None:
            for i, frame in enumerate(self.data):
                if frame['step'] == self.robot_start_step:
                    start_idx = i
                    break
        else:
            # 如果没有机器人介入数据，可能跳过或从头开始
            # 既然是做机器人视角数据集，如果没有机器人介入，这数据可能无效
            return []

        for i in range(start_idx, len(self.data)):
            frame = self.data[i]
            human_agent = frame['human_agent']
            
            raw_action = human_agent.get('atomic_action', 'Idle')
            raw_target = human_agent.get('action_target', 'None')
            
            current_held_obj = frame_holding_map.get(i, None)
            is_holding = (current_held_obj is not None)
            
            is_same_segment = False
            if current_segment:
                if (current_segment['raw_action'] == raw_action and 
                    current_segment['raw_target'] == raw_target):
                    is_same_segment = True
            
            if is_same_segment:
                current_segment['end_step'] = frame['step']
                current_segment['end_idx'] = i
            else:
                if current_segment:
                    segments.append(current_segment)
                
                clean_target = self._clean_obj_name(raw_target)
                
                current_segment = {
                    "segment_id": len(segments),
                    "start_step": frame['step'],
                    "end_step": frame['step'],
                    "start_idx": i,
                    "end_idx": i,
                    "raw_action": raw_action,
                    "raw_target": raw_target,
                    "clean_action": self._clean_action_name(raw_action),
                    "clean_target": clean_target,
                    "is_holding": is_holding,
                    "held_obj": current_held_obj
                }
        
        if current_segment:
            segments.append(current_segment)
            
        return segments

    def generate_cot_for_segment(self, segment):
        """生成 CoT"""
        start_idx = segment['start_idx']
        
        # 1. 观察层
        target_str = segment['clean_target'] if segment['clean_target'] else "nothing"
        obs_text = f"Observation: I see the human {segment['clean_action']} {target_str}. "
        
        if segment['is_holding'] and segment['held_obj']:
            held_clean = self._clean_obj_name(segment['held_obj'])
            if "Place" not in segment['raw_action']:
                obs_text += f"The human is currently holding {held_clean}. "
        
        # 2. 上下文推理
        reasoning_text = "Reasoning: "
        
        obj_state_desc = ""
        if segment['raw_target'] and segment['raw_target'] != "None":
            obj_state_desc = self._get_object_state_description(start_idx, segment['raw_target'])
            
        if segment['start_step'] == self.robot_start_step:
            reasoning_text += "I have just started observing. "
            if obj_state_desc:
                reasoning_text += f"{obj_state_desc} "
            if segment['is_holding'] and "Pick" not in segment['raw_action']:
                held_name = self._clean_obj_name(segment['held_obj'])
                reasoning_text += f"Since the human is already holding {held_name}, they must have picked it up before I arrived. "
        else:
            reasoning_text += "Following the previous action, "
            if segment['is_holding'] and "Nav" in segment['raw_action']:
                reasoning_text += "the human is transporting the held object. "
            elif not segment['is_holding'] and "Nav" in segment['raw_action']:
                reasoning_text += "the human is moving towards the next target. "

        reasoning_text += f"This aligns with the instruction '{self.instruction}'. "

        # 3. 意图层
        intent_text = "Intent: "
        action = segment['clean_action']
        
        if "Navigating" in action:
            if segment['is_holding']:
                intent_text += f"Transporting {self._clean_obj_name(segment['held_obj'])} to destination."
            else:
                intent_text += f"Moving to pick up {target_str}."
        elif "Picking" in action:
            intent_text += f"Acquiring {target_str}."
        elif "Placing" in action:
            intent_text += f"Placing {target_str} at the target location."
        elif "Looking for" in action:
            intent_text += f"Searching for {target_str}."
        else:
            intent_text += f"Executing {action}."

        return f"{obs_text}\n{reasoning_text}\n{intent_text}"

    def generate_mcq(self, segment):
        """生成单选题"""
        correct_action = segment['clean_action']
        target = segment['clean_target'] if segment['clean_target'] else "something"
        
        if "Navigating" in correct_action and segment['is_holding']:
            correct_answer = f"Transporting {self._clean_obj_name(segment['held_obj'])}"
        else:
            correct_answer = f"{correct_action} {target}"
        
        distractors = [
            "Resting",
            "Cleaning up",
            "Interacting with robot",
            f"Looking for {target}"
        ]
        
        if "Navigating" in correct_action:
            distractors.append(f"Picking up {target}")
        elif "Picking" in correct_action:
            distractors.append(f"Placing {target}")
            
        selected_distractors = random.sample(distractors, 3)
        options = selected_distractors + [correct_answer]
        random.shuffle(options)
        
        labels = ['A', 'B', 'C', 'D']
        correct_idx = options.index(correct_answer)
        
        return {
            "question": "What is the human doing right now?",
            "options": {l: opt for l, opt in zip(labels, options)},
            "answer": labels[correct_idx],
            "answer_text": correct_answer
        }

def process_all():
    json_files = glob.glob(os.path.join(METADATA_DIR, "*.json"))
    print(f"Found {len(json_files)} metadata files.")
    
    final_dataset = []
    
    for fpath in json_files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            if not data: continue
            
            ep_id = os.path.basename(fpath).replace(".json", "")
            
            # 初始化 Generator
            generator = AnnotationGenerator(data, ep_id)
            
            # 获取片段
            segments = generator.segment_atomic_actions()
            
            annotated_segments = []
            for seg in segments:
                cot = generator.generate_cot_for_segment(seg)
                mcq = generator.generate_mcq(seg)
                
                out_seg = {
                    "segment_id": seg['segment_id'],
                    "start_step": seg['start_step'],
                    "end_step": seg['end_step'],
                    "label": seg['label'] if 'label' in seg else f"{seg['clean_action']} {seg['clean_target']}",
                    "chain_of_thought": cot,
                    "mcq": mcq
                }
                annotated_segments.append(out_seg)
            
            if annotated_segments:
                entry = {
                    "episode_id": ep_id,
                    "instruction": generator.instruction,
                    "annotations": annotated_segments
                }
                final_dataset.append(entry)
            
        except Exception as e:
            print(f"Error processing {fpath}: {e}")
            import traceback
            traceback.print_exc()
        
    out_path = os.path.join(OUTPUT_DIR, "final_training_data.json")
    with open(out_path, 'w') as f:
        json.dump(final_dataset, f, indent=2)
    print(f"Done. Saved to {out_path}")

if __name__ == "__main__":
    process_all()