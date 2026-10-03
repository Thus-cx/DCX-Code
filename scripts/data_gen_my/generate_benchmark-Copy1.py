import json
import os
import glob
import numpy as np
import random
from collections import defaultdict

# --- 路径配置 ---
METADATA_DIR = "/root/autodl-tmp/partnr-planner-main/my_videos/metadata"
OUTPUT_DIR = "/root/autodl-tmp/partnr-planner-main/my_videos/benchmark_dataset"  # 新的输出目录，用于存放清洗好的评测数据
os.makedirs(OUTPUT_DIR, exist_ok=True)

class BenchmarkGenerator:
    def __init__(self, episode_data, episode_id):
        self.episode_id = episode_id
        
        # 1. 自动适配数据格式 (Dict/List)
        if isinstance(episode_data, dict) and "steps" in episode_data:
            self.data = episode_data["steps"]
            self.instruction = episode_data.get("instruction", "Unknown instruction")
        elif isinstance(episode_data, list):
            self.data = episode_data
            self.instruction = episode_data[0].get("instruction", "Unknown instruction") if episode_data else ""
        else:
            self.data = []
            self.instruction = ""

        # 2. 找到机器人介入时间
        self.robot_start_step = self._find_robot_start_step()

    def _find_robot_start_step(self):
        """找到机器人 robot_active 首次为 True 的帧"""
        for frame in self.data:
            if frame.get('robot_active', False):
                return frame['step']
        return None

    def _clean_obj_name(self, raw_name):
        """清洗物体名称: candle_0 -> candle"""
        if raw_name in ["None", "Unknown", None, ""]:
            return None
        # 去掉 _数字 后缀，保留语义名称
        return raw_name.split('_')[0]

    def _clean_action_name(self, raw_name):
        """清洗动作名称"""
        name = raw_name.replace("Oracle", "").replace("Skill", "").replace("Tool", "")
        mapping = {
            "Nav": "Walking",
            "Pick": "Picking up",
            "Place": "Placing",
            "FindObject": "Searching for",
            "Rearrange": "Rearranging"
        }
        return mapping.get(name, name)

    def extract_subtasks(self):
        """
        [核心逻辑]：基于全过程上帝视角，切分出所有完整的子任务 (Pick -> Place)
        返回: list of subtask dicts
        """
        subtasks = []
        
        # 状态机变量
        current_subtask = None # {type, object, start_step, actions:[]}
        
        for i, frame in enumerate(self.data):
            human = frame['human_agent']
            raw_action = human.get('atomic_action', '')
            raw_target = human.get('action_target', 'None')
            step_num = frame['step']
            
            # --- 状态机逻辑 ---
            
            # 1. 检测子任务开始：Pick
            if "Pick" in raw_action and current_subtask is None:
                current_subtask = {
                    "subtask_id": f"{self.episode_id}_sub_{len(subtasks)}",
                    "type": "Transport",
                    "target_object_id": raw_target,
                    "target_object_name": self._clean_obj_name(raw_target),
                    "start_step": step_num,
                    "end_step": None, # 尚未结束
                    "actions": []
                }
            
            # 2. 记录动作流
            if current_subtask:
                # 简化：只记录每一帧的关键信息
                action_info = {
                    "step": step_num,
                    "action": self._clean_action_name(raw_action),
                    "target": self._clean_obj_name(raw_target),
                    "target_id": raw_target,
                    # 标记此动作机器人是否可见
                    "robot_visible": (self.robot_start_step is not None and step_num >= self.robot_start_step)
                }
                current_subtask['actions'].append(action_info)
                
                # 3. 检测子任务结束：Place
                # 只有当 Place 的对象和 Pick 的对象一致（或逻辑上连贯）时才闭环
                # 这里简化：只要发生 Place，且当前处于 Pick 状态，就视为该任务结束
                if "Place" in raw_action:
                    current_subtask['end_step'] = step_num
                    # 记录 Place 的容器 (通常 Place 的 target 是桌子/柜子)
                    current_subtask['target_receptacle'] = self._clean_obj_name(raw_target)
                    
                    subtasks.append(current_subtask)
                    current_subtask = None # 重置，准备下一个任务

        return subtasks

    def generate_benchmark_sample(self):
        """
        生成符合要求的评测样本：
        只针对 Robot 介入时正在进行的那个子任务，截取视频片段和 Log
        """
        if self.robot_start_step is None:
            return None # 机器人没开机，跳过

        # 1. 获取所有子任务
        all_subtasks = self.extract_subtasks()
        
        # 2. 定位：机器人介入时，人类正在做哪个子任务？
        # 判定标准：robot_start_step 在 subtask [start, end] 之间
        # 或者 robot_start_step 在 subtask start 之前 (即机器人看着人类开始做)
        active_subtask = None
        
        for task in all_subtasks:
            # 情况 A: 中途介入 (Interrupted Observation)
            if task['start_step'] <= self.robot_start_step <= task['end_step']:
                task['observation_type'] = "partial_cold_start"
                active_subtask = task
                break
            
            # 情况 B: 完整观测 (Full Observation) - 机器人先醒，人类后干活
            # 只要 robot_start 在任务结束前，就算观测到了有效部分
            if self.robot_start_step < task['start_step']:
                task['observation_type'] = "full_observation"
                active_subtask = task
                break
                
        if not active_subtask:
            return None # 机器人醒来时，人类可能在休息，没有进行搬运任务

        # 3. 构建 Prompt 素材 (LLM CoT Generation Context)
        # 将信息分为 "机器人看见的" (Visible) 和 "机器人没看见的" (History)
        
        visible_actions = [a for a in active_subtask['actions'] if a['step'] >= self.robot_start_step]
        unseen_history = [a for a in active_subtask['actions'] if a['step'] < self.robot_start_step]
        
        # 确定“持有状态”推断依据
        # 如果有 unseen history 且其中包含 Pick，说明机器人一睁眼人就拿着东西
        start_status = "not_holding"
        held_object = None
        for act in unseen_history:
            if "Picking" in act['action']:
                start_status = "already_holding"
                held_object = act['target']
                break
        
        # 4. 构建 LLM Prompt 数据结构
        llm_context = {
            "instruction": self.instruction,
            "robot_start_step": self.robot_start_step,
            "observation_type": active_subtask['observation_type'],
            "initial_perception": {
                "status": start_status,
                "held_object_if_any": held_object,
                "inference_logic": "If status is 'already_holding', the model implies a Pick action occurred in unseen history."
            },
            "visual_event_stream": [
                f"Step {a['step']}: Human is {a['action']} {a['target'] if a['target'] else ''}" 
                for a in visible_actions
            ],
            # 仅供 Ground Truth 验证，不作为 Visual Prompt 输入
            "ground_truth_intent": f"Transport {active_subtask['target_object_name']} to {active_subtask.get('target_receptacle', 'destination')}"
        }

        # 5. 构建 MCQ (单选题)
        # 选项 A: 正确答案
        correct_option = llm_context['ground_truth_intent']
        
        # 选项 B/C: 同一 Instruction 下的其他子任务 (Internal Distractors)
        other_tasks = []
        for t in all_subtasks:
            if t != active_subtask:
                desc = f"Transport {t['target_object_name']} to {t.get('target_receptacle', 'destination')}"
                other_tasks.append(desc)
        
        # 选项 D/E: 外部干扰项 (External Distractors - 如果内部不够)
        fallback_distractors = [
            f"Clean the {active_subtask['target_object_name']}",
            f"Find the {active_subtask['target_object_name']}",
            "Resting",
            "Interacting with robot"
        ]
        
        distractors = (other_tasks + fallback_distractors)[:3] # 取前3个
        
        options = [correct_option] + distractors
        random.shuffle(options)
        
        # 加上 "None of the above"
        options.append("None of the above")
        
        # 找到正确选项的 Label (A/B/C/D/E)
        labels = ["A", "B", "C", "D", "E"]
        answer_idx = options.index(correct_option)
        
        mcq_data = {
            "question": "Based on the robot's observation, what subtask is the human currently performing?",
            "options": {labels[i]: opt for i, opt in enumerate(options)},
            "answer": labels[answer_idx],
            "answer_text": correct_option
        }

        # 6. 最终封装
        return {
            "sample_id": active_subtask['subtask_id'],
            "video_range": [self.robot_start_step, active_subtask['end_step']],
            "instruction": self.instruction,
            "llm_generation_prompt": llm_context, # 给 LLM 生成 CoT 用
            "evaluation_data": { # 给评测用
                "mcq": mcq_data,
                "ground_truth_cot_facts": { # 用于给 LLM 生成的 CoT 打分的事实点
                    "seen_pick": (active_subtask['observation_type'] == "full_observation"),
                    "object_name": active_subtask['target_object_name'],
                    "action_sequence": [a['action'] for a in visible_actions]
                }
            }
        }

def process_benchmark_generation():
    json_files = glob.glob(os.path.join(METADATA_DIR, "*.json"))
    print(f"Found {len(json_files)} metadata files.")
    
    benchmark_dataset = []
    
    for fpath in json_files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            
            if not data: continue
            
            ep_id = os.path.basename(fpath).replace(".json", "")
            generator = BenchmarkGenerator(data, ep_id)
            
            sample = generator.generate_benchmark_sample()
            
            if sample:
                benchmark_dataset.append(sample)
                print(f"Generated sample for Episode {ep_id}: {sample['llm_generation_prompt']['ground_truth_intent']}")
            else:
                print(f"Skipped Episode {ep_id}: No active subtask overlap.")
                
        except Exception as e:
            print(f"Error processing {fpath}: {e}")
            import traceback
            traceback.print_exc()
            
    # 保存结果
    out_path = os.path.join(OUTPUT_DIR, "benchmark_v1.json")
    with open(out_path, 'w') as f:
        json.dump(benchmark_dataset, f, indent=2)
    
    print(f"\nBenchmark generation complete. Saved {len(benchmark_dataset)} samples to {out_path}")
    print("Next step: Use 'llm_generation_prompt' field to query GPT-4 for natural language CoT.")

if __name__ == "__main__":
    process_benchmark_generation()