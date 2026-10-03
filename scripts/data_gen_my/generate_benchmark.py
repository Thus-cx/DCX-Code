import json
import os
import glob
from collections import defaultdict

# --- 配置 ---
METADATA_DIR = "/root/autodl-tmp/partnr-planner-main/my_videos/metadata"
OUTPUT_DIR = "/root/autodl-tmp/partnr-planner-main/my_videos/benchmark_dataset_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class BenchmarkGeneratorV2:
    def __init__(self, episode_data, episode_id):
        self.episode_id = episode_id
        # 兼容处理
        if isinstance(episode_data, dict) and "steps" in episode_data:
            self.data = episode_data["steps"]
            self.instruction = episode_data.get("instruction", "")
        else:
            self.data = episode_data if isinstance(episode_data, list) else []
            self.instruction = self.data[0].get("instruction", "") if self.data else ""
            
        self.robot_start_step = self._find_robot_start_step()

    def _find_robot_start_step(self):
        for frame in self.data:
            if frame.get('robot_active', False): return frame['step']
        return None

    def _clean_name(self, name):
        if not name or name in ["None", "Unknown"]: return None
        return name.replace("Oracle", "").replace("Skill", "").replace("Tool", "")

    def segment_subtasks(self):
        """
        【核心改进】：基于 High-Level Action (Task) + Target 的变化来切分子任务
        这能自动适应 Rearrange, Clean, PowerOn 等所有任务类型
        """
        subtasks = []
        current_subtask = None
        
        for i, frame in enumerate(self.data):
            human = frame['human_agent']
            step = frame['step']
            
            # 读取 High-Level Action (例如 OracleRearrangeSkill, OracleCleanSkill)
            # 注意：在 run_data_gen 中你需要确保记录了 'action' (高层) 和 'atomic_action' (原子)
            hl_action = human.get('action', 'Idle') 
            atomic_action = human.get('atomic_action', 'Idle')
            # 使用 high-level action 的 target (通常更稳定)，如果没有则用 atomic 的
            # 这里假设 run_data_gen 记录的 action_target 主要是 atomic 的 target
            # 我们需要一种机制判断当前高层任务的目标。
            # 简化策略：如果 atomic_action 是 Nav/Pick/Interact，它的 target 通常就是子任务 target
            target = human.get('action_target', 'None')
            
            # 过滤掉一些非实质性动作
            if hl_action in ["Idle", "FindObjectTool", "WaitSkill"]:
                continue

            # 定义当前帧的特征签名
            # 如果是 Rearrange，我们更关心操作对象；如果是 Clean，也是操作对象
            task_signature = (hl_action, target)
            
            # 判断是否开启新片段
            if current_subtask is None:
                current_subtask = {
                    "subtask_id": f"{self.episode_id}_sub_{len(subtasks)}",
                    "task_type": self._clean_name(hl_action),
                    "target_object": self._clean_name(target), # 语义化名称 e.g. apple
                    "target_id": target, # 原始ID e.g. apple_0
                    "start_step": step,
                    "actions": []
                }
            
            # 判断是否切换了子任务
            # 规则：高层动作变了 OR (高层动作没变 但 目标对象变了)
            # 注意：Nav 过程中的 target 可能是中间点，需要平滑处理
            # 改进：只在发生关键原子动作（Pick, Open, Toggle, Clean）时确认 Target 变更
            
            is_different_task = False
            if current_subtask['task_type'] != self._clean_name(hl_action):
                is_different_task = True
            elif target != "None" and "apple" in target and target != current_subtask['target_id']:
                # 简单的启发式：如果目标明确变了 (比如从 apple_0 变成 apple_1)，算新任务
                # 但要小心 Nav 过程中的 target 切换
                if "Pick" in atomic_action or "Open" in atomic_action or "Toggle" in atomic_action:
                     is_different_task = True

            if is_different_task:
                current_subtask['end_step'] = self.data[i-1]['step']
                subtasks.append(current_subtask)
                
                # Start new
                current_subtask = {
                    "subtask_id": f"{self.episode_id}_sub_{len(subtasks)}",
                    "task_type": self._clean_name(hl_action),
                    "target_object": self._clean_name(target),
                    "target_id": target,
                    "start_step": step,
                    "actions": []
                }

            # 记录原子动作
            current_subtask['actions'].append({
                "step": step,
                "atomic": self._clean_name(atomic_action),
                "target": self._clean_name(target)
            })
            
        # 最后一个
        if current_subtask:
            current_subtask['end_step'] = self.data[-1]['step']
            subtasks.append(current_subtask)
            
        return subtasks

    def generate(self):
        if self.robot_start_step is None: return None
        
        all_subtasks = self.segment_subtasks()
        
        # 找到 Robot 介入时的子任务
        active_task = None
        for task in all_subtasks:
            if task['start_step'] <= self.robot_start_step <= task['end_step']:
                active_task = task
                break
            # 或者机器人一醒来这个任务还没开始（但在机器人醒后不久开始了）
            if self.robot_start_step < task['start_step']:
                active_task = task
                break
        
        if not active_task: return None
        
        # 截取机器人可见部分
        visible_actions = [a for a in active_task['actions'] if a['step'] >= self.robot_start_step]
        
        # 构建 Prompt
        llm_prompt = {
            "instruction": self.instruction,
            "robot_start_step": self.robot_start_step,
            "visible_events": [f"Human is {a['atomic']} {a['target']}" for a in visible_actions],
            "task_type_hint": active_task['task_type'], # 告诉 LLM 这是 Rearrange 还是 Clean
            "target_hint": active_task['target_object']
        }
        
        return {
            "sample_id": active_task['subtask_id'],
            "llm_prompt": llm_prompt,
            "video_range": [self.robot_start_step, active_task['end_step']]
        }

def process():
    files = glob.glob(os.path.join(METADATA_DIR, "*.json"))
    dataset = []
    for f in files:
        with open(f) as fp: data = json.load(fp)
        ep_id = os.path.basename(f).split('.')[0]
        gen = BenchmarkGeneratorV2(data, ep_id)
        sample = gen.generate()
        if sample: dataset.append(sample)
        
    with open(os.path.join(OUTPUT_DIR, "benchmark_data.json"), "w") as f:
        json.dump(dataset, f, indent=2)

if __name__ == "__main__":
    process()