import json
import torch
import os
import re
import numpy as np
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ================= 配置区域 =================
MODEL_PATH = "../Qwen2-VL-7B-Instruct" 
BENCHMARK_FILE = "test_my_videos_2/final_benchmark_rovi/rovi_final_exam.json"
RESULT_FILE = "results/rovi_results_multidim_analysis.json"
TEST_VIEW = "ego_raw" # 可选: 'ego_raw', 'third_raw', 'global'

# 显存优化
MAX_PIXELS = 768 * 768 
FPS = 2.0 

# 定义统计分组
SLICE_KEYS = ["p20", "p40", "p60", "p80", "init", "transit", "converge"]
DIFFICULTY_KEYS = ["Easy", "Medium", "Hard", "Unknown"]
INTENT_KEYS = ["Placement", "Storage", "Maintenance", "Disposal"]

# ================= 辅助函数 =================

def fix_ground_truth_logic(item):
    """
    根据 Instruction 强制修正 GT Intent。
    """
    instruction = item['instruction'].lower()
    original_intent = item['ground_truth']['intent_category']
    original_target = item['ground_truth']['target_category']

    # 1. 强制修正 Maintenance
    if any(v in instruction for v in ["clean", "wash", "fill", "pour", "scrub"]):
        return "Maintenance", original_target

    # 2. 强制修正 Disposal
    if any(v in instruction for v in ["throw", "discard", "trash", "dispose"]):
        return "Disposal", original_target

    return original_intent, original_target

def extract_json_from_text(text):
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            json_str = match.group(0)
        else:
            return None, "No JSON found"

    try:
        data = json.loads(json_str)
        return data, None
    except json.JSONDecodeError as e:
        return None, f"JSON Decode Error: {str(e)}"

def eval_response(pred_data, gt_intent, gt_target):
    if not pred_data:
        return {"intent_correct": False, "target_hit": False, "joint_success": False, "error": "Parse Failed"}
    
    # 1. 意图评估
    pred_intent = pred_data.get("intent_category", "").lower()
    intent_correct = (pred_intent == gt_intent.lower())
    
    # 2. 目标评估 (Top-3 Recall)
    pred_targets = [t.lower() for t in pred_data.get("predicted_target", [])]
    target_hit = False
    gt_target_lower = gt_target.lower()
    
    for pt in pred_targets:
        if gt_target_lower in pt or pt in gt_target_lower:
            target_hit = True
            break
            
    return {
        "intent_correct": intent_correct,
        "target_hit": target_hit,
        "joint_success": (intent_correct and target_hit)
    }

class StatsBucket:
    def __init__(self, name):
        self.name = name
        self.total = 0
        self.metrics = {"intent_correct": 0, "target_hit": 0, "joint_success": 0}
        self.by_category = {} # 用于计算 Macro

    def update(self, eval_res, intent_cat):
        self.total += 1
        if eval_res["intent_correct"]: self.metrics["intent_correct"] += 1
        if eval_res["target_hit"]: self.metrics["target_hit"] += 1
        if eval_res["joint_success"]: self.metrics["joint_success"] += 1
        
        if intent_cat not in self.by_category:
            self.by_category[intent_cat] = {"total": 0, "joint": 0}
        self.by_category[intent_cat]["total"] += 1
        if eval_res["joint_success"]: self.by_category[intent_cat]["joint"] += 1

    def get_metrics(self):
        if self.total == 0: return (0, 0.0, 0.0, 0.0, 0.0)
        sr = (self.metrics["joint_success"] / self.total) * 100
        acc = (self.metrics["intent_correct"] / self.total) * 100
        rec = (self.metrics["target_hit"] / self.total) * 100
        
        # Macro SR
        cat_srs = [v["joint"]/v["total"] for v in self.by_category.values()]
        macro_sr = (sum(cat_srs)/len(cat_srs)*100) if cat_srs else 0.0
        
        return (self.total, sr, acc, rec, macro_sr)

# ================= 主程序 =================

def main():
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto"
        )
    except:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
        )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    with open(BENCHMARK_FILE, 'r') as f:
        exam_questions = json.load(f)

    # === 初始化多维统计桶 ===
    # 1. Slice Stats (原始)
    stats_slice = {k: StatsBucket(k) for k in SLICE_KEYS}
    
    # 2. Slice x Difficulty Stats (新增)
    # key格式: "p20_Easy", "transit_Hard"
    stats_slice_diff = {}
    for s in SLICE_KEYS:
        for d in DIFFICULTY_KEYS:
            stats_slice_diff[f"{s}_{d}"] = StatsBucket(f"{s}_{d}")
            
    # 3. Intent x Difficulty Stats (新增)
    # key格式: "Storage_Easy", "Placement_Hard"
    stats_intent_diff = {}
    for i in INTENT_KEYS:
        for d in DIFFICULTY_KEYS:
            stats_intent_diff[f"{i}_{d}"] = StatsBucket(f"{i}_{d}")

    results = []
    
    print(f"Running Inference on {len(exam_questions)} clips. View: {TEST_VIEW}")

    for item in tqdm(exam_questions):
        video_path = item['video_paths'].get(TEST_VIEW)
        if not video_path or not os.path.exists(video_path):
            continue

        # 1. 获取关键属性
        fixed_intent, fixed_target = fix_ground_truth_logic(item)
        slice_param = item['slice_param']
        difficulty = item.get('difficulty', 'Unknown')
        
        # 2. 构造推理
        messages = [
            {"role": "system", "content": item['system_prompt']},
            {"role": "user", "content": [
                {"type": "video", "video": video_path, "max_pixels": MAX_PIXELS, "fps": FPS},
                {"type": "text", "text": item['user_prompt']},
            ]}
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)

        output_text = processor.batch_decode(
            [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)], 
            skip_special_tokens=True
        )[0]

        # 3. 解析与评分
        parsed_json, error_msg = extract_json_from_text(output_text)
        eval_res = eval_response(parsed_json, fixed_intent, fixed_target)

        # 4. === 更新多维统计 ===
        if not eval_res.get('error'):
            # (1) Slice Only
            if slice_param in stats_slice:
                stats_slice[slice_param].update(eval_res, fixed_intent)
            
            # (2) Slice x Difficulty
            sd_key = f"{slice_param}_{difficulty}"
            if sd_key in stats_slice_diff:
                stats_slice_diff[sd_key].update(eval_res, fixed_intent)
                
            # (3) Intent x Difficulty
            id_key = f"{fixed_intent}_{difficulty}"
            if id_key in stats_intent_diff:
                stats_intent_diff[id_key].update(eval_res, fixed_intent)

        # 5. 保存结果
        res_entry = item.copy()
        res_entry.update({
            "fixed_gt_intent": fixed_intent,
            "fixed_gt_target": fixed_target,
            "model_output": output_text,
            "parsed_json": parsed_json,
            "evaluation": eval_res
        })
        for k in ['system_prompt', 'user_prompt', 'candidate_list']: 
            if k in res_entry: del res_entry[k]
        results.append(res_entry)

        if len(results) % 20 == 0:
            with open(RESULT_FILE, 'w') as f:
                json.dump(results, f, indent=2)

    with open(RESULT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    # ================= 打印多维分析报表 =================
    
    print("\n" + "="*100)
    print(f"ROVI MULTI-DIMENSIONAL ANALYSIS (View: {TEST_VIEW})")
    print("="*100)
    
    # 表头模板
    row_fmt = "{:<15} | {:<5} | {:<10} | {:<10} | {:<10} | {:<10}"
    header = row_fmt.format("Group", "N", "Joint SR", "Int Acc", "Tgt Rec", "Macro SR")
    
    # --- TABLE 1: Slice Analysis (Original) ---
    print("\n[TABLE 1] Slice Analysis (Overall)")
    print("-" * 100)
    print(header)
    print("-" * 100)
    for k in SLICE_KEYS:
        if k in stats_slice:
            n, sr, acc, rec, macro = stats_slice[k].get_metrics()
            print(row_fmt.format(k, n, f"{sr:.1f}%", f"{acc:.1f}%", f"{rec:.1f}%", f"{macro:.1f}%"))

    # --- TABLE 2: Slice x Difficulty Analysis ---
    print("\n[TABLE 2] Slice x Difficulty Analysis")
    print("-" * 100)
    print(f"{'Slice':<10} | {'Diff':<8} | {'N':<5} | {'Joint SR':<10} | {'Int Acc':<10} | {'Tgt Rec':<10}")
    print("-" * 100)
    
    for s in SLICE_KEYS:
        # 只打印 Easy, Medium, Hard (忽略 Unknown 以节省版面)
        for d in ["Easy", "Medium", "Hard"]:
            key = f"{s}_{d}"
            if key in stats_slice_diff:
                n, sr, acc, rec, _ = stats_slice_diff[key].get_metrics()
                if n > 0: # 只打印有数据的行
                    print(f"{s:<10} | {d:<8} | {n:<5} | {sr:.1f}%      | {acc:.1f}%      | {rec:.1f}%")
        print("-" * 65) # 分隔线

    # --- TABLE 3: Intent x Difficulty Analysis ---
    print("\n[TABLE 3] Intent x Difficulty Analysis")
    print("-" * 100)
    print(f"{'Intent':<15} | {'Diff':<8} | {'N':<5} | {'Joint SR':<10} | {'Int Acc':<10} | {'Tgt Rec':<10}")
    print("-" * 100)
    
    for i in INTENT_KEYS:
        for d in ["Easy", "Medium", "Hard"]:
            key = f"{i}_{d}"
            if key in stats_intent_diff:
                n, sr, acc, rec, _ = stats_intent_diff[key].get_metrics()
                if n > 0:
                    print(f"{i:<15} | {d:<8} | {n:<5} | {sr:.1f}%      | {acc:.1f}%      | {rec:.1f}%")
        print("-" * 70)

    print("="*100)
    print(f"Detailed results saved to {RESULT_FILE}")

if __name__ == "__main__":
    main()