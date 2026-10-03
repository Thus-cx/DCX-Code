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

# 1. [修改] 替换为真实场景的考卷路径
BENCHMARK_FILE = "real_raw_videos/train_dataset/real_final_exam.json"  
RESULT_FILE = "final_results/qwen_real_results_global_nframes_8.json"

# 2. [修改] 真实场景的视角名称没有 _raw 后缀
TEST_VIEW = "global"  # 你的消融实验可以在这里切换 ego / third / global

# 显存优化
MAX_PIXELS = 768 * 768
TEST_NFRAMES = 8

# 3. [修改] 严格对应真实场景代码中生成的 slice_param
SLICE_KEYS = ["transit_30", "transit_60", "transit_90", "phase_early", "phase_mid", "phase_late"]

# 4. [修改] 增加 "Unknown"，因为你的 10 个真实 episode 没有手动打难度标签
DIFFICULTY_KEYS = ["Easy", "Medium", "Hard", "Unknown"]


# ================= 辅助函数 (保持原样) =================

def extract_json_from_text(text):
    """鲁棒的 JSON 提取器"""
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


def eval_response(pred_data, gt):
    """核心评估逻辑：独立准确率与一致性得分 (SCA)"""
    if not pred_data or not isinstance(pred_data, dict):
        return {k: False for k in
                ["acc_area", "acc_status", "acc_obj", "acc_furn", "acc_aff", "acc_action", "acc_target", "acc_intent",
                 "sca_reasoning", "sca_full"]}

    def match(p, g):
        if p is None or g is None: return False
        return str(p).lower().strip() == str(g).lower().strip()

    p_area = pred_data.get("Q1_target_area", "")
    p_status = pred_data.get("Q2_is_holding_object", "")
    p_obj = pred_data.get("Q3_held_object_category", "")
    p_furn = pred_data.get("Q4_target_furniture", "")
    p_aff = pred_data.get("Q5_furniture_affordance", "")
    p_triplet = pred_data.get("Q6_final_intent_triplet", [])

    acc_area = match(p_area, gt.get("target_area"))
    acc_status = match(p_status, gt.get("is_holding_object"))
    acc_obj = match(p_obj, gt.get("held_object_category"))
    acc_furn = match(p_furn, gt.get("target_furniture"))
    acc_aff = match(p_aff, gt.get("furniture_affordance"))

    gt_triplet = gt.get("final_intent_triplet", ["", "", ""])
    if isinstance(p_triplet, list) and len(p_triplet) == 3:
        acc_action = match(p_triplet[0], gt_triplet[0])
        acc_target = match(p_triplet[2], gt_triplet[2])
        acc_intent = acc_action and match(p_triplet[1], gt_triplet[1]) and acc_target
    else:
        acc_action, acc_target, acc_intent = False, False, False

    sca_reasoning = (acc_area and acc_status and acc_furn and acc_aff and acc_action and acc_target)
    sca_full = (sca_reasoning and acc_obj)

    return {
        "acc_area": acc_area, "acc_status": acc_status, "acc_obj": acc_obj,
        "acc_furn": acc_furn, "acc_aff": acc_aff, "acc_action": acc_action,
        "acc_target": acc_target, "acc_intent": acc_intent,
        "sca_reasoning": sca_reasoning, "sca_full": sca_full
    }


class StatsBucket:
    def __init__(self, name):
        self.name = name
        self.total = 0
        self.metrics = {
            "acc_area": 0, "acc_status": 0, "acc_obj": 0,
            "acc_furn": 0, "acc_aff": 0, "acc_action": 0,
            "acc_intent": 0, "sca_reasoning": 0, "sca_full": 0
        }

    def update(self, eval_res):
        self.total += 1
        for k in self.metrics:
            if eval_res.get(k, False):
                self.metrics[k] += 1

    def get_metrics_dict(self):
        if self.total == 0:
            res = {k: 0.0 for k in self.metrics}
            res["total"] = 0
            return res

        res = {"total": self.total}
        for k in self.metrics:
            res[k] = (self.metrics[k] / self.total) * 100
        return res


# ================= 主程序 (保持核心推理逻辑不变) =================

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

    with open(BENCHMARK_FILE, 'r', encoding='utf-8') as f:
        exam_questions = json.load(f)

    stats_slice = {k: StatsBucket(k) for k in SLICE_KEYS}
    stats_slice_diff = {}
    for s in SLICE_KEYS:
        for d in DIFFICULTY_KEYS:
            stats_slice_diff[f"{s}_{d}"] = StatsBucket(f"{s}_{d}")

    results = []
    print(f"Running Inference on {len(exam_questions)} clips. View: {TEST_VIEW}")

    for item in tqdm(exam_questions):
        video_path = item['video_paths'].get(TEST_VIEW)
        if not video_path or not os.path.exists(video_path):
            continue

        slice_param = item['slice_param']
        difficulty = item.get('difficulty', 'Unknown')
        gt_data = item['ground_truth']

        messages = [
            {"role": "system", "content": item['system_prompt']},
            {"role": "user", "content": [
                {"type": "video", "video": video_path, "max_pixels": MAX_PIXELS, "nframes": TEST_NFRAMES},
                {"type": "text", "text": item['user_prompt']},
            ]}
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(
            "cuda")

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)

        output_text = processor.batch_decode(
            [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)],
            skip_special_tokens=True
        )[0]

        parsed_json, error_msg = extract_json_from_text(output_text)
        eval_res = eval_response(parsed_json, gt_data)

        if slice_param in stats_slice:
            stats_slice[slice_param].update(eval_res)

        sd_key = f"{slice_param}_{difficulty}"
        if sd_key in stats_slice_diff:
            stats_slice_diff[sd_key].update(eval_res)

        res_entry = item.copy()
        res_entry.update({
            "model_output": output_text,
            "parsed_json": parsed_json,
            "evaluation": eval_res,
            "parse_error": error_msg
        })
        for k in ['system_prompt', 'user_prompt']:
            if k in res_entry: del res_entry[k]
        results.append(res_entry)

        if len(results) % 20 == 0:
            os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
            with open(RESULT_FILE, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    with open(RESULT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ================= 打印多维诊断分析报表 =================

    print("\n" + "=" * 120)
    print(f"CoT-HRC REAL-WORLD DIAGNOSTIC ANALYSIS REPORT (View: {TEST_VIEW})")
    print("=" * 120)

    row_fmt = "{:<15} | {:<4} | {:<6} {:<6} {:<6} {:<6} {:<6} | {:<8} | {:<8} | {:<8}"
    header = row_fmt.format("Group", "N", "Area", "Status", "Obj", "Furn", "Aff", "Intent", "SCA_R", "SCA_F")

    print("\n[TABLE 1] Slice Analysis (Overall)")
    print("-" * 120)
    print("Note: SCA_R = Reasoning Consistency | SCA_F = Full Consistency")
    print("-" * 120)
    print(header)
    print("-" * 120)

    for k in SLICE_KEYS:
        if k in stats_slice:
            res = stats_slice[k].get_metrics_dict()
            if res["total"] > 0:
                print(row_fmt.format(
                    k, res["total"],
                    f"{res['acc_area']:.0f}%", f"{res['acc_status']:.0f}%", f"{res['acc_obj']:.0f}%",
                    f"{res['acc_furn']:.0f}%", f"{res['acc_aff']:.0f}%",
                    f"{res['acc_intent']:.1f}%", f"{res['sca_reasoning']:.1f}%", f"{res['sca_full']:.1f}%"
                ))

    print("\n[TABLE 2] Degradation by Difficulty (SCA_Reasoning Focus)")
    print("-" * 120)
    print(f"{'Slice Param':<15} | {'Easy (N)':<15} | {'Medium (N)':<15} | {'Hard (N)':<15} | {'Unknown (N)':<15}")
    print("-" * 120)

    for s in SLICE_KEYS:
        row_strs = [f"{s:<15}"]
        for d in DIFFICULTY_KEYS:
            key = f"{s}_{d}"
            if key in stats_slice_diff:
                res = stats_slice_diff[key].get_metrics_dict()
                if res["total"] > 0:
                    row_strs.append(f"{res['sca_reasoning']:.1f}% ({res['total']})".ljust(15))
                else:
                    row_strs.append("-".ljust(15))
        if len(row_strs) > 1:
            print(" | ".join(row_strs))

    print("=" * 120)
    print(f"Detailed results saved to {RESULT_FILE}")


if __name__ == "__main__":
    main()