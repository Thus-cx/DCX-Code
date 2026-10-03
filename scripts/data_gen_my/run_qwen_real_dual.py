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

BENCHMARK_FILE = "real_raw_videos/train_dataset/real_final_exam.json"  
# 修改结果保存路径，标明是双视角融合测试
RESULT_FILE = "final_results/qwen_real_results_dual_nframes_8.json"

# 显存优化
MAX_PIXELS = 768 * 768
TEST_NFRAMES = 8  # 注意：双视频相当于输入了 16 帧，显存占用会翻倍，如果 OOM 可以调成 4

# 统计分组
SLICE_KEYS = ["transit_30", "transit_60", "transit_90", "phase_early", "phase_mid", "phase_late"]
DIFFICULTY_KEYS = ["Easy", "Medium", "Hard", "Unknown"]


# ================= 辅助函数 =================

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

    with open(BENCHMARK_FILE, 'r', encoding='utf-8') as f:
        exam_questions = json.load(f)

    stats_slice = {k: StatsBucket(k) for k in SLICE_KEYS}
    stats_slice_diff = {}
    for s in SLICE_KEYS:
        for d in DIFFICULTY_KEYS:
            stats_slice_diff[f"{s}_{d}"] = StatsBucket(f"{s}_{d}")

    results = []

    print(f"Running Inference on {len(exam_questions)} clips.")
    print(f"Mode: Multi-view Fusion (Third-person + Global)")

    for item in tqdm(exam_questions):
        # 1. 同时获取两个视角的视频路径
        video_path_third = item['video_paths'].get('third')
        video_path_global = item['video_paths'].get('global')

        # 只要有一个视角的视频丢失，就跳过该样本以保证对比实验的公平性
        if not video_path_third or not os.path.exists(video_path_third) or \
           not video_path_global or not os.path.exists(video_path_global):
            continue

        slice_param = item['slice_param']
        difficulty = item.get('difficulty', 'Unknown')
        gt_data = item['ground_truth']

        # 2. 修改 Prompt：动态添加提示信息，告知模型现在传入了两个视频
        enhanced_user_prompt = item['user_prompt'] + "\nNote: You are provided with TWO synchronized video views of the exact same action. The first video is the third-person view, and the second video is the global view. Please synthesize information from both views to make your judgment."

        # 3. 核心修改：在 content 列表中连续推入两个视频对象
        messages = [
            {"role": "system", "content": item['system_prompt']},
            {"role": "user", "content": [
                {"type": "video", "video": video_path_third, "max_pixels": MAX_PIXELS, "nframes": TEST_NFRAMES},
                {"type": "video", "video": video_path_global, "max_pixels": MAX_PIXELS, "nframes": TEST_NFRAMES},
                {"type": "text", "text": enhanced_user_prompt},
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
    print(f"CoT-HRC MULTI-VIEW FUSION REPORT (Views: Third + Global)")
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
    print(f"Detailed fusion results saved to {RESULT_FILE}")


if __name__ == "__main__":
    main()