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
BENCHMARK_FILE = "test_dataset/final_benchmark_0302/final_exam.json"  # 请确认你的考题路径
RESULT_FILE = "final_results/qwen_results_ego_raw_nframes_16.json"
TEST_VIEW = "ego_raw"  # 你的消融实验可以在这里切换 ego_raw / ego_ann / third_raw

# 显存优化
MAX_PIXELS = 768 * 768
# FPS = 2.0
TEST_NFRAMES = 16

# 定义新的统计分组 (严格对应你的切片策略)
SLICE_KEYS = ["transit_30", "transit_60", "transit_90", "pre_transit", "transit_30_60", "transit_60_90", "post_transit"]
DIFFICULTY_KEYS = ["Easy", "Medium", "Hard"]


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
    """
    核心评估逻辑：严格对应论文 5.2 节的独立准确率与一致性得分 (SCA)
    """
    # 如果输出不是有效字典，全判错
    if not pred_data or not isinstance(pred_data, dict):
        return {k: False for k in
                ["acc_area", "acc_status", "acc_obj", "acc_furn", "acc_aff", "acc_action", "acc_target", "acc_intent",
                 "sca_reasoning", "sca_full"]}

    # 字符串清洗与匹配函数
    def match(p, g):
        if p is None or g is None: return False
        return str(p).lower().strip() == str(g).lower().strip()

    # 1. 提取预测结果
    p_area = pred_data.get("Q1_target_area", "")
    p_status = pred_data.get("Q2_is_holding_object", "")
    p_obj = pred_data.get("Q3_held_object_category", "")
    p_furn = pred_data.get("Q4_target_furniture", "")
    p_aff = pred_data.get("Q5_furniture_affordance", "")
    p_triplet = pred_data.get("Q6_final_intent_triplet", [])

    # 2. 计算独立节点准确率 (Independent Node Accuracy)
    acc_area = match(p_area, gt.get("target_area"))
    acc_status = match(p_status, gt.get("is_holding_object"))
    acc_obj = match(p_obj, gt.get("held_object_category"))
    acc_furn = match(p_furn, gt.get("target_furniture"))
    acc_aff = match(p_aff, gt.get("furniture_affordance"))

    # 解析三元组 [Action, Object, Target]
    gt_triplet = gt.get("final_intent_triplet", ["", "", ""])
    if isinstance(p_triplet, list) and len(p_triplet) == 3:
        acc_action = match(p_triplet[0], gt_triplet[0])
        acc_target = match(p_triplet[2], gt_triplet[2])
        # 完整三元组正确要求：Action对、Object对、Target对
        acc_intent = acc_action and match(p_triplet[1], gt_triplet[1]) and acc_target
    else:
        acc_action, acc_target, acc_intent = False, False, False

    # 3. 计算阶梯一致性得分 (SCA)
    # 逻辑溯因一致性 (不包含极难看清的 held_object)
    sca_reasoning = (acc_area and acc_status and acc_furn and acc_aff and acc_action and acc_target)

    # 全链路严格一致性 (必须连手里拿的物体也看清且猜对)
    sca_full = (sca_reasoning and acc_obj)

    return {
        "acc_area": acc_area,
        "acc_status": acc_status,
        "acc_obj": acc_obj,
        "acc_furn": acc_furn,
        "acc_aff": acc_aff,
        "acc_action": acc_action,
        "acc_target": acc_target,
        "acc_intent": acc_intent,
        "sca_reasoning": sca_reasoning,
        "sca_full": sca_full
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

    with open(BENCHMARK_FILE, 'r') as f:
        exam_questions = json.load(f)

    # === 初始化多维统计桶 ===
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

        # 构造推理 Prompt
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

        # 解析与严格评分
        parsed_json, error_msg = extract_json_from_text(output_text)
        eval_res = eval_response(parsed_json, gt_data)

        # 更新统计
        if slice_param in stats_slice:
            stats_slice[slice_param].update(eval_res)

        sd_key = f"{slice_param}_{difficulty}"
        if sd_key in stats_slice_diff:
            stats_slice_diff[sd_key].update(eval_res)

        # 保存明细
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

        # 定期保存
        if len(results) % 20 == 0:
            os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
            with open(RESULT_FILE, 'w') as f:
                json.dump(results, f, indent=2)

    with open(RESULT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    # ================= 打印多维诊断分析报表 =================

    print("\n" + "=" * 120)
    print(f"CoT-HRC DIAGNOSTIC ANALYSIS REPORT (View: {TEST_VIEW})")
    print("=" * 120)

    # 格式化打印宽度
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
    print(f"{'Slice Param':<15} | {'Easy (N)':<15} | {'Medium (N)':<15} | {'Hard (N)':<15}")
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