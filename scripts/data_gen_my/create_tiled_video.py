import json
import torch
import os
import re
import cv2
import numpy as np
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ================= 配置区域 =================
MODEL_PATH = "/root/autodl-tmp/Qwen2-VL-7B-Instruct"  # 根据你的实际路径修改
BENCHMARK_FILE = "test_dataset_strong/final_benchmark_0302/final_exam.json"
TILED_VIDEO_DIR = "test_dataset_strong/final_benchmark_0302/tiled_videos"
RESULT_FILE = "test_dataset_strong/final_benchmark_0302/teacher_results_tiled_nframes_8.json"

# 显存与采样优化 (教师模型同屏多视角，建议 8 帧即可覆盖动作，防止 OOM)
MAX_PIXELS = 768 * 768
TEST_NFRAMES = 8

# 定义统计分组 (严格对应你的切片策略)
SLICE_KEYS = ["transit_30", "transit_60", "transit_90", "pre_transit", "transit_30_60", "transit_60_90", "post_transit"]
DIFFICULTY_KEYS = ["Easy", "Medium", "Hard"]

# ================= 1. 视频拼接与提取工具 =================

def create_tiled_video(global_path, third_path, ego_path, output_path):
    """逐帧读取三个视角的视频，同步拼接为一个品字形三合一网格视频"""
    if os.path.exists(output_path):
        return True

    cap_td = cv2.VideoCapture(global_path)
    cap_tp = cv2.VideoCapture(third_path)
    cap_ego = cv2.VideoCapture(ego_path)
    
    if not (cap_td.isOpened() and cap_tp.isOpened() and cap_ego.isOpened()):
        print(f"\n[Warning] 视频流读取失败，跳过: {output_path}")
        return False

    fps = cap_td.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 10.0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (1024, 1024))
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret_td, frame_td = cap_td.read()
        ret_tp, frame_tp = cap_tp.read()
        ret_ego, frame_ego = cap_ego.read()
        
        if not (ret_td and ret_tp and ret_ego):
            break
            
        frame_td = cv2.resize(frame_td, (512, 512))
        frame_tp = cv2.resize(frame_tp, (512, 512))
        frame_ego = cv2.resize(frame_ego, (512, 512))
        
        canvas = np.zeros((1024, 1024, 3), dtype=np.uint8)
        canvas[0:512, 0:512] = frame_td
        canvas[0:512, 512:1024] = frame_tp
        canvas[512:1024, 256:768] = frame_ego 
        
        cv2.putText(canvas, 'Global View', (20, 40), font, 1, (255, 255, 255), 2)
        cv2.putText(canvas, 'Third-person View', (532, 40), font, 1, (255, 255, 255), 2)
        cv2.putText(canvas, 'Robot Ego-View', (276, 552), font, 1, (255, 255, 255), 2)
        
        out.write(canvas)

    cap_td.release()
    cap_tp.release()
    cap_ego.release()
    out.release()
    return True

def extract_json_robustly(text):
    """最鲁棒的 JSON 提取器：避免正则截断，直接寻找括号对齐"""
    try:
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = text[start_idx : end_idx + 1]
            return json.loads(json_str), None
    except json.JSONDecodeError as e:
        return None, f"JSON Decode Error: {str(e)}"
    return None, "No JSON found"

# ================= 2. 核心评估与统计逻辑 =================

def eval_response(pred_data, gt):
    """核心评估逻辑：包含基础准确率、SCA、CRA 和幻觉追踪"""
    if not pred_data or not isinstance(pred_data, dict):
        return {k: False for k in 
                ["acc_area", "acc_status", "acc_obj", "acc_furn", "acc_aff", "acc_action", "acc_target", "acc_intent", 
                 "sca_reasoning", "sca_full", "is_cra_eligible", "is_cra_success", "is_obj_hallucinated"]}

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

    # 1. 阶梯一致性得分 (SCA)
    sca_reasoning = (acc_area and acc_status and acc_furn and acc_aff and acc_action and acc_target)
    sca_full = (sca_reasoning and acc_obj)

    # 2. CRA 条件推理指标 (前提：区域和物体都识别正确)
    is_cra_eligible = acc_area and acc_obj
    is_cra_success = is_cra_eligible and acc_aff

    # 3. 幻觉率追踪 (如果没猜对物体，且回答的不是 Unknown，则视为编造幻觉)
    is_obj_hallucinated = (not acc_obj) and (str(p_obj).lower().strip() != "unknown")

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
        "sca_full": sca_full,
        "is_cra_eligible": is_cra_eligible,
        "is_cra_success": is_cra_success,
        "is_obj_hallucinated": is_obj_hallucinated
    }


class StatsBucket:
    def __init__(self, name):
        self.name = name
        self.total = 0
        self.metrics = {
            "acc_area": 0, "acc_status": 0, "acc_obj": 0,
            "acc_furn": 0, "acc_aff": 0, "acc_action": 0,
            "acc_intent": 0, "sca_reasoning": 0, "sca_full": 0,
            "cra_eligible_count": 0, "cra_success_count": 0,
            "hallucinated_count": 0
        }

    def update(self, eval_res):
        self.total += 1
        for k in ["acc_area", "acc_status", "acc_obj", "acc_furn", "acc_aff", "acc_action", "acc_intent", "sca_reasoning", "sca_full"]:
            if eval_res.get(k, False):
                self.metrics[k] += 1
        
        if eval_res.get("is_cra_eligible", False):
            self.metrics["cra_eligible_count"] += 1
            if eval_res.get("is_cra_success", False):
                self.metrics["cra_success_count"] += 1
                
        if eval_res.get("is_obj_hallucinated", False):
            self.metrics["hallucinated_count"] += 1

    def get_metrics_dict(self):
        if self.total == 0:
            res = {k: 0.0 for k in self.metrics if not k.endswith("_count")}
            res.update({"cra": 0.0, "hallucination_rate": 0.0, "total": 0})
            return res

        res = {"total": self.total}
        for k in ["acc_area", "acc_status", "acc_obj", "acc_furn", "acc_aff", "acc_action", "acc_intent", "sca_reasoning", "sca_full"]:
            res[k] = (self.metrics[k] / self.total) * 100
            
        res["cra"] = (self.metrics["cra_success_count"] / self.metrics["cra_eligible_count"] * 100) if self.metrics["cra_eligible_count"] > 0 else 0.0
        res["hallucination_rate"] = (self.metrics["hallucinated_count"] / self.total) * 100
        return res

# ================= 3. 主程序与推理循环 =================

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

    # 载入或初始化断点结果
    results = []
    processed_qids = set()
    if os.path.exists(RESULT_FILE):
        with open(RESULT_FILE, 'r') as f:
            results = json.load(f)
            processed_qids = {r['question_id'] for r in results}

    stats_slice = {k: StatsBucket(k) for k in SLICE_KEYS}
    stats_slice_diff = {f"{s}_{d}": StatsBucket(f"{s}_{d}") for s in SLICE_KEYS for d in DIFFICULTY_KEYS}

    print(f"Running Inference on {len(exam_questions)} clips. Target: Teacher Model (Tiled Views)")

    for item in tqdm(exam_questions):
        if item['question_id'] in processed_qids:
            # 恢复统计桶
            eval_res = next((r['evaluation'] for r in results if r['question_id'] == item['question_id']), None)
            if eval_res:
                sp, diff = item['slice_param'], item.get('difficulty', 'Unknown')
                if sp in stats_slice: stats_slice[sp].update(eval_res)
                if f"{sp}_{diff}" in stats_slice_diff: stats_slice_diff[f"{sp}_{diff}"].update(eval_res)
            continue

        # 提取三种视角的原始路径
        paths = item.get('video_paths', {})
        v_global = paths.get('global_ann')
        v_third = paths.get('third_ann')
        v_ego = paths.get('ego_ann')

        if not (v_global and v_third and v_ego):
            continue

        # 生成拼接视频
        tiled_out_path = os.path.join(TILED_VIDEO_DIR, f"{item['question_id']}_tiled.mp4")
        if not create_tiled_video(v_global, v_third, v_ego, tiled_out_path):
            continue

        slice_param = item['slice_param']
        difficulty = item.get('difficulty', 'Unknown')
        gt_data = item['ground_truth']

        # 【核心】：动态替换 Prompt，告知模型正在看多视角拼接视频
        teacher_sys_prompt = item['system_prompt'].replace(
            "analyzing a human-robot collaboration video from a quadruped robot's egocentric perspective. Due to the low-angle view, the human's torso might cause severe physical occlusion.",
            "analyzing a multi-view tiled video containing three synchronized perspectives: Top-left is the Global (Top-down) view, Top-right is the Human Third-person view, and Bottom-center is the Robot's Egocentric view."
        ).replace(
            "Perform a step-by-step diagnostic reasoning to anticipate the human's final intent.",
            "Perform a step-by-step diagnostic reasoning to anticipate the human's final intent, which means predicting human's final target and intent according to the video."
        )

        messages = [
            {"role": "system", "content": teacher_sys_prompt},
            {"role": "user", "content": [
                {"type": "video", "video": tiled_out_path, "max_pixels": MAX_PIXELS, "nframes": TEST_NFRAMES},
                {"type": "text", "text": item['user_prompt']},
            ]}
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)

        # with torch.no_grad():
        #     generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=512, 
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None
            )

        output_text = processor.batch_decode(
            [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)],
            skip_special_tokens=True
        )[0]

        # 解析与评分
        parsed_json, error_msg = extract_json_robustly(output_text)
        eval_res = eval_response(parsed_json, gt_data)

        # 更新统计
        if slice_param in stats_slice:
            stats_slice[slice_param].update(eval_res)
        sd_key = f"{slice_param}_{difficulty}"
        if sd_key in stats_slice_diff:
            stats_slice_diff[sd_key].update(eval_res)

        # 保存明细 (记录完整的超参数 Context，确保数据可溯源)
        res_entry = item.copy()
        res_entry.update({
            "experiment_config": {
                "model_path": MODEL_PATH,
                "test_view": "tiled_multi_view",
                "nframes": TEST_NFRAMES,
                "max_pixels": MAX_PIXELS
            },
            "model_output": output_text,
            "parsed_json": parsed_json,
            "evaluation": eval_res,
            "parse_error": error_msg
        })
        for k in ['system_prompt', 'user_prompt']:
            if k in res_entry: del res_entry[k]
            
        results.append(res_entry)
        processed_qids.add(item['question_id'])

        # 定期保存
        if len(results) % 10 == 0:
            os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
            with open(RESULT_FILE, 'w') as f:
                json.dump(results, f, indent=2)

    with open(RESULT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    # ================= 打印多维诊断分析报表 =================

    print("\n" + "=" * 135)
    print("CoT-HRC TEACHER MODEL DIAGNOSTIC ANALYSIS (View: Tiled Multi-view)")
    print("=" * 135)

    # 扩展了报表宽度，加入了 CRA 和 Hallucination (幻觉率)
    row_fmt = "{:<15} | {:<4} | {:<6} {:<6} {:<6} {:<6} {:<6} | {:<8} | {:<6} | {:<6} | {:<6} | {:<6}"
    header = row_fmt.format("Group", "N", "Area", "Status", "Obj", "Furn", "Aff", "Intent", "SCA_R", "SCA_F", "CRA", "Halluc")

    print("\n[TABLE 1] Slice Analysis (Overall)")
    print("-" * 135)
    print("Note: SCA_R=Reasoning Consistency | CRA=Conditional Reasoning Acc | Halluc=Object Hallucination Rate")
    print("-" * 135)
    print(header)
    print("-" * 135)

    for k in SLICE_KEYS:
        if k in stats_slice:
            res = stats_slice[k].get_metrics_dict()
            if res["total"] > 0:
                print(row_fmt.format(
                    k, res["total"],
                    f"{res['acc_area']:.0f}%", f"{res['acc_status']:.0f}%", f"{res['acc_obj']:.0f}%",
                    f"{res['acc_furn']:.0f}%", f"{res['acc_aff']:.0f}%",
                    f"{res['acc_intent']:.1f}%", f"{res['sca_reasoning']:.1f}%", f"{res['sca_full']:.1f}%",
                    f"{res['cra']:.1f}%", f"{res['hallucination_rate']:.1f}%"
                ))

    print("\n[TABLE 2] Degradation by Difficulty (SCA_Reasoning / CRA)")
    print("-" * 135)
    print(f"{'Slice Param':<15} | {'Easy (N)':<25} | {'Medium (N)':<25} | {'Hard (N)':<25}")
    print("-" * 135)

    for s in SLICE_KEYS:
        row_strs = [f"{s:<15}"]
        for d in DIFFICULTY_KEYS:
            key = f"{s}_{d}"
            if key in stats_slice_diff:
                res = stats_slice_diff[key].get_metrics_dict()
                if res["total"] > 0:
                    metric_str = f"S:{res['sca_reasoning']:.1f}% C:{res['cra']:.0f}% ({res['total']})"
                    row_strs.append(metric_str.ljust(25))
                else:
                    row_strs.append("-".ljust(25))
        if len(row_strs) > 1:
            print(" | ".join(row_strs))

    print("=" * 135)
    print(f"Detailed results saved to {RESULT_FILE}")

if __name__ == "__main__":
    main()