import json
import os
import re
import cv2
import base64
import numpy as np
from tqdm import tqdm
from openai import OpenAI

# ================= API 配置区域 =================
API_KEY = "fk240582-3VHuaYldm3YOi5bc4eB25uTa6bfKLTjT"
BASE_URL = "https://oa.api2d.net"  # API2D 的官方请求接口

# 切换你想测试的模型： "gpt-4o" 或 "gemini-1.5-pro"
TEST_MODEL = "gpt-4o" 

# ================= 原始配置区域 =================
BENCHMARK_FILE = "test_dataset_strong/final_benchmark_0302/final_exam_sampled_1200.json"  

# 🚨 修改点 1：改了输出文件名，千万别覆盖了你之前跑的数据！
RESULT_FILE = f"final_results/{TEST_MODEL}_results_oracle_dual_nframes_8_part_570_951.json"

TEST_VIEW = "oracle_dual"  
TEST_NFRAMES = 8

SLICE_KEYS = ["transit_30", "transit_60", "transit_90", "pre_transit", "transit_30_60", "transit_60_90", "post_transit"]
DIFFICULTY_KEYS = ["Easy", "Medium", "Hard"]

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= 视频抽帧辅助函数 =================
def encode_image_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def extract_frames_from_video(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0: return []
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    base64_frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            base64_frames.append(encode_image_to_base64(frame))
    cap.release()
    return base64_frames

# ================= 评估逻辑 =================
def extract_json_from_text(text):
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match: return json.loads(match.group(1)), None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match: return json.loads(match.group(0)), None
    return None, "No JSON found"

def eval_response(pred_data, gt):
    if not pred_data or not isinstance(pred_data, dict):
        return {k: False for k in ["acc_area", "acc_status", "acc_obj", "acc_furn", "acc_aff", "acc_action", "acc_target", "acc_intent", "sca_reasoning", "sca_full"]}
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
        self.name = name; self.total = 0
        self.metrics = {"acc_area": 0, "acc_status": 0, "acc_obj": 0, "acc_furn": 0, "acc_aff": 0, "acc_action": 0, "acc_intent": 0, "sca_reasoning": 0, "sca_full": 0}
    def update(self, eval_res):
        self.total += 1
        for k in self.metrics:
            if eval_res.get(k, False): self.metrics[k] += 1
    def get_metrics_dict(self):
        if self.total == 0:
            res = {k: 0.0 for k in self.metrics}; res["total"] = 0; return res
        res = {"total": self.total}
        for k in self.metrics: res[k] = (self.metrics[k] / self.total) * 100
        return res

# ================= 主程序 =================

def main():
    with open(BENCHMARK_FILE, 'r') as f:
        exam_questions = json.load(f)

    # 🚨 修改点 2：在送入循环前，严格切片提取你想要的范围
    # Python 索引从 0 开始，且左闭右开。 [570:952] 恰好包含索引 570 到 951 对应的 382 个样本。
    target_questions = exam_questions[570:952]

    stats_slice = {k: StatsBucket(k) for k in SLICE_KEYS}
    stats_slice_diff = {}
    for s in SLICE_KEYS:
        for d in DIFFICULTY_KEYS:
            stats_slice_diff[f"{s}_{d}"] = StatsBucket(f"{s}_{d}")

    results = []

    print(f"Running API Inference using model: [{TEST_MODEL}] on {len(target_questions)} clips (Index 570-951). View: {TEST_VIEW}")

    # 🚨 修改点 3：让 tqdm 遍历切片后的 target_questions
    for item in tqdm(target_questions):
        video_path = item['video_paths'].get(TEST_VIEW)
        if not video_path or not os.path.exists(video_path):
            continue

        slice_param = item['slice_param']
        difficulty = item.get('difficulty', 'Unknown')
        gt_data = item['ground_truth']

        base64_frames = extract_frames_from_video(video_path, TEST_NFRAMES)
        if not base64_frames:
            continue

        user_content = [{"type": "text", "text": item['user_prompt']}]
        for b64_img in base64_frames:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_img}", "detail": "low"}
            })

        messages = [
            {"role": "system", "content": item['system_prompt']},
            {"role": "user", "content": user_content}
        ]

        output_text = ""
        try:
            response = client.chat.completions.create(
                model=TEST_MODEL,
                messages=messages,
                temperature=0.1,  
                max_tokens=512,
                response_format={"type": "json_object"} 
            )
            output_text = response.choices[0].message.content
        except Exception as e:
            print(f"\nAPI Error on sample {item.get('id', 'unknown')}: {e}")
            output_text = "{}" 

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

        if len(results) % 10 == 0:
            os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
            with open(RESULT_FILE, 'w') as f:
                json.dump(results, f, indent=2)

    with open(RESULT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    # 打印报表
    print("\n" + "=" * 120)
    print(f"REPORT FOR PARTIAL RUN (Index 570-951) - Model: {TEST_MODEL} | View: {TEST_VIEW}")
    print("=" * 120)

    row_fmt = "{:<15} | {:<4} | {:<6} {:<6} {:<6} {:<6} {:<6} | {:<8} | {:<8} | {:<8}"
    header = row_fmt.format("Group", "N", "Area", "Status", "Obj", "Furn", "Aff", "Intent", "SCA_R", "SCA_F")

    print("\n[TABLE 1] Slice Analysis (Overall)")
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

    print("=" * 120)
    print(f"Partial results saved to {RESULT_FILE}")

if __name__ == "__main__":
    main()