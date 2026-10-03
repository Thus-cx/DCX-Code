import json
import os
import re
import cv2
import base64
import numpy as np
from tqdm import tqdm
from openai import OpenAI
import time

# ================= GPTGod 配置区域 =================
# ⚠️ 修改点 1：替换为你的 GPTGod Token
API_KEY = "sk-nmkly5m2zk2y0l3k4mk2x442k5nonln5nmkly5m2zk2y0l3k"  
BASE_URL = "https://api.gptgod.online/v1"
TEST_MODEL = "gemini-2.5-pro"

# ================= 原始配置区域 =================
BENCHMARK_FILE = "test_dataset_strong/final_benchmark_0302/final_exam_sampled_1200.json"
# ⚠️ 修改点 2：独立的结果文件，避免冲突
RESULT_FILE = f"./{TEST_MODEL}_results_third_raw_nframes_8_GPTGod_Part2.json"
TEST_VIEW = "third_raw"
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
            frame = cv2.resize(frame, (768, 768), interpolation=cv2.INTER_AREA)
            base64_frames.append(encode_image_to_base64(frame))
    cap.release()
    return base64_frames

# ================= 评估逻辑 =================
def extract_json_from_text(text):
    import ast
    if not text or not isinstance(text, str): return None, "Empty input"
    text = text.strip().replace('\xa0', ' ').replace('\u200b', '').replace('\u3000', ' ')
    try: return json.loads(text), None
    except: pass
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match: json_str = match.group(1)
    else:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match: json_str = match.group(1)
        else: return None, "No JSON found"
    try: return json.loads(json_str), None
    except Exception as e: return None, f"Decode Error: {e}"

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
    else: acc_action, acc_target, acc_intent = False, False, False
    sca_reasoning = (acc_area and acc_status and acc_furn and acc_aff and acc_action and acc_target)
    sca_full = (sca_reasoning and acc_obj)
    return {"acc_area": acc_area, "acc_status": acc_status, "acc_obj": acc_obj, "acc_furn": acc_furn, "acc_aff": acc_aff, "acc_action": acc_action, "acc_target": acc_target, "acc_intent": acc_intent, "sca_reasoning": sca_reasoning, "sca_full": sca_full}

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

    # ⚠️ 修改点 3：切除前 201 个，只跑后面的所有样本
    exam_questions = exam_questions[201:]
    print(f"[*] GPTGod 任务启动，共分配剩余的 {len(exam_questions)} 个样本。")

    stats_slice = {k: StatsBucket(k) for k in SLICE_KEYS}
    stats_slice_diff = {f"{s}_{d}": StatsBucket(f"{s}_{d}") for s in SLICE_KEYS for d in DIFFICULTY_KEYS}
    results = []; processed_ids = set()

    if os.path.exists(RESULT_FILE):
        try:
            with open(RESULT_FILE, 'r') as f: results = json.load(f)
            for r in results:
                sample_id = r.get('id') or r['video_paths'].get(TEST_VIEW)
                processed_ids.add(sample_id)
                slice_param = r['slice_param']; difficulty = r.get('difficulty', 'Unknown'); eval_res = r['evaluation']
                if slice_param in stats_slice: stats_slice[slice_param].update(eval_res)
                sd_key = f"{slice_param}_{difficulty}"
                if sd_key in stats_slice_diff: stats_slice_diff[sd_key].update(eval_res)
        except Exception as e: pass

    for item in tqdm(exam_questions):
        video_path = item['video_paths'].get(TEST_VIEW)
        if not video_path or not os.path.exists(video_path): continue
        current_sample_id = item.get('id') or video_path
        if current_sample_id in processed_ids: continue

        slice_param = item['slice_param']; difficulty = item.get('difficulty', 'Unknown'); gt_data = item['ground_truth']
        base64_frames = extract_frames_from_video(video_path, TEST_NFRAMES)
        if not base64_frames: continue

        sys_prompt = item['system_prompt'].replace(
            "Output a SINGLE JSON object exactly matching this structure, answering the 6 progressive questions:",
            "Output a SINGLE JSON object exactly matching this structure, answering the 6 progressive questions, do not output the thinking progress, only output the choices of the 6 questions in json format:")
        
        user_content = [{"type": "text", "text": f"System Instruction:\n{sys_prompt}\n\nUser Task:\n{item['user_prompt']}"}]
        for b64_img in base64_frames: user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}", "detail": "auto"}})

        # output_text = "{}"
        # for attempt in range(3):
        #     try:
        #         response = client.chat.completions.create(
        #             model=TEST_MODEL, messages=[{"role": "user", "content": user_content}],
        #             temperature=0.1, max_tokens=2048, response_format={"type": "json_object"}
        #         )
        #         output_text = response.choices[0].message.content
        #         break
        #     except Exception as e:
        #         if attempt < 2: time.sleep(5 * (attempt + 1))
        output_text = "{}"
        api_success = False  # 新增一个成功标志位

        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=TEST_MODEL, messages=[{"role": "user", "content": user_content}],
                    temperature=0.1, max_tokens=2048, response_format={"type": "json_object"}
                )
                output_text = response.choices[0].message.content
                api_success = True
                break
            except Exception as e:
                print(f"\n[!] ⚠️ 第 {attempt + 1} 次请求失败 | 样本 ID: {current_sample_id} | 错误: {e}")
                if attempt < 2: 
                    time.sleep(5 * (attempt + 1))
                else:
                    print(f"\n[!] ❌ 严重错误: 样本 {current_sample_id} 连续 3 次请求彻底失败！")

        # 核心改动：如果 3 次都失败了，跳过后续解析和保存，不把残缺数据写入文件
        if not api_success:
            print(f"[*] ⏭️ 跳过该样本的保存，以免污染测试集结果。你可以稍后重新运行以重试该样本。\n")
            continue

        parsed_json, error_msg = extract_json_from_text(output_text)
        eval_res = eval_response(parsed_json, gt_data)

        if slice_param in stats_slice: stats_slice[slice_param].update(eval_res)
        sd_key = f"{slice_param}_{difficulty}"
        if sd_key in stats_slice_diff: stats_slice_diff[sd_key].update(eval_res)

        res_entry = item.copy()
        res_entry.update({"model_output": output_text, "parsed_json": parsed_json, "evaluation": eval_res, "parse_error": error_msg})
        for k in ['system_prompt', 'user_prompt']: res_entry.pop(k, None)
        
        results.append(res_entry); processed_ids.add(current_sample_id)

        if len(results) % 5 == 0:
            os.makedirs(os.path.dirname(RESULT_FILE) or '.', exist_ok=True)
            with open(RESULT_FILE, 'w') as f: json.dump(results, f, indent=2)
        time.sleep(2)

    os.makedirs(os.path.dirname(RESULT_FILE) or '.', exist_ok=True)
    with open(RESULT_FILE, 'w') as f: json.dump(results, f, indent=2)
    print(f"\n[*] GPTGod 剩余样本跑批完成！结果保存在: {RESULT_FILE}")

if __name__ == "__main__":
    main()