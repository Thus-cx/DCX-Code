import json
import os
import re
import cv2
import base64
import numpy as np
from tqdm import tqdm
from openai import OpenAI
import time

# ================= API 配置区域 =================
API_KEY = "fk240582-3VHuaYldm3YOi5bc4eB25uTa6bfKLTjT"  # 你的 API2D Forward Key
# ⚠️ 核心修改 1：API2D 必须加上 /v1 后缀
BASE_URL = "https://oa.api2d.net/v1"

# 切换你想测试的模型： "gpt-4o" 或 "gemini-2.5-pro-preview-05-06"
TEST_MODEL = "gemini-2.5-pro-preview-05-06"

# ================= 原始配置区域 =================
BENCHMARK_FILE = "test_dataset_strong/final_benchmark_0302/final_exam_sampled_1200.json"
RESULT_FILE = f"./{TEST_MODEL}_results_third_raw_201.json"
TEST_VIEW = "third_raw"
TEST_NFRAMES = 8

SLICE_KEYS = ["transit_30", "transit_60", "transit_90", "pre_transit", "transit_30_60", "transit_60_90", "post_transit"]
DIFFICULTY_KEYS = ["Easy", "Medium", "Hard"]

# ⚠️ 核心修改 2：使用极简的 OpenAI 客户端初始化，它会自动处理 fk 的鉴权
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ================= 视频抽帧辅助函数 =================

def encode_image_to_base64(img):
    """将 OpenCV 图像矩阵转换为 Base64 字符串"""
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')


def extract_frames_from_video(video_path, num_frames):
    """均匀抽取指定数量的视频帧并返回 Base64 列表"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        return []

    # 均匀采样索引
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    base64_frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # VLM 通常不需要太高的分辨率，这里可以选择性 resize 降低 Token 成本
            frame = cv2.resize(frame, (768, 768), interpolation=cv2.INTER_AREA)
            base64_frames.append(encode_image_to_base64(frame))

    cap.release()
    return base64_frames


# ================= 评估逻辑 (保留原版) =================

def extract_json_from_text(text):
    """极其鲁棒的 JSON 提取器（带有不可见字符清洗与多种格式容错）"""
    import ast

    if not text or not isinstance(text, str):
        return None, "Empty or invalid text input"

    text = text.strip()
    text = text.replace('\xa0', ' ').replace('\u200b', '').replace('\u3000', ' ')

    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        json_str = match.group(1)
    else:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            return None, "No JSON found in text"

    try:
        data = json.loads(json_str)
        return data, None
    except json.JSONDecodeError as e:
        try:
            data = ast.literal_eval(json_str)
            if isinstance(data, dict):
                return data, None
        except Exception:
            pass
        return None, f"JSON Decode Error: {str(e)}"


def eval_response(pred_data, gt):
    """核心评估逻辑：严格对应独立准确率与一致性得分 (SCA)"""
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
    with open(BENCHMARK_FILE, 'r') as f:
        exam_questions = json.load(f)

    stats_slice = {k: StatsBucket(k) for k in SLICE_KEYS}
    stats_slice_diff = {}
    for s in SLICE_KEYS:
        for d in DIFFICULTY_KEYS:
            stats_slice_diff[f"{s}_{d}"] = StatsBucket(f"{s}_{d}")

    results = []
    processed_ids = set()

    # 断点续传逻辑
    if os.path.exists(RESULT_FILE):
        print(f"[*] 发现历史运行记录 '{RESULT_FILE}'，正在加载断点...")
        try:
            with open(RESULT_FILE, 'r') as f:
                results = json.load(f)

            for r in results:
                sample_id = r.get('id') or r['video_paths'].get(TEST_VIEW)
                processed_ids.add(sample_id)
                slice_param = r['slice_param']
                difficulty = r.get('difficulty', 'Unknown')
                eval_res = r['evaluation']

                if slice_param in stats_slice:
                    stats_slice[slice_param].update(eval_res)
                sd_key = f"{slice_param}_{difficulty}"
                if sd_key in stats_slice_diff:
                    stats_slice_diff[sd_key].update(eval_res)

            print(f"[*] 成功恢复了 {len(results)} 个已处理样本的数据，即将继续评估剩余样本。")
        except Exception as e:
            print(f"[!] 警告: 读取历史记录失败 ({e})，将从头开始。")

    print(f"\nRunning API Inference using model: [{TEST_MODEL}] on {len(exam_questions)} clips. View: {TEST_VIEW}")

    for item in tqdm(exam_questions):
        video_path = item['video_paths'].get(TEST_VIEW)
        if not video_path or not os.path.exists(video_path):
            continue

        current_sample_id = item.get('id') or video_path
        if current_sample_id in processed_ids:
            continue

        slice_param = item['slice_param']
        difficulty = item.get('difficulty', 'Unknown')
        gt_data = item['ground_truth']

        # 1. 抽取视频帧
        base64_frames = extract_frames_from_video(video_path, TEST_NFRAMES)
        if not base64_frames:
            continue

        # 2. 构造适合 OpenAI API 的多模态 Prompt
        system_prompt_processed = item['system_prompt'].replace(
            "You are an embodied AI assistant analyzing a human-robot collaboration video from a quadruped robot's egocentric perspective.\nDue to the low-angle view, the human's torso might cause severe physical occlusion.\n\n**Task:**\nPerform a step-by-step diagnostic reasoning to anticipate the human's final intent.\nOutput a SINGLE JSON object exactly matching this structure, answering the 6 progressive questions:",
            "You are an embodied AI assistant analyzing a human-robot collaboration video from a quadruped robot's egocentric perspective.\nDue to the low-angle view, the human's torso might cause severe physical occlusion.\n\n**Task:**\nPerform a step-by-step diagnostic reasoning to anticipate the human's final intent.\nOutput a SINGLE JSON object exactly matching this structure, answering the 6 progressive questions, do not output the thinking progress, only output the choices of the 6 questions in json format:")
        combined_text = f"System Instruction:\n{system_prompt_processed}\n\nUser Task:\n{item['user_prompt']}"

        # ⚠️ 核心修改 3：回退到 OpenAI 的 message 格式（API2D 完美支持此格式）
        user_content = [{"type": "text", "text": combined_text}]
        for b64_img in base64_frames:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_img}",
                    "detail": "auto"
                }
            })

        messages = [{"role": "user", "content": user_content}]

        # 3. 发送 API 请求 (使用优雅的 openai 库调用)
        output_text = "{}"
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=TEST_MODEL,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=4096,
                    response_format={"type": "json_object"}  # 强制返回 JSON
                )
                output_text = response.choices[0].message.content
                break

            except Exception as e:
                print(f"\n[Attempt {attempt + 1}/{max_retries}] API Error on sample {item.get('id', 'unknown')}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                else:
                    print(f"Failed completely after {max_retries} attempts.")

        # 4. 解析与严格评分
        parsed_json, error_msg = extract_json_from_text(output_text)
        eval_res = eval_response(parsed_json, gt_data)

        # 5. 更新统计
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
        processed_ids.add(current_sample_id)

        if len(results) > 0 and len(results) % 5 == 0:
            dirname = os.path.dirname(RESULT_FILE)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            with open(RESULT_FILE, 'w') as f:
                json.dump(results, f, indent=2)

        time.sleep(2)  # 给中转站留点喘息时间

    # 全量跑完后的最终保存
    dirname = os.path.dirname(RESULT_FILE)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(RESULT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    # ================= 打印报表 =================
    print("\n" + "=" * 120)
    print(f"CoT-HRC DIAGNOSTIC ANALYSIS REPORT (Model: {TEST_MODEL} | View: {TEST_VIEW})")
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