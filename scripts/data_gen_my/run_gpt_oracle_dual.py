import json
import os
import re
import cv2
import base64
import numpy as np
from tqdm import tqdm
from openai import OpenAI
import time

# ================= 阿里云百炼/API2D 配置 =================
API_KEY = "sk-gT6T9rFX7vLUQxpP1342641c96734c7bAeA5AeCc9c664aC5" # ⚠️ 记得换成新生成的 Key
BASE_URL = "https://api.v3.cm/v1"  # API 接口地址

TEST_MODEL = "gpt-4o" 

# ================= 路径配置区域 =================
# 1. 全集文件路径
BENCHMARK_FILE = "test_dataset_strong/final_benchmark_0302/final_exam.json"  

# 2. 历史结果文件 (如果有之前跑过的单视角或旧双视角文件，可以在这里融合，没有就留空或填不存在的路径)
EXISTING_RESULT_FILE = "final_results/gpt-4o_results_oracle_dual_nframes_8_merged.json"

# 3. 最终输出的完整断点文件
RESULT_FILE = f"final_results/{TEST_MODEL}_results_oracle_dual_nframes_8_FULL.json"

# 🚨 核心配置：双视角物理视图
DUAL_VIEWS = ["third_raw", "global_ann"]  
TEST_NFRAMES = 8

SLICE_KEYS = ["transit_30", "transit_60", "transit_90", "pre_transit", "transit_30_60", "transit_60_90", "post_transit"]
DIFFICULTY_KEYS = ["Easy", "Medium", "Hard"]

# 初始化 API 客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= 辅助函数 =================

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

def get_unique_key(item):
    """提取样本的唯一标识符，兼容不同格式"""
    if item.get("question_id"):
        return str(item.get("question_id"))
    elif item.get("id"):
        return str(item.get("id"))
    return str(item.get("video_paths", {}))

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
            if eval_res.get(k, False): self.metrics[k] += 1
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
    print(f"[*] 加载全集数据: {BENCHMARK_FILE}")
    with open(BENCHMARK_FILE, 'r') as f:
        exam_questions = json.load(f)

    stats_slice = {k: StatsBucket(k) for k in SLICE_KEYS}
    stats_slice_diff = {}
    for s in SLICE_KEYS:
        for d in DIFFICULTY_KEYS:
            stats_slice_diff[f"{s}_{d}"] = StatsBucket(f"{s}_{d}")

    results = []
    processed_ids = set()

    # ================= 核心：断点续传 & 融合历史样本 =================
    if os.path.exists(EXISTING_RESULT_FILE):
        print(f"[*] 发现历史运行记录 '{EXISTING_RESULT_FILE}'，正在加载...")
        try:
            with open(EXISTING_RESULT_FILE, 'r') as f:
                existing_results = json.load(f)
            for r in existing_results:
                uid = get_unique_key(r)
                processed_ids.add(uid)
                results.append(r)
                # 恢复计分板
                slice_param = r.get('slice_param')
                difficulty = r.get('difficulty', 'Unknown')
                eval_res = r.get('evaluation', {})
                if slice_param in stats_slice:
                    stats_slice[slice_param].update(eval_res)
                sd_key = f"{slice_param}_{difficulty}"
                if sd_key in stats_slice_diff:
                    stats_slice_diff[sd_key].update(eval_res)
            print(f"[*] 成功加载了 {len(existing_results)} 个历史样本。")
        except Exception as e:
            print(f"[!] 读取历史记录失败: {e}")

    # 如果中途跑断了，从最近生成的 RESULT_FILE 文件中恢复断点
    if os.path.exists(RESULT_FILE):
        print(f"[*] 发现最近的 FULL 断点文件 '{RESULT_FILE}'，正在补充加载...")
        try:
            with open(RESULT_FILE, 'r') as f:
                full_results = json.load(f)
            for r in full_results:
                uid = get_unique_key(r)
                if uid not in processed_ids:
                    processed_ids.add(uid)
                    results.append(r)
                    # 恢复计分板
                    slice_param = r.get('slice_param')
                    difficulty = r.get('difficulty', 'Unknown')
                    eval_res = r.get('evaluation', {})
                    if slice_param in stats_slice:
                        stats_slice[slice_param].update(eval_res)
                    sd_key = f"{slice_param}_{difficulty}"
                    if sd_key in stats_slice_diff:
                        stats_slice_diff[sd_key].update(eval_res)
            print(f"[*] 目前总计已完成 {len(processed_ids)} 个样本的评估。")
        except Exception as e:
            pass
    # ====================================================================

    print(f"\nRunning API Inference using model: [{TEST_MODEL}] on {len(exam_questions)} clips. Dual Views: {DUAL_VIEWS}")

    for item in tqdm(exam_questions):
        # 跳过已处理样本
        uid = get_unique_key(item)
        if uid in processed_ids:
            continue

        # 🚨 获取两个视角的视频路径
        video_path_1 = item['video_paths'].get(DUAL_VIEWS[0])
        video_path_2 = item['video_paths'].get(DUAL_VIEWS[1])
        
        # 路径校验
        if not video_path_1 or not os.path.exists(video_path_1) or \
           not video_path_2 or not os.path.exists(video_path_2):
            print(f"\n[!] 跳过样本 {uid}：缺失对应视角视频文件。")
            continue

        slice_param = item['slice_param']
        difficulty = item.get('difficulty', 'Unknown')
        gt_data = item['ground_truth']

        # 抽取双视角帧
        base64_frames_1 = extract_frames_from_video(video_path_1, TEST_NFRAMES)
        base64_frames_2 = extract_frames_from_video(video_path_2, TEST_NFRAMES)
        
        if not base64_frames_1 or not base64_frames_2:
            print(f"\n[!] 跳过样本 {uid}：抽帧失败。")
            continue

        # 构造双视角 Prompt
        user_content = [
            {"type": "text", "text": item['user_prompt'] + f"\n\n接下来我将提供同一场景下两个不同视角的视频抽帧，请结合多视角信息进行综合推理。"}
        ]
        
        user_content.append({"type": "text", "text": f"\n\n--- 以下是视角 1 ({DUAL_VIEWS[0]}) 的序列帧 ---"})
        for b64_img in base64_frames_1:
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}", "detail": "low"}})

        user_content.append({"type": "text", "text": f"\n\n--- 以下是视角 2 ({DUAL_VIEWS[1]}) 的序列帧 ---"})
        for b64_img in base64_frames_2:
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}", "detail": "low"}})

        messages = [
            {"role": "system", "content": item['system_prompt']},
            {"role": "user", "content": user_content}
        ]

        # 失败重试机制
        output_text = "{}"
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=TEST_MODEL,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=512,
                    response_format={"type": "json_object"} 
                )
                output_text = response.choices[0].message.content
                break # 成功则跳出重试循环
            except Exception as e:
                print(f"\n[Attempt {attempt+1}/{max_retries}] API Error on {uid}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(3) # 遇到错误稍微休息一下再试
                else:
                    output_text = "{}"

        # 评估逻辑
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
        processed_ids.add(uid)

        # 定期保存与目录创建保护
        if len(results) > 0 and len(results) % 10 == 0:
            dirname = os.path.dirname(RESULT_FILE)
            if dirname: os.makedirs(dirname, exist_ok=True)
            with open(RESULT_FILE, 'w') as f:
                json.dump(results, f, indent=2)

        time.sleep(1) # API 保护性休眠

    # 最终保存
    dirname = os.path.dirname(RESULT_FILE)
    if dirname: os.makedirs(dirname, exist_ok=True)
    with open(RESULT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    # ================= 打印报表 =================
    print("\n" + "=" * 120)
    print(f"CoT-HRC DIAGNOSTIC ANALYSIS REPORT (Model: {TEST_MODEL} | Views: {DUAL_VIEWS} | FULL SET)")
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

if __name__ == "__main__":
    main()