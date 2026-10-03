import json
import torch
import os
import re
import av
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

# ================= 配置区域 =================
# 确保这里是你刚才下载的 OneVision 路径
MODEL_PATH = "/root/autodl-tmp/llava-onevision-qwen2-7b-ov-hf"
BENCHMARK_FILE = "test_dataset/final_benchmark_0302/final_exam.json" 
RESULT_FILE = "final_results/llava_onevision_results_ego_raw.json"
TEST_VIEW = "ego_raw"  # 可切换: ego_raw / ego_ann / third_raw / oracle_dual

# 显存保护：单视角抽 8 帧，双视角各抽 4 帧（总计 8 帧，保底 24GB 显存不爆）
NUM_FRAMES_SINGLE = 8
NUM_FRAMES_DUAL = 4  

SLICE_KEYS = ["transit_30", "transit_60", "transit_90", "pre_transit", "transit_30_60", "transit_60_90", "post_transit"]
DIFFICULTY_KEYS = ["Easy", "Medium", "Hard"]

# ================= 视频抽帧函数 =================
def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index: break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def sample_video_frames(video_path, num_frames=8):
    container = av.open(video_path)
    container.streams.video[0].thread_type = "AUTO" 
    total_frames = container.streams.video[0].frames
    
    if total_frames > 0:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        clip = read_video_pyav(container, indices)
    else:
        decoded = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        indices = np.linspace(0, len(decoded) - 1, num_frames, dtype=int)
        clip = np.stack([decoded[i] for i in indices])
    return clip

# ================= JSON解析与评估逻辑 =================
def extract_json_from_text(text):
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match: json_str = match.group(1)
    else:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match: json_str = match.group(0)
        else: return None, "No JSON found"
    try: return json.loads(json_str), None
    except json.JSONDecodeError as e: return None, f"JSON Decode Error: {str(e)}"

def eval_response(pred_data, gt):
    if not pred_data or not isinstance(pred_data, dict):
        return {k: False for k in ["acc_area", "acc_status", "acc_obj", "acc_furn", "acc_aff", "acc_action", "acc_target", "acc_intent", "sca_reasoning", "sca_full"]}

    def match(p, g): return str(p).lower().strip() == str(g).lower().strip() if p and g else False

    res = {
        "acc_area": match(pred_data.get("Q1_target_area"), gt.get("target_area")),
        "acc_status": match(pred_data.get("Q2_is_holding_object"), gt.get("is_holding_object")),
        "acc_obj": match(pred_data.get("Q3_held_object_category"), gt.get("held_object_category")),
        "acc_furn": match(pred_data.get("Q4_target_furniture"), gt.get("target_furniture")),
        "acc_aff": match(pred_data.get("Q5_furniture_affordance"), gt.get("furniture_affordance")),
    }
    gt_triplet = gt.get("final_intent_triplet", ["", "", ""])
    p_triplet = pred_data.get("Q6_final_intent_triplet", [])
    if isinstance(p_triplet, list) and len(p_triplet) == 3:
        res["acc_action"] = match(p_triplet[0], gt_triplet[0])
        res["acc_target"] = match(p_triplet[2], gt_triplet[2])
        res["acc_intent"] = res["acc_action"] and match(p_triplet[1], gt_triplet[1]) and res["acc_target"]
    else:
        res.update({"acc_action": False, "acc_target": False, "acc_intent": False})

    res["sca_reasoning"] = (res["acc_area"] and res["acc_status"] and res["acc_furn"] and res["acc_aff"] and res["acc_action"] and res["acc_target"])
    res["sca_full"] = (res["sca_reasoning"] and res["acc_obj"])
    return res

class StatsBucket:
    def __init__(self, name):
        self.name, self.total = name, 0
        self.metrics = {k: 0 for k in ["acc_area", "acc_status", "acc_obj", "acc_furn", "acc_aff", "acc_action", "acc_intent", "sca_reasoning", "sca_full"]}
    def update(self, eval_res):
        self.total += 1
        for k in self.metrics:
            if eval_res.get(k): self.metrics[k] += 1
    def get_metrics_dict(self):
        if self.total == 0: return {**{k: 0.0 for k in self.metrics}, "total": 0}
        return {"total": self.total, **{k: (self.metrics[k] / self.total) * 100 for k in self.metrics}}

# ================= 主程序 =================
def main():
    abs_model_path = os.path.abspath(MODEL_PATH)
    print(f"Loading LLaVA-OneVision from {abs_model_path}...")
    
    # 使用 AutoProcessor 自动拉取对应的处理器
    processor = AutoProcessor.from_pretrained(abs_model_path, local_files_only=True)
    
    # 加载模型，不使用 flash-attn，保证环境鲁棒性
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        abs_model_path, 
        torch_dtype=torch.bfloat16, 
        local_files_only=True,
        low_cpu_mem_usage=False,
        attn_implementation="sdpa"
    ).cuda().eval()

    with open(BENCHMARK_FILE, 'r') as f:
        exam_questions = json.load(f)

    stats_slice = {k: StatsBucket(k) for k in SLICE_KEYS}
    stats_slice_diff = {f"{s}_{d}": StatsBucket(f"{s}_{d}") for s in SLICE_KEYS for d in DIFFICULTY_KEYS}
    results = []

    print(f"Running Inference on {len(exam_questions)} clips. View: {TEST_VIEW}")

    for item in tqdm(exam_questions):
        try:
            raw_system_prompt = item['system_prompt']
            user_text = item['user_prompt']

            # ---------------------------------------------------------
            # 核心：使用 Qwen2 格式的 Message 列表构建 Prompt
            # ---------------------------------------------------------
            if TEST_VIEW == "oracle_dual":
                v1, v2 = item['video_paths'].get('third_raw'), item['video_paths'].get('global')
                if not (v1 and os.path.exists(v1) and v2 and os.path.exists(v2)): continue
                
                clip_third = sample_video_frames(v1, NUM_FRAMES_DUAL)
                clip_global = sample_video_frames(v2, NUM_FRAMES_DUAL)
                video_inputs = [clip_third, clip_global]
                
                processed_sys = raw_system_prompt.replace(
                    "from a quadruped robot's egocentric perspective",
                    "from TWO simultaneous perspectives: third-person and global overhead"
                )
                text_prompt = f"Video 1 is third-person view. Video 2 is global overhead view.\n{processed_sys}\nBased on both videos, {user_text}"
                
                # 告诉处理器这里有两个视频
                conversation = [
                    {"role": "user", "content": [
                        {"type": "video"},
                        {"type": "video"},
                        {"type": "text", "text": text_prompt}
                    ]}
                ]
            else:
                v_path = item['video_paths'].get(TEST_VIEW)
                if not (v_path and os.path.exists(v_path)): continue
                
                video_inputs = [sample_video_frames(v_path, NUM_FRAMES_SINGLE)]
                
                if 'third' in TEST_VIEW:
                    processed_sys = raw_system_prompt.replace("egocentric perspective", "third-person perspective")
                elif 'global' in TEST_VIEW:
                    processed_sys = raw_system_prompt.replace("egocentric perspective", "global overhead perspective")
                else:
                    processed_sys = raw_system_prompt
                    
                text_prompt = f"{processed_sys}\n{user_text}"
                
                conversation = [
                    {"role": "user", "content": [
                        {"type": "video"},
                        {"type": "text", "text": text_prompt}
                    ]}
                ]

            # 应用 Chat Template 自动拼装带 <|im_start|> 的 Prompt
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

            # 推理生成 (加入了截断保护)
            inputs = processor(
                text=prompt, 
                videos=video_inputs, 
                return_tensors="pt",
                truncation=True,
                max_length=4096
            ).to("cuda")

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

            # ---------------------------------------------------------
            # 解码与后处理
            # ---------------------------------------------------------
            # 截取新生成的 Token（去掉 Prompt 部分）
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()

            parsed_json, error_msg = extract_json_from_text(output_text)
            eval_res = eval_response(parsed_json, item['ground_truth'])

            # 统计与保存
            s_param, diff = item['slice_param'], item.get('difficulty', 'Unknown')
            if s_param in stats_slice: stats_slice[s_param].update(eval_res)
            if f"{s_param}_{diff}" in stats_slice_diff: stats_slice_diff[f"{s_param}_{diff}"].update(eval_res)

            res_entry = {**item, "model_output": output_text, "parsed_json": parsed_json, "evaluation": eval_res, "parse_error": error_msg}
            for k in ['system_prompt', 'user_prompt']:
                if k in res_entry: del res_entry[k]
            results.append(res_entry)

            if len(results) % 20 == 0:
                os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
                with open(RESULT_FILE, 'w') as f: json.dump(results, f, indent=2)

        except Exception as e:
            print(f"Error processing item: {e}")
            continue

    with open(RESULT_FILE, 'w') as f: json.dump(results, f, indent=2)
    print(f"\nDone! Results saved to {RESULT_FILE}")

if __name__ == "__main__":
    main()