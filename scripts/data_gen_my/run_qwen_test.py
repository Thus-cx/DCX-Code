import json
import torch
import os
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ================= 配置 =================
MODEL_PATH = "../Qwen2-VL-7B-Instruct"  # 请修改为你本地的模型路径
BENCHMARK_FILE = "test_my_videos_2/final_benchmark_clips_2/final_test_benchmark_third_ann.json"
RESULT_FILE = "test_my_videos/results/qwen_inference_results_204.json"
MAX_PIXELS = 480 * 480  # 限制分辨率防止爆显存
FPS = 1.0  # 抽帧率 (每秒1帧足够做意图推理)


# =======================================

def main():
    # 1. 加载模型
    print(f"Loading model from {MODEL_PATH}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",  # 如果显卡不支持 FlashAttn2，请删掉这行
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # 2. 加载数据
    with open(BENCHMARK_FILE, 'r') as f:
        questions = json.load(f)

    results = []

    print(f"Starting inference on {len(questions)} items...")

    for item in tqdm(questions):
        video_path = item['video_path']
        if not os.path.exists(video_path):
            print(f"Warning: Video not found {video_path}, skipping.")
            continue

        # 构造输入
        prompt_text = f"{item['question']}\n\nOptions:\nA. {item['options']['A']}\nB. {item['options']['B']}\nC. {item['options']['C']}\nD. {item['options']['D']}\n\nOutput the reasoning first, then conclude with 'Answer: X'."

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": MAX_PIXELS,
                        "fps": FPS,
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        # 预处理
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # 生成
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # 解析答案 (简单的规则匹配)
        # 假设模型输出最后会包含 "Answer: A"
        pred_option = "Unknown"
        if "Answer: A" in output_text:
            pred_option = "A"
        elif "Answer: B" in output_text:
            pred_option = "B"
        elif "Answer: C" in output_text:
            pred_option = "C"
        elif "Answer: D" in output_text:
            pred_option = "D"
        # 备用：检查最后一个出现的 ABCD
        elif len(output_text) > 0:
            # 这里可以加更复杂的 regex，暂时简单处理
            pass

        # 记录结果
        result_entry = item.copy()
        result_entry['model_output'] = output_text
        result_entry['pred_option'] = pred_option
        result_entry['is_correct'] = (pred_option == item['correct_option'])

        results.append(result_entry)

        # 实时保存防止中断
        if len(results) % 10 == 0:
            with open(RESULT_FILE, 'w') as f:
                json.dump(results, f, indent=2)

    # 最终保存
    with open(RESULT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print("Inference finished.")


if __name__ == "__main__":
    main()