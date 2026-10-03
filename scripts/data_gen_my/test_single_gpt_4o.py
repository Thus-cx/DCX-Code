import json
import os
import re
import cv2
import base64
import numpy as np
from openai import OpenAI

# ================= 你的配置 =================
API_KEY = "AIzaSyApmT22LSZzenKSV67i0yrikD_58G3tMRA"  # 记得填上你的 Key
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
TEST_MODEL = "gemini-1.5-flash"

BENCHMARK_FILE = "test_dataset_strong/final_benchmark_0302/final_exam.json"  
TEST_VIEW = "ego_raw"  
TEST_NFRAMES = 8

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

def extract_json_from_text(text):
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match: return json.loads(match.group(1)), None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match: return json.loads(match.group(0)), None
    return None, "No JSON found"

# ================= 测试主逻辑 =================
def test_one_sample():
    print(f"[{TEST_MODEL}] 开始加载数据集并寻找第1个测试视频...")
    with open(BENCHMARK_FILE, 'r') as f:
        exam_questions = json.load(f)

    # 找第一个有效的样本
    test_item = None
    for item in exam_questions:
        if TEST_VIEW in item.get('video_paths', {}) and os.path.exists(item['video_paths'][TEST_VIEW]):
            test_item = item
            break
            
    if not test_item:
        print("错误：没有找到有效的视频路径，请检查 BENCHMARK_FILE 和视频是否存在！")
        return

    video_path = test_item['video_paths'][TEST_VIEW]
    print(f"找到测试视频: {video_path}")
    print("正在抽帧...")
    
    base64_frames = extract_frames_from_video(video_path, TEST_NFRAMES)
    if not base64_frames:
        print("抽帧失败，视频可能已损坏。")
        return

    print("开始调用 API ... (请耐心等待10-20秒)")
    user_content = [{"type": "text", "text": test_item['user_prompt']}]
    for b64_img in base64_frames:
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}", "detail": "low"}})

    messages = [
        {"role": "system", "content": test_item['system_prompt']},
        {"role": "user", "content": user_content}
    ]

    try:
        response = client.chat.completions.create(
            model=TEST_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=512,
            response_format={"type": "json_object"}
        )
        raw_output = response.choices[0].message.content
        
        print("\n" + "="*50)
        print("✅ API 原始返回文本 (Raw Output):")
        print("="*50)
        print(raw_output)
        print("="*50)

        # 测试解析逻辑
        parsed_json, error = extract_json_from_text(raw_output)
        
        if error:
            print(f"\n❌ JSON 解析失败: {error}")
        else:
            print("\n✅ JSON 成功解析！转化为的 Python 字典如下:")
            # 使用 json.dumps 美化打印出来，确保包含你需要的 Q1 到 Q6 字段
            print(json.dumps(parsed_json, indent=4, ensure_ascii=False))

    except Exception as e:
        print(f"\n❌ API 调用报错: {e}")

if __name__ == "__main__":
    test_one_sample()