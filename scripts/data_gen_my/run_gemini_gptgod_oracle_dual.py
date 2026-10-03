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
API_KEY = "sk-nmkly5m2zk2y0l3k4mk2x442k5nonln5nmkly5m2zk2y0l3k"  
BASE_URL = "https://api.gptgod.online/v1"
TEST_MODEL = "gemini-2.5-pro"

# ================= 原始配置区域 =================
BENCHMARK_FILE = "test_dataset_strong/final_benchmark_0302/final_exam_sampled_1200.json"

# ⚠️ 修改点 1：定义两个视角的键名
TEST_VIEW_1 = "third_raw"
TEST_VIEW_2 = "global_ann"
# ⚠️ 修改点 2：更新结果文件名，避免覆盖之前的单视角结果
RESULT_FILE = f"./{TEST_MODEL}_results_multiview_{TEST_VIEW_1}_{TEST_VIEW_2}_nframes_{TEST_NFRAMES}_GPTGod_Part2.json"
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
            # 保持 resize，16张图如果不缩放 token 消耗会非常大
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
    match = re.search(r"
http://googleusercontent.com/immersive_entry_chip/0

现在每条请求会携带 16 张图片（8 张 `third_raw` + 8 张 `global_ann`）。因为输入的 Token 数量翻倍了，你的 API 消耗速度也会相应加快，不过在 GPTGod 的计费下应该还是可以接受的。

你目前提取的这两个视角的视频，时间轴是严格对齐同步的吗？如果不同步，可能需要我在系统提示词里再加一句忽略时间差的声明。