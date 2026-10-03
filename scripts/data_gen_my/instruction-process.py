import json
import gzip
from openai import OpenAI
from tqdm import tqdm

ds_api = 'sk-d73823efeec249ac913d8a545c20ae24'
client = OpenAI(
    api_key=ds_api,
    base_url="https://api.deepseek.com"
)
def rewrite_instruction_to_intent(instruction):
    system_prompt = """
    You are an expert linguistic annotator for an Embodied AI human-robot collaboration dataset. 
    Your task is to rewrite detailed, low-level robotic instructions into high-level, natural human intents. These instuctions and intents are some common senarios and examples in the human's daily life.

    Rules for rewriting:
    1. If the objects have a strong functional connection, abstract them into a functional semantic goal (e.g., "prepare a snack"). DO NOT mention specific objects.
    2. If the objects have no logical functional connection (pure spatial rearrangement), abstract them using a common categorical hypernym (e.g., "toys") or their shared spatial origin (e.g., "items left in the bedroom").
    3. If the instruction start with a summary sentence which summarize other sentences in the instruction, use the first sentence as the result.
    4. Keep the rewritten instruction to a single, concise sentence.
    5. Output strictly in JSON format with keys: "reasoning" and "high_level_intent".
    6. CRITICAL: ALL output text, including reasoning and the final intent, MUST be in English.
    """
    few_shot_examples = [
        {"role": "user",
         "content": "Help me prepare a snack. Move both apples, the piece of bread, and the butter dish from the kitchen counter to the glass table."},
        {"role": "assistant",
         "content": "{\"reasoning\": \"Objects (apples, bread, butter) have a strong functional connection related to food. The user explicitly mentions 'prepare a snack'.\", \"high_level_intent\": \"Help me prepare a snack on the glass table.\"}"},

        {"role": "user",
         "content": "Help me move the toy fire truck and the toy bee from the chair in the kitchen to a chair in the living room."},
        {"role": "assistant",
         "content": "{\"reasoning\": \"Objects are a toy fire truck and a toy bee. No unified functional task, but they share the category 'toys'.\", \"high_level_intent\": \"Help me relocate the toys from the kitchen to the living room.\"}"},

        {"role": "user",
         "content": "Help me move the cushion from the bedroom chair to the living room chair. Move the toy construction set from the bedroom table to the living room table."},
        {"role": "assistant",
         "content": "{\"reasoning\": \"Objects are a cushion and a toy set. No obvious shared category. But they share the same starting room (bedroom).\", \"high_level_intent\": \"Help me tidy up the items left in the bedroom and bring them to the living room.\"}"}
    ]
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(few_shot_examples)
    messages.append({"role": "user", "content": instruction})
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=8100
        )
        result_str = response.choices[0].message.content
        return json.loads(result_str)
    except json.JSONDecodeError as e:
        # 记录原始内容和错误详情
        print(f"JSON 解析失败，指令: {instruction}...")  # 打印部分指令
        print(f"错误信息: {e}")
        print(f"原始响应内容: {result_str}")  # 注意 result_str 可能未定义如果异常发生在解析前？需调整作用域
        # 将错误内容保存到文件以便复查
        with open("error_responses.txt", "a", encoding="utf-8") as f:
            f.write(f"Instruction: {instruction}\n")
            f.write(f"Response: {result_str}\n")
            f.write("-" * 80 + "\n")
        return None
    except Exception as e:
        print(f"API 调用发生错误：{e}")
        return None

if __name__ == "__main__":
    test_instruction = "Place the candle and candle holder on the white table in the living room. Move the plant to the same table."
    print(f"原始指令：{test_instruction}\n")
    print("正在调用api处理...\n")
    output = rewrite_instruction_to_intent(test_instruction)
    if output:
        print("推理过程（Reasoning）：", output.get("reasoning"))
        print("\n最终高层意图（High-level Intent）：", output.get("high_level_intent"))
    val_new_file_path = "/root/autodl-tmp/partnr-planner-main/data/datasets/partnr_episodes/v0_0/final_test_datasets.json.gz"
    with gzip.open(val_new_file_path, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    episodes = data['episodes']
    processed_data = []
    error_episodes = []
    for episode in tqdm(episodes, desc="Processing episodes' instructions", unit="episode"):
        couple = {}
        instruction = episode['instruction']
        processed_intent = rewrite_instruction_to_intent(instruction)
        couple['episode_id'] = episode['episode_id']
        couple['instruction'] = episode['instruction']
        if processed_intent is not None:
            couple['reasoning'] = processed_intent.get("reasoning")
            couple['processed_intent'] = processed_intent.get("high_level_intent")
        else:
            couple['reasoning'] = None
            couple['processed_intent'] = None
            error_episodes.append(episode['episode_id'])
        processed_data.append(couple)
    processed_file = "/root/autodl-tmp/partnr-planner-main/data/datasets/partnr_episodes/v0_0/processed_instructions.json.gz"
    with gzip.open(processed_file, 'wt', encoding="utf-8") as f:
        json.dump(processed_data, f, indent=2)
    print(f"error_episodes: {error_episodes}")
        
    