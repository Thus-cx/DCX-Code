import os
import hydra
from omegaconf import OmegaConf
from habitat_llm.llm.hf_model import HFModel

# 模拟配置
cfg = OmegaConf.create({
    "model_name": "/root/autodl-tmp/LLM-Research/Meta-Llama-3-8B-Instruct",
    "tokenizer_name": "/root/autodl-tmp/LLM-Research/Meta-Llama-3-8B-Instruct",
    "max_new_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9,
    "torch_dtype": "bfloat16",
    "device_map": "auto"
})

print(f"正在尝试加载模型: {cfg.model_name}")

try:
    # 实例化模型
    llm = HFModel(cfg)
    print("✅ 模型加载成功！")
    
    # 简单测试生成
    prompt = "Role: Robot. Task: Check functionality. Action:"
    print("正在测试生成...")
    response = llm.generate(prompt)
    print(f"生成结果: {response}")
    
except Exception as e:
    import traceback
    print(f"❌ 加载失败: {e}")
    traceback.print_exc()
    print("请检查路径是否正确，以及文件夹下是否包含 config.json 和 model.safetensors 等文件。")