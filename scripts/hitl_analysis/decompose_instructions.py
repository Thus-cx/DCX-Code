#!/usr/bin/env python3
"""
脚本：decompose_instructions.py
功能：使用Few-shot示例提升LLM的指令分解质量
用法：python decompose_instructions.py --input data/datasets/partnr_episodes/v0_0/val_mini.json.gz --output data/datasets/partnr_episodes/v0_0/val_mini_with_subtasks.json.gz --examples examples_prompts.txt
"""

import gzip
import json
import time
import argparse
import traceback
from pathlib import Path
from typing import List, Dict, Any, Tuple

# 使用transformers直接加载模型
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class FewShotInstructionDecomposer:
    """使用Few-shot示例的指令分解器"""
    
    def __init__(self, model_path: str, examples_file: str = None):
        print(f"正在加载模型: {model_path}")
        
        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 加载模型
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            if not torch.cuda.is_available():
                self.model = self.model.to("cpu")
        except Exception as e:
            print(f"警告: 使用默认方式加载模型: {e}")
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # 生成配置
        self.generation_config = {
            "max_new_tokens": 500,  # 增加token数量以包含更详细的分解
            "temperature": 0.1,
            "do_sample": False,
            "top_p": 1.0,
            "repetition_penalty": 1.1,  # 增加重复惩罚以避免重复
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        
        # 加载示例
        self.examples = []
        if examples_file and Path(examples_file).exists():
            self.examples = self.load_examples(examples_file)
            print(f"✓ 加载了 {len(self.examples)} 个few-shot示例")
        else:
            print("⚠ 未找到示例文件，将使用zero-shot模式")
        
        print(f"✓ 模型加载完成")
    
    def load_examples(self, examples_file: str) -> List[Tuple[str, List[str]]]:
        """
        从文本文件加载few-shot示例
        
        假设文件格式如下：
        Instruction: [instruction text]
        Subtasks: ["subtask1", "subtask2", "subtask3"]
        
        示例之间用空行分隔
        """
        examples = []
        
        try:
            with open(examples_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 按空行分割示例
            example_blocks = content.strip().split('\n\n')
            
            for block in example_blocks:
                lines = block.strip().split('\n')
                if len(lines) < 2:
                    continue
                
                instruction = None
                subtasks = None
                
                for line in lines:
                    line = line.strip()
                    if line.startswith("Instruction:"):
                        instruction = line[len("Instruction:"):].strip()
                    elif line.startswith("Subtasks:"):
                        # 尝试解析JSON数组
                        subtasks_str = line[len("Subtasks:"):].strip()
                        try:
                            subtasks = json.loads(subtasks_str)
                            if isinstance(subtasks, list):
                                # 确保所有子任务都是字符串
                                subtasks = [str(s) for s in subtasks]
                        except json.JSONDecodeError:
                            # 如果不是JSON格式，尝试其他格式
                            subtasks = [s.strip() for s in subtasks_str.split(',')]
                
                if instruction and subtasks:
                    examples.append((instruction, subtasks))
        
        except Exception as e:
            print(f"警告: 加载示例文件时出错: {e}")
        
        return examples
    
    def build_few_shot_prompt(self, instruction: str) -> str:
        """
        构建包含few-shot示例的提示词
        
        Args:
            instruction: 当前要分解的指令
            
        Returns:
            完整的提示词
        """
        # 系统指令
        system_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert at decomposing household task instructions into clear, actionable subtasks for robotic agents.

Your task is to break down complex instructions into a logical sequence of simple, specific subtasks.

IMPORTANT GUIDELINES:
1. Each subtask should be a SINGLE, ACTIONABLE step
2. Use NATURAL LANGUAGE, not tool calls or code
3. Be SPECIFIC about objects and locations mentioned in the instruction
4. Maintain LOGICAL ORDER (first things first)
5. Take the temporal constraint into consideration
6. Do NOT include agent assignments (like "Agent 1:")
7. Do NOT include tool names (like "Navigate to...")
8. Output format: Return ONLY a JSON array of strings, where each string is one subtask
9. Example: ["first subtask", "second subtask", "third subtask"]<|eot_id|>

"""
        
        # 添加few-shot示例
        prompt = system_prompt
        
        for i, (example_instruction, example_subtasks) in enumerate(self.examples):
            prompt += f"""<|start_header_id|>user<|end_header_id|>

Instruction: {example_instruction}

Please decompose this instruction into sequential subtasks. Return ONLY a JSON array:<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>

Subtasks: {json.dumps(example_subtasks, ensure_ascii=False)}<|eot_id|>

"""
        
        # 添加当前指令
        prompt += f"""<|start_header_id|>user<|end_header_id|>

Instruction: {instruction}

Please decompose this instruction into sequential subtasks. Return ONLY a JSON array:<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>

Subtasks:"""
        
        return prompt
    
    def decompose_instruction(self, instruction: str) -> List[str]:
        """分解单个指令"""
        # 构建提示词
        prompt = self.build_few_shot_prompt(instruction)
        
        try:
            # 编码输入
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **self.generation_config
                )
            
            # 解码
            input_length = inputs.input_ids.shape[1]
            response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            
            # 清理响应
            response = response.strip()
            
            # 尝试解析JSON数组
            try:
                # 找到第一个'['和最后一个']'
                start_idx = response.find('[')
                end_idx = response.rfind(']')
                
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx+1]
                    subtasks = json.loads(json_str)
                    
                    # 验证结果类型
                    if isinstance(subtasks, list) and all(isinstance(s, str) for s in subtasks):
                        return subtasks
                    
            except (json.JSONDecodeError, ValueError) as e:
                print(f"JSON解析失败: {e}")
                print(f"原始响应前200字符: {response[:200]}...")
            
            # 如果JSON解析失败，尝试提取编号列表
            subtasks = []
            for line in response.split('\n'):
                line = line.strip()
                # 匹配 "1. 子任务" 或 "1) 子任务" 格式
                if line and line[0].isdigit():
                    # 移除编号和分隔符
                    if '.' in line:
                        parts = line.split('.', 1)
                    elif ')' in line:
                        parts = line.split(')', 1)
                    else:
                        continue
                        
                    if len(parts) > 1:
                        subtask = parts[1].strip()
                        if subtask and len(subtask) > 3:  # 确保有实际内容
                            subtasks.append(subtask)
            
            # 如果还是失败，使用简单规则分解
            if not subtasks:
                print(f"警告: 无法解析LLM响应，使用简单规则分解")
                subtasks = self.simple_decomposition(instruction)
                
            return subtasks
            
        except Exception as e:
            print(f"错误: 分解指令时出错: {e}")
            traceback.print_exc()
            # 出错时使用简单规则分解
            return self.simple_decomposition(instruction)
    
    def simple_decomposition(self, instruction: str) -> List[str]:
        """
        简单的指令分解规则（作为LLM失败的备选方案）
        
        Args:
            instruction: 原始任务指令
            
        Returns:
            子任务列表
        """
        subtasks = []
        
        # 按句号分割
        sentences = instruction.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # 按"and"分割
                if ' and ' in sentence.lower():
                    parts = sentence.split(' and ')
                    for part in parts:
                        part = part.strip()
                        if part:
                            subtasks.append(part)
                else:
                    subtasks.append(sentence)
        
        # 如果还是没有有效的子任务，返回原始指令
        if not subtasks:
            subtasks = [instruction]
            
        return subtasks
    
    def batch_decompose(self, instructions: List[str]) -> List[List[str]]:
        """
        批量分解指令
        
        Args:
            instructions: 指令列表
            
        Returns:
            子任务列表的列表
        """
        all_subtasks = []
        
        for i, instruction in enumerate(instructions):
            print(f"处理指令 {i+1}/{len(instructions)}: {instruction[:60]}...")
            
            start_time = time.time()
            subtasks = self.decompose_instruction(instruction)
            elapsed = time.time() - start_time
            
            all_subtasks.append(subtasks)
            
            # 打印示例
            if i < 3:
                print(f"  耗时: {elapsed:.2f}秒")
                print(f"  子任务数: {len(subtasks)}")
                for j, subtask in enumerate(subtasks, 1):
                    print(f"    {j}. {subtask}")
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 小延迟避免过热
            time.sleep(0.5)
        
        return all_subtasks


def load_and_process_dataset(input_path: str, max_episodes: int = None) -> Tuple[List[Dict], Dict]:
    """
    直接加载JSON.gz文件并进行处理
    
    Args:
        input_path: 输入文件路径
        max_episodes: 最大处理episode数量
        
    Returns:
        (episode字典列表, 原始数据集的其他数据)
    """
    print(f"正在加载数据集: {input_path}")
    
    # 检查文件是否存在
    if not Path(input_path).exists():
        print(f"✗ 错误: 文件不存在: {input_path}")
        return [], {}
    
    try:
        # 读取gzip压缩的JSON文件
        with gzip.open(input_path, 'rt', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取episodes
        episodes = data.get("episodes", [])
        
        print(f"✓ 成功加载 {len(episodes)} 个episode")
        
        # 限制episode数量
        if max_episodes and max_episodes < len(episodes):
            episodes = episodes[:max_episodes]
            print(f"测试模式: 仅处理前 {len(episodes)} 个episode")
        
        return episodes, data
        
    except Exception as e:
        print(f"✗ 加载数据集时出错: {e}")
        traceback.print_exc()
        return [], {}


def save_dataset_with_subtasks(episodes: List[Dict], original_data: Dict, output_path: str):
    """
    保存带子任务的数据集
    
    Args:
        episodes: 处理后的episode列表
        original_data: 原始数据集的其他数据
        output_path: 输出文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 创建输出数据结构
    output_data = original_data.copy()
    output_data["episodes"] = episodes
    
    # 保存为gzip压缩文件
    try:
        with gzip.open(output_path, 'wt', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 数据集已保存到: {output_path}")
        
    except Exception as e:
        print(f"✗ 保存数据集时出错: {e}")
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="使用Few-shot示例提升LLM的指令分解质量")
    parser.add_argument("--input", required=True, 
                       help="输入数据集路径 (.json.gz)")
    parser.add_argument("--output", required=True,
                       help="输出数据集路径 (.json.gz)")
    parser.add_argument("--model-path", default="../LLM-Research/Meta-Llama-3-8B-Instruct",
                       help="Llama模型路径（本地路径或HuggingFace模型名）")
    parser.add_argument("--examples", default="scripts/hitl_analysis/examples_prompts.txt",
                       help="Few-shot示例文件路径")
    parser.add_argument("--max-episodes", type=int, default=None,
                       help="最大处理episode数量（用于测试）")
    parser.add_argument("--use-cpu", action="store_true",
                       help="强制使用CPU（如果有GPU但内存不足）")
    
    args = parser.parse_args()
    
    # 设置设备
    if args.use_cpu:
        print("⚠ 强制使用CPU模式")
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    print("=" * 60)
    print("PARTNR指令分解工具 (Few-shot版本)")
    print("=" * 60)
    
    # 首先加载整个数据集以保留其他数据
    print("正在加载原始数据集...")
    try:
        with gzip.open(args.input, 'rt', encoding='utf-8') as f:
            original_data = json.load(f)
        
        episodes = original_data.get("episodes", [])
        print(f"✓ 加载了 {len(episodes)} 个episode")
        
    except Exception as e:
        print(f"✗ 加载原始数据集失败: {e}")
        return
    
    # 限制episode数量
    if args.max_episodes and args.max_episodes < len(episodes):
        episodes = episodes[:args.max_episodes]
        print(f"测试模式: 仅处理前 {len(episodes)} 个episode")
    
    # 提取所有指令
    instructions = [episode.get("instruction", "") for episode in episodes]
    
    # 检查是否有空的指令
    empty_instructions = sum(1 for instr in instructions if not instr.strip())
    if empty_instructions > 0:
        print(f"警告: 发现 {empty_instructions} 个空指令")
    
    # 初始化分解器
    print("=" * 60)
    print(f"正在初始化LLM模型和Few-shot示例...")
    
    try:
        decomposer = FewShotInstructionDecomposer(args.model_path, args.examples)
    except Exception as e:
        print(f"✗ 初始化模型失败: {e}")
        print("请检查模型路径是否正确，以及是否有足够的GPU内存")
        print(f"模型路径: {args.model_path}")
        print(f"示例文件: {args.examples}")
        return
    
    # 批量分解指令
    print("=" * 60)
    print(f"开始分解 {len(instructions)} 条指令...")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        all_subtasks = decomposer.batch_decompose(instructions)
    except Exception as e:
        print(f"✗ 分解指令时发生错误: {e}")
        traceback.print_exc()
        return
    
    elapsed_time = time.time() - start_time
    print("=" * 60)
    print(f"✓ 分解完成! 总耗时: {elapsed_time:.1f}秒")
    print(f"  平均每条指令: {elapsed_time/len(instructions):.1f}秒")
    
    # 将子任务添加回episode
    print("=" * 60)
    print("将子任务添加回episode...")
    
    success_count = 0
    for episode, subtasks in zip(episodes, all_subtasks):
        # 添加subtasks字段
        episode["subtasks"] = subtasks
        success_count += 1
    
    print(f"✓ 成功处理 {success_count}/{len(episodes)} 个episode")
    
    # 保存数据集
    print("=" * 60)
    save_dataset_with_subtasks(episodes, original_data, args.output)
    
    # 统计信息
    print("=" * 60)
    total_subtasks = sum(len(s) for s in all_subtasks)
    avg_subtasks = total_subtasks / len(all_subtasks) if all_subtasks else 0
    
    print("处理完成统计:")
    print(f"  ✓ 总episode数: {len(episodes)}")
    print(f"  ✓ 总子任务数: {total_subtasks}")
    print(f"  ✓ 平均每个指令的子任务数: {avg_subtasks:.2f}")
    
    # 打印更多示例
    print("\n分解示例:")
    print("=" * 60)
    for i, (episode, subtasks) in enumerate(zip(episodes[:3], all_subtasks[:3])):
        episode_id = episode.get("episode_id", f"#{i}")
        instruction = episode.get("instruction", "")
        
        print(f"\n示例 {i+1} (Episode {episode_id}):")
        print(f"  指令: {instruction}")
        print(f"  子任务: {subtasks}")
        for j, subtask in enumerate(subtasks, 1):
            print(f"    {j}. {subtask}")
    
    print("\n" + "=" * 60)
    print("完成! 请检查输出文件:")
    print(f"  {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()