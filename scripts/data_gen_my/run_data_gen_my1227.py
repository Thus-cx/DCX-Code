import sys
import os
import json
import cv2
import numpy as np
import hydra
import magnum as mn
from omegaconf import OmegaConf

# --- 修正导入 ---
# DCX 的核心环境接口
from habitat_llm.agent.env import EnvironmentInterface
from habitat_llm.planner.centralized_llm_planner import CentralizedLLMPlanner

# 引入我们写的小脑 (确保 spot_follower.py 在同一目录下或 PYTHONPATH 中)
# 如果报错找不到，请确保在该脚本同级目录下创建了 spot_follower.py
sys.path.append("scripts/data_gen_my")
from spot_follower_my import SpotFollower
import habitat_llm.agent.env.dataset
import habitat_llm.agent.env.sensors

# 设置环境变量
os.environ["HABITAT_LLM_CONF_DIR"] = "habitat_llm/conf"

def save_video(frames, filepath, fps=10):
    if len(frames) == 0:
        return
    h, w, _ = frames[0].shape
    # 使用 mp4v 编码
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, fps, (w, h))
    for frame in frames:
        # Habitat (RGB) -> OpenCV (BGR)
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

@hydra.main(version_base=None, config_path="../../habitat_llm/conf", config_name="examples/data_gen_config_my")
def main(config):
    # 1. 强制修改 World Model 以欺骗 LLM (只保留 agent_0/Human)
    if "world_model" in config and "agents" in config.world_model:
        human_desc = config.world_model.agents.get("agent_0", None)
        # 覆盖为仅含 agent_0 的字典
        config.world_model.agents = {"agent_0": human_desc} if human_desc else {}
        print("已修改 World Model: LLM 仅可见 Human Agent。")

    # 2. 初始化环境 (使用 EnvironmentInterface)
    # 这一步会自动 reset 环境
    config.habitat.simulator.agents_order = ["agent_0", "agent_1"]
    env_interface = EnvironmentInterface(config)
    
    # 获取底层的 habitat simulator 对象 (用于获取绝对坐标)
    sim = env_interface.sim
    
    # 获取 Agent 实体
    # agent_0 是 Human, agent_1 是 Spot
    # 注意：在 EnvironmentInterface 中，agents 存储在 sim.agents_mgr 或直接通过 sim.get_agent(i)
    # habitat-lab 0.2.x / 0.3.x 差异：通常是 agents_mgr
    human_agent = sim.agents_mgr[0].articulated_agent
    spot_agent = sim.agents_mgr[1].articulated_agent
    
    # 3. 初始化模块
    # 大脑：LLM Planner
    # 需要传入 env_interface，它包含了 dataset 和 partial observation 逻辑
    planner = CentralizedLLMPlanner(config.planner, env_interface)
    
    # 小脑：Spot Follower (我们自定义的脚本)
    spot_brain = SpotFollower(spot_agent, human_agent)

    # 4. 准备数据保存
    episode_id = env_interface.env.current_episode().episode_id
    save_dir = f"output_dataset/episode_{episode_id}"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"开始生成 Episode: {episode_id}")
    instruction_text = env_interface.env.current_episode().instruction.instruction_text
    print(f"当前任务: {instruction_text}")

    video_ego_frames = []
    video_3rd_frames = []
    trace_log = []
    
    done = False
    step_count = 0
    max_steps = 1500

    # 初始 Observation
    # EnvironmentInterface 在 init 时已经 reset，我们需要获取当前的 obs
    # 但 EnvironmentInterface 没有直接暴露 last_obs，我们需要手动 step 或 parse
    # 通常 planner.get_next_action 会处理 obs 获取
    # 这里我们模拟一次空 step 来刷新 obs
    observation = sim.get_sensor_observations()

    while not done and step_count < max_steps:
        # --- A. Planner 决策 (High-Level) ---
        print(f"\n[Step {step_count}] Planner 正在思考...")
        try:
            # Planner 内部会自动调用 env_interface.get_observations()
            plan_step = planner.get_next_action(observation)
        except Exception as e:
            print(f"Planner 报错: {e}")
            break

        # 记录思维链
        current_thought = getattr(planner, "last_thought", "No thought captured")
        print(f"LLM Thought: {current_thought}")
        
        # 保存 Log
        trace_log.append({
            "step": step_count,
            "role": "planner",
            "thought": current_thought,
            "instruction_to_human": str(plan_step)
        })
        
        # 判断任务结束
        if plan_step.is_done:
            print("Planner 判定任务完成 (Done).")
            done = True
            break

        # --- B. Skill 执行 (Low-Level Unrolling) ---
        current_skill = plan_step.skill
        skill_name = current_skill.__class__.__name__
        print(f"Human 执行 Skill: {skill_name}")
        
        # 某些 Skill 需要 reset
        # current_skill.reset(observation) # 视具体 Skill 实现而定

        # 手动循环 Skill 的执行直到完成
        # 注意：is_done 需要传入当前的 observation
        while not current_skill.is_done(observation) and step_count < max_steps:
            
            # 1. 计算 Human 动作 (High-Level Policy -> Low-Level Action)
            # step() 返回动作字典，如 {'action': 'agent_0_arm_action', ...}
            human_action = current_skill.step(observation)
            
            # 2. 计算 Spot 动作 (Scripted Policy)
            # 获取最新的 observation 里的位置信息可能不准，建议直接用 sim 获取绝对位置
            # spot_follower 内部我们改为直接读取 sim state，不依赖 obs
            spot_action = spot_brain.get_action()
            
            # 3. 合并动作
            combined_action = {**human_action, **spot_action}
            
            # 4. 物理步进 (Environment Interface step)
            # 注意：EnvironmentInterface.step 可能会处理 info 和 reward，
            # 但这里我们直接调 underlying sim.step 可能更灵活，防止 env_interface 内部做过多 assert
            # 不过为了保持兼容，我们尝试用 sim.step
            observation = sim.step(combined_action)
            step_count += 1
            
            # 5. 采集图像
            # 注意：Key 名称通常包含 agent 前缀
            # agent_1 (Spot): jaw_rgb
            # agent_0 (Human): third_person_rgb
            
            # 调试：打印一次 key 看看
            if step_count == 1:
                print("Available Sensor Keys:", observation.keys())

            # 尝试提取
            ego_key = "agent_1_jaw_rgb"
            third_key = "agent_0_third_person_rgb"
            
            if ego_key in observation:
                video_ego_frames.append(observation[ego_key])
            if third_key in observation:
                video_3rd_frames.append(observation[third_key])

    # 5. 保存数据
    print(f"正在保存数据到 {save_dir} ...")
    save_video(video_ego_frames, f"{save_dir}/robot_ego.mp4")
    save_video(video_3rd_frames, f"{save_dir}/third_person.mp4")
    
    metadata = {
        "instruction": instruction_text,
        "success": env_interface.get_metrics().get("success", False),
        "total_steps": step_count,
        "trace": trace_log
    }
    
    with open(f"{save_dir}/annotation.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        
    env_interface.close()

if __name__ == "__main__":
    main()