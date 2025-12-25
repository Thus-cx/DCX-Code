import os
import habitat_sim

# 尝试不同的渲染后端配置
render_backends = [
    {"HABITAT_SIM_HEADLESS": "1", "MAGNUM_DISABLE_EXTENSIONS": "1"},
    {"HABITAT_SIM_HEADLESS": "1", "MAGNUM_GPU_VALIDATION": "0"},
    {"LIBGL_ALWAYS_SOFTWARE": "1", "HABITAT_SIM_HEADLESS": "1"},
]

for i, env_vars in enumerate(render_backends):
    print(f"尝试配置 {i+1}: {env_vars}")
    
    # 设置环境变量
    for key, value in env_vars.items():
        os.environ[key] = value
    
    try:
        # 重新导入 habitat_sim 以确保使用新的环境变量
        import importlib
        importlib.reload(habitat_sim)
        
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = ""
        backend_cfg.enable_physics = False
        backend_cfg.create_renderer = i != 2  # 对于软件渲染，禁用渲染器创建
        
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        
        with habitat_sim.Simulator(cfg) as sim:
            print(f"✓ 配置 {i+1} 成功")
            
            from habitat_llm.tests.test_constrained_generation import test_grammar_generation
            test_grammar_generation()
            print("✓ 测试通过")
            break
            
    except Exception as e:
        print(f"✗ 配置 {i+1} 失败: {e}")
        continue
else:
    print("所有配置尝试都失败了")
