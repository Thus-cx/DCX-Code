import os
os.environ['__EGL_VENDOR_LIBRARY_FILENAMES'] = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
os.environ['EGL_PLATFORM'] = 'surfaceless'
os.environ['EGL_DEVICE_ID'] = '0'
os.environ['HABITAT_SIM_HEADLESS'] = '1'

print("=== 使用 NVIDIA EGL 配置 ===")

try:
    import habitat_sim
    from habitat_sim import Simulator
    
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = ""
    backend_cfg.enable_physics = False
    
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    
    with habitat_sim.Simulator(cfg) as sim:
        print("✓ Habitat-sim NVIDIA EGL 模式启动成功")
        
        from habitat_llm.tests.test_constrained_generation import test_grammar_generation
        test_grammar_generation()
        print("✓ 测试通过")
        
except Exception as e:
    print(f"✗ 错误: {e}")
    import traceback
    traceback.print_exc()
