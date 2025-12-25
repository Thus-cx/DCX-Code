# test_sensor.py
import habitat_sim
import os

os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

def make_cfg(scene_path):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.create_renderer = True
    sim_cfg.gpu_device_id = 0
    return sim_cfg

def make_agent_cfg():
    agent_cfg = habitat_sim.AgentConfiguration()
    sensor_spec = habitat_sim.SensorSpec()
    sensor_spec.uuid = "test_rgb"
    sensor_spec.resolution = [128, 128]
    sensor_spec.position = [0.0, 1.5, 0.0]
    sensor_spec.orientation = [0.0, 0.0, 0.0]
    sensor_spec.hfov = 90
    sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    agent_cfg.sensor_specifications = [sensor_spec]
    return agent_cfg

try:
    # 尝试加载场景
    scene_path = "data/fpss/stages/102817140.glb"
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"{scene_path}")

    # 构造 config
    sim_cfg = make_cfg(scene_path)
    agent_cfg = make_agent_cfg()

    # 创建 simulator
    print("Attempting to create simulator with COLOR sensor...")
    backend_cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(backend_cfg)

    # 获取观测
    obs = sim.get_sensor_observations()[0]["test_rgb"]
    print(f"✅ Success! Got observation shape: {obs.shape}, dtype: {obs.dtype}")
    sim.close()

except Exception as e:
    print(f"❌ Failed to create COLOR sensor: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
