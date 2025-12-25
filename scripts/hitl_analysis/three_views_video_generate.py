import os
import json
import gzip
import math
import numpy as np
import magnum as mn
import cv2
import habitat_sim
from habitat.articulated_agent_controllers import HumanoidSeqPoseController

# ================= 用户配置区域 =================
# 请根据你的实际路径修改以下变量
DATA_ROOT = "../../data"  # PARTNR 项目的 data 文件夹根目录
EPISODE_JSON_PATH = "../../data/hitl_data/p5_multi_val/processed/best/0.json" # 你提供的 json 路径

# 资源路径 (相对于 DATA_ROOT)
SCENE_DATASET_CONFIG = "hssd-hab/hssd-hab-partnr.scene_dataset_config.json" 
ROBOT_URDF = "robots/hab_spot_arm/urdf/hab_spot_arm.urdf"
HUMAN_URDF = "humanoids/humanoid_data/female_2/female_2.urdf" # 可以换成 male_1 等
HUMAN_MOTION_PKL = "humanoids/humanoid_data/walking_motion_processed_smplx.pkl"

# 输出设置
OUTPUT_VIDEO_NAME = "partnr_hitl_replay.mp4"
FPS = 30
RESOLUTION = (640, 480) # (宽, 高)

# 代理索引 (根据 0.json 的 user_index_to_agent_index_map 确认)
# 通常 PARTNR 中: Agent 0 = Robot (Spot), Agent 1 = Human
AGENT_ID_ROBOT = 1
AGENT_ID_HUMAN = 0

# ===============================================

def load_json_data(path):
    print(f"Loading data from {path}...")
    if path.endswith(".gz"):
        with gzip.open(path, 'rt') as f:
            return json.load(f)
    else:
        with open(path, 'r') as f:
            return json.load(f)

def make_cfg(scene_id):
    # 1. 配置仿真器后端
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = os.path.join(DATA_ROOT, SCENE_DATASET_CONFIG)
    sim_cfg.scene_id = scene_id
    sim_cfg.enable_physics = True # 必须开启物理以加载关节体
    sim_cfg.gpu_device_id = 0
    sim_cfg.load_semantic_mesh = True # 如果需要语义分割可开启

    # 2. 配置传感器 (这里我们配置给 Agent 0, 也就是机器人，作为主视角)
    # 注意：我们稍后会在循环中动态移动这些传感器来实现不同视角
    
    # 视角 A: 机器人第一人称 (RGB)
    robot_cam = habitat_sim.CameraSensorSpec()
    robot_cam.uuid = "robot_ego_rgb"
    robot_cam.sensor_type = habitat_sim.SensorType.COLOR
    robot_cam.resolution =, RESOLUTION]
    robot_cam.position = [0.0, 0.0, 0.0] # 稍后会绑定到机器人头部
    
    # 视角 B: 跟踪人类的第三人称 (RGB)
    track_cam = habitat_sim.CameraSensorSpec()
    track_cam.uuid = "track_human_rgb"
    track_cam.sensor_type = habitat_sim.SensorType.COLOR
    track_cam.resolution =, RESOLUTION]
    
    # 视角 C: 全局上帝视角 (RGB)
    global_cam = habitat_sim.CameraSensorSpec()
    global_cam.uuid = "global_rgb"
    global_cam.sensor_type = habitat_sim.SensorType.COLOR
    global_cam.resolution =, RESOLUTION]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [robot_cam, track_cam, global_cam]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def get_transform_from_state(state_dict):
    # 从 JSON state 解析位置和旋转
    pos = state_dict['position']
    rot_quat = state_dict['rotation'] # [x, y, z, w] 或者是 [w, x, y, z]? 
    # Habitat JSON 通常是 [w, x, y, z] 或者 [x, y, z, w]。
    # mn.Quaternion 构造函数是 (vector, scalar) 即 (x,y,z, w)
    # 或者是 (w, x, y, z) 取决于库。
    # 让我们假设 JSON 是 [x, y, z, w] (SciPy 风格) 或 [w, x, y, z] (Habitat 内部风格)
    # 这里做一个简单的长度归一化检查，通常 w 是实部
    
    # 尝试构建 Magnum 向量
    translation = mn.Vector3(pos)
    
    # 注意：0.json 的 rotation 格式如果是 [0.0, 0.9, 0.0, 0.2] 这种，通常是四元数
    # Habitat-Sim 的 quaternion 是 mn.Quaternion(imag, real) -> (Vector3, float)
    # 如果数据源是 habitat-lab 生成的，它通常是 [x, y, z, w]
    rotation = mn.Quaternion(mn.Vector3(rot_quat[0:3]), rot_quat[2])
    
    return translation, rotation

def main():
    # 1. 加载数据
    record_data = load_json_data(EPISODE_JSON_PATH)
    episode_info = record_data['episode']
    scene_id = episode_info['scene_id'] # 例如 "102817140"
    
    # 注意：scene_id 在 json 中可能只是哈希码，我们需要完整的.glb 路径
    # 通常 HSSD 场景位于 data/scene_datasets/hssd/scenes/{scene_id}.glb
    # 但我们使用 scene_dataset_config 加载，所以可以直接尝试用 scene_id (或者带扩展名)
    full_scene_id = f"{scene_id}.glb" # 尝试加上后缀
    
    print(f"Scene ID: {full_scene_id}")
    print(f"Total Frames: {len(record_data['frames'])}")

    # 2. 初始化仿真器
    try:
        cfg = make_cfg(full_scene_id)
        sim = habitat_sim.Simulator(cfg)
    except Exception as e:
        print(f"Error initializing simulator: {e}")
        print("Tip: Check if SCENE_DATASET_CONFIG path is correct.")
        return

    # 3. 加载 Articulated Agents (机器人和人类)
    ao_mgr = sim.get_articulated_object_manager()
    
    # 加载 Spot 机器人
    robot_urdf_file = os.path.join(DATA_ROOT, ROBOT_URDF)
    spot = ao_mgr.add_articulated_object_from_urdf(robot_urdf_file)
    if not spot:
        print(f"Failed to load Spot from {robot_urdf_file}")
        return
    print(f"Loaded Spot Robot. ID: {spot.object_id}")

    # 加载人类 (SMPL-X URDF)
    human_urdf_file = os.path.join(DATA_ROOT, HUMAN_URDF)
    human = ao_mgr.add_articulated_object_from_urdf(human_urdf_file)
    if not human:
        print(f"Failed to load Human from {human_urdf_file}")
        return
    print(f"Loaded Humanoid. ID: {human.object_id}")

    # 4. 初始化人类动作控制器
    # 我们使用 SeqPoseController 来读取.pkl 并计算走路姿态
    motion_path = os.path.join(DATA_ROOT, HUMAN_MOTION_PKL)
    human_controller = HumanoidSeqPoseController(motion_pose_path=motion_path)
    
    # 5. 准备视频录制
    # 我们将三个视角拼接成一个宽视频
    video_width = RESOLUTION * 3
    video_height = RESOLUTION[1]
    video_writer = cv2.VideoWriter(
        OUTPUT_VIDEO_NAME, 
        cv2.VideoWriter_fourcc(*'mp4v'), 
        FPS, 
        (video_width, video_height)
    )

    # 6. 主循环：逐帧渲染
    # 我们跳过空白帧，只处理有效帧
    valid_frames = [f for f in record_data['frames'] if 'agent_states' in f]
    
    # 用于计算人类移动速度
    prev_human_pos = None
    
    print("Starting rendering loop...")
    for i, frame in enumerate(valid_frames):
        agent_states = frame['agent_states']
        object_states = frame.get('object_states',)
        
        # --- A. 更新机器人状态 ---
        if len(agent_states) > AGENT_ID_ROBOT:
            r_state = agent_states
            r_trans, r_rot = get_transform_from_state(r_state)
            
            # 设置 Spot 的基座位置 (修正高度，Spot 的 URDF 原点通常在中心)
            # data 中的 position 通常是地面投影点，可能需要 +z 偏移
            # 这里先直接设置，观察是否有穿模
            spot.translation = r_trans
            spot.rotation = r_rot
            
            # 简单的关节归位 (Spot 的腿部关节)
            # 如果不设置，它会是僵尸状态。我们可以给它一个默认站立姿态
            # 这里简化处理，只做刚体移动

        # --- B. 更新人类状态 (混合驱动) ---
        if len(agent_states) > AGENT_ID_HUMAN:
            h_state = agent_states
            h_trans, h_rot = get_transform_from_state(h_state)
            
            # 1. 强制设定根节点位置 (Ground Truth)
            human.translation = h_trans
            human.rotation = h_rot
            
            # 2. 计算是否在移动
            is_moving = False
            if prev_human_pos is not None:
                dist = (mn.Vector3(h_trans) - mn.Vector3(prev_human_pos)).length()
                speed = dist * FPS # 估算速度 m/s
                if speed > 0.1: # 阈值
                    is_moving = True
            
            # 3. 更新动作控制器
            # 如果在移动，播放下一帧走路动画；否则保持/重置
            if is_moving:
                human_controller.next_pose(cycle=True)
            else:
                # 如果静止，这行代码会让它回到站立姿态 (或者你可以不调用，保持上一帧动作)
                # human_controller.calculate_stop_pose() 
                pass 

            # 4. 获取并应用关节角度
            # get_pose() 返回的是 Pose 对象，包含 joints (四元数列表)
            pose = human_controller.get_pose()
            human.set_joint_positions(pose.joints)
            
            prev_human_pos = h_trans

        # --- C. 更新物体状态 (Replay Objects) ---
        # 如果你想让场景里的杯子、碗也跟着动
        # 注意：这需要 get_object_manager 并且通过 object_id 找到对应物体
        # 由于 ID 映射比较复杂，这里暂时略过，只关注 Agent

        # --- D. 更新相机位置 ---
        
        # 1. Robot Ego Camera (绑定到 Spot 头部)
        # 假设 Spot 头部 Link ID 是最后一个或特定的。
        # 简单起见，我们计算一个相对于 Spot 基座的偏移量
        # Spot 高度约 0.5m，头在前方 0.4m
        robot_head_local_offset = mn.Vector3(0.4, 0.5, 0.0) 
        robot_cam_pos = spot.transformation.transform_point(robot_head_local_offset)
        
        sim.agents._sensors["robot_ego_rgb"].set_transformation_from_spec() # 重置
        # 这一步是 hack，直接修改 sensor 的 scene node
        sensor_node = sim.agents._sensors["robot_ego_rgb"]._sensor_object.object
        sensor_node.translation = robot_cam_pos
        sensor_node.rotation = spot.rotation # 相机朝向跟随机器人

        # 2. Tracking Camera (注视人类)
        human_pos = human.translation
        # 相机位置：人类身后 2 米，高 1.8 米
        # 简单算法：当前人类位置 + 固定偏移 (更高级的需要平滑)
        cam_pos = human_pos + mn.Vector3(2.0, 1.8, 2.0) 
        sim.agents._sensors["track_human_rgb"].set_transformation_from_spec()
        sensor_node_track = sim.agents._sensors["track_human_rgb"]._sensor_object.object
        sensor_node_track.translation = cam_pos
        # 计算 LookAt 矩阵
        look_at_mat = mn.Matrix4.look_at(cam_pos, human_pos + mn.Vector3(0,1,0), mn.Vector3(0,1,0))
        sensor_node_track.rotation = mn.Quaternion.from_matrix(look_at_mat.rotation())

        # 3. Global Camera (上帝视角)
        sim.agents._sensors["global_rgb"].set_transformation_from_spec()
        sensor_node_global = sim.agents._sensors["global_rgb"]._sensor_object.object
        # 设置在房间中心高处，垂直向下
        global_pos = human_pos + mn.Vector3(0, 5.0, 0.1) # 稍微偏移以免万向节死锁
        sensor_node_global.translation = global_pos
        look_at_mat_global = mn.Matrix4.look_at(global_pos, human_pos, mn.Vector3(0,0,1)) # Z-up in camera space
        sensor_node_global.rotation = mn.Quaternion.from_matrix(look_at_mat_global.rotation())

        # --- E. 渲染 ---
        obs = sim.get_sensor_observations()
        
        # 拼接图像
        rgb_robot = obs["robot_ego_rgb"][..., :3] # 去掉 Alpha 通道
        rgb_track = obs["track_human_rgb"][..., :3]
        rgb_global = obs["global_rgb"][..., :3]
        
        # 转换为 BGR (OpenCV 格式)
        frame_img = np.hstack((rgb_robot, rgb_track, rgb_global))
        frame_img = frame_img[..., ::-1] # RGB to BGR
        
        video_writer.write(frame_img)

        if i % 50 == 0:
            print(f"Processed frame {i}/{len(valid_frames)}")

    # 清理
    video_writer.release()
    sim.close()
    print(f"Saved video to {OUTPUT_VIDEO_NAME}")

if __name__ == "__main__":
    main()