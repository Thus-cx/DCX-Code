import numpy as np
import magnum as mn

class SpotFollower:
    def __init__(self, spot_agent, human_agent, target_dist=2.5):
        self.spot = spot_agent
        self.human = human_agent
        self.target_dist = target_dist
        
        self.kp_lin = 1.5
        self.kp_ang = 3.0
        self.max_lin_vel = 1.0
        self.max_ang_vel = 1.0

    def get_action(self):
        # 1. 获取变换矩阵 (Matrix4)
        spot_trans = self.spot.base_transformation
        human_trans = self.human.base_transformation
        
        # 2. 提取位置 (Vector3)
        spot_pos = spot_trans.translation
        human_pos = human_trans.translation
        
        # 3. 计算 Spot 的前向向量 (假设 -Z 是前方，视具体 URDF 而定，Spot 通常是 +X 或 -Z)
        # 经过查阅 Habitat Spot URDF，通常 X 轴是前方
        spot_forward = spot_trans.transform_vector(mn.Vector3(1, 0, 0))

        # 4. 距离控制
        vec_to_human = human_pos - spot_pos
        dist = vec_to_human.length()
        
        err_dist = dist - self.target_dist
        if abs(err_dist) < 0.1:
            lin_vel = 0.0
        else:
            lin_vel = np.clip(self.kp_lin * err_dist, -self.max_lin_vel, self.max_lin_vel)

        # 5. 角度控制
        # 投影到 2D 平面
        v_h_2d = mn.Vector3(vec_to_human.x, 0, vec_to_human.z).normalized()
        v_f_2d = mn.Vector3(spot_forward.x, 0, spot_forward.z).normalized()
        
        # 计算角度差
        # cross Y > 0 -> 左转, < 0 -> 右转
        cross = mn.math.cross(v_f_2d, v_h_2d)
        angle = mn.math.angle(v_f_2d, v_h_2d)
        angle_rad = float(angle)
        
        if cross.y < 0:
            angle_rad = -angle_rad
            
        ang_vel = np.clip(self.kp_ang * angle_rad, -self.max_ang_vel, self.max_ang_vel)

        return {
            "action": "agent_1_base_velocity",
            "action_args": {
                "lin_vel": lin_vel,
                "ang_vel": ang_vel
            }
        }