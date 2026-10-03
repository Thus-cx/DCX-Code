import numpy as np
import magnum as mn
import habitat_sim

class SpotFollower:
    def __init__(self, spot_agent, human_agent, sim, target_dist=2.0, stop_dist=0.5):
        self.spot = spot_agent
        self.human = human_agent
        self.sim = sim  # 需要传入 sim 实例以访问 pathfinder
        self.target_dist = target_dist
        self.stop_dist = stop_dist # 距离误差小于此值时不移动
        
        # PID 参数
        self.kp_lin = 2.0
        self.kp_ang = 5.0
        self.max_lin_vel = 1.0 # m/s
        self.max_ang_vel = 1.5 # rad/s

    def get_action(self):
        # 1. 获取变换矩阵和位置
        spot_trans = self.spot.base_transformation
        human_trans = self.human.base_transformation
        spot_pos = spot_trans.translation
        human_pos = human_trans.translation

        # 2. 计算目标位置 (保持在人类附近 target_dist 处)
        # 策略：目标点设在人类当前位置，但 Spot 会在距离 target_dist 处停下
        # 为了更智能，我们应该让 Spot 尝试移动到人类和 Spot 连线上的某一点，或者单纯朝向人类
        
        # 简单策略：目标就是人类的位置
        target_pos = human_pos
        
        # 3. 计算局部坐标系下的目标向量
        # 将目标位置转换到 Spot 的局部坐标系
        # Spot URDF 通常 +X 是前方, +Y 是上方, +Z 是右方 (需根据实际 URDF 确认，这里假设标准 Habitat 约定 -Z 前)
        # 但 Spot 特例通常是 +X 前。我们使用 magnum 自动处理
        spot_inv_trans = spot_trans.inverted()
        local_target = spot_inv_trans.transform_point(target_pos)
        
        # local_target.x: 前方距离
        # local_target.y: 高度差
        # local_target.z: 左右偏移 (左是+还是-取决于坐标系，通常右手系中，若X前Y上，则Z右)

        # 4. 计算距离和角度误差
        # 在水平面上计算距离 (忽略高度差)
        dist_to_target = np.sqrt(local_target.x**2 + local_target.z**2)
        
        # 计算需要旋转的角度 (目标在局部坐标系的角度)
        # atan2(y, x) -> atan2(z, x) 
        angle_error = np.arctan2(local_target.z, local_target.x)

        # 5. 计算控制指令
        lin_vel = 0.0
        ang_vel = 0.0

        # 角度控制 (优先调整朝向)
        if abs(angle_error) > 0.1:
            ang_vel = self.kp_ang * angle_error
            # 限制旋转速度
            ang_vel = np.clip(ang_vel, -self.max_ang_vel, self.max_ang_vel)
            
            # 如果角度偏差太大，先原地旋转，不前进
            if abs(angle_error) > 0.5:
                lin_vel = 0.0
            else:
                # 角度较小时，允许前进
                if dist_to_target > self.target_dist:
                    lin_vel = self.kp_lin * (dist_to_target - self.target_dist)
        else:
            # 角度对准了，处理距离
            if dist_to_target > self.target_dist:
                lin_vel = self.kp_lin * (dist_to_target - self.target_dist)

        # 6. NavMesh 防碰撞检查 (Raycast)
        # 如果前方有障碍物，强制减速或停止
        # 这里使用简单的 step 检查，或者假设 PathFinder 会处理碰撞
        # 为了安全，如果线速度过大，我们限幅
        lin_vel = np.clip(lin_vel, -self.max_lin_vel, self.max_lin_vel)

        # 7. 返回 Action
        # Habitat BaseVelAction 参数通常是 [linear_velocity, angular_velocity]
        # 注意：某些配置可能是 [ang, lin]，需根据 config 确认。通常是 [lin, ang]
        return [lin_vel, ang_vel]