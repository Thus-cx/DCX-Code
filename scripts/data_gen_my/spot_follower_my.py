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
        spot_trans = self.spot.base_transformation
        human_trans = self.human.base_transformation
        
        spot_pos = spot_trans.translation
        human_pos = human_trans.translation
        
        # Spot 坐标系通常 X 朝前
        spot_forward = spot_trans.transform_vector(mn.Vector3(1, 0, 0))

        vec_to_human = human_pos - spot_pos
        vec_to_human_flat = mn.Vector3(vec_to_human.x, 0, vec_to_human.z)
        dist = vec_to_human_flat.length()
        
        lin_vel = 0.0
        err_dist = dist - self.target_dist
        if abs(err_dist) > 0.1:
            lin_vel = np.clip(self.kp_lin * err_dist, -self.max_lin_vel, self.max_lin_vel)

        ang_vel = 0.0
        if vec_to_human_flat.length() > 0.01:
            dir_to_human = vec_to_human_flat.normalized()
            spot_forward_flat = mn.Vector3(spot_forward.x, 0, spot_forward.z).normalized()
            
            cross = mn.math.cross(spot_forward_flat, dir_to_human)
            angle = mn.math.angle(spot_forward_flat, dir_to_human)
            angle_rad = float(angle)
            if cross.y < 0:
                angle_rad = -angle_rad
                
            ang_vel = np.clip(self.kp_ang * angle_rad, -self.max_ang_vel, self.max_ang_vel)

        return [lin_vel, ang_vel]