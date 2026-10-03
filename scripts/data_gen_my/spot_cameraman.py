import numpy as np
import habitat_sim
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_to_list, quaternion_from_coeff
from habitat_sim.utils.common import quat_from_two_vectors, quat_rotate_vector
from numpy.ma.core import angle
from scipy.spatial.distance import euclidean
from collections import deque  # [MOD] 需要用到双端队列

class SpotCameraman:
    def __init__(self, sim, robot_agent_uid=1, human_agent_uid=0):
        self.sim = sim
        self.robot_uid = robot_agent_uid
        self.human_uid = human_agent_uid
        
        # 配置参数
        self.config = {
            # --- 位置生成参数 ---
            "min_start_dist": 0.8,  # 0.6
            "max_start_dist": 1.2,  # 1.0
            "max_geo_dist_ratio": 1.25,
            "navmesh_search_trials": 50,

            # --- 动态观察参数 ---
            "start_observation_step_range": (100, 200),
            "optimal_view_dist": 1.8,  #1.5
            "focus_bias_decay": 0.98,
            "dist_converge_rate": 0.005,
            "initial_yaw_noise_rad": 0.5,
            
            # --- [MOD] 困难模式参数 (Hard Mode) ---
            "latency_frames": 6,         # 延迟帧数 (6帧 ≈ 0.6秒)
            "noise_scale": 0.05,         # 持续的速度噪声
            "random_close_up_prob": 0.02, # 每帧有 2% 的概率切换跟随距离
            "close_up_dist": 0.6,        # 贴脸距离 (Bucket B)
            "far_dist": 2.5              # 远景距离
        }

        # 内部状态变量
        self.current_step = 0
        self.start_step = 0
        self.target_dist = 0.0
        self.current_yaw_bias = 0.0
        self.is_active = False
        self.pending_start_pos = None
        self.pending_start_rot = None
        
        # [MOD] 新增：动作缓冲区与动态目标距离
        self.action_buffer = deque(maxlen=self.config["latency_frames"] + 1)
        self.current_target_mode_dist = self.config["optimal_view_dist"] # 当前期望距离

    def reset(self, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng
        self.current_step = 0
        self.is_active = False
        self.start_step = self.rng.randint(*self.config["start_observation_step_range"])
        self.target_dist = self.config["min_start_dist"]
        self.current_yaw_bias = self.rng.uniform(-self.config["initial_yaw_noise_rad"], self.config["initial_yaw_noise_rad"])
        
        self.pending_start_pos = None
        self.pending_start_rot = None
        
        # [MOD] 重置缓冲区
        self.action_buffer.clear()
        self.current_target_mode_dist = self.config["optimal_view_dist"]

    def _get_valid_observation_pose(self, rng):
        # ... (保持原样，无需修改) ...
        # (为了节省篇幅，这里省略这部分代码，请保留你原有的实现)
        human_state = self.sim.get_agent_state(self.human_uid)
        human_pos = human_state.position
        pf = self.sim.pathfinder

        for _ in range(self.config["navmesh_search_trials"]):
            angle = rng.uniform(0, 2*np.pi)
            dist = rng.uniform(self.config["min_start_dist"], self.config["max_start_dist"])
            offset = np.array([np.cos(angle)*dist, 0.0, np.sin(angle)*dist])
            candidate_pos = human_pos + offset
            snapped_pos = pf.snap_point(candidate_pos)
            if np.isnan(snapped_pos).any(): continue
            clearance = pf.island_radius(snapped_pos)
            if clearance < 0.3: continue
            geo_dist = self.sim.geodesic_distance(human_pos, snapped_pos)
            euclidean_dist = np.linalg.norm(human_pos-snapped_pos)
            if geo_dist == float('inf') or geo_dist > euclidean_dist*self.config["max_geo_dist_ratio"]: continue
            look_vec = human_pos-snapped_pos
            look_vec[1] = 0
            if np.linalg.norm(look_vec) < 1e-3: continue
            look_vec = look_vec/np.linalg.norm(look_vec)
            initial_forward = np.array([0, 0, -1], dtype=np.float32)
            rotation = quat_from_two_vectors(initial_forward, look_vec)
            return snapped_pos, rotation
        return None, None

    def step(self):
        """
        每个时间步调用，返回控制动作
        """
        self.current_step += 1
        
        # 1. 启动前的等待
        if self.current_step < self.start_step:
            return None
            
        # 2. 启动帧：瞬移到初始位置
        if self.current_step == self.start_step:
            start_pos, start_rot = self._get_valid_observation_pose(self.rng)
            self.pending_start_pos = start_pos
            self.pending_start_rot = start_rot
            if self.pending_start_pos is not None:
                agent_data = self.sim.agents_mgr[self.robot_uid]
                robot_agent = agent_data.articulated_agent
                teleport_pos = np.array(self.pending_start_pos)
                teleport_pos[1] += 0.65
                robot_agent.base_pos = teleport_pos
                robot_agent.base_velocity = np.zeros(3)
                robot_agent.base_angular_velocity = np.zeros(3)

                angle, axis = habitat_sim.utils.common.quat_to_angle_axis(self.pending_start_rot)
                if axis[1] < 0: angle = -angle
                robot_agent.base_rot = angle
                
                print(f"[SpotCameraman] Activated at step {self.current_step}")
                self.is_active = True
                
                # [MOD] 启动时填满 buffer 为 0，防止刚开始就乱动
                for _ in range(self.config["latency_frames"]):
                    self.action_buffer.append(np.array([0.0, 0.0], dtype=np.float32))
                    
                return np.array([0.0, 0.0], dtype=np.float32)

        self.is_active = True
        human_pos = self.sim.get_agent_state(self.human_uid).position
        robot_state = self.sim.get_agent_state(self.robot_uid)
        robot_pos = robot_state.position

        # --- [MOD] 动态目标距离更新 (制造贴脸效果) ---
        if self.rng.rand() < self.config["random_close_up_prob"]:
            # 随机切换期望距离：贴脸、正常、或者远景
            choice = self.rng.choice([self.config["close_up_dist"], self.config["optimal_view_dist"], self.config["far_dist"]], p=[0.4, 0.4, 0.2])
            self.current_target_mode_dist = choice
            
        # 平滑过渡 current_target_mode_dist
        self.target_dist = (1 - self.config["dist_converge_rate"]) * self.target_dist + \
                           self.config["dist_converge_rate"] * self.current_target_mode_dist

        # --- 原始的控制逻辑 (NavMesh Pathfinding) ---
        # (保持你原有的逻辑不变，只修改最后返回部分)
        path = habitat_sim.ShortestPath()
        path.requested_start = robot_pos
        path.requested_end = human_pos
        found_path = self.sim.pathfinder.find_path(path)
        
        if found_path and len(path.points) > 1:
            move_target = path.points[1]
        else:
            move_target = human_pos
            
        vec_to_human = human_pos - robot_pos
        vec_to_path = move_target - robot_pos
        
        angle_to_human = np.arctan2(vec_to_human[0], vec_to_human[2])
        angle_to_path = np.arctan2(vec_to_path[0], vec_to_path[2])
        
        angle_diff = abs(angle_to_human - angle_to_path)
        while angle_diff > np.pi: angle_diff -= 2*np.pi
        angle_diff = abs(angle_diff)
        
        if angle_diff > 1.0:
            target_yaw = angle_to_path
            is_focusing_human = False
        else:
            target_yaw = angle_to_human
            is_focusing_human = True
            
        forward_local = np.array([0, 0, -1], dtype=np.float32)
        robot_forward_global = habitat_sim.utils.common.quat_rotate_vector(robot_state.rotation, forward_local)
        current_yaw = np.arctan2(robot_forward_global[0], robot_forward_global[2])
        
        yaw_error = target_yaw - current_yaw
        while yaw_error > np.pi: yaw_error -= 2*np.pi
        while yaw_error < -np.pi: yaw_error += 2*np.pi
        
        # [MOD] 叠加持续噪声，而不是只在初始时衰减
        # 这会让机器人在对准时依然会有细微的抖动
        continuous_noise = self.rng.normal(0, self.config["noise_scale"])
        final_yaw_error = yaw_error + continuous_noise
        
        # 初始噪声衰减 (保留原逻辑)
        self.current_yaw_bias *= self.config["focus_bias_decay"]
        if is_focusing_human:
            final_yaw_error += self.current_yaw_bias

        ang_vel = np.clip(final_yaw_error * 2.0, -1.0, 1.0)
        
        geodesic_dist = self.sim.geodesic_distance(robot_pos, human_pos)
        if geodesic_dist == float('inf'): geodesic_dist = np.linalg.norm(human_pos - robot_pos)
        
        dist_error = geodesic_dist - self.target_dist
        raw_lin_vel = np.clip(dist_error * 1.5, -1.0, 1.0)
        
        # 拟人化迟疑 (保留)
        if abs(yaw_error) > 0.5:
            lin_vel = 0.0
        elif abs(yaw_error) > 0.2:
            lin_vel = raw_lin_vel * 0.5
        else:
            lin_vel = raw_lin_vel

        # --- [MOD] 核心修改：延迟与缓冲 ---
        ideal_action = np.array([lin_vel, ang_vel], dtype=np.float32)
        
        # 1. 存入缓冲区
        self.action_buffer.append(ideal_action)
        
        # 2. 取出延迟动作 (活在 6 帧之前)
        # 缓冲区满时 popleft 拿出最早加入的；如果没满（刚启动），就拿 ideal_action
        if len(self.action_buffer) >= self.config["latency_frames"]:
            delayed_action = self.action_buffer[0] # 这里我们只是读取队首，不pop，依赖 deque 的 maxlen 自动挤出? 
            # 实际上 deque 不会自动从头部挤出，maxlen 是挤出尾部还是头部？
            # deque.append() 加到右边，maxlen 挤出左边(popleft)。
            # 所以 action_buffer[0] 永远是最早加入的那个。
            # 为了实现 FIFO 延迟，我们应该 append，并且每次只读 [0]。
            # 由于 maxlen 限制，每次 append 新的，旧的如果超过长度就会被自动挤出（丢弃）。
            # 不对，deque 是 append 挤出左端吗？是的。
            # "Once a bounded length deque is full, when new items are added, a corresponding number of items are discarded from the opposite end."
            # 所以 append (右) 会挤出 [0] (左)。
            # 所以 action_buffer[0] 始终是当前存活的最老的动作。
            
            final_action = delayed_action
        else:
            final_action = ideal_action

        return final_action