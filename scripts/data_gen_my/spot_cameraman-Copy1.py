import numpy as np
import habitat_sim
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_to_list, quaternion_from_coeff
from habitat_sim.utils.common import quat_from_two_vectors, quat_rotate_vector
from numpy.ma.core import angle
from scipy.spatial.distance import euclidean
from wandb.wandb_agent import agent


class SpotCameraman:
    def __init__(self, sim, robot_agent_uid=1, human_agent_uid=0):
        self.sim = sim
        self.robot_uid = robot_agent_uid
        self.human_uid = human_agent_uid
        # 配置参数
        self.config = {
            # 位置生成参数
            "min_start_dist": 0.6, # 初始距离人类的最小距离（米）
            "max_start_dist": 1.0, # 初始距离人类的最大距离（米）
            "max_geo_dist_ratio": 1.25, # 测地距离/欧式距离 的最大比值（用于判断是否绕远路/隔墙）
            "navmesh_search_trials": 50, # 寻找合法初始点的最大尝试次数

            # 动态观察参数
            "start_observation_step_range": (100, 200), # 随机开始介入的时间步范围
            "optimal_view_dist": 1.5, #最终希望达到的最佳观察距离
            "focus_bias_decay": 0.98, # 视角偏差的衰减系数（模拟从没对准到对准的过程）
            "dist_converge_rate": 0.005, # 距离调整的收敛速率

            # 初始噪声（模拟刚开始不知道往哪看）
            "initial_yaw_noise_rad": 0.5, # 初始偏航角噪声（弧度）
        }

        # 内部状态变量
        self.current_step = 0
        self.start_step = 0
        self.target_dist = 0.0
        self.current_yaw_bias = 0.0
        self.is_active = False
        self.pending_start_pos = None
        self.pending_start_rot = None

    def reset(self, rng=None):
        """
        每个episode开始时调用
        功能：
        1. 随机决定介入时间
        2. 计算并强制设置机器人的初始位置（在人类附近、同房间、NavMesh上）
        :param rng:
        :return:
        """
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng
        self.current_step = 0
        self.is_active = False
        # 1. 随机决定何时开始观察
        self.start_step = self.rng.randint(*self.config["start_observation_step_range"])
        # 2. 设置动态调整的初始状态
        # current_human_pos = self.sim.get_agent_state(self.human_uid).position
        self.target_dist = self.config["min_start_dist"]
        self.current_yaw_bias = self.rng.uniform(-self.config["initial_yaw_noise_rad"], self.config["initial_yaw_noise_rad"]) # 随机一个初始视角偏差
        # 3. 寻找并设置合法的初始位置
        # start_pos, start_rot = self._get_valid_observation_pose(rng)
        # 新增存储
        # self.pending_start_pos = start_pos
        # self.pending_start_rot = start_rot
        self.pending_start_pos = None
        self.pending_start_rot = None

        # if start_pos is not None:
        #     # new_state = habitat_sim.AgentState()
        #     # new_state.position = start_pos
        #     # new_state.rotation = start_rot
        #     # # 强制设定机器人位置
        #     # print("agents_mgr: ", self.sim.agents_mgr)
        #     # robot_agent_sim = self.sim.agents_mgr[self.robot_uid]
        #     # # robot_agent = robot_agent_sim.articulated_agent
        #     # # robot_agent = self.sim.get_agent(self.robot_uid)
        #     # robot_agent_sim.set_state(new_state)
        #     agent_data = self.sim.agents_mgr[self.robot_uid]
        #     robot_agent = agent_data.articulated_agent
        #     robot_agent.base_pos = start_pos
        #     angle, axis = habitat_sim.utils.common.quat_to_angle_axis(start_rot)
        #     if axis[1] < 0:
        #         angle = -angle
        #     robot_agent.base_rot = angle
        # else:
        #     print("[SpotCameraman] Warning: Could not find valid start pose nearby!!!")


    def _get_valid_observation_pose(self, rng):
        """
        获取当前观察人类的可行初始位置
        相关参考：
        - habitat_sim.nav.PathFinder(sim.pathfinder)
        - sim.pathfinder.snap_point()
        - sim.pathfinder.geodesic_distance()
        :return:
        """
        human_state = self.sim.get_agent_state(self.human_uid)
        human_pos = human_state.position
        pf = self.sim.pathfinder

        for _ in range(self.config["navmesh_search_trials"]):
            # 1. 随机采样级坐标
            angle = rng.uniform(0, 2*np.pi)
            dist = rng.uniform(self.config["min_start_dist"], self.config["max_start_dist"])
            # 2. 计算候选点 （假设在同一水平面，habitat中Y轴为垂直地面的高度轴）
            offset = np.array([np.cos(angle)*dist, 0.0, np.sin(angle)*dist])
            candidate_pos = human_pos + offset
            # 3. 吸附到NavMesh
            # snap_point会返回最近的可导航点，如果太远或无可达点，可能返回NaN
            snapped_pos = pf.snap_point(candidate_pos)
            # 如果返回了nan，直接跳过这次尝试，进行下次尝试
            if np.isnan(snapped_pos).any():
                continue
            # 新增优化：空间宽敞度检测
            clearance = pf.island_radius(snapped_pos) # 到该点最近的障碍物的距离
            if clearance < 0.3:
                continue
            # 4. 验证连通性，避免隔墙
            geo_dist = self.sim.geodesic_distance(human_pos, snapped_pos) # 走路距离
            euclidean_dist = np.linalg.norm(human_pos-snapped_pos) # 直线距离
            # 如果走路距离inf不可达或者远大于直线距离（说明绕了一大圈），则丢弃
            if geo_dist == float('inf') or geo_dist > euclidean_dist*self.config["max_geo_dist_ratio"]:
                continue
            # 5. 生成对应的旋转，朝向人类
            look_vec = human_pos-snapped_pos
            look_vec[1] = 0 # 忽略高度差，水平旋转
            if np.linalg.norm(look_vec) < 1e-3:
                continue
            look_vec = look_vec/np.linalg.norm(look_vec) # 朝向单位向量
            # habitat中-Z方向是前方，需要旋转-Z对准look_vec
            # 需要计算从初始-Z（[0，0，-1]）旋转到look_vec的四元数
            initial_forward = np.array([0, 0, -1], dtype=np.float32)
            rotation = quat_from_two_vectors(initial_forward, look_vec)
            return snapped_pos, rotation

        return None, None

    def step(self):
        """
        每个时间步调用，返回控制动作
        :return:low_level_actions["agent_1"]的值，查看low_level_actions中值的要求！！！
        """
        self.current_step += 1
        if self.current_step < self.start_step:
            return None
        if self.current_step == self.start_step:
            start_pos, start_rot = self._get_valid_observation_pose(self.rng)
            self.pending_start_pos = start_pos
            self.pending_start_rot = start_rot
            if self.pending_start_pos is not None:
                agent_data = self.sim.agents_mgr[self.robot_uid]
                robot_agent = agent_data.articulated_agent
                # 增加高度偏移，防止埋入地下被弹回
                teleport_pos = np.array(self.pending_start_pos)
                teleport_pos[1] += 0.65
                robot_agent.base_pos = teleport_pos
                # 显式将线速度和角速度置零
                robot_agent.base_velocity = np.zeros(3)
                robot_agent.base_angular_velocity = np.zeros(3)

                angle, axis = habitat_sim.utils.common.quat_to_angle_axis(self.pending_start_rot)
                if axis[1] < 0: angle = -angle
                robot_agent.base_rot = angle
                human_pos = self.sim.get_agent_state(self.human_uid).position
                print(f"[Human Position] Human is at {human_pos}")
                print(f"[SpotCameraman] Teleported to {self.pending_start_pos} at step {self.current_step}")
                self.is_active = True
                return np.array([0.0, 0.0], dtype=np.float32)
        self.is_active = True
        human_pos = self.sim.get_agent_state(self.human_uid).position
        robot_state = self.sim.get_agent_state(self.robot_uid)
        robot_pos = robot_state.position
        # 动态参数更新
        # 1. 目标距离平滑过渡：慢慢从“近处特写”变为“全景观察”
        self.target_dist = (1-self.config["dist_converge_rate"])*self.target_dist + self.config["dist_converge_rate"]*self.config["optimal_view_dist"]
        # 2. 视角偏差衰减：慢慢修正对不准的问题
        self.current_yaw_bias *= self.config["focus_bias_decay"]

        # 新增修改：使用PathFinder寻找路径
        path = habitat_sim.ShortestPath()
        path.requested_start = robot_pos
        path.requested_end = human_pos
        found_path = self.sim.pathfinder.find_path(path)
        # 如果找到路径，且路径点多于1个，朝向第一个路径点走，这样可以自动绕过障碍物
        if found_path and len(path.points) > 1:
            move_target = path.points[1]
        else: # 如果没有路径，则直线走
            move_target = human_pos
        vec_to_human = human_pos - robot_pos # 朝向人的方向
        vec_to_path = move_target - robot_pos # 实际该走的方向
        # 计算各种目标角度
        angle_to_human = np.arctan2(vec_to_human[0], vec_to_human[2])
        angle_to_path = np.arctan2(vec_to_path[0], vec_to_path[2])
        # 决定看人还是看路？优先看路
        angle_diff = abs(angle_to_human - angle_to_path)
        while angle_diff > np.pi: angle_diff -= 2*np.pi
        angle_diff = abs(angle_diff)
        if angle_diff > 1.0: # 约60度
            target_yaw = angle_to_path
            is_focusing_human = False
        else:
            target_yaw = angle_to_human
            is_focusing_human = True
        # 计算当前角度误差
        forward_local = np.array([0, 0, -1], dtype=np.float32)
        robot_forward_global = habitat_sim.utils.common.quat_rotate_vector(robot_state.rotation, forward_local)
        current_yaw = np.arctan2(robot_forward_global[0], robot_forward_global[2])
        yaw_error = target_yaw - current_yaw
        while yaw_error > np.pi: yaw_error -= 2*np.pi
        while yaw_error < -np.pi: yaw_error += 2*np.pi
        # 叠加初始噪声，只在看人时生效，看路时要精准
        if is_focusing_human:
            final_yaw_error = yaw_error + self.current_yaw_bias
        else:
            final_yaw_error = yaw_error
        # 计算角速度
        ang_vel = np.clip(final_yaw_error*2.0, -1.0, 1.0)
        # 计算线速度
        geodesic_dist = self.sim.geodesic_distance(robot_pos, human_pos) # 计算与人类的实际路径距离
        if geodesic_dist == float('inf'): geodesic_dist = np.linalg.norm(human_pos - robot_pos)
        dist_error = geodesic_dist - self.target_dist
        raw_lin_vel = np.clip(dist_error*1.5, -1.0, 1.0)
        # 拟人化迟疑，如果需要大幅度转身，它会先减速或停下，先转身再走，避免侧身滑步的假感,也模拟丢失目标后寻找的过程
        if abs(yaw_error) > 0.5:
            lin_vel = 0.0
        elif abs(yaw_error) > 0.2:
            lin_vel = raw_lin_vel * 0.5
        else:
            lin_vel = raw_lin_vel
        return np.array([lin_vel, ang_vel], dtype=np.float32)
        # # 计算控制动作
        # # 1. 线速度计算
        # vec_to_human = human_pos - robot_pos
        # curr_dist = np.linalg.norm(vec_to_human)
        # dist_error = curr_dist - self.target_dist
        # lin_vel = np.clip(dist_error*1.5, -1.0, 1.0)
        # # 2. 角速度计算
        # forward_local = np.array([0, 0, -1], dtype=np.float32)
        # target_yaw = np.arctan2(vec_to_human[0], vec_to_human[2])
        # # robot_forward_global = robot_state.rotation.transform_vector(forward_local)
        # robot_forward_global = quat_rotate_vector(robot_state.rotation, forward_local)
        # current_yaw = np.arctan2(robot_forward_global[0], robot_forward_global[2])
        # yaw_error = target_yaw - current_yaw
        # while yaw_error > np.pi: yaw_error -= 2*np.pi
        # while yaw_error < -np.pi: yaw_error += 2*np.pi
        # final_yaw_error = yaw_error + self.current_yaw_bias
        # ang_vel = np.clip(final_yaw_error*2.0, -1.0, 1.0)
        # return np.array([lin_vel, ang_vel], dtype=np.float32)