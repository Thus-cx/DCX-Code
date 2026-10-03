import numpy as np
import cv2
import magnum as mn
import habitat_sim


class ProjectionAnnotator:
    def __init__(self, sim, sensor_uuid):
        """
        [V4.0 Raycast版]
        不再依赖 Depth Sensor，改用物理射线检测遮挡。
        完美解决 Depth 不跟随的问题。
        """
        self.sim = sim
        self.sensor_uuid = sensor_uuid
        self.K = None
        self.W = 0
        self.H = 0
        self.aom = self.sim.get_articulated_object_manager()
        # 自动初始化内参
        self._init_intrinsics()

    def _init_intrinsics(self):
        try:
            sensors = self.sim.agents[0]._sensors
            # 模糊匹配 UUID
            if self.sensor_uuid not in sensors:
                valid_keys = list(sensors.keys())
                candidates = [k for k in valid_keys if self.sensor_uuid in k]
                if candidates:
                    self.sensor_uuid = candidates[0]

            if self.sensor_uuid in sensors:
                sensor = sensors[self.sensor_uuid]
                self.H, self.W = sensor.specification().resolution
                hfov_deg = float(sensor.specification().hfov)
                self.hfov_rad = np.deg2rad(hfov_deg)

                # 计算初始焦距
                self.fx = (self.W / 2.0) / np.tan(self.hfov_rad / 2.0)
                self.fy = self.fx
                self.cx = self.W / 2.0
                self.cy = self.H / 2.0

                print(f"[Annotator] Init {self.sensor_uuid}: {self.W}x{self.H}, Raycast Mode Ready")
        except Exception as e:
            print(f"[Annotator] Init failed: {e}")
            self.hfov_rad = np.deg2rad(90)  # Fallback

    def get_camera_transform(self):
        """获取相机在世界坐标系的变换 (T_cam_world)"""
        try:
            sensor = self.sim.agents[0]._sensors.get(self.sensor_uuid)
            if not sensor: return None
            # render_camera.camera_matrix 是 World-to-Camera
            # 我们需要 Camera-to-World (即相机在世界中的位置) 来发射射线
            T_world_cam = np.array(sensor.render_camera.camera_matrix)

            # 求逆得到 T_cam_world (相机本身的位置和朝向)
            T_cam_world = np.linalg.inv(T_world_cam)
            return T_world_cam, T_cam_world
        except Exception:
            return None, None

    def project_3d_point(self, point_3d, T_world_cam):
        """3D -> 2D 投影 (同 V3.0)"""
        p_world = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])
        p_cam = T_world_cam @ p_world

        z_depth = -p_cam[2]
        if z_depth < 0.1: return None  # 在相机后面

        x_ndc = p_cam[0] / z_depth
        y_ndc = p_cam[1] / z_depth

        u = self.fx * x_ndc + self.cx
        v = self.cy - (self.fy * y_ndc)  # Y-Flip fix

        return (int(u), int(v))

    def _is_occluded_by_human(self, camera_pos, target_pos, target_dist):
        # print("Enter human occlude check")
        human_positions = []
        if hasattr(self.sim, "agents_mgr"):
            # 遍历所有智能体 (包括机器人和人类)
            for agent_data in self.sim.agents_mgr._all_agent_data:
                # 忽略没有物理实体的智能体 (比如纯相机)
                if agent_data.articulated_agent is None:
                    continue

                # 获取智能体基座位置
                # sim_obj 是底层的 ArticulatedObject
                human_positions.append(agent_data.articulated_agent.sim_obj.translation)

            # 策略 2: 如果策略 1 失败 (比如 self.sim 只是纯 C++ Simulator), 回退到字符串匹配
        if not human_positions:
            human_objs = self.aom.get_objects_by_handle_substring("human")
            for _, obj in human_objs.items():
                human_positions.append(obj.translation)

            # 如果没找到任何人类/智能体，直接返回不遮挡
        if not human_positions:
            print("!!!!!!!!!Can not find human position!!!!!!!!")
            return False

            # 定义人类/机器人圆柱体参数 (单位: 米)
            # 稍微调大一点半径以确保遮挡更严格
        AGENT_RADIUS = 0.5
        AGENT_HEIGHT = 1.65

        ray_vec = target_pos - camera_pos
        ray_dir = ray_vec.normalized()
        # print(f"human positions: {human_positions}")
        for human_pos in human_positions:
            # --- 圆柱体求交逻辑 (与之前相同) ---
            vec_cam_human = human_pos - camera_pos
            proj_dist = mn.math.dot(vec_cam_human, ray_dir)

            if proj_dist < 0.1 or proj_dist > (target_dist - 0.1):
                continue

            closest_point_on_ray = camera_pos + proj_dist * ray_dir

            dx = closest_point_on_ray.x - human_pos.x
            dz = closest_point_on_ray.z - human_pos.z
            horizontal_dist = np.sqrt(dx * dx + dz * dz)

            if horizontal_dist > AGENT_RADIUS:
                continue

            hit_height = closest_point_on_ray.y
            # human_base_y = human_pos.y
            human_base_y = 0.0
            # 检查高度范围
            if human_base_y < hit_height < (human_base_y + AGENT_HEIGHT):
                return True

        return False

    def check_occlusion_raycast(self, camera_pos, target_pos, target_obj_id):
        """
        [核心] 使用物理射线检测遮挡
        camera_pos: 相机世界坐标 (Vector3)
        target_pos: 目标点世界坐标 (Vector3)
        target_obj_id: 目标物体的 object_id，用于确认打到的是不是它
        """
        # 1. 构建射线
        ray_dir = target_pos - camera_pos
        ray_length = ray_dir.length()

        if ray_length < 0.01: return True  # 重合

        if self._is_occluded_by_human(camera_pos, target_pos, ray_length):
            return True

        ray_dir = ray_dir.normalized()
        ray = habitat_sim.geo.Ray(camera_pos, ray_dir)

        # 2. 发射射线
        # cast_ray 返回 RaycastResults
        raycast_results = self.sim.cast_ray(ray, max_distance=ray_length)

        # 3. 检查击中结果
        if raycast_results.has_hits():
            hit = raycast_results.hits[0]
            hit_dist = hit.ray_distance

            # 如果击中点比目标点更近 (且误差超过阈值)，说明被遮挡
            # 允许 10cm 的误差 (穿模容忍)
            if hit_dist < (ray_length - 0.1):
                # 进一步检查：打到的东西是不是目标物体本身？
                # 如果打到的是目标物体(ID匹配)，则不算遮挡
                if hit.object_id == target_obj_id:
                    return False
                return True  # 被其他物体遮挡

        return False  # 无遮挡

    def annotate_frame(self, rgb_img, world_graph):
        """
        [V4.0 主入口] 不需要 depth_img 参数
        """
        # 动态分辨率适配
        current_h, current_w = rgb_img.shape[:2]
        if current_h != self.H or current_w != self.W:
            self.H, self.W = current_h, current_w
            if not hasattr(self, 'hfov_rad'): self._init_intrinsics()
            self.fx = (self.W / 2.0) / np.tan(self.hfov_rad / 2.0)
            self.fy = self.fx
            self.cx = self.W / 2.0
            self.cy = self.H / 2.0

        # 获取矩阵
        T_world_cam, T_cam_world_mat = self.get_camera_transform()
        if T_world_cam is None: return rgb_img

        # 提取相机位置 (Translation)
        cam_pos_np = T_cam_world_mat[:3, 3]
        cam_pos_mn = mn.Vector3(cam_pos_np[0], cam_pos_np[1], cam_pos_np[2])

        annotated_img = rgb_img.copy()
        rom = self.sim.get_rigid_object_manager()

        for obj in world_graph.get_all_objects():
            sim_handle = getattr(obj, "sim_handle", obj.properties.get("handle", None))
            if not sim_handle: continue

            try:
                if not rom.get_library_has_handle(sim_handle): continue
                sim_obj = rom.get_object_by_handle(sim_handle)
                obj_id = sim_obj.object_id  # 获取物理 ID 用于射线校验

                # A. 计算 AABB 角点
                aabb = sim_obj.aabb
                trans = sim_obj.transformation

                local_pts = [
                    mn.Vector3(x, y, z)
                    for x in [aabb.min.x, aabb.max.x]
                    for y in [aabb.min.y, aabb.max.y]
                    for z in [aabb.min.z, aabb.max.z]
                ]

                projected_points = []
                visible_corner_count = 0

                # B. 投影 + 射线检测
                for pt in local_pts:
                    pt_world = trans.transform_point(pt)  # Magnum Vector3

                    # 1. 投影到 2D
                    pt_np = np.array([pt_world.x, pt_world.y, pt_world.z])
                    uv = self.project_3d_point(pt_np, T_world_cam)

                    if uv:
                        projected_points.append(uv)

                        # 2. [关键] 射线检测该角点是否可见
                        # 只对边框上的点做检测
                        if not self.check_occlusion_raycast(cam_pos_mn, pt_world, obj_id):
                            visible_corner_count += 1

                if not projected_points: continue

                # C. 遮挡判定策略
                # 检查中心点
                center_world = sim_obj.translation
                center_visible = not self.check_occlusion_raycast(cam_pos_mn, center_world, obj_id)

                # 如果中心点被遮挡，且可见角点少于 2 个 -> 视为完全遮挡
                if not center_visible and visible_corner_count < 2:
                    continue

                # D. 绘制
                pts = np.array(projected_points)
                x1, y1 = np.min(pts, axis=0)
                x2, y2 = np.max(pts, axis=0)

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(self.W, x2), min(self.H, y2)

                if (x2 - x1) < 10 or (y2 - y1) < 10: continue
                if (x2 - x1)*(y2 - y1) < 400: continue

                obj_name = obj.name.split(':')[0]
                color = (0, 255, 0)

                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)

                label = obj_name
                font_scale = 0.6 if self.W > 800 else 0.4
                thickness = 2 if self.W > 800 else 1
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.rectangle(annotated_img, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
                cv2.putText(annotated_img, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

            except Exception:
                pass

        return annotated_img


class FurnitureVisibilityChecker:
    def __init__(self, sim, sensor_uuid):
        self.sim = sim
        self.sensor_uuid = sensor_uuid
        self.K = None
        self.W = 0
        self.H = 0
        self.aom = self.sim.get_articulated_object_manager()
        self._init_intrinsics()

    def _init_intrinsics(self):
        try:
            sensors = self.sim.agents[0]._sensors
            if self.sensor_uuid not in sensors:
                valid_keys = list(sensors.keys())
                candidates = [k for k in valid_keys if self.sensor_uuid in k]
                if candidates:
                    self.sensor_uuid = candidates[0]

            if self.sensor_uuid in sensors:
                sensor = sensors[self.sensor_uuid]
                self.H, self.W = sensor.specification().resolution
                hfov_deg = float(sensor.specification().hfov)
                self.hfov_rad = np.deg2rad(hfov_deg)
                self.fx = (self.W / 2.0) / np.tan(self.hfov_rad / 2.0)
                self.fy = self.fx
                self.cx = self.W / 2.0
                self.cy = self.H / 2.0
        except Exception as e:
            print(f"[VisibilityChecker] Init failed: {e}")
            self.hfov_rad = np.deg2rad(90)

    def get_camera_transform(self):
        try:
            sensor = self.sim.agents[0]._sensors.get(self.sensor_uuid)
            if not sensor: return None, None
            T_world_cam = np.array(sensor.render_camera.camera_matrix)
            T_cam_world = np.linalg.inv(T_world_cam)
            return T_world_cam, T_cam_world
        except Exception:
            return None, None

    def project_3d_point(self, point_3d, T_world_cam):
        p_world = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])
        p_cam = T_world_cam @ p_world
        z_depth = -p_cam[2]
        if z_depth < 0.1: return None
        x_ndc = p_cam[0] / z_depth
        y_ndc = p_cam[1] / z_depth
        u = self.fx * x_ndc + self.cx
        v = self.cy - (self.fy * y_ndc)
        return (int(u), int(v))

    def _is_occluded_by_human(self, camera_pos, target_pos, target_dist):
        human_positions = []
        if hasattr(self.sim, "agents_mgr"):
            for agent_data in self.sim.agents_mgr._all_agent_data:
                if agent_data.articulated_agent is None: continue
                human_positions.append(agent_data.articulated_agent.sim_obj.translation)

        if not human_positions: return False

        AGENT_RADIUS = 0.5
        AGENT_HEIGHT = 1.65
        ray_vec = target_pos - camera_pos
        ray_dir = ray_vec.normalized()

        for human_pos in human_positions:
            vec_cam_human = human_pos - camera_pos
            proj_dist = mn.math.dot(vec_cam_human, ray_dir)
            if proj_dist < 0.1 or proj_dist > (target_dist - 0.1): continue

            closest_point_on_ray = camera_pos + proj_dist * ray_dir
            dx = closest_point_on_ray.x - human_pos.x
            dz = closest_point_on_ray.z - human_pos.z
            horizontal_dist = np.sqrt(dx * dx + dz * dz)

            if horizontal_dist > AGENT_RADIUS: continue
            hit_height = closest_point_on_ray.y
            human_base_y = 0.0
            if human_base_y < hit_height < (human_base_y + AGENT_HEIGHT):
                return True
        return False

    def check_occlusion_raycast(self, camera_pos, target_pos, target_obj_id):
        ray_dir = target_pos - camera_pos
        ray_length = ray_dir.length()
        if ray_length < 0.01: return True
        if self._is_occluded_by_human(camera_pos, target_pos, ray_length): return True

        ray_dir = ray_dir.normalized()
        ray = habitat_sim.geo.Ray(camera_pos, ray_dir)
        raycast_results = self.sim.cast_ray(ray, max_distance=ray_length)

        if raycast_results.has_hits():
            hit = raycast_results.hits[0]
            hit_dist = hit.ray_distance
            if hit_dist < (ray_length - 0.1):
                if hit.object_id == target_obj_id: return False
                return True
        return False

    def get_visible_furniture_handles(self, world_graph):
        """核心方法：扫描所有家具，返回当前物理上可见的家具 sim_handle 列表"""
        visible_handles = set()
        T_world_cam, T_cam_world_mat = self.get_camera_transform()
        if T_world_cam is None: return visible_handles

        cam_pos_np = T_cam_world_mat[:3, 3]
        cam_pos_mn = mn.Vector3(cam_pos_np[0], cam_pos_np[1], cam_pos_np[2])
        rom = self.sim.get_rigid_object_manager()

        for obj in world_graph.get_all_furnitures():  # [注意这里只查家具]
            sim_handle = getattr(obj, "sim_handle", obj.properties.get("handle", None))
            if not sim_handle: continue

            try:
                if not rom.get_library_has_handle(sim_handle): continue
                sim_obj = rom.get_object_by_handle(sim_handle)
                obj_id = sim_obj.object_id

                # 1. 检查中心点是否在视野(FOV)内，并且无遮挡
                center_world = sim_obj.translation
                center_np = np.array([center_world.x, center_world.y, center_world.z])
                uv = self.project_3d_point(center_np, T_world_cam)

                # 如果投影点在画面之外，直接判定不可见
                if not uv or uv[0] < 0 or uv[0] > self.W or uv[1] < 0 or uv[1] > self.H:
                    # 如果中心点不在画面内，为了性能考虑(家具很大)，我们可以放宽，也可以直接跳过。
                    # 这里采取严谨策略：尝试测角点
                    pass

                    # 2. 射线遮挡判定
                center_visible = not self.check_occlusion_raycast(cam_pos_mn, center_world, obj_id)

                if center_visible:
                    visible_handles.add(sim_handle)
                    continue

                # 3. 如果中心被遮挡，检查边框角点
                aabb = sim_obj.aabb
                trans = sim_obj.transformation
                local_pts = [
                    mn.Vector3(x, y, z)
                    for x in [aabb.min.x, aabb.max.x]
                    for y in [aabb.min.y, aabb.max.y]
                    for z in [aabb.min.z, aabb.max.z]
                ]

                visible_corner_count = 0
                for pt in local_pts:
                    pt_world = trans.transform_point(pt)
                    pt_np = np.array([pt_world.x, pt_world.y, pt_world.z])
                    uv_corner = self.project_3d_point(pt_np, T_world_cam)

                    # 只有在画面内的角点才有意义
                    if uv_corner and 0 <= uv_corner[0] <= self.W and 0 <= uv_corner[1] <= self.H:
                        if not self.check_occlusion_raycast(cam_pos_mn, pt_world, obj_id):
                            visible_corner_count += 1
                            if visible_corner_count >= 2:  # 只要有两个角点可见就算可见
                                visible_handles.add(sim_handle)
                                break

            except Exception:
                pass

        return visible_handles