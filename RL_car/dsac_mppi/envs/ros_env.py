import gymnasium as gym
import rospy
import numpy as np
from gymnasium import spaces
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SetModelState, GetModelState, SpawnModel
from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker, MarkerArray
import math
import tf.transformations
from dsac_mppi.envs.frenet_utils import FrenetTransform, compute_tracking_reward

class MyCarEnv(gym.Env):
    collision_reward = -1000.0
    success_reward = 1000.0
    soft_out_of_bounds_limit = 3.0
    hard_out_of_bounds_limit = 4.0
    soft_out_of_bounds_penalty = -50.0
    hard_out_of_bounds_reward = -300.0
    shaping_reward_low = -50.0
    shaping_reward_high = 50.0

    LOW_LEVEL_ACTION_LOW = np.array([-1.0, -1.0], dtype=np.float32)
    LOW_LEVEL_ACTION_HIGH = np.array([2.0, 1.0], dtype=np.float32)
    HIGH_LEVEL_ACTION_LOW = np.array([0.8, -2.2, 0.2], dtype=np.float32)
    HIGH_LEVEL_ACTION_HIGH = np.array([3.5, 2.2, 1.2], dtype=np.float32)

    def __init__(self, control_mode="low_level_velocity", dynamics_model="inertia", curriculum_stage=None):
        super(MyCarEnv, self).__init__()
        self.control_mode = str(control_mode)
        self.dynamics_model = str(dynamics_model)
        if self.control_mode not in ("low_level_velocity", "high_level_frenet"):
            raise ValueError(f"Unsupported control_mode: {self.control_mode}")
        if self.dynamics_model not in ("inertia", "ideal"):
            raise ValueError(f"Unsupported dynamics_model: {self.dynamics_model}")
        self.curriculum_stage = curriculum_stage
        
        try:
            rospy.init_node('my_car_rl_node', anonymous=True)
        except rospy.ROSException:
            pass
        
        if self.control_mode == "high_level_frenet":
            self.action_space = spaces.Box(
                low=self.HIGH_LEVEL_ACTION_LOW.copy(),
                high=self.HIGH_LEVEL_ACTION_HIGH.copy(),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Box(
                low=self.LOW_LEVEL_ACTION_LOW.copy(),
                high=self.LOW_LEVEL_ACTION_HIGH.copy(),
                dtype=np.float32,
            )

        self.n_laser_beams = 400 
        self.max_laser_range = 10.0
        self.map_size = 40.0 
        self.goal_reach_threshold = 0.4
        
        # Frenet坐标系
        self.observation_space = spaces.Box(low=-1.0, high=1.0,
                                            shape=(self.n_laser_beams + 4,),
                                            dtype=np.float32)
        
        # 路径可视化
        self.pub_path_marker = rospy.Publisher('/global_path_marker', MarkerArray, queue_size=1, latch=True)
        self.pub_mppi_trajectories_marker = rospy.Publisher('/mppi_trajectories_marker', MarkerArray, queue_size=1)
        self.pub_mppi_action_marker = rospy.Publisher('/mppi_action_marker', MarkerArray, queue_size=1)
        # 小车姿态箭头可视化
        self.pub_car_pose_marker = rospy.Publisher('/car_pose_marker', Marker, queue_size=1)
        self.frenet_transform = None
        self.start_pos = np.array([0.0, 0.0])
        self.last_frenet_s = 0.0
        self.last_abs_frenet_d = 0.0
        
        self.pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        rospy.wait_for_service('/gazebo/get_model_state')
        self.get_state_proxy = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        self.spawn_model_proxy = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        
        self.target_pos = np.array([0.0, 0.0])
        self.current_pos = np.array([0.0, 0.0])
        self.current_yaw = 0.0
        self.last_distance_to_goal = None
        self.step_count = 0
        self.max_steps = 500

        # 3-DOF 惯性模型状态 (Surge, Sway, Yaw)
        self.velocity = np.array([0.0, 0.0, 0.0])  # [u, v, r]
        self.last_cmd_velocity = np.array([0.0, 0.0, 0.0], dtype=float)
        self.mass = 2.0    # 质量系数
        self.damping = 0.5  # 阻尼系数
        self.dt = 0.1       # 控制周期 (秒)

        # 激光雷达异步回调缓存
        self.latest_scan = None
        rospy.Subscriber('/scan', LaserScan, self._scan_callback, queue_size=1)

        # 机器人状态异步回调缓存
        self.usv_pos = np.array([0.0, 0.0])
        self.usv_yaw = 0.0
        self.dynamic_obstacles = []
        self.dynamic_obstacle_prefixes = ("obs_", "dynamic_obstacle_")
        self.dynamic_obstacle_radius = 0.4
        rospy.Subscriber('/gazebo/model_states', ModelStates, self._model_states_callback, queue_size=1)

        # ROS 仿真时间频率控制器（锁定 MDP 步长为 10Hz，对应 dt=0.1）
        self.rate = rospy.Rate(10)
    
    def _get_robot_position(self):
        return self.usv_pos.copy(), self.usv_yaw

    def get_planner_state(self):
        """Return a read-only snapshot used by evaluation-time action optimizers."""
        current_pos, current_yaw = self._get_robot_position()
        planner_velocity = self.velocity if self.dynamics_model == "inertia" else self.last_cmd_velocity
        return {
            "position": current_pos.copy(),
            "yaw": float(current_yaw),
            "velocity": planner_velocity.copy(),
            "target_position": self.target_pos.copy(),
            "frenet_transform": self.frenet_transform,
            "scan": self.latest_scan,
            "dynamic_obstacles": [dict(obstacle) for obstacle in self.dynamic_obstacles],
            "action_low": self.LOW_LEVEL_ACTION_LOW.copy(),
            "action_high": self.LOW_LEVEL_ACTION_HIGH.copy(),
            "control_mode": self.control_mode,
            "dynamics_model": self.dynamics_model,
            "high_level_action_low": self.HIGH_LEVEL_ACTION_LOW.copy(),
            "high_level_action_high": self.HIGH_LEVEL_ACTION_HIGH.copy(),
            "dt": float(self.dt),
            "mass": float(self.mass),
            "damping": float(self.damping),
            "max_laser_range": float(self.max_laser_range),
            "last_action": self.last_action.copy() if hasattr(self, "last_action") else np.zeros(2, dtype=np.float32),
            "last_high_level_action": self.last_high_level_action.copy()
            if hasattr(self, "last_high_level_action")
            else np.zeros(3, dtype=np.float32),
        }

    def _scan_callback(self, msg):
        self.latest_scan = msg

    def _model_states_callback(self, msg):
        dynamic_obstacles = []
        try:
            idx = msg.name.index('usv')
            pos = msg.pose[idx].position
            ori = msg.pose[idx].orientation
            self.usv_pos = np.array([pos.x, pos.y])
            _, _, yaw = tf.transformations.euler_from_quaternion((ori.x, ori.y, ori.z, ori.w))
            self.usv_yaw = yaw
        except ValueError:
            pass
        for idx, name in enumerate(msg.name):
            if name == 'usv' or not name.startswith(self.dynamic_obstacle_prefixes):
                continue
            pos = msg.pose[idx].position
            twist = msg.twist[idx] if idx < len(msg.twist) else None
            if twist is None:
                velocity = np.zeros(2, dtype=float)
            else:
                velocity = np.array([twist.linear.x, twist.linear.y], dtype=float)
            dynamic_obstacles.append({
                "name": name,
                "position": np.array([pos.x, pos.y], dtype=float),
                "velocity": velocity,
                "radius": float(self.dynamic_obstacle_radius),
            })
        self.dynamic_obstacles = dynamic_obstacles

    def step(self, action):
        requested_action = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.control_mode == "high_level_frenet" and requested_action.size >= 3:
            executed_action, high_level_info = self.high_level_action_to_low_level(requested_action)
            self.last_high_level_action = requested_action[:3].astype(np.float32)
        else:
            executed_action = np.clip(requested_action[:2], self.LOW_LEVEL_ACTION_LOW, self.LOW_LEVEL_ACTION_HIGH)
            high_level_info = {}

        vel_msg = self._low_level_action_to_twist(executed_action)
        self.pub_cmd_vel.publish(vel_msg)

        # 等待当前控制量被 Gazebo 物理步实际执行后，再采样下一状态。
        self.rate.sleep()

        # 4. 获取激光雷达数据（非阻塞回调）
        while self.latest_scan is None:
            rospy.sleep(0.01)
        scan_data = self.latest_scan

        # 5. 获取机器人当前状态与目标距离
        self.current_pos, self.current_yaw = self._get_robot_position()
        dist_to_goal = np.linalg.norm(self.target_pos - self.current_pos)

        # 可视化小车姿态箭头
        self._visualize_car_pose()

        # 4. 计算 Frenet 坐标系相关信息
        if self.frenet_transform is not None:
            # 统一使用 self.current_pos 进行一次计算即可
            frenet_s, frenet_d = self.frenet_transform.cartesian_to_frenet(self.current_pos)
            heading_to_path = self.frenet_transform.get_heading_error(self.current_yaw, frenet_s)
            
            path_length = self.frenet_transform.path_length
            distance_remaining = path_length - frenet_s
            distance_remaining = max(distance_remaining, 0.0) 
        else:
            frenet_s = dist_to_goal
            frenet_d = 0.0
            heading_to_path = 0.0
            path_length = dist_to_goal + 1.0
            distance_remaining = dist_to_goal

        # 5. 构建状态观测 (Observation)
        laser_state = self._process_scan_data(scan_data)
        frenet_s_norm = (frenet_s / path_length) * 2 - 1
        frenet_d_norm = np.clip(frenet_d / 3.0, -1.0, 1.0)
        heading_norm = heading_to_path / math.pi
        remaining_norm = (distance_remaining / path_length) * 2 - 1

        obs = np.concatenate((laser_state, [frenet_s_norm, frenet_d_norm, heading_norm, remaining_norm])).astype(np.float32)

        # 6. 初始化回合状态
        terminated = False
        truncated = False
        reward = 0.0
        terminal_reason = "running"
        self.step_count += 1

        # 获取最小障碍物距离
        min_laser_dist = np.min(scan_data.ranges if scan_data else [0])
        if min_laser_dist == float('inf'):
            min_laser_dist = self.max_laser_range

        # 初始化上一帧的 s 值 (用于计算 delta_s)
        if not hasattr(self, 'last_frenet_s'):
            self.last_frenet_s = frenet_s
        
        # 计算单步推进距离 (防止后退刷分)
        delta_s = frenet_s - self.last_frenet_s

        # ---------------------------------------------------------
        # 7. 核心奖励函数逻辑 (Reward Design)
        # ---------------------------------------------------------
        
        # 7.1 终结状态判别：碰撞惩罚与到达终点奖励

        abs_frenet_d = abs(float(frenet_d))
        soft_out_of_bounds = abs_frenet_d > self.soft_out_of_bounds_limit
        hard_out_of_bounds = abs_frenet_d > self.hard_out_of_bounds_limit

        if min_laser_dist < 0.25:
            reward = self.collision_reward
            terminated = True
            terminal_reason = "collision"
            print("Collision detected!")

        elif hard_out_of_bounds:
            reward = self.hard_out_of_bounds_reward
            terminated = True
            terminal_reason = "out_of_bounds"
            print("Out of bounds!")

        elif distance_remaining < self.goal_reach_threshold and abs_frenet_d <= 1.0:
            reward = self.success_reward
            terminated = True
            terminal_reason = "success"
            print("Goal reached!")
            
        else:
            reward_dict = compute_tracking_reward(
                delta_s,
                frenet_d,
                heading_to_path,
                min_obstacle_dist=min_laser_dist,
                previous_abs_frenet_d=self.last_abs_frenet_d,
                action=executed_action,
                previous_action=self.last_action,
            )
            reward = float(np.clip(reward_dict["total"], self.shaping_reward_low, self.shaping_reward_high))
            if soft_out_of_bounds:
                reward = min(reward + self.soft_out_of_bounds_penalty, self.soft_out_of_bounds_penalty)

        # ---------------------------------------------------------

        # 8. 超时截断判断
        if self.step_count >= self.max_steps:
            truncated = True
            if terminal_reason == "running":
                terminal_reason = "timeout"
            print(f"Timeout at {self.step_count} steps.")

        # 9. 更新上一帧的 s 值
        self.last_frenet_s = frenet_s
        self.last_abs_frenet_d = abs(float(frenet_d))
        # 10. 更新上一帧动作（用于下一step的动作平滑度惩罚）
        self.last_action = np.array([float(executed_action[0]), float(executed_action[1])], dtype=np.float32)

        info = {
            "current_position": self.current_pos.copy(),
            "current_yaw": float(self.current_yaw),
            "target_position": self.target_pos.copy(),
            "distance_to_goal": float(dist_to_goal),
            "distance_remaining": float(distance_remaining),
            "frenet_s": float(frenet_s),
            "frenet_d": float(frenet_d),
            "heading_to_path": float(heading_to_path),
            "min_laser_dist": float(min_laser_dist),
            "is_success": bool(distance_remaining < self.goal_reach_threshold and abs_frenet_d <= 1.0),
            "is_collision": bool(min_laser_dist < 0.25),
            "is_timeout": bool(truncated),
            "is_out_of_bounds": bool(soft_out_of_bounds),
            "terminal_reason": terminal_reason,
            "control_mode": self.control_mode,
            "dynamics_model": self.dynamics_model,
            "requested_action": requested_action.astype(np.float32),
            "executed_low_level_action": executed_action.astype(np.float32),
        }
        info.update(high_level_info)

        return obs, reward, terminated, truncated, info

    def high_level_action_to_low_level(self, action):
        """Convert DSAC's high-level Frenet target into an ideal low-level command."""
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        padded = self.HIGH_LEVEL_ACTION_LOW.copy()
        padded[: min(3, action.size)] = action[:3]
        delta_s, target_d, target_speed = np.clip(
            padded,
            self.HIGH_LEVEL_ACTION_LOW,
            self.HIGH_LEVEL_ACTION_HIGH,
        )
        target_point = self.high_level_action_to_target_point(np.array([delta_s, target_d, target_speed], dtype=np.float32))
        position = np.asarray(self.current_pos, dtype=float)
        yaw = float(self.current_yaw)
        target_vec = np.asarray(target_point, dtype=float) - position
        target_heading = math.atan2(float(target_vec[1]), float(target_vec[0]))
        heading_error = self._wrap_angle(target_heading - yaw)
        heading_scale = max(0.2, math.cos(float(np.clip(heading_error, -math.pi / 2, math.pi / 2))))
        surge = float(target_speed) * heading_scale
        yaw_cmd = float(np.clip(1.8 * heading_error, self.LOW_LEVEL_ACTION_LOW[1], self.LOW_LEVEL_ACTION_HIGH[1]))
        low_level = np.array(
            [
                np.clip(surge, self.LOW_LEVEL_ACTION_LOW[0], self.LOW_LEVEL_ACTION_HIGH[0]),
                yaw_cmd,
            ],
            dtype=np.float32,
        )
        return low_level, {
            "high_level_delta_s": float(delta_s),
            "high_level_target_d": float(target_d),
            "high_level_target_speed": float(target_speed),
            "high_level_target_point": np.asarray(target_point, dtype=np.float32),
            "high_level_heading_error": float(heading_error),
        }

    def high_level_action_to_target_point(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size < 3:
            padded = self.HIGH_LEVEL_ACTION_LOW.copy()
            padded[: action.size] = action
            action = padded
        delta_s, target_d, _ = np.clip(action[:3], self.HIGH_LEVEL_ACTION_LOW, self.HIGH_LEVEL_ACTION_HIGH)
        if self.frenet_transform is None:
            heading = np.array([math.cos(self.current_yaw), math.sin(self.current_yaw)], dtype=float)
            lateral = np.array([-heading[1], heading[0]], dtype=float)
            return np.asarray(self.current_pos, dtype=float) + heading * float(delta_s) + lateral * float(target_d)
        current_s, _ = self.frenet_transform.cartesian_to_frenet(np.asarray(self.current_pos, dtype=float))
        target_s = float(np.clip(current_s + float(delta_s), 0.0, self.frenet_transform.path_length))
        return self.frenet_transform.frenet_to_cartesian(target_s, float(target_d))

    def _low_level_action_to_twist(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        surge = float(action[0]) if action.size > 0 else 0.0
        yaw_cmd = float(action[1]) if action.size > 1 else 0.0
        if self.dynamics_model == "inertia":
            control_input = np.array([surge, 0.0, yaw_cmd], dtype=float)
            acceleration = (control_input / self.mass) - (self.damping * self.velocity)
            self.velocity = self.velocity + acceleration * self.dt
            cmd = self.velocity.copy()
        else:
            cmd = np.array([surge, 0.0, yaw_cmd], dtype=float)
            self.last_cmd_velocity = cmd.copy()
        vel_msg = Twist()
        vel_msg.linear.x = float(cmd[0])
        vel_msg.linear.y = float(cmd[1])
        vel_msg.angular.z = float(cmd[2])
        return vel_msg

    @staticmethod
    def _wrap_angle(angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pub_cmd_vel.publish(Twist())
        self.step_count = 0
        # 重置 3-DOF 惯性速度状态
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.last_cmd_velocity = np.array([0.0, 0.0, 0.0], dtype=float)
        # 重置动作平滑度状态
        self.last_action = np.array([0.0, 0.0])
        self.last_high_level_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.last_abs_frenet_d = 0.0

        stage_cfg = self._curriculum_stage_config()
        range_limit = float(stage_cfg["range_limit"])
        start_x = np.random.uniform(-range_limit/2, range_limit/2)
        start_y = np.random.uniform(-range_limit/2, range_limit/2)
        self.start_pos = np.array([start_x, start_y])

        while True:
            goal_x = np.random.uniform(-range_limit/2, range_limit/2)
            goal_y = np.random.uniform(-range_limit/2, range_limit/2)
            goal_distance = np.linalg.norm([goal_x - start_x, goal_y - start_y])
            if stage_cfg["min_goal_distance"] <= goal_distance <= stage_cfg["max_goal_distance"]:
                self.target_pos = np.array([goal_x, goal_y])
                break

        # 初始化Frenet坐标系（随机曲率增强泛化）
        curve_offset = np.random.uniform(-stage_cfg["curve_offset"], stage_cfg["curve_offset"])
        self.frenet_transform = FrenetTransform(self.start_pos, self.target_pos, curve_offset=curve_offset)

        # 可视化
        self._update_marker("marker_start", start_x, start_y, "Blue")
        self._update_marker("marker_goal", self.target_pos[0], self.target_pos[1], "Red")
        self._visualize_path()

        # 移动机器人
        state_msg = ModelState()
        state_msg.model_name = 'usv'
        state_msg.pose.position.x = start_x
        state_msg.pose.position.y = start_y
        state_msg.pose.position.z = 0.05

        # 先计算起点到目标的直线大方向作为基准参考
        path_vec = self.target_pos - self.start_pos
        base_yaw = math.atan2(path_vec[1], path_vec[0])

        # 给一个正负 0.5 弧度（约30度）的随机偏差，降低初始探索难度
        yaw = base_yaw + np.random.uniform(-0.5, 0.5)

        # 确保 yaw 在 -pi 到 pi 之间
        while yaw > math.pi: yaw -= 2 * math.pi
        while yaw < -math.pi: yaw += 2 * math.pi
        
        q = tf.transformations.quaternion_from_euler(0, 0, yaw)
        state_msg.pose.orientation.x = q[0]
        state_msg.pose.orientation.y = q[1]
        state_msg.pose.orientation.z = q[2]
        state_msg.pose.orientation.w = q[3]

        state_msg.twist.linear.x = 0
        state_msg.twist.linear.y = 0
        state_msg.twist.angular.z = 0

        try:
            self.set_state_proxy(state_msg)
        except rospy.ServiceException:
            pass

        self.current_pos = np.array([start_x, start_y])
        self.current_yaw = yaw
        self.last_frenet_s = 0.0
        self.last_abs_frenet_d = 0.0
        path_length = self.frenet_transform.path_length
        self.max_steps = int((path_length / 0.2) * 10 * 2) + 200

        print("Reset: Start(%.1f,%.1f) -> Goal(%.1f,%.1f) | Path Length: %.2fm" %
              (start_x, start_y, self.target_pos[0], self.target_pos[1], path_length))

        while self.latest_scan is None:
            rospy.sleep(0.01)
        data = self.latest_scan

        return self._build_obs(data, path_length), {}

    def _curriculum_stage_config(self):
        stage = 4 if self.curriculum_stage is None else int(self.curriculum_stage)
        configs = {
            0: {"range_limit": 8.0, "min_goal_distance": 2.0, "max_goal_distance": 5.0, "curve_offset": 0.0},
            1: {"range_limit": 12.0, "min_goal_distance": 4.0, "max_goal_distance": 8.0, "curve_offset": 0.8},
            2: {"range_limit": 14.0, "min_goal_distance": 5.0, "max_goal_distance": 10.0, "curve_offset": 1.4},
            3: {"range_limit": 18.0, "min_goal_distance": 6.0, "max_goal_distance": 14.0, "curve_offset": 2.0},
            4: {"range_limit": 20.0, "min_goal_distance": 4.0, "max_goal_distance": 28.0, "curve_offset": 2.5},
        }
        return configs.get(stage, configs[4])

    def _visualize_path(self):
        if self.frenet_transform is None:
            return

        # 生成路径点
        path_points = self.frenet_transform.generate_path_points(num_points=50)
        marker_array = MarkerArray()

        # --- 1. 路径线 (Line Strip) ---
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "global_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP  
        marker.action = Marker.ADD
        marker.scale.x = 0.1  # 线宽
        marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0; marker.color.a = 0.8

        marker.pose.orientation.w = 1.0
        
        for point in path_points:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = 0.1  
            marker.points.append(p)
        marker_array.markers.append(marker)

        # 计算路径方向用于箭头朝向
        path_vector = self.target_pos - self.start_pos
        angle = math.atan2(path_vector[1], path_vector[0])
        q = tf.transformations.quaternion_from_euler(0, 0, angle)

        # --- 2. 起点箭头 (Start Arrow) ---
        start_marker = Marker()
        start_marker.header.frame_id = "odom"
        start_marker.header.stamp = rospy.Time.now()
        start_marker.ns = "global_path"
        start_marker.id = 1
        start_marker.type = Marker.ARROW
        start_marker.action = Marker.ADD
        start_marker.pose.position.x = self.start_pos[0]
        start_marker.pose.position.y = self.start_pos[1]
        start_marker.pose.position.z = 0.1
        start_marker.pose.orientation.x = q[0]
        start_marker.pose.orientation.y = q[1]
        start_marker.pose.orientation.z = q[2]
        start_marker.pose.orientation.w = q[3]
        start_marker.scale.x = 1.0; start_marker.scale.y = 0.2; start_marker.scale.z = 0.2 
        start_marker.color.r = 0.0; start_marker.color.g = 0.0; start_marker.color.b = 1.0; start_marker.color.a = 1.0 
        marker_array.markers.append(start_marker)

        # --- 3. 终点箭头 (End Arrow) ---
        end_marker = Marker()
        end_marker.header.frame_id = "odom"
        end_marker.header.stamp = rospy.Time.now()
        end_marker.ns = "global_path"
        end_marker.id = 2
        end_marker.type = Marker.ARROW
        end_marker.action = Marker.ADD
        end_marker.pose.position.x = self.target_pos[0]
        end_marker.pose.position.y = self.target_pos[1]
        end_marker.pose.position.z = 0.1
        end_marker.pose.orientation.x = q[0]
        end_marker.pose.orientation.y = q[1]
        end_marker.pose.orientation.z = q[2]
        end_marker.pose.orientation.w = q[3]
        end_marker.scale.x = 1.0; end_marker.scale.y = 0.2; end_marker.scale.z = 0.2
        end_marker.color.r = 1.0; end_marker.color.g = 0.0; end_marker.color.b = 0.0; end_marker.color.a = 1.0 
        marker_array.markers.append(end_marker)

        # 发布
        self.pub_path_marker.publish(marker_array)

    def publish_mppi_trajectories(self, sampled, weighted, frame_id="odom"):
        marker_array = MarkerArray()
        stamp = rospy.Time(0)
        lifetime = rospy.Duration(0.8)

        clear_marker = Marker()
        clear_marker.header.frame_id = frame_id
        clear_marker.header.stamp = stamp
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        for marker_id, trajectory in enumerate(sampled or []):
            marker = self._make_trajectory_marker(
                trajectory,
                frame_id=frame_id,
                stamp=stamp,
                namespace="mppi_samples",
                marker_id=marker_id,
                color=(0.2, 1.0, 0.2, 0.18),
                line_width=0.012,
                lifetime=lifetime,
                z=0.16,
            )
            if marker is not None:
                marker_array.markers.append(marker)

        weighted_marker = self._make_trajectory_marker(
            weighted,
            frame_id=frame_id,
            stamp=stamp,
            namespace="mppi_weighted",
            marker_id=0,
            color=(1.0, 0.0, 0.0, 1.0),
            line_width=0.04,
            lifetime=lifetime,
            z=0.2,
        )
        if weighted_marker is not None:
            marker_array.markers.append(weighted_marker)

        self.pub_mppi_trajectories_marker.publish(marker_array)

    def publish_mppi_action_commands(self, raw_action, optimized_action, planner_state=None, frame_id="odom"):
        marker_array = MarkerArray()
        stamp = rospy.Time(0)
        lifetime = rospy.Duration(0.8)

        clear_marker = Marker()
        clear_marker.header.frame_id = frame_id
        clear_marker.header.stamp = stamp
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        if planner_state is None:
            position = self.current_pos
            yaw = self.current_yaw
        else:
            position = np.asarray(planner_state.get("position", self.current_pos), dtype=float)
            yaw = float(planner_state.get("yaw", self.current_yaw))

        raw_marker = self._make_action_arrow_marker(
            raw_action,
            position=position,
            yaw=yaw,
            lateral_offset=0.24,
            frame_id=frame_id,
            stamp=stamp,
            namespace="rl_raw_action",
            marker_id=0,
            color=(0.1, 0.65, 1.0, 0.95),
            lifetime=lifetime,
            z=0.42,
        )
        optimized_marker = self._make_action_arrow_marker(
            optimized_action,
            position=position,
            yaw=yaw,
            lateral_offset=-0.24,
            frame_id=frame_id,
            stamp=stamp,
            namespace="mppi_final_action",
            marker_id=0,
            color=(1.0, 0.25, 0.0, 0.95),
            lifetime=lifetime,
            z=0.48,
        )
        marker_array.markers.extend([raw_marker, optimized_marker])
        self.pub_mppi_action_marker.publish(marker_array)

    def _make_action_arrow_marker(
        self,
        action,
        position,
        yaw,
        lateral_offset,
        frame_id,
        stamp,
        namespace,
        marker_id,
        color,
        lifetime,
        z,
    ):
        action = np.asarray(action, dtype=float).reshape(-1)
        surge = float(action[0]) if action.size > 0 else 0.0
        yaw_cmd = float(action[1]) if action.size > 1 else 0.0
        direction = float(yaw + 0.8 * yaw_cmd)
        if surge < 0.0:
            direction += math.pi
        length = float(np.clip(0.35 + 0.45 * abs(surge), 0.25, 1.4))
        heading_vec = np.array([math.cos(direction), math.sin(direction)], dtype=float)
        lateral_vec = np.array([-math.sin(yaw), math.cos(yaw)], dtype=float)
        start = np.asarray(position[:2], dtype=float) + lateral_vec * float(lateral_offset)
        end = start + heading_vec * length

        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = int(marker_id)
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.lifetime = lifetime
        marker.scale.x = 0.035
        marker.scale.y = 0.09
        marker.scale.z = 0.14
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = float(color[3])

        p0 = Point()
        p0.x = float(start[0])
        p0.y = float(start[1])
        p0.z = float(z)
        p1 = Point()
        p1.x = float(end[0])
        p1.y = float(end[1])
        p1.z = float(z)
        marker.points.extend([p0, p1])
        return marker

    def _make_trajectory_marker(
        self,
        trajectory,
        frame_id,
        stamp,
        namespace,
        marker_id,
        color,
        line_width,
        lifetime,
        z,
    ):
        if trajectory is None:
            return None
        points = np.asarray(trajectory, dtype=float).reshape(-1, 2)
        if len(points) < 2:
            return None

        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = int(marker_id)
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.lifetime = lifetime
        marker.pose.orientation.w = 1.0
        marker.scale.x = float(line_width)
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = float(color[3])
        for point in points:
            p = Point()
            p.x = float(point[0])
            p.y = float(point[1])
            p.z = float(z)
            marker.points.append(p)
        return marker

    def _visualize_car_pose(self):
        """发布小车当前位置和朝向的箭头 Marker"""
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "car_pose"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # 位置
        marker.pose.position.x = self.current_pos[0]
        marker.pose.position.y = self.current_pos[1]
        marker.pose.position.z = 0.2  

        # 朝向 (根据 yaw 转换成四元数)
        q = tf.transformations.quaternion_from_euler(0, 0, self.current_yaw)
        marker.pose.orientation.x = q[0]
        marker.pose.orientation.y = q[1]
        marker.pose.orientation.z = q[2]
        marker.pose.orientation.w = q[3]

        # 箭头尺寸
        marker.scale.x = 1.0  
        marker.scale.y = 0.15 
        marker.scale.z = 0.15 

        # 颜色 (紫色)
        marker.color.r = 0.8
        marker.color.g = 0.2
        marker.color.b = 0.8
        marker.color.a = 1.0

        self.pub_car_pose_marker.publish(marker)

    def _update_marker(self, marker_name, x, y, color):
        sdf_xml = """<sdf version="1.6">
        <model name="%s">
            <static>0</static>
            <link name="link">
            <gravity>0</gravity>
            <visual name="visual">
                <geometry><sphere><radius>0.25</radius></sphere></geometry>
                <material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>Gazebo/%s</name></script></material>
                <cast_shadows>0</cast_shadows>
            </visual>
            </link>
        </model>
        </sdf>""" % (marker_name, color)

        state_msg = ModelState()
        state_msg.model_name = marker_name
        state_msg.pose.position.x = x
        state_msg.pose.position.y = y
        state_msg.pose.position.z = 0.5
        state_msg.pose.orientation.w = 1.0
        state_msg.twist.linear.x = 0
        state_msg.twist.linear.y = 0
        state_msg.twist.angular.z = 0

        try:
            resp = self.set_state_proxy(state_msg)
            if not resp.success:
                raise rospy.ServiceException("Model not found")
        except rospy.ServiceException:
            try:
                initial_pose = Pose()
                initial_pose.position.x = x
                initial_pose.position.y = y
                initial_pose.position.z = 0.5
                initial_pose.orientation.w = 1.0
                self.spawn_model_proxy(marker_name, sdf_xml, "", initial_pose, "world")
            except Exception:
                pass

    def _process_scan_data(self, data):
        if data is None:
            return np.zeros(self.n_laser_beams, dtype=np.float32)

        raw_ranges = np.array(data.ranges)
        raw_ranges = np.nan_to_num(raw_ranges, nan=self.max_laser_range, posinf=self.max_laser_range, neginf=self.max_laser_range)
        raw_ranges = np.clip(raw_ranges, 0, self.max_laser_range)

        if len(raw_ranges) == self.n_laser_beams:
            processed = raw_ranges
        else:
            x_old = np.linspace(0, 1, len(raw_ranges))
            x_new = np.linspace(0, 1, self.n_laser_beams)
            processed = np.interp(x_new, x_old, raw_ranges)

        return (processed / self.max_laser_range).astype(np.float32)

    def _build_obs(self, scan_data, path_length):
        if self.frenet_transform is not None:
            frenet_s, frenet_d = self.frenet_transform.cartesian_to_frenet(self.current_pos)
            
            # 使用算好的 frenet_s 传给 get_heading_error 即可
            heading_to_path = self.frenet_transform.get_heading_error(self.current_yaw, frenet_s)
            
            distance_remaining = max(0.0, path_length - frenet_s)
        else:
            frenet_s = 0.0
            frenet_d = 0.0
            heading_to_path = 0.0
            distance_remaining = path_length

        frenet_s_norm = (frenet_s / path_length) * 2 - 1
        frenet_d_norm = np.clip(frenet_d / 3.0, -1.0, 1.0)
        heading_norm = heading_to_path / math.pi
        remaining_norm = (distance_remaining / path_length) * 2 - 1

        laser_state = self._process_scan_data(scan_data)
        return np.concatenate((laser_state, [frenet_s_norm, frenet_d_norm, heading_norm, remaining_norm])).astype(np.float32)
