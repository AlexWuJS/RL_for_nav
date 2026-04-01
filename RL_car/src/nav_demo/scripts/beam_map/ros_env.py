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
from frenet_utils import FrenetTransform, frenet_reward

class MyCarEnv(gym.Env):
    def __init__(self):
        super(MyCarEnv, self).__init__()
        
        try:
            rospy.init_node('my_car_rl_node', anonymous=True)
        except rospy.ROSException:
            pass
        
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]),
                                       high=np.array([2.0, 1.0]),
                                       dtype=np.float32)

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
        # 小车姿态箭头可视化
        self.pub_car_pose_marker = rospy.Publisher('/car_pose_marker', Marker, queue_size=1)
        self.frenet_transform = None
        self.start_pos = np.array([0.0, 0.0])
        self.last_frenet_s = 0.0
        
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
        self.mass = 2.0    # 质量系数
        self.damping = 0.5  # 阻尼系数
        self.dt = 0.1       # 控制周期 (秒)

        # 激光雷达异步回调缓存
        self.latest_scan = None
        rospy.Subscriber('/scan', LaserScan, self._scan_callback, queue_size=1)

        # 机器人状态异步回调缓存
        self.usv_pos = np.array([0.0, 0.0])
        self.usv_yaw = 0.0
        rospy.Subscriber('/gazebo/model_states', ModelStates, self._model_states_callback, queue_size=1)

        # ROS 仿真时间频率控制器（锁定 MDP 步长为 10Hz，对应 dt=0.1）
        self.rate = rospy.Rate(10)
    
    def _get_robot_position(self):
        return self.usv_pos.copy(), self.usv_yaw

    def _scan_callback(self, msg):
        self.latest_scan = msg

    def _model_states_callback(self, msg):
        try:
            idx = msg.name.index('usv')
            pos = msg.pose[idx].position
            ori = msg.pose[idx].orientation
            self.usv_pos = np.array([pos.x, pos.y])
            _, _, yaw = tf.transformations.euler_from_quaternion((ori.x, ori.y, ori.z, ori.w))
            self.usv_yaw = yaw
        except ValueError:
            pass

    def step(self, action):
        # 1. 解析动作为控制输入 (欠驱动 USV: action[0]=Surge期望, action[1]=Yaw期望)
        control_input = np.array([
            float(action[0]),  # 期望 Surge 力/速度
            0.0,               # Sway 无直接控制输入 (欠驱动)
            float(action[1])   # 期望 Yaw 角速度
        ])

        # 2. 惯性结算: a = F/m - damping*v, V_{t+1} = V_t + a*dt
        acceleration = (control_input / self.mass) - (self.damping * self.velocity)
        self.velocity = self.velocity + acceleration * self.dt

        # 3. 发布带有惯性的实际速度指令
        vel_msg = Twist()
        vel_msg.linear.x = self.velocity[0]   # 实际 Surge 速度
        vel_msg.linear.y = self.velocity[1]   # 实际 Sway 速度
        vel_msg.angular.z = self.velocity[2]  # 实际 Yaw 角速度
        self.pub_cmd_vel.publish(vel_msg)

        # 2. 获取激光雷达数据（非阻塞回调）
        while self.latest_scan is None:
            rospy.sleep(0.01)
        scan_data = self.latest_scan

        # 3. 获取机器人当前状态与目标距离
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
        max_frenet_d_allowed = 3.0

        if min_laser_dist < 0.25:
            reward = -100.0
            terminated = True
            print("Collision detected!")
            
        elif abs(frenet_d) > max_frenet_d_allowed:  # 越界死亡逻辑
            reward = -100.0
            terminated = True
            print(f"Out of bounds! Deviated {frenet_d:.2f}m from path.")
            
        elif distance_remaining < self.goal_reach_threshold:
            reward = +1000.0
            terminated = True
            print("Goal reached!")
            
        else:
            # 7.2 路径跟踪奖励 (Frenet，放大2.5倍鼓励探索)
            frenet_reward_dict = frenet_reward(delta_s, frenet_d, heading_to_path)
            frenet_amplification_factor = 2.5
            reward += frenet_reward_dict['total'] * frenet_amplification_factor

            # 7.3 安全避障惩罚（收缩边界，k=5.0，上限10.0）
            safe_distance = 0.45
            if min_laser_dist < safe_distance:
                k = 5.0
                penalty = math.exp(k * (safe_distance - min_laser_dist)) - 1
                reward -= min(penalty, 10.0)  # 上限-10

            # 7.4 动作平滑度约束（削弱系数，降低前期探索阻力）
            surge_change_penalty = abs(action[0] - self.last_action[0]) * 0.2
            yaw_change_penalty = abs(action[1] - self.last_action[1]) * 0.1
            reward -= surge_change_penalty + yaw_change_penalty

            # 7.5 存活时间惩罚
            reward -= 0.1

        # ---------------------------------------------------------

        # 8. 超时截断判断
        if self.step_count >= self.max_steps:
            truncated = True
            print(f"Timeout at {self.step_count} steps.")

        # 9. 更新上一帧的 s 值
        self.last_frenet_s = frenet_s
        # 10. 更新上一帧动作（用于下一step的动作平滑度惩罚）
        self.last_action = np.array([float(action[0]), float(action[1])])

        # 11. 锁定 MDP 步长为仿真时间 10Hz
        self.rate.sleep()

        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pub_cmd_vel.publish(Twist())
        self.step_count = 0
        # 重置 3-DOF 惯性速度状态
        self.velocity = np.array([0.0, 0.0, 0.0])
        # 重置动作平滑度状态
        self.last_action = np.array([0.0, 0.0])

        range_limit = 20.0
        start_x = np.random.uniform(-range_limit/2, range_limit/2)
        start_y = np.random.uniform(-range_limit/2, range_limit/2)
        self.start_pos = np.array([start_x, start_y])

        while True:
            goal_x = np.random.uniform(-range_limit/2, range_limit/2)
            goal_y = np.random.uniform(-range_limit/2, range_limit/2)
            if np.linalg.norm([goal_x - start_x, goal_y - start_y]) > 4.0:
                self.target_pos = np.array([goal_x, goal_y])
                break

        # 初始化Frenet坐标系（随机曲率增强泛化）
        curve_offset = np.random.uniform(-2.5, 2.5)
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
        path_length = self.frenet_transform.path_length
        self.max_steps = int((path_length / 0.2) * 10 * 2) + 200

        print("Reset: Start(%.1f,%.1f) -> Goal(%.1f,%.1f) | Path Length: %.2fm" %
              (start_x, start_y, self.target_pos[0], self.target_pos[1], path_length))

        while self.latest_scan is None:
            rospy.sleep(0.01)
        data = self.latest_scan

        return self._build_obs(data, path_length), {}

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