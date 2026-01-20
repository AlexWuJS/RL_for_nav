import gym
import rospy
import numpy as np
import math
from gym import spaces
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty

class MyRobotEnv(gym.Env):
    def __init__(self):
        super(MyRobotEnv, self).__init__()
        
        # 1. 初始化 ROS 节点（如果是单独运行训练脚本，需要在这里 init）
        # rospy.init_node('gym_env_node', anonymous=True) 

        # 2. 定义动作空间 (SAC 输出是连续值)
        # 假设机器人: [线速度 v, 角速度 w]
        # 我们让神经网络输出 [-1, 1]，在 step 里再映射回真实速度
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # 3. 定义观测空间 (输入给神经网络的数据)
        # 假设我们用: 24个方向的雷达距离 + 目标距离 + 目标角度 + 当前线速度 + 当前角速度 = 28维
        self.n_laser = 24
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_laser + 4,), dtype=np.float32)

        # 4. ROS 连接
        self.pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.sub_scan = rospy.Subscriber('/scan', LaserScan, self._scan_callback)
        self.sub_odom = rospy.Subscriber('/odom', Odometry, self._odom_callback)
        
        # 仿真控制服务 (用于 Reset 时重置机器人位置)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        # 内部变量
        self.scan_data = None
        self.current_pose = None
        self.target_pose = [5.0, 0.0] # 示例目标点 (x, y)
        self.max_v = 0.5  # 机器人最大线速度
        self.max_w = 1.0  # 机器人最大角速度

    def _scan_callback(self, msg):
        self.scan_data = msg

    def _odom_callback(self, msg):
        self.current_pose = msg.pose.pose

    def step(self, action):
        # --- 1. 执行动作 ---
        # 将 SAC 输出的 [-1, 1] 映射到实际速度
        vel_cmd = Twist()
        vel_cmd.linear.x = (action[0] + 1) / 2 * self.max_v # 映射到 [0, max_v]
        vel_cmd.angular.z = action[1] * self.max_w          # 映射到 [-max_w, max_w]
        self.pub_cmd_vel.publish(vel_cmd)

        # 等待动作执行一小段时间 (控制频率)
        rospy.sleep(0.1) 

        # --- 2. 获取环境状态 ---
        obs = self._get_observation()

        # --- 3. 计算奖励 (Reward Shaping) ---
        # 这里的逻辑决定了训练效果的好坏
        dist_to_goal = self._get_dist_to_goal()
        reward = 0
        done = False
        
        # 奖励设计:
        # A. 靠近目标奖励 (Distance Reward)
        reward += (self.previous_dist - dist_to_goal) * 100 
        self.previous_dist = dist_to_goal

        # B. 碰撞惩罚
        if self._check_collision():
            reward -= 200
            done = True
            print("Collision!")

        # C. 到达目标奖励
        if dist_to_goal < 0.3:
            reward += 200
            done = True
            print("Goal Reached!")

        # D. 时间惩罚 (强迫快速到达)
        reward -= 0.1

        return obs, reward, done, {}

    def reset(self):
        # 1. 重置 Gazebo 仿真
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)
            
        # 2. 等待传感器数据刷新
        rospy.sleep(0.5) 
        
        # 3. 设置初始状态变量
        self.previous_dist = self._get_dist_to_goal()
        
        return self._get_observation()

    def _get_observation(self):
        # 处理雷达数据：降采样 + 归一化
        # 假设 scan_data 已经收到
        if self.scan_data is None:
            return np.zeros(self.observation_space.shape)
            
        # 简化雷达: 取 24 个扇区的最小值，并归一化到 [0, 1] (假设最大距离3.5米)
        full_ranges = np.array(self.scan_data.ranges)
        full_ranges = np.nan_to_num(full_ranges, posinf=3.5, neginf=0.0) # 处理无效值
        len_scan = len(full_ranges)
        step = len_scan // self.n_laser
        laser_state = []
        for i in range(self.n_laser):
            min_val = np.min(full_ranges[i*step : (i+1)*step])
            laser_state.append(np.clip(min_val / 3.5, 0, 1))

        # 计算目标相对位置 (极坐标)
        dx = self.target_pose[0] - self.current_pose.position.x
        dy = self.target_pose[1] - self.current_pose.position.y
        dist = math.sqrt(dx**2 + dy**2)
        # 算出机器人当前的朝向 yaw
        # (这里省略四元数转欧拉角的代码，实际需添加 tf 转换)
        robot_yaw = 0.0 # 需自行实现 get_yaw_from_quaternion(self.current_pose.orientation)
        angle_to_goal = math.atan2(dy, dx) - robot_yaw
        
        # 拼接所有状态
        obs = np.concatenate((
            laser_state, 
            [dist, angle_to_goal],
            [0.0, 0.0] # 占位，填入当前速度
        ))
        return obs

    def _check_collision(self):
        # 如果雷达探测到极近距离，视为碰撞
        if self.scan_data and np.min(self.scan_data.ranges) < 0.2:
            return True
        return False
        
    def _get_dist_to_goal(self):
        if self.current_pose is None: return 100.0
        dx = self.target_pose[0] - self.current_pose.position.x
        dy = self.target_pose[1] - self.current_pose.position.y
        return math.sqrt(dx**2 + dy**2)