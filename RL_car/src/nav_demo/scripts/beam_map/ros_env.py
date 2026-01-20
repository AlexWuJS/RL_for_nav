import gymnasium as gym
import rospy
import numpy as np
from gymnasium import spaces
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
import torch
import torch.nn as nn
import math
from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import Pose

class MyCarEnv(gym.Env):
    def __init__(self):
        super(MyCarEnv, self).__init__()
        
        try:
            rospy.init_node('my_car_rl_node', anonymous=True)
        except rospy.ROSException:
            pass
        
        self.action_space = spaces.Box(low=np.array([0.0, -1.0]), high=np.array([2.0, 1.0]), dtype=np.float32)
        
        # === 修改 1: 增加雷达采样数 ===
        # 假设你的 URDF 改成了 2000，这里我们降采样到 1000 给网络，既保留细节又减少计算
        self.n_laser_beams = 1000 
        self.max_laser_range = 20.0 # 雷达最大探测距离
        self.map_size = 40.0 # 地图大小40 * 40米
        self.goal_reach_threshold = 0.3 # 到达目标点的距离阈值
        
        #状态空间
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_laser_beams +2,), dtype=np.float32)
        
        self.pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        rospy.wait_for_service('/gazebo/get_model_state')
        self.get_state_proxy = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

        #内部变量
        self.target_pos = np.array([0.0, 0.0])
        self.current_pos = np.array([0.0, 0.0])
        self.current_yaw = 0.0
        self.last_distance_to_goal = None
    
    def _get_robot_position(self):
        try:
            res = self.get_state_proxy('car', 'world')
            pos = res.pose.position
            ori = res.pose.orientation

            #简单的四元数转欧拉角
            # siny_cosp = 2 * (w * z + x * y)
            # cosy_cosp = 1 - 2 * (y * y + z * z)
            siny_cosp = 2.0 * (ori.w * ori.z + ori.x * ori.y)
            cosy_cosp = 1 - 2 * (ori.y * ori.y + ori.z * ori.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)

            return np.array([pos.x, pos.y]), yaw
        except rospy.ServiceException as e:
            print("获取机器人状态失败: %s" % e)
            return np.array([0.0, 0.0]), 0.0

    def step(self, action):
        linear_vel = float(action[0])
        angular_vel = float(action[1])
        
        vel_msg = Twist()
        vel_msg.linear.x = linear_vel
        vel_msg.angular.z = angular_vel
        self.pub_cmd_vel.publish(vel_msg)
        
        # rospy.sleep(0.05) 
        
        scan_data = None
        while scan_data is None:
            try:
                scan_data = rospy.wait_for_message('/scan', LaserScan, timeout=0.1)
            except:
                pass
        
        # A. 为了计算奖励，先处理原始数据（保持单位为米）
        # 这里我们临时用一个不归一化的简单处理来算 min_dist
        raw_ranges = np.array(scan_data.ranges)
        raw_ranges[np.isinf(raw_ranges)] = self.max_laser_range
        raw_ranges[np.isnan(raw_ranges)] = self.max_laser_range
        min_laser_dist = np.min(raw_ranges) # 真实的物理距离

        #获取自身位置
        self.current_pos, self.current_yaw = self._get_robot_position()

        #计算相对于目标的距离和角度
        dist_to_goal = np.linalg.norm(self.target_pos - self.current_pos)

        #计算目标相对与机器人的角度差
        #计算全局角度
        angle_to_goal = math.atan2(self.target_pos[1] - self.current_pos[1], self.target_pos[0] - self.current_pos[0])
        #相对角度 = 全局目标角度 - 机器人当前朝向
        heading_error = angle_to_goal - self.current_yaw
        #归一化角度到[-pi,pi]
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi

        #整合状态
        #处理雷达
        laser_state = self._process_scan_data(scan_data)

        #归一化导航信息,方便网络训练
        #距离归一化
        norm_dist = dist_to_goal / self.map_size
        #角度归一化到[-1,1]
        norm_heading = heading_error / math.pi

        #拼接[雷达 1000 + 距离 1 + 角度 1]
        obs = np.concatenate((laser_state,[norm_dist,norm_heading])).astype(np.float32)

        # 计算奖励 (使用真实的米 min_dist_meters)
        terminated = False
        truncated = False
        reward = 0.0

        #A.碰撞惩罚
        min_laser_dist = np.min(scan_data.ranges) if scan_data.ranges else 0
        
        
        if min_laser_dist < 0.25: # 撞墙判定使用真实距离
            reward = -50.0
            terminated = True

        #B.到达目标奖励
        elif dist_to_goal < self.goal_reach_threshold:
            reward = 100.0
            terminated = True
        
        #C.导航奖励
        else:
            #c1:靠近奖励（如果当前距离比上一步近，给正分）
            if self.last_distance_to_goal is not None:
                reward += (self.last_distance_to_goal - dist_to_goal) * 10.0
            #c2:时间/生存惩罚（鼓励快速到达）
            reward -= 0.05
        
        self.last_distance_to_goal = dist_to_goal

        return obs, reward, terminated, truncated, {}
                
        # 返回给神经网络的是归一化后的数据
        return laser_state_norm, reward, terminated, truncated, {}

    def _process_scan_data(self, data):
        if data is None:
            return np.zeros(self.n_laser_beams, dtype=np.float32)
            
        raw_ranges = np.array(data.ranges)
        raw_ranges[np.isinf(raw_ranges)] = self.max_laser_range
        raw_ranges[np.isnan(raw_ranges)] = self.max_laser_range
        
        bin_size = len(raw_ranges) // self.n_laser_beams
        processed_ranges = []
        for i in range(self.n_laser_beams):
            segment = raw_ranges[i*bin_size : (i+1)*bin_size]
            if len(segment) > 0:
                min_val = np.min(segment)
            else:
                min_val = self.max_laser_range
            processed_ranges.append(min_val)
            
        # === 修改 3: 归一化 ===
        # 将 [0, 30] 映射到 [0, 1]，这对神经网络训练非常重要
        processed_ranges = np.array(processed_ranges, dtype=np.float32)
        return processed_ranges / self.max_laser_range
    
    def _update_marker(self, marker_name, x, y, color):
            """
            生成或移动一个【无重力、无碰撞、可移动】的幽灵球体
            """
            # === 核心修改：SDF 定义 ===
            # 1. <static>0</static>: 让它变成动态物体，这样才能被 SetModelState 移动
            # 2. <gravity>0</gravity>: 关掉重力，让它悬浮在半空
            # 3. <inertial>: 动态物体必须有质量，给个极小值
            # 4. <collision>: 故意不写，实现“无碰撞”
            sdf_xml = f"""
            <sdf version="1.6">
            <model name="{marker_name}">
                <static>0</static> 
                <link name="link">
                <gravity>0</gravity>
                <inertial>
                    <mass>0.001</mass>
                    <inertia>
                    <ixx>0.0001</ixx><ixy>0</ixy><ixz>0</ixz>
                    <iyy>0.0001</iyy><iyz>0</iyz>
                    <izz>0.0001</izz>
                    </inertia>
                </inertial>
                <visual name="visual">
                    <geometry><sphere><radius>0.3</radius></sphere></geometry>
                    <material>
                    <script>
                        <uri>file://media/materials/scripts/gazebo.material</uri>
                        <name>Gazebo/{color}</name>
                    </script>
                    </material>
                    <cast_shadows>0</cast_shadows>
                </visual>
                <!-- 没有 collision 标签，车可以直接穿过去 -->
                </link>
            </model>
            </sdf>
            """

            # 尝试移动模型
            state_msg = ModelState()
            state_msg.model_name = marker_name
            state_msg.pose.position.x = x
            state_msg.pose.position.y = y
            state_msg.pose.position.z = 0.5 # 高度
            state_msg.pose.orientation.w = 1.0
            # 强制速度为0，防止之前如果有速度残留导致它漂移
            state_msg.twist.linear.x = 0
            state_msg.twist.linear.y = 0
            state_msg.twist.angular.z = 0
            
            try:
                # 发送移动指令
                resp = self.set_state_proxy(state_msg)
                
                # 如果移动失败（通常是因为模型还不存在），则生成它
                if not resp.success:
                    raise rospy.ServiceException("Model not found")
            
            except rospy.ServiceException:
                # 如果报错，说明模型还没生成，调用 spawn
                try:
                    spawn_proxy = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
                    spawn_proxy(marker_name, sdf_xml, "", Pose(), "world")
                    
                    # 生成后立刻再设置一次位置，确保万无一失
                    rospy.sleep(0.05) 
                    self.set_state_proxy(state_msg)
                except rospy.ServiceException as e:
                    pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 随机生成目标点和起点
        #活动范围
        range_limit = 40.0

        # 随机生成起点
        start_x = np.random.uniform(-range_limit/2, range_limit/2)
        start_y = np.random.uniform(-range_limit/2, range_limit/2)

        # 随机生成目标点
        while True:
            goal_x = np.random.uniform(-range_limit/2, range_limit/2)
            goal_y = np.random.uniform(-range_limit/2, range_limit/2)
            if np.linalg.norm(np.array([goal_x - start_x, goal_y - start_y])) > 5.0:
                self.target_pos = np.array([goal_x, goal_y])
                break
        
        # 1. 标记终点 (红色球)
        self._update_marker("marker_goal", self.target_pos[0], self.target_pos[1], "Red")
        
        # 2. 标记起点 (蓝色球)
        self._update_marker("marker_start", start_x, start_y, "Blue")
        
        state_msg = ModelState()
        state_msg.model_name = 'car'
        state_msg.pose.position.x = start_x
        state_msg.pose.position.y = start_y
        state_msg.pose.position.z = 0.0

        #随即朝向
        yaw = np.random.uniform(-math.pi, math.pi)
        state_msg.pose.orientation.z = math.sin(yaw / 2)
        state_msg.pose.orientation.w = math.cos(yaw / 2)

        #速度清零
        state_msg.twist.linear.x = 0.0
        state_msg.twist.linear.y = 0.0
        state_msg.twist.angular.z = 0.0



        # state_msg.model_name = 'car' 
        # state_msg.pose.position.x = 0.0
        # state_msg.pose.position.y = 0.0
        # state_msg.pose.position.z = 0.0
        # state_msg.pose.orientation.w = 1
        # state_msg.twist.linear.x = 0
        # state_msg.twist.linear.y = 0
        # state_msg.twist.angular.z = 0

        try:
            self.set_state_proxy(state_msg)
        except rospy.ServiceException as e:
            print("瞬移服务调用失败: %s" % e)

        #初始化内部变量
        self.current_pos = np.array([start_x, start_y])
        self.current_yaw = yaw
        self.last_distance_to_goal = np.linalg.norm(self.target_pos - self.current_pos)
        
        #获取初始化观测
        data = None
        retry_count = 0
        while data is None and retry_count < 10:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=0.5)
            except:
                retry_count += 1

        laser_state = self._process_scan_data(data)
        
        # if data is None:
        #     obs = np.zeros(self.n_laser_beams, dtype=np.float32)
        # else:
        #     obs = self._process_scan_data(data)

        #初始化导航信息
        dist = self.last_distance_to_goal
        #计算初始相对角度
        angle_to_goal = math.atan2(self.target_pos[1] - start_y, self.target_pos[0] - start_x)
        heading_error = angle_to_goal - self.current_yaw
        
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi

        norm_dist = dist / self.map_size
        norm_heading = heading_error / math.pi

        obs = np.concatenate((laser_state, [norm_dist, norm_heading])).astype(np.float32)

        print(f"Reset: Start({start_x:.1f},{start_y:.1f}) -> Goal({self.target_pos[0]:.1f},{self.target_pos[1]:.1f})")

        return obs, {}