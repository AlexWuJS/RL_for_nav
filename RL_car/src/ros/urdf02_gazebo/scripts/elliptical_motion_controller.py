#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from gazebo_msgs.srv import GetModelState, SetModelState, GetWorldProperties
from gazebo_msgs.msg import ModelState
import random
import time
import math
import tf.transformations

class RandomObstacleController:
    def __init__(self):
        rospy.init_node('random_obstacle_controller', anonymous=False)
        
        # 1. 障碍物前缀
        self.obstacle_prefix = "obs_"
        self.obstacles = []
        
        # 2. 活动范围 (40x40地图，限制在 +/- 19米内)
        self.bounds = {
            'x_min': -19.0, 'x_max': 19.0,
            'y_min': -19.0, 'y_max': 19.0
        }
        
        # 3. === 核心修改：分级速度配置 ===
        # 机器人最高速 2.0 m/s
        self.speed_configs = {
            'small': (1.2, 2.0),  # 小物体：飞快 (模拟敏捷目标)
            'med':   (0.6, 1.2),  # 中物体：中速 (模拟行人)
            'large': (0.2, 0.5)   # 大物体：缓慢 (模拟重型设备)
        }
        
        # 存储状态
        self.bot_states = {}

        # 连接服务
        rospy.wait_for_service('/gazebo/get_world_properties')
        rospy.wait_for_service('/gazebo/get_model_state')
        rospy.wait_for_service('/gazebo/set_model_state')
        
        self.get_world_props = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
        self.get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        
        # 初始化列表
        self.refresh_obstacles()
        rospy.loginfo(f"控制器启动: 找到 {len(self.obstacles)} 个障碍物")

    def refresh_obstacles(self):
        """自动获取所有障碍物"""
        try:
            resp = self.get_world_props()
            all_models = resp.model_names
            # 过滤出所有以 obs_ 开头的模型
            self.obstacles = [name for name in all_models if name.startswith(self.obstacle_prefix)]
        except rospy.ServiceException as e:
            rospy.logerr(f"无法获取模型列表: {e}")

    def get_obstacle_speed(self, name):
        """根据名字判断应该用什么速度"""
        if "small" in name:
            return random.uniform(*self.speed_configs['small'])
        elif "med" in name:
            return random.uniform(*self.speed_configs['med'])
        elif "large" in name:
            return random.uniform(*self.speed_configs['large'])
        else:
            # 默认速度 (如果名字里没写大小)
            return random.uniform(0.5, 1.0)

    def generate_new_target(self, model_name):
        """生成新目标"""
        
        # 1. 获取当前高度 (Z轴)
        try:
            current_state = self.get_state(model_name, 'world')
            start_x = current_state.pose.position.x
            start_y = current_state.pose.position.y
            fixed_z = current_state.pose.position.z 
        except:
            start_x, start_y, fixed_z = 0, 0, 0.5 
            
        # 2. 随机生成终点
        target_x = random.uniform(self.bounds['x_min'], self.bounds['x_max'])
        target_y = random.uniform(self.bounds['y_min'], self.bounds['y_max'])
        
        # 计算距离
        dist = math.sqrt((target_x - start_x)**2 + (target_y - start_y)**2)
        
        # 防止原地踏步
        if dist < 0.5:
            target_x += 1.0
            dist += 1.0

        # 3. === 关键：获取对应等级的速度 ===
        speed = self.get_obstacle_speed(model_name)
        
        # 计算耗时
        duration = dist / speed
        
        # 更新状态
        self.bot_states[model_name] = {
            'start_pos': (start_x, start_y),
            'target_pos': (target_x, target_y),
            'z_height': fixed_z,
            'start_time': time.time(),
            'duration': duration,
            'speed': speed
        }

    def run(self):
        rate = rospy.Rate(30) # 30Hz 更新频率
        
        while not rospy.is_shutdown():
            now = time.time()
            
            for name in self.obstacles:
                # 检查是否需要新目标
                if name not in self.bot_states or \
                   (now - self.bot_states[name]['start_time'] > self.bot_states[name]['duration']):
                    self.generate_new_target(name)
                
                state = self.bot_states[name]
                elapsed = now - state['start_time']
                
                # 进度计算
                if state['duration'] <= 0:
                    progress = 1.0
                else:
                    progress = elapsed / state['duration']
                
                if progress > 1.0: progress = 1.0
                
                # --- 位置插值 ---
                sx, sy = state['start_pos']
                ex, ey = state['target_pos']
                
                current_x = sx + (ex - sx) * progress
                current_y = sy + (ey - sy) * progress
                
                # --- 速度与朝向计算 ---
                if progress < 1.0 and state['duration'] > 0:
                    dx = ex - sx
                    dy = ey - sy
                    
                    vx = dx / state['duration']
                    vy = dy / state['duration']
                    
                    # 计算朝向 (Yaw)
                    yaw = math.atan2(dy, dx)
                    # 转换为四元数
                    quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
                else:
                    vx, vy = 0, 0
                    quaternion = None

                # --- 发送指令 ---
                msg = ModelState()
                msg.model_name = name
                msg.pose.position.x = current_x
                msg.pose.position.y = current_y
                msg.pose.position.z = state['z_height'] # 保持原高度
                
                # 如果在运动，更新朝向；如果在静止，保持不动(这里简单略过orientation更新)
                if quaternion is not None:
                    msg.pose.orientation.x = quaternion[0]
                    msg.pose.orientation.y = quaternion[1]
                    msg.pose.orientation.z = quaternion[2]
                    msg.pose.orientation.w = quaternion[3]
                
                msg.twist.linear.x = vx
                msg.twist.linear.y = vy
                
                self.set_state(msg)
                
            rate.sleep()

if __name__ == '__main__':
    try:
        controller = RandomObstacleController()
        controller.run()
    except rospy.ROSInterruptException:
        pass