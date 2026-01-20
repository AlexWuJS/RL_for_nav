#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from gazebo_msgs.srv import GetModelState, SetModelState
from gazebo_msgs.msg import ModelState
import random
import time
import math

class RandomObstacleController:
    def __init__(self):
        rospy.init_node('random_obstacle_controller', anonymous=False)
        
        # 1. 这里填入SDF文件中那三个障碍物的名字
        self.obstacles = [
            'moving_box_square',
            'moving_box_patrol',
            'moving_cylinder_diamond',
            'moving_cylinder_diamond_2',
            'moving_sphere_yellow',
            'moving_box_purple'

        ]
        
        # 2. 定义活动范围 (墙是 +/- 20米，我们设为 +/- 18米以防撞墙)
        self.bounds = {
            'x_min': -18.0, 'x_max': 18.0,
            'y_min': -18.0, 'y_max': 18.0
        }
        
        # 3. 速度设置
        self.speed_range = (2.0, 3.0)  # 最小速度2m/s，最大5m/s
        
        # 4. 存储每个障碍物的状态
        # 结构: {'name': {'start_pos': (x,y), 'target_pos': (x,y), 'start_time': t, 'duration': d}}
        self.bot_states = {}
        
        # 连接 Gazebo 服务
        rospy.wait_for_service('/gazebo/get_model_state')
        rospy.wait_for_service('/gazebo/set_model_state')
        self.get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        
        rospy.loginfo("随机障碍物控制器已启动...")

    def generate_new_target(self, model_name):
        """生成一个新的随机目标点和到达时间"""
        
        # 获取当前位置作为起点
        try:
            current_state = self.get_state(model_name, 'world')
            start_x = current_state.pose.position.x
            start_y = current_state.pose.position.y
        except:
            start_x, start_y = 0, 0 # 如果获取失败，默认从原点开始
            
        # 随机生成终点
        target_x = random.uniform(self.bounds['x_min'], self.bounds['x_max'])
        target_y = random.uniform(self.bounds['y_min'], self.bounds['y_max'])
        
        # 计算距离
        dist = math.sqrt((target_x - start_x)**2 + (target_y - start_y)**2)
        
        # 随机速度
        speed = random.uniform(self.speed_range[0], self.speed_range[1])
        
        # 计算需要的时间
        duration = dist / speed
        
        # 更新状态
        self.bot_states[model_name] = {
            'start_pos': (start_x, start_y),
            'target_pos': (target_x, target_y),
            'start_time': time.time(),
            'duration': duration,
            'speed': speed
        }
        # rospy.loginfo(f"{model_name} 新目标: ({target_x:.1f}, {target_y:.1f}), 耗时: {duration:.1f}s")

    def run(self):
        rate = rospy.Rate(30) # 30Hz，高频率保证移动平滑
        
        while not rospy.is_shutdown():
            now = time.time()
            
            for name in self.obstacles:
                # 1. 如果是第一次运行，或者已经到达目标(时间到了)，生成新目标
                if name not in self.bot_states or \
                   (now - self.bot_states[name]['start_time'] > self.bot_states[name]['duration']):
                    self.generate_new_target(name)
                
                # 2. 计算当前应该在哪 (线性插值)
                state = self.bot_states[name]
                elapsed = now - state['start_time']
                
                # 进度 0.0 ~ 1.0
                progress = elapsed / state['duration']
                if progress > 1.0: progress = 1.0
                
                start_x, start_y = state['start_pos']
                end_x, end_y = state['target_pos']
                
                # 插值公式: 当前 = 起点 + (差距 * 进度)
                current_x = start_x + (end_x - start_x) * progress
                current_y = start_y + (end_y - start_y) * progress
                
                # 3. 计算速度向量 (为了让物理引擎知道它在动，碰撞效果更好)
                # 方向向量
                dx = end_x - start_x
                dy = end_y - start_y
                
                # 归一化后乘以速度
                # 注意：如果就在终点，速度设为0
                if progress < 1.0 and state['duration'] > 0:
                    vx = (dx / state['duration'])
                    vy = (dy / state['duration'])
                else:
                    vx, vy = 0, 0

                # 4. 发送命令给 Gazebo
                msg = ModelState()
                msg.model_name = name
                msg.pose.position.x = current_x
                msg.pose.position.y = current_y
                msg.pose.position.z = 0.6 # 保持高度，防止掉地里
                
                # 保持朝向不变 (你可以改为朝向运动方向，这里为了简单设为固定)
                msg.pose.orientation.w = 1.0
                
                # 设置速度
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