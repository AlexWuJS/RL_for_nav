#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from gazebo_msgs.srv import GetModelState, SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Twist, Point, Quaternion
import math
import time

def obstacle_controller():
    """
    控制两个动态障碍物的运动脚本 - 通过Gazebo服务直接控制
    """
    rospy.init_node('obstacle_controller', anonymous=True)
    
    # 等待Gazebo服务准备好
    rospy.wait_for_service('/gazebo/set_model_state')
    rospy.wait_for_service('/gazebo/get_model_state')
    
    set_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    
    rate = rospy.Rate(10)  # 10 Hz
    
    rospy.loginfo("障碍物控制器已启动...")
    
    time_counter = 0
    
    while not rospy.is_shutdown():
        try:
            # 计算障碍物1的新位置和速度
            t1 = time_counter * 0.02
            obs1_x = 2.0 + 2.0 * math.sin(t1)
            obs1_y = 1.5 + 1.5 * math.cos(t1)
            obs1_z = 0.5
            
            obs1_vel_x = 2.0 * 0.02 * math.cos(t1)
            obs1_vel_y = -1.5 * 0.02 * math.sin(t1)
            
            # 计算障碍物2的新位置和速度
            t2 = time_counter * 0.015
            obs2_x = -2.5 + 2.0 * math.cos(t2)
            obs2_y = 2.0 + 2.0 * math.sin(t2)
            obs2_z = 0.5
            
            obs2_vel_x = -2.0 * 0.015 * math.sin(t2)
            obs2_vel_y = 2.0 * 0.015 * math.cos(t2)
            
            # 创建模型状态消息 - 障碍物1
            state1 = ModelState()
            state1.model_name = 'dynamic_obstacle_1'
            state1.pose = Pose()
            state1.pose.position = Point(x=obs1_x, y=obs1_y, z=obs1_z)
            state1.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)
            state1.twist = Twist()
            state1.twist.linear.x = obs1_vel_x
            state1.twist.linear.y = obs1_vel_y
            state1.twist.linear.z = 0
            state1.twist.angular.z = 0.1
            
            # 创建模型状态消息 - 障碍物2
            state2 = ModelState()
            state2.model_name = 'dynamic_obstacle_2'
            state2.pose = Pose()
            state2.pose.position = Point(x=obs2_x, y=obs2_y, z=obs2_z)
            state2.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)
            state2.twist = Twist()
            state2.twist.linear.x = obs2_vel_x
            state2.twist.linear.y = obs2_vel_y
            state2.twist.linear.z = 0
            state2.twist.angular.z = -0.08
            
            # 发送服务调用
            set_state_service(state1)
            set_state_service(state2)
            
            time_counter += 1
            
        except rospy.ServiceException as e:
            rospy.logwarn("服务调用失败: %s" % e)
        
        rate.sleep()

if __name__ == '__main__':
    try:
        obstacle_controller()
    except rospy.ROSInterruptException:
        pass
