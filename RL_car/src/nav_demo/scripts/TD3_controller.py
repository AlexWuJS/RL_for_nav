#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from algos import TD3  # 这里导入你提供的TD3算法类
import torch

class TD3Controller:
    def __init__(self):
        rospy.init_node('td3_controller', anonymous=True)

        # === 参数配置 ===
        self.lidar_dim = 1800
        self.pos_dim = 3
        self.action_dim = 2  # 例如 (线速度, 角速度)
        self.max_action = 1.0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # === 加载TD3模型 ===
        self.policy = TD3.TD3(
            lidar_state_dim=self.lidar_dim,
            position_state_dim=self.pos_dim,
            lidar_feature_dim=50,
            action_dim=self.action_dim,
            max_action=self.max_action,
            hidden_dim=256,
            device=self.device
        )
        self.policy.load('/path/to/your/trained_model')  # 载入训练好的参数
        self.policy.eval_mode()

        # === ROS 订阅与发布 ===
        self.lidar_data = None
        self.position_data = None
        rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        rospy.loginfo("TD3 Controller Initialized.")
        self.control_loop()

    def lidar_callback(self, msg):
        # 将雷达数据处理成 numpy 数组
        self.lidar_data = np.array(msg.ranges, dtype=np.float32)
        self.lidar_data = np.clip(self.lidar_data, 0, 10)
        self.lidar_data = np.nan_to_num(self.lidar_data)

    def odom_callback(self, msg):
        # 从里程计中提取位置与朝向
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        self.position_data = np.array([pos.x, pos.y, ori.z], dtype=np.float32)

    def control_loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.lidar_data is not None and self.position_data is not None:
                action = self.policy.select_action(self.lidar_data, self.position_data)

                cmd = Twist()
                cmd.linear.x = float(action[0])
                cmd.angular.z = float(action[1])
                self.cmd_pub.publish(cmd)
            rate.sleep()

if __name__ == '__main__':
    TD3Controller()
