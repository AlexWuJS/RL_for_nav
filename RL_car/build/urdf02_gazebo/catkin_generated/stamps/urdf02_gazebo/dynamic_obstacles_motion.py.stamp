#!/usr/bin/env python3
"""
Dynamic obstacles motion controller.
Makes obstacles 1 and 3-8 move back and forth along the y-axis.
"""

import rospy
from gazebo_msgs.srv import GetModelState, SetModelState
from gazebo_msgs.msg import ModelState
import math
import time

class DynamicObstaclesController:
    def __init__(self):
        rospy.init_node('dynamic_obstacles_motion', anonymous=False)
        
        # List of obstacles to control (relative to obstacle_1)
        self.obstacles = [
            'dynamic_obstacle_1',
            'dynamic_obstacle_3',
            'dynamic_obstacle_4',
            'dynamic_obstacle_5',
            'dynamic_obstacle_6',
            'dynamic_obstacle_7',
            'dynamic_obstacle_8',
        ]
        
        # Base positions (y-coordinate is what varies)
        self.base_positions = {
            'dynamic_obstacle_1': (100, 75),
            'dynamic_obstacle_3': (100, 95),
            'dynamic_obstacle_4': (100, 115),
            'dynamic_obstacle_5': (100, 135),
            'dynamic_obstacle_6': (100, 55),
            'dynamic_obstacle_7': (100, 35),
            'dynamic_obstacle_8': (100, 15),
        }
        
        # Motion parameters
        self.amplitude = 15.0  # +/- 15 meters along y-axis
        self.frequency = 0.5   # 0.5 Hz (period of 2 seconds)
        self.start_time = time.time()
        
        # Service clients
        rospy.wait_for_service('/gazebo/get_model_state', timeout=5)
        rospy.wait_for_service('/gazebo/set_model_state', timeout=5)
        self.get_state_client = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_state_client = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        
        rospy.loginfo("Dynamic Obstacles Controller initialized")
        
    def run(self):
        """Main control loop"""
        rate = rospy.Rate(10)  # 10 Hz update rate
        
        while not rospy.is_shutdown():
            try:
                elapsed = time.time() - self.start_time
                
                for obstacle in self.obstacles:
                    base_x, base_y = self.base_positions[obstacle]
                    
                    # Sinusoidal motion: oscillate along y-axis
                    # y = base_y + amplitude * sin(2*pi*frequency*elapsed)
                    y_offset = self.amplitude * math.sin(2 * math.pi * self.frequency * elapsed)
                    new_y = base_y + y_offset
                    
                    # Try to get current state (to preserve z, roll, pitch, yaw)
                    try:
                        state = self.get_state_client(model_name=obstacle, relative_entity_name="world")
                        z = state.pose.position.z
                        qx = state.pose.orientation.x
                        qy = state.pose.orientation.y
                        qz = state.pose.orientation.z
                        qw = state.pose.orientation.w
                    except rospy.ServiceException as e:
                        rospy.logwarn(f"Failed to get state of {obstacle}: {e}")
                        # Use defaults if fetch fails
                        z = 0.5
                        qx, qy, qz, qw = 0, 0, 0, 1
                    
                    # Create and send new state
                    model_state = ModelState()
                    model_state.model_name = obstacle
                    model_state.pose.position.x = base_x
                    model_state.pose.position.y = new_y
                    model_state.pose.position.z = z
                    model_state.pose.orientation.x = qx
                    model_state.pose.orientation.y = qy
                    model_state.pose.orientation.z = qz
                    model_state.pose.orientation.w = qw
                    
                    # Set linear velocity (optional, for smooth motion in physics)
                    vy = self.amplitude * 2 * math.pi * self.frequency * math.cos(2 * math.pi * self.frequency * elapsed)
                    model_state.twist.linear.x = 0
                    model_state.twist.linear.y = vy
                    model_state.twist.linear.z = 0
                    
                    try:
                        self.set_state_client(model_state)
                    except rospy.ServiceException as e:
                        rospy.logwarn(f"Failed to set state of {obstacle}: {e}")
                        
            except Exception as e:
                rospy.logerr(f"Error in control loop: {e}")
            
            rate.sleep()

if __name__ == '__main__':
    try:
        controller = DynamicObstaclesController()
        controller.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Dynamic obstacles controller stopped")
