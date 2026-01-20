#!/usr/bin/env python3
"""
Dynamic obstacles motion controller.
Controls all red dynamic obstacles to move along the y-axis in sinusoidal pattern.
"""

import rospy
from gazebo_msgs.srv import GetModelState, SetModelState
from gazebo_msgs.msg import ModelState
import math
import time

class ObstaclesMotionController:
    def __init__(self):
        rospy.init_node('obstacles_motion_controller', anonymous=False)
        
        # List of all obstacles to control (red obstacles on y-axis)
        self.obstacles = [
            'dynamic_obstacle_1',
            'dynamic_obstacle_3',
            'dynamic_obstacle_4',
            'dynamic_obstacle_5',
            'dynamic_obstacle_6',
            'dynamic_obstacle_7',
        ]
        
        # Base positions (x is constant, y varies with motion)
        self.base_positions = {
            'dynamic_obstacle_1': (100, 75),
            'dynamic_obstacle_3': (100, -25),
            'dynamic_obstacle_4': (100, -50),
            'dynamic_obstacle_5': (100, -75),
            'dynamic_obstacle_6': (100, -100),
            'dynamic_obstacle_7': (100, -125),
        }
        
        # Motion parameters
        self.amplitude = 15.0  # +/- 15 meters along y-axis
        self.frequency = 0.5   # 0.5 Hz (period of 2 seconds)
        self.start_time = time.time()
        
        # Service clients
        try:
            rospy.wait_for_service('/gazebo/get_model_state', timeout=5)
            rospy.wait_for_service('/gazebo/set_model_state', timeout=5)
            self.get_state_client = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            self.set_state_client = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            rospy.loginfo("Obstacles Motion Controller initialized successfully")
        except rospy.ROSException as e:
            rospy.logerr(f"Failed to connect to Gazebo services: {e}")
            raise
        
    def run(self):
        """Main control loop"""
        rate = rospy.Rate(10)  # 10 Hz update rate
        
        while not rospy.is_shutdown():
            try:
                elapsed = time.time() - self.start_time
                
                for obstacle in self.obstacles:
                    if obstacle not in self.base_positions:
                        rospy.logwarn(f"Obstacle {obstacle} not in base_positions")
                        continue
                        
                    base_x, base_y = self.base_positions[obstacle]
                    
                    # Sinusoidal motion: oscillate along y-axis
                    # y = base_y + amplitude * sin(2*pi*frequency*elapsed)
                    y_offset = self.amplitude * math.sin(2 * math.pi * self.frequency * elapsed)
                    new_y = base_y + y_offset
                    
                    # Try to get current state (to preserve z, orientation)
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
                    
                    # Set linear velocity (for smooth motion)
                    vy = self.amplitude * 2 * math.pi * self.frequency * math.cos(2 * math.pi * self.frequency * elapsed)
                    model_state.twist.linear.x = 0
                    model_state.twist.linear.y = vy
                    model_state.twist.linear.z = 0
                    model_state.twist.angular.x = 0
                    model_state.twist.angular.y = 0
                    model_state.twist.angular.z = 0
                    
                    try:
                        self.set_state_client(model_state)
                    except rospy.ServiceException as e:
                        rospy.logwarn(f"Failed to set state of {obstacle}: {e}")
                        
            except Exception as e:
                rospy.logerr(f"Error in control loop: {e}")
            
            rate.sleep()

if __name__ == '__main__':
    try:
        controller = ObstaclesMotionController()
        controller.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Obstacles motion controller stopped")
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")
