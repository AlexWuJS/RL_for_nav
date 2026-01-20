#!/usr/bin/env python3
"""
Elliptical motion controller for dynamic obstacles.
Makes obstacles 3-7 perform elliptical motion around specified base positions.
"""

import rospy
from gazebo_msgs.srv import GetModelState, SetModelState
from gazebo_msgs.msg import ModelState
import math
import time

class EllipticalMotionController:
    def __init__(self):
        rospy.init_node('elliptical_motion_controller', anonymous=False)
        
        # List of obstacles to control
        self.obstacles = [
            'dynamic_obstacle_3',
            'dynamic_obstacle_4',
            'dynamic_obstacle_5',
            'dynamic_obstacle_6',
            'dynamic_obstacle_7',
        ]
        
        # Base positions (center of elliptical motion) -- spacing halved
        self.base_positions = {
            'dynamic_obstacle_3': (0, -50),
            'dynamic_obstacle_4': (0, -75),
            'dynamic_obstacle_5': (0, -100),
            'dynamic_obstacle_6': (0, -125),
            'dynamic_obstacle_7': (0, -150),
        }
        
        # Elliptical motion parameters
        self.x_amplitude = 20.0  # X-axis amplitude (semi-major axis)
        self.y_amplitude = 10.0  # Y-axis amplitude (semi-minor axis)
        self.frequency = 0.15    # 0.15 Hz (speed halved)
        self.start_time = time.time()
        
        # Service clients
        try:
            rospy.wait_for_service('/gazebo/get_model_state', timeout=5)
            rospy.wait_for_service('/gazebo/set_model_state', timeout=5)
            self.get_state_client = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            self.set_state_client = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            rospy.loginfo("Elliptical Motion Controller initialized")
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
                    theta = 2 * math.pi * self.frequency * elapsed
                    new_x = base_x + self.x_amplitude * math.cos(theta)
                    new_y = base_y + self.y_amplitude * math.sin(theta)
                    
                    # Try to get current state
                    try:
                        state = self.get_state_client(model_name=obstacle, relative_entity_name="world")
                        z = state.pose.position.z
                        qx = state.pose.orientation.x
                        qy = state.pose.orientation.y
                        qz = state.pose.orientation.z
                        qw = state.pose.orientation.w
                    except rospy.ServiceException as e:
                        rospy.logwarn(f"Failed to get state of {obstacle}: {e}")
                        z = 0.5
                        qx, qy, qz, qw = 0, 0, 0, 1
                    
                    # Create and send new state
                    model_state = ModelState()
                    model_state.model_name = obstacle
                    model_state.pose.position.x = new_x
                    model_state.pose.position.y = new_y
                    model_state.pose.position.z = z
                    model_state.pose.orientation.x = qx
                    model_state.pose.orientation.y = qy
                    model_state.pose.orientation.z = qz
                    model_state.pose.orientation.w = qw
                    
                    # Velocity (derivative of position)
                    vx = -self.x_amplitude * 2 * math.pi * self.frequency * math.sin(theta)
                    vy = self.y_amplitude * 2 * math.pi * self.frequency * math.cos(theta)
                    model_state.twist.linear.x = vx
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
        controller = EllipticalMotionController()
        controller.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Elliptical motion controller stopped")
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")
