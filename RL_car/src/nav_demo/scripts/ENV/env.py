import gym
import numpy as np
import rospy
from gazebo_msgs.srv import SetModelState #用于Gazebo重置机器人位置
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import LaserScan #读取激光雷达数据
from nav_msgs.msg import Odometry #读取里程计数据
from geometry_msgs.msg import Twist

def __init__(self):
    super(Env, self).__init__()
    #初始化ROS节点，使环境能与Gazebo通信
    rospy.init_node('nav_env', anonymous=True)

    #1.定义动作空间和状态空间
    self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,)) #线速度和角速度
    self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(720+3,)) #激光雷达数据

    # 2.订阅激光雷达和里程计数据
    self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback)
    self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
    # 发布速度命令
    self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    #3.调用Gazebo设置状态
    rospy.wait_for_service('/gazebo/set_model_state')
    self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

    #数据缓存
    self.lidar = None
    self.position = None
    self.yaw = None

    #自定义目标点
    self.goal = np.array([5.0, 5.0])

def _lidar_callback(self, msg):
    self.lidar = np.array(msg.ranges)

def _odom_callback(self, msg):
    self.position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
    #提取yaw
    q = msg.pose.pose.orientation
    self.yaw = self.quat_to_yaw(q)

def _get_obs(self):
    #等待传感器数据准备好
    while self.lidar is None or self.position is None or self.yaw is None:
        rospy.sleep(0.01)

    # lidar截取720维特征
    lidar_feat = self.lidar[:720]
    # 计算目标相对位置
    delta = self.goal - self.position
    dist_to_goal = np.linalg.norm(delta)
    angle_to_goal = np.arctan2(delta[1], delta[0])
    heading_error = angle_to_goal - self.yaw
    # 组合状态
    return np.concatenate([lidar_feat,[dist_to_goal,heading_error,self.yaw]])

def reset(self):
    #重置机器人位置
    state_msg = ModelState()
    state_msg.model_name = 'robot' #根据实际模型名称修改
    state_msg.pose.position.x = np.random.uniform(-1, 1)
    state_msg.pose.position.y = np.random.uniform(-1, 1)
    state_msg.pose.position.z = 0.0

    self.set_model_state(state_msg)

    #等待一段时间让Gazebo更新状态
    rospy.sleep(0.5) #等待仿真稳定

    return self._get_obs()