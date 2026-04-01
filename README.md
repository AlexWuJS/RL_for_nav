分支frenet_reward是小车模型
分支USV是USV模型，修改动力学，加惯性

算法代码在RL_car/src/nav_demo/scripts/beam_map

frenet_utils.py是笛卡尔坐标系转换成frenet坐标的脚本

ros_env.py是仿真环境配置和奖励函数

train.py是训练脚本

test01.py是测试脚本

运行
1.进入目录RL_car/RL_car:roslaunch urdf02_gazebo demo03_env.launch

2.执行训练或者运行脚本
  cd RL_car/RL_car/src/nav_demo/scripts/beam_map
  python train.py/python test01.py
