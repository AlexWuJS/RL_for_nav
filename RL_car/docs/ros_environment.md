# ROS/Gazebo 仿真环境说明

Gymnasium 环境代码位于：

```text
dsac_mppi/envs/ros_env.py
```

Frenet 路径、跟线奖励和障碍物相关工具位于：

```text
dsac_mppi/envs/frenet_utils.py
```

## ROS 资源目录

ROS package 和仿真资源统一放在：

```text
src/ros/
```

主要资源包括：

- `src/ros/nav_demo/launch/`：导航相关 launch 文件。
- `src/ros/nav_demo/config/`：RViz 和导航配置。
- `src/ros/nav_demo/map/`：地图文件。
- `src/ros/urdf02_gazebo/launch/`：Gazebo 启动文件。
- `src/ros/urdf02_gazebo/worlds/`：Gazebo world。
- `src/ros/urdf02_gazebo/urdf/`：机器人 URDF/xacro。

## 运行依赖

`MyCarEnv` 运行时需要以下 ROS/Gazebo topic 或 service 可用：

- `/scan`
- `/cmd_vel`
- `/gazebo/model_states`
- `/gazebo/set_model_state`
- `/gazebo/get_model_state`
- `/gazebo/spawn_sdf_model`

## MPPI 需要的环境接口

`MyCarEnv.get_planner_state()` 是 MPPI wrapper 读取规划状态的入口，里面包含位置、速度、yaw、雷达、目标点、Frenet transform 等信息。

如果 DSAC+MPPI 评估启动后卡住，优先检查 Gazebo 是否已经启动，以及上述 topic/service 是否存在。
