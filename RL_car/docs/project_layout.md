# 项目结构说明

当前项目按 DSAC+MPPI 工作流整理。

```text
RL_car/
  dsac_mppi/
    algorithms/
    controllers/
    envs/
  scripts/
    train/
    test/
    analysis/
  src/
    ros/
  data/
    <model_name>/
      models/
      logs/
      eval/
  docs/
  tests/
```

## 模块边界

- `dsac_mppi/algorithms/`：强化学习算法实现，目前保留 DSAC。
- `dsac_mppi/controllers/`：MPPI 优化器、RL-driven MPPI 和 Gym wrapper。
- `dsac_mppi/envs/`：ROS/Gazebo 环境、Frenet 工具和奖励函数。
- `scripts/train/`：训练入口脚本。
- `scripts/test/`：评估脚本和 Gazebo 单次运行脚本。
- `scripts/analysis/`：绘图和结果分析脚本。
- `src/ros/`：ROS package、launch、world、地图、RViz、URDF 等资源。
- `data/`：按模型名保存模型、日志和评估结果。
- `docs/`：一个文件对应一个算法、脚本或模块说明。
- `tests/`：不依赖 Gazebo 的单元测试。

## 数据管理约定

不同模型统一放在：

```text
data/<model_name>/
```

其中：

- `models/` 保存模型。
- `logs/` 保存训练日志。
- `eval/` 保存评估结果。

Python 缓存、catkin 编译产物和模型权重文件由 `.gitignore` 忽略。
