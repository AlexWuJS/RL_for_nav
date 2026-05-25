# 项目结构说明

本仓库是一个用于 USV 导航实验的 ROS catkin 工作空间，包含强化学习训练、MPPI 控制器、Gazebo 仿真资源和评估脚本。

## 顶层目录

```text
RL_car/
  src/
    nav_demo/
      scripts/
        beam_map/
        SAC/
        ENV/
    urdf01_rviz/
    urdf02_gazebo/
    my_urdf/
  tests/
  docs/
  build/
  devel/
```

## 核心导航代码

`src/nav_demo/scripts/beam_map/` 是当前主要使用的训练、评估和控制器目录。

- `ros_env.py`：基于 ROS/Gazebo 的 Gymnasium 环境。负责发布 `/cmd_vel`，读取激光雷达和 Gazebo 模型状态，计算 Frenet 观测、奖励和终止条件。
- `train.py`：基础 SAC 训练入口，动作空间为 2 维 `[surge, yaw]`。
- `dsac.py`：严格复现用的 DSAC actor、distributional critic、replay buffer、trainer 和模型保存/加载逻辑。
- `train_dsac.py`：DSAC 离线策略训练入口。
- `compare_sac_mppi.py`：离线评估和绘图入口，支持 baseline、shield、层级 MPPI 和 RL-driven MPPI 等模式。
- `test01.py`：Gazebo 中单次运行某个策略或控制器模式的脚本。
- `mppi_dbas.py`：已有低干预 MPPI/DBaS 优化器，以及层级 MPPI 相关辅助逻辑。
- `mppi_dbas_wrapper.py`：低干预 MPPI/DBaS 的 Gym wrapper。
- `hierarchical_mppi_wrapper.py`：将高层 SAC intent 映射到底层 MPPI 控制的 wrapper。
- `rl_driven_mppi.py`：论文式 RL-driven MPPI 实现，包含 SAC/DSAC policy adapter、在线优化器、TransitionModel 接口和 wrapper。
- `plot_comparison_curves.py`：根据评估输出生成 summary 和 trace 图表。

## ROS/Gazebo 资源

- `src/urdf02_gazebo/`：Gazebo launch 文件、world、地图、动态障碍物脚本和机器人仿真资源。
- `src/urdf01_rviz/`：RViz 配置和可视化 URDF 示例。
- `src/my_urdf/`：额外 URDF 包文件。

## 测试目录

`tests/` 中是无需启动 ROS/Gazebo 即可运行的 Python 单元测试。

- `test_mppi_dbas_low_intervention.py`：低干预 MPPI/DBaS 行为测试。
- `test_hierarchical_sac_mppi.py`：层级 wrapper 和 intent 解码测试。
- `test_plot_comparison_curves.py`：绘图颜色和 summary 健壮性测试。
- `test_dsac.py`：DSAC actor、distributional critic、replay buffer、adapter 和 strict terminal Q 测试。
- `test_rl_driven_mppi.py`：RL-driven MPPI 初始化、guided rollout、top-Z 更新、方差下限、动作边界和消融行为测试。

从仓库根目录运行：

```bash
python -m unittest discover -s RL_car/tests
```

## 训练和评估产物

训练脚本通常会在 `src/nav_demo/scripts/beam_map/` 下写入模型目录和日志目录，例如：

- `training_usv_v2_results/`
- `training_dsac_usv_results/`
- `training_hierarchical_mppi_v*_results/`
- `logs_hierarchical_mppi_v*/`
- `sac_*_log/`

评估脚本通常会写入以下目录：

- `comparison_results/`
- `comparison_rl_driven_mppi/`
- `comparison_rl_driven_mppi_ablation/`
- `comparison_dsac_rl_driven_mppi/`
- `comparison_dsac_rl_driven_mppi_ablation/`

每个评估输出目录可能包含：

- `<mode>_metrics.csv`：按 episode 统计的指标。
- `traces/<mode>_episode_*.csv`：每一步的 trace 数据。
- `summary.json`：聚合指标和配对对比结果。
- `plots/`：启用 `--plot` 后生成的图表。
