# 项目结构说明

当前分支已整理为 DSAC+MPPI 专用结构，核心说明见：

- `docs/project_layout.md`
- `docs/dsac.md`
- `docs/rl_driven_mppi.md`
- `docs/ros_environment.md`

## 常用目录

```text
dsac_mppi/        DSAC、MPPI、ROS/Gazebo 环境代码
scripts/train/    训练入口
scripts/test/     测试与评估入口
scripts/analysis/ 绘图与结果分析
src/ros/          ROS package、launch、world、urdf、rviz、map
data/             按模型名保存模型、日志和评估结果
docs/             一个文件对应一个算法或脚本说明
tests/            单元测试
```

从 `RL_car/` 目录运行 Python 模块，例如：

```bash
python -m scripts.train.train_dsac --model-name dsac_high_level
```
