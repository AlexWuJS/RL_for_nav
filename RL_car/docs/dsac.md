# DSAC 算法说明

`dsac_mppi.algorithms.dsac` 是当前项目使用的 Distributional SAC 实现，负责训练和加载 DSAC 策略模型。

## 文件内容

- `DSACConfig`：训练参数和模型超参数配置。
- `DSACPolicy`：actor、分布式 critic、动作缩放、模型保存/加载和推理接口。
- `DSACReplayBuffer`：用于帧堆叠观测的经验回放池。
- `DSACTrainer`：和 ROS/Gazebo 环境交互的在线训练循环。

## 模型保存格式

每个 DSAC 模型目录包含：

```text
config.json
model.pt
```

高层课程训练默认模型路径为：

```text
data/dsac_high_level/models/best_model
```

## 使用注意

- 旧低层模型动作为二维：`[surge, yaw]`。
- 新高层模型动作为三维：`[delta_s, target_d, target_speed]`，表示 Frenet 前视距离、横向目标偏移和目标速度。
- 高层 DSAC 负责选择全局绕障意图；MPPI 负责把高层意图转成局部安全、平滑的低层动作。
- 高层 DSAC+MPPI 评估时不再直接把 DSAC 动作作为 MPPI 速度均值，而是先把 Frenet 目标转换为低层参考动作。
- 训练时会根据封装后的 ROS 环境自动推断观测维度。
