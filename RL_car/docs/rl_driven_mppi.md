# RL-driven MPPI 内容说明

`dsac_mppi.controllers.rl_driven_mppi` 实现 DSAC 引导的 MPPI 控制器。

## 作用

RL-driven MPPI 会利用 DSAC 的 actor 和 critic 辅助在线轨迹采样：

- DSAC actor 提供动作序列的初始均值。
- DSAC 随机策略提供 guided rollout 的采样动作。
- DSAC critic 在开启时提供终端代价。
- MPPI 根据采样结果更新动作序列，并将第一个动作发送给环境执行。

## 主要运行模式

- `pure_mppi`：只运行 MPPI，不使用 DSAC 初始化、guided rollout 和 terminal Q。
- `dsac_rl_driven_mppi`：完整 DSAC + RL-driven MPPI。
- `dsac_rl_driven_mppi_no_hss`：关闭 guided rollout 采样。
- `dsac_rl_driven_mppi_fixed_sigma`：关闭在线方差更新。
- `dsac_rl_driven_mppi_no_q`：关闭 DSAC critic 终端代价。

## 重要类

- `RLDrivenMPPIConfig`：控制器参数。
- `DSACPolicyAdapter`：把 DSAC 模型适配成 MPPI 可调用的策略接口。
- `RLDrivenMPPIOptimizer`：在线 MPPI 优化器。
- `RLDrivenMPPIActionWrapper`：Gymnasium action wrapper，用于把 MPPI 接到环境 step 前。

## 环境要求

底层环境需要提供 `get_planner_state()`，MPPI 会从这里读取当前位置、速度、雷达、Frenet 路径等规划状态。
