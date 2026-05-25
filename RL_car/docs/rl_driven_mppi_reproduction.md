# RL-Driven MPPI 复现说明

本文档记录当前项目中论文式 DSAC + RL-driven MPPI 控制器的复现方式。该实现基于现有 USV ROS/Gazebo 仿真环境，重点严格复现论文中的离线 DSAC 策略、分布式 critic 终端代价，以及在线 RL-driven MPPI 控制流程。

## 算法映射

论文使用离线训练得到的随机强化学习策略来加速在线 MPPI。当前严格复现版本通过 `DSACPolicyAdapter` 接入项目内实现的 DSAC actor 和 distributional critic；旧的 `SB3SacPolicyAdapter` 仍保留为 SAC-compatible 对比模式。

已实现的论文机制：

- RL 初始化：离线策略提供 MPPI 初始均值序列 `U0`，adapter 提供初始动作标准差。
- 混合采样策略：每个在线控制步开始时，从离线策略采样 guided rollouts，并在后续 MPPI 迭代中复用。
- 均值和方差同步更新：每轮 MPPI 选择 top-Z 条候选控制序列，同时更新 `U` 和 `Sigma`，并使用 `sigma_min` 作为方差下限。
- 终端价值代价：严格模式必须读取 DSAC distributional critic 作为 terminal cost；如果 critic 不可用，strict 模式会显式失败，不再静默退回普通 rollout cost。

USV rollout 模型复用了 `mppi_dbas.py` 中已有的短时域 3-DOF 近似动力学、Frenet 代价和激光障碍物代价，因此新控制器可以和已有低干预 MPPI、层级 SAC-MPPI 方案直接对比。

## 主要文件

- `src/nav_demo/scripts/beam_map/dsac.py`：DSAC actor、distributional critic、replay buffer、trainer 和模型保存/加载逻辑。
- `src/nav_demo/scripts/beam_map/train_dsac.py`：DSAC 训练入口。
- `src/nav_demo/scripts/beam_map/rl_driven_mppi.py`：RL-driven MPPI 配置、优化器、SAC/DSAC adapter、TransitionModel 接口和 Gym wrapper。
- `src/nav_demo/scripts/beam_map/compare_sac_mppi.py`：离线评估模式、summary 输出和 debug 指标。
- `src/nav_demo/scripts/beam_map/test01.py`：Gazebo 单次运行模式接入。
- `tests/test_dsac.py`：DSAC actor、distributional critic、replay buffer、adapter 和 strict terminal Q 测试。
- `tests/test_rl_driven_mppi.py`：使用 fake planner state 的单元测试，覆盖初始化、HSS、top-Z 更新、方差下限、动作边界和消融开关。

## 运行命令

进入脚本目录：

```bash
cd RL_car/src/nav_demo/scripts/beam_map
```

训练离线 DSAC 策略：

```bash
python train_dsac.py --total-timesteps 300000
```

训练产物默认保存到：

```text
./training_dsac_usv_results/best_model
./training_dsac_usv_results/final_model_dsac
```

如果只想运行旧 SAC-compatible 对比模式，可继续训练 SAC：

```bash
python train.py --total-timesteps 300000
```

快速评估严格 DSAC + RL-driven MPPI：

```bash
python compare_sac_mppi.py \
  --dsac-model ./training_dsac_usv_results/best_model \
  --mode dsac_rl_driven_mppi \
  --episode 10 \
  --output-dir ./comparison_dsac_rl_driven_mppi \
  --plot
```

运行严格 DSAC-RLMPPI 消融实验：

```bash
python compare_sac_mppi.py \
  --baseline-model ./training_usv_v2_results/best_model \
  --dsac-model ./training_dsac_usv_results/best_model \
  --mode ablation_dsac_rlmppi \
  --episode 30 \
  --output-dir ./comparison_dsac_rl_driven_mppi_ablation \
  --plot
```

运行旧 SAC-compatible RL-driven MPPI：

```bash
python compare_sac_mppi.py \
  --model ./training_usv_v2_results/best_model \
  --mode rl_driven_mppi \
  --episode 10 \
  --output-dir ./comparison_rl_driven_mppi \
  --plot
```

运行论文机制消融实验：

```bash
python compare_sac_mppi.py \
  --model ./training_usv_v2_results/best_model \
  --mode ablation_rlmppi \
  --episode 30 \
  --output-dir ./comparison_rl_driven_mppi_ablation \
  --plot
```

Gazebo 单次导航运行严格模式：

```bash
python test01.py \
  --mode dsac_rl_driven_mppi \
  --model ./training_dsac_usv_results/best_model
```

Gazebo 单次导航运行旧兼容模式：

```bash
python test01.py \
  --mode rl_driven_mppi \
  --model ./training_usv_v2_results/best_model
```

## 评估模式

- `dsac`：直接执行 DSAC actor，不经过 MPPI。
- `dsac_rl_driven_mppi`：严格 DSAC + RL-driven MPPI。
- `dsac_rl_driven_mppi_no_hss`：严格 DSAC-RLMPPI，但关闭 guided rollouts。
- `dsac_rl_driven_mppi_fixed_sigma`：严格 DSAC-RLMPPI，但关闭方差更新。
- `dsac_rl_driven_mppi_no_q`：DSAC-RLMPPI 消融，关闭 terminal critic cost。
- `ablation_dsac_rlmppi`：依次运行 `baseline`、`pure_mppi`、`dsac`、严格 DSAC-RLMPPI 和三个消融版本。
- `pure_mppi`：纯 MPPI，不使用 RL 初始化、HSS 和终端 Q。
- `rl_driven_mppi`：旧 SAC-compatible RL-driven MPPI。
- `rl_driven_mppi_no_hss`：关闭 guided rollouts。
- `rl_driven_mppi_fixed_sigma`：关闭方差更新。
- `rl_driven_mppi_no_q`：关闭终端 critic cost。
- `ablation_rlmppi`：依次运行 `baseline`、`pure_mppi`、完整 RLMPPI 和三个消融版本。

## 输出指标

评估脚本会生成每个模式的 CSV、每步 trace、`summary.json`，如果启用 `--plot` 还会生成图表。

RLMPPI 专属字段包括：

- `rlmppi_hss_enabled`
- `rlmppi_terminal_q_enabled`
- `rlmppi_terminal_q_used`
- `rlmppi_update_sigma`
- `rlmppi_num_rl_rollouts`
- `rlmppi_num_mppi_rollouts`
- `rlmppi_num_iterations`
- `rlmppi_top_z`
- `rlmppi_sigma_mean`
- `rlmppi_cost_best`
- `rlmppi_online_time_ms`

## 当前限制

- 严格复现范围是 DSAC + RLMPPI 机制，任务环境仍是当前 USV ROS/Gazebo，不重建论文 UAV 6-DOF 任务。
- 在线 rollout 当前使用现有 USV 3-DOF 近似动力学；代码中已预留 `TransitionModel` 接口，后续可接入神经网络转移模型。
- 训练和运行严格 DSAC 需要安装 PyTorch；如果当前 Python 环境没有 `torch`，DSAC 相关单元测试会跳过，训练脚本无法运行。
