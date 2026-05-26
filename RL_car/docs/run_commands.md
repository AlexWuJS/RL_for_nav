# 项目运行指令

本文档整理当前 USV ROS/Gazebo 项目中 DSAC、RL-driven MPPI、严格 DSAC + RL-driven MPPI 的常用运行命令。默认从仓库根目录进入脚本目录后执行：

```bash
cd RL_car/src/nav_demo/scripts/beam_map
```

如果你的工程路径是双层 `RL_car/RL_car`，则常见路径为：

```bash
cd ~/RL_car/RL_car/src/nav_demo/scripts/beam_map
```

## 环境要求

- ROS/Gazebo 仿真环境已启动，且 `/scan`、`/cmd_vel`、`/gazebo/model_states` 等话题可用。
- Python 环境需要安装 `torch`、`gymnasium`、`stable-baselines3`、`numpy`、`matplotlib` 等依赖。
- DSAC 训练和严格 DSAC-RLMPPI 运行需要 PyTorch。
- 如果只运行部分单元测试，缺少 `torch` 时 DSAC 相关测试会跳过，但训练脚本不能运行。

## 单元测试

从仓库根目录运行：

```bash
python -m unittest discover -s RL_car/tests
```

编译检查核心脚本：

```bash
python -m py_compile \
  RL_car/src/nav_demo/scripts/beam_map/frenet_utils.py \
  RL_car/src/nav_demo/scripts/beam_map/ros_env.py \
  RL_car/src/nav_demo/scripts/beam_map/dsac.py \
  RL_car/src/nav_demo/scripts/beam_map/train_dsac.py \
  RL_car/src/nav_demo/scripts/beam_map/mppi_dbas.py \
  RL_car/src/nav_demo/scripts/beam_map/rl_driven_mppi.py \
  RL_car/src/nav_demo/scripts/beam_map/compare_sac_mppi.py \
  RL_car/src/nav_demo/scripts/beam_map/test01.py
```

## 训练 DSAC

快速 smoke 训练，用来检查流程是否能跑通：

```bash
python train_dsac.py \
  --total-timesteps 5000 \
  --save-dir ./training_dsac_smoke_results \
  --log-dir ./logs_dsac_smoke
```

正式训练离线 DSAC 策略：

```bash
python train_dsac.py \
  --total-timesteps 300000 \
  --save-dir ./training_dsac_usv_results \
  --log-dir ./logs_dsac
```

默认输出：

```text
./training_dsac_usv_results/best_model
./training_dsac_usv_results/final_model_dsac
```

注意：当前奖励函数已改为软边界跟线奖励，旧 DSAC 模型不会自动具备新的跟线行为。建议重新训练后再评估 `dsac` 和 `dsac_rl_driven_mppi`。

## 训练 SAC Baseline

如果需要运行旧 SAC baseline 或 SAC-compatible RLMPPI：

```bash
python train.py --total-timesteps 300000
```

默认输出：

```text
./training_usv_v2_results/best_model
```

## 离线评估

单独评估 DSAC，不经过 MPPI：

```bash
python compare_sac_mppi.py \
  --dsac-model ./training_dsac_usv_results/best_model \
  --mode dsac \
  --episode 10 \
  --output-dir ./comparison_dsac_only \
  --plot
```

评估严格 DSAC + RL-driven MPPI：

```bash
python compare_sac_mppi.py \
  --dsac-model ./training_dsac_usv_results/best_model \
  --mode dsac_rl_driven_mppi \
  --episode 10 \
  --output-dir ./comparison_dsac_rl_driven_mppi \
  --plot
```

运行严格 DSAC-RLMPPI 消融对比：

```bash
python compare_sac_mppi.py \
  --baseline-model ./training_usv_v2_results/best_model \
  --dsac-model ./training_dsac_usv_results/best_model \
  --mode ablation_dsac_rlmppi \
  --episode 30 \
  --output-dir ./comparison_dsac_rl_driven_mppi_ablation \
  --plot
```

评估纯 MPPI：

```bash
python compare_sac_mppi.py \
  --mode pure_mppi \
  --episode 10 \
  --output-dir ./comparison_pure_mppi \
  --plot
```

评估旧 SAC-compatible RL-driven MPPI：

```bash
python compare_sac_mppi.py \
  --model ./training_usv_v2_results/best_model \
  --mode rl_driven_mppi \
  --episode 10 \
  --output-dir ./comparison_rl_driven_mppi \
  --plot
```

## Gazebo 单次运行

严格 DSAC + RL-driven MPPI：

```bash
python test01.py \
  --mode dsac_rl_driven_mppi \
  --model ./training_dsac_usv_results/best_model
```

单独 DSAC：

```bash
python test01.py \
  --mode dsac \
  --model ./training_dsac_usv_results/best_model
```

旧 SAC-compatible RL-driven MPPI：

```bash
python test01.py \
  --mode rl_driven_mppi \
  --model ./training_usv_v2_results/best_model
```

## 常用模式说明

- `dsac`：只执行 DSAC actor，不加 MPPI。
- `pure_mppi`：不使用 RL 初始化、guided rollouts 和 terminal critic。
- `dsac_rl_driven_mppi`：严格 DSAC + RL-driven MPPI，使用 DSAC actor 初始化、guided rollouts、top-Z 更新和 DSAC distributional critic terminal cost。
- `dsac_rl_driven_mppi_no_hss`：关闭 guided rollouts。
- `dsac_rl_driven_mppi_fixed_sigma`：关闭方差更新。
- `dsac_rl_driven_mppi_no_q`：关闭 terminal critic cost。
- `ablation_dsac_rlmppi`：批量运行 baseline、pure MPPI、DSAC、完整 DSAC-RLMPPI 和消融模式。

## 输出文件

评估命令会在 `--output-dir` 指定目录中生成：

- 每个模式的 episode CSV。
- 每步 trace CSV。
- `summary.json`。
- 启用 `--plot` 后生成对比图。

重点观察指标：

- `mean_success`
- `mean_collision`
- `mean_out_of_bounds`
- `mean_reward`
- `mean_frenet_abs_d`
- `mean_rlmppi_terminal_q_used`
- `mean_rlmppi_online_time_ms`

## 推荐调试顺序

1. 先跑单元测试，确认代码没有接口错误。
2. 用 `train_dsac.py --total-timesteps 5000` 做 smoke 训练。
3. 用 `--mode dsac` 评估新 DSAC 是否学会跟线。
4. 再用 `--mode dsac_rl_driven_mppi` 评估 MPPI 是否进一步改善。
5. 如果 DSAC-only 的 `mean_frenet_abs_d` 仍然很高，先继续调奖励或增加训练步数，不要先调 MPPI。
