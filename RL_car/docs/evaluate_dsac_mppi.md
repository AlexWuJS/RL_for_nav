# DSAC 和 DSAC+MPPI 评估脚本使用说明

评估入口为：

```bash
cd ~/RL_car/RL_car
python -m scripts.test.compare_dsac_mppi \
  --model data/dsac_high_level/models/best_model \
  --mode dsac_rl_driven_mppi \
  --run-name dsac_rlmppi_eval \
  --plot
```

## 可用模式

- `dsac`：只运行 DSAC 策略。
- `pure_mppi`：只运行 MPPI，不使用 DSAC 引导。
- `dsac_rl_driven_mppi`：完整 DSAC + RL-driven MPPI。
- `dsac_rl_driven_mppi_no_hss`：关闭 guided rollout。
- `dsac_rl_driven_mppi_fixed_sigma`：关闭在线 sigma 更新。
- `dsac_rl_driven_mppi_no_q`：关闭 terminal Q。
- `ablation_dsac_rlmppi`：批量运行 DSAC、pure MPPI、完整 DSAC+MPPI 和消融模式。

## 输出目录

默认评估结果保存到：

```text
data/<model-name>/eval/<run-name>/
```

每个评估目录通常包含：

- `<mode>_metrics.csv`：每个 episode 的统计结果。
- `traces/<mode>_episode_*.csv`：每一步的 trace 数据。
- `summary.json`：聚合结果。
- `plots/`：使用 `--plot` 时生成的曲线图。

## Gazebo 单次运行

如果想在 Gazebo 中交互式运行一次策略：

```bash
python -m scripts.test.run_dsac_mppi \
  --model data/dsac_high_level/models/best_model \
  --mode dsac_rl_driven_mppi \
  --max-steps 500 \
  --log-every 10
```

这个脚本适合观察 USV 动作、RViz marker 和 MPPI debug 信息。高层模型会自动使用 `high_level_frenet + first_order` 环境；旧二维模型会自动使用低层速度环境。
