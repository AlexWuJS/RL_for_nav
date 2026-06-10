# DSAC 训练脚本使用说明

DSAC 训练入口为：

```bash
cd ~/RL_car/RL_car
python -m scripts.train.train_dsac
```

## 常用命令

快速 smoke 训练，用来确认流程能跑通：

```bash
python -m scripts.train.train_dsac \
  --model-name dsac_smoke \
  --control-mode high_level_frenet \
  --dynamics-model first_order \
  --curriculum auto \
  --total-timesteps 5000 \
  --learning-starts 1000
```

正式训练：

```bash
python -m scripts.train.train_dsac \
  --model-name dsac_high_level \
  --control-mode high_level_frenet \
  --dynamics-model first_order \
  --curriculum auto \
  --total-timesteps 300000 \
  --log-interval 1000 \
  --episode-log-interval 100
```

## 输出目录

当 `--model-name dsac_high_level` 时，默认输出为：

```text
data/dsac_high_level/models/best_model
data/dsac_high_level/models/final_model_dsac
data/dsac_high_level/logs/
data/dsac_high_level/logs/tensorboard/
```

查看 TensorBoard：

```bash
tensorboard --logdir data/dsac_high_level/logs/tensorboard --port 6006
```

浏览器打开 `http://localhost:6006`，重点看 `rollout/*`、`train/*` 和 `curriculum/*`。

## 参数说明

- `--model-name`：模型名称，同时决定 `data/<model-name>/` 输出目录。
- `--total-timesteps`：训练总步数。
- `--learning-starts`：开始更新网络前收集的环境步数。
- `--frame-stack`：观测帧堆叠数量，默认 4。
- `--control-mode`：`high_level_frenet` 表示 DSAC 输出 `[delta_s, target_d, target_speed]`；`low_level_velocity` 表示旧二维速度动作。
- `--dynamics-model`：`first_order` 表示默认 USV 一阶滞后模型；`ideal` 表示理想速度跟踪调试模式；`inertia` 会兼容映射到 `first_order`。
- `--curriculum`：`auto` 表示按最近训练回合自动推进课程阶段；`off` 表示直接使用完整随机环境。
- `--log-interval`：按环境 step 输出一次训练状态表，并写入 TensorBoard，默认 1000。
- `--episode-log-interval`：按 episode 输出一次汇总表，默认 100；设为 0 可关闭终端 episode 汇总。
- `--save-dir`：手动指定模型输出目录；通常不需要设置。
- `--log-dir`：手动指定日志目录；通常不需要设置。
- `--device`：`auto`、`cpu` 或 `cuda`。

建议优先使用 `--model-name` 管理不同模型，不要手动把输出散放到脚本目录。

## 课程学习

`--curriculum auto` 会从短距离、低曲率、无复杂障碍的阶段开始，逐步增加路径长度、曲率和障碍难度。推进标准为最近窗口内成功率、碰撞率和平均横向误差达标。

`best_model` 现在按训练过程中的 rolling 指标保存，不会在训练结束时被 `final_model_dsac` 无条件覆盖。
