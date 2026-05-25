# SAC + MPPI-DBaS 低干预改进工作记录

日期：2026-05-08

分支：`codex/usv-mppi-dbas`

相关提交：

- `08cee8f Implement shield-first MPPI DBaS`
- `a3afc5b Adapt comparison env to model observation space`
- `01b195e Fix comparison plots for shield modes`
- `fc6f53e Add shield modes to Gazebo test script`

## 1. 修改目标

原始问题是：`SAC + MPPI-DBaS` 的效果不如纯 SAC。评估结果显示，问题主要不是 SAC 策略本身，而是 MPPI 作为后处理控制器时过度改动作，破坏了 SAC 已学到的路径跟踪和推进节奏。

本轮采用“不重训”的低干预路线：

- 默认尽量执行 SAC 原动作。
- 只有在预测碰撞、TTC 风险、越界风险、前方距离过近时，才让 shield/fallback 或 MPPI 介入。
- 优先验证 `shield_only`，再观察 MPPI 是否适合执行。
- 保留 teacher 记录能力，用于后续判断 MPPI 候选是否真的优于 SAC。

## 2. 主要代码修改

### 2.1 `mppi_dbas.py`

文件：

`RL_car/src/nav_demo/scripts/beam_map/mppi_dbas.py`

主要修改：

- 调整 `MPPIDBaSConfig` 默认行为：
  - `always_run_mppi=False`
  - `final_safety_check=True`
  - 降低 residual 动作幅度，特别限制 yaw 大幅偏转。
- 新增低干预决策逻辑：
  - SAC 安全时直接执行 SAC。
  - SAC 不安全时先考虑 fallback。
  - MPPI 候选只有在通过安全收益、进度、Frenet 横向误差、动作信任域等检查后才允许执行。
- 改 MPPI 接收准则：
  - 不再因为“最小障碍距离略大”就接收 MPPI。
  - 若进度下降、横向误差恶化、预测越界或奖励收益不足，则拒绝 MPPI。
  - fallback 优先级高于 MPPI，除非 MPPI 明显更优。
- 改 MPPI 代价函数：
  - 对齐 `ros_env.py` 的真实奖励结构。
  - 使用 Frenet 进度、横向误差、航向误差、动作平滑、终止惩罚。
  - 避障项改为渐进 barrier：只在距离低于安全距离时增强惩罚。
  - Frenet 横向惩罚改为分段形式：中心走廊内不罚，缓冲区软罚，越界区硬罚。
- 增加 debug/teacher 字段：
  - `action_source`
  - `mppi_active`
  - `mppi_accept`
  - `mppi_reject`
  - `mppi_decision_reason`
  - `teacher_mppi_would_accept`
  - `predicted_reward_sac`
  - `predicted_reward_mppi`
  - `predicted_reward_delta`
  - `mppi_pred_collision`
  - `mppi_pred_out_of_bounds`

### 2.2 `mppi_dbas_wrapper.py`

文件：

`RL_car/src/nav_demo/scripts/beam_map/mppi_dbas_wrapper.py`

作用：

- 在环境 `step(action)` 前，把 SAC 动作交给 `MPPIDBaSOptimizer`。
- 执行优化后动作或 fallback 动作。
- 把 debug 信息写入 `info`，方便评估和仿真打印。

### 2.3 `compare_sac_mppi.py`

文件：

`RL_car/src/nav_demo/scripts/beam_map/compare_sac_mppi.py`

主要修改：

- 新增/明确评估模式：
  - `baseline`
  - `shield_only`
  - `shield_first`
  - `shield_mppi_teacher`
  - `shield_mppi_execute`
- `--mode both` 现在对比：
  - `baseline`
  - `shield_first`
- `--mode ablation` 现在对比：
  - `baseline`
  - `shield_only`
  - `shield_mppi_teacher`
  - `shield_mppi_execute`
- 增加 `--obs-mode {auto,flat,dict}`：
  - 自动检测模型 observation space。
  - 修复了模型期望 Dict observation、但当前环境输出 flat observation 时的加载报错。
- 增加 `RadarDictObservationWrapper`：
  - 将 flat lidar observation 转为 `{radar_image, kinematics}`。
  - 用于兼容旧的 Dict observation 模型。
- `--episodes` 增加别名 `--episode`。
- 评估时输出更多 MPPI/shield trace 字段，便于分析介入行为。

### 2.4 `plot_comparison_curves.py`

文件：

`RL_car/src/nav_demo/scripts/beam_map/plot_comparison_curves.py`

主要修改：

- 修复画图报错：

```text
ValueError: Invalid RGBA argument: None
```

原因是新模式没有配置颜色，`color_for()` 返回了 `None`。

修复内容：

- 为新模式补充颜色：
  - `shield_first`
  - `shield_only`
  - `shield_mppi_teacher`
  - `shield_mppi_execute`
- 为未知模式增加稳定 fallback 颜色，避免以后新增 mode 再次导致 Matplotlib 崩溃。

### 2.5 `test01.py`

文件：

`RL_car/src/nav_demo/scripts/beam_map/test01.py`

主要修改：

- 将原本只运行纯 SAC 的仿真脚本扩展为可选择模式：
  - `baseline`
  - `shield_only`
  - `shield_first`
  - `shield_mppi_teacher`
  - `shield_mppi_execute`
- 默认仍是 `baseline`，不改变原始运行方式。
- 当选择 shield 模式时，自动套用 `MppiDbaSActionWrapper`。
- 运行时打印 shield debug：
  - 当前动作来源：`sac` / `fallback` / `mppi`
  - 是否激活 MPPI
  - 是否接收 MPPI
  - 决策原因
  - 动作变化量
  - 当前障碍距离

### 2.6 测试文件

新增：

`RL_car/tests/test_mppi_dbas_low_intervention.py`

覆盖低干预 MPPI/DBaS 的核心行为。

新增：

`RL_car/tests/test_plot_comparison_curves.py`

覆盖新模式颜色映射，防止画图再次因为 `None` 颜色崩溃。

## 3. 已验证的命令

在本地验证过：

```bash
python -m unittest discover -s RL_car/tests
```

结果：

```text
Ran 7 tests
OK
```

语法检查：

```bash
python -m py_compile \
  RL_car/src/nav_demo/scripts/beam_map/mppi_dbas.py \
  RL_car/src/nav_demo/scripts/beam_map/mppi_dbas_wrapper.py \
  RL_car/src/nav_demo/scripts/beam_map/compare_sac_mppi.py \
  RL_car/src/nav_demo/scripts/beam_map/plot_comparison_curves.py \
  RL_car/src/nav_demo/scripts/beam_map/test01.py
```

## 4. 评估运行指令

进入目录：

```bash
cd ~/RL_car/RL_car/src/nav_demo/scripts/beam_map
```

推荐使用当前 v2 模型：

```bash
python compare_sac_mppi.py \
  --model ./training_usv_v2_results/best_model \
  --mode ablation \
  --episode 30 \
  --output-dir ./comparison_results_low_intervention \
  --plot
```

如果只想快速看 baseline 和低干预 MPPI：

```bash
python compare_sac_mppi.py \
  --model ./training_usv_v2_results/best_model \
  --mode both \
  --episode 30 \
  --output-dir ./comparison_results_low_intervention \
  --plot
```

如果只跑 `shield_only`：

```bash
python compare_sac_mppi.py \
  --model ./training_usv_v2_results/best_model \
  --mode shield_only \
  --episode 30 \
  --output-dir ./comparison_results_shield_only \
  --plot
```

如果模型 observation space 不匹配，可以手动指定：

```bash
python compare_sac_mppi.py \
  --model ./training_results/best_model \
  --mode baseline \
  --episode 5 \
  --obs-mode dict
```

一般情况下推荐保留默认：

```bash
--obs-mode auto
```

## 5. 单独补画图指令

如果评估已经跑完，但画图失败或想重新生成图，不需要重跑 episode：

```bash
python plot_comparison_curves.py \
  --result-dir ./comparison_results_low_intervention \
  --output-dir ./comparison_results_low_intervention/plots \
  --max-steps 180
```

生成目录：

```text
./comparison_results_low_intervention/plots
```

常看图：

- `summary_bars.png`
- `episode_curves.png`
- `terminal_outcomes.png`
- `trace_curves.png`
- `reward_alignment.png`
- `action_source.png`

## 6. Gazebo/ROS 仿真运行指令

进入目录：

```bash
cd ~/RL_car/RL_car/src/nav_demo/scripts/beam_map
```

原始 SAC：

```bash
python test01.py --mode baseline
```

推荐先看当前效果最好的 shield：

```bash
python test01.py --mode shield_only
```

看低干预 MPPI 执行版：

```bash
python test01.py --mode shield_first
```

看 teacher 记录模式：

```bash
python test01.py --mode shield_mppi_teacher
```

看 MPPI 执行版：

```bash
python test01.py --mode shield_mppi_execute
```

指定模型路径：

```bash
python test01.py \
  --mode shield_only \
  --model ./training_usv_v2_results/best_model
```

减少打印频率：

```bash
python test01.py --mode shield_only --log-every 30
```

关闭周期性 debug 打印，只在动作来源变化时打印：

```bash
python test01.py --mode shield_only --log-every 0
```

运行时重点看：

```text
source=sac
source=fallback
source=mppi
```

含义：

- `source=sac`：执行 SAC 原动作。
- `source=fallback`：shield 安全兜底介入。
- `source=mppi`：MPPI 候选被接收并执行。

当前建议重点观察 `shield_only`：

- 是否明显减少撞障碍。
- 是否减少越界。
- fallback 是否只在危险时介入。
- 船是否仍能保持 SAC 原本的推进节奏。

## 7. 当前结果解读

你给出的 30 episode ablation 图中：

| 模式 | Success | Collision | Out of Bounds | Mean Reward | Min Obstacle Distance | Mean \|Frenet d\| |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.70 | 0.23 | 0.07 | 1183.24 | 1.04 | 1.23 |
| shield_mppi_execute | 0.37 | 0.20 | 0.43 | 704.74 | 1.25 | 0.44 |
| shield_mppi_teacher | 0.63 | 0.23 | 0.20 | 975.65 | 1.15 | 0.49 |
| shield_only | 0.87 | 0.13 | 0.00 | 1406.43 | 0.95 | 0.58 |

结论：

- `shield_only` 是当前最好方案。
- `shield_only` 成功率最高，碰撞率更低，越界率为 0，平均奖励最高。
- `shield_mppi_execute` 虽然最小障碍距离最高、Frenet 横向误差最低，但成功率明显下降，越界率严重升高。
- 这说明 MPPI 当前倾向于局部避障或局部贴路径，但会破坏全局任务节奏，尤其容易把船推向边界。
- 当前不建议默认执行 MPPI。
- 推荐把 `shield_only` 作为主线方案，把 MPPI 保留在 teacher/诊断层继续校准。

## 8. 推荐下一步

短期：

- 仿真中优先运行：

```bash
python test01.py --mode shield_only
```

- 同场景对比：

```bash
python test01.py --mode baseline
```

- 观察 `source=fallback` 是否集中出现在近障碍或高风险时刻。

中期：

- 检查 `shield_mppi_teacher` 是否真的不改变执行动作。
- 如果 teacher 模式仍明显影响结果，需要检查 wrapper 或 config 是否改变了 fallback 行为。
- 继续收紧 MPPI 接收条件，特别是越界风险：
  - MPPI 候选不能增加预测越界风险。
  - yaw residual 继续降低。
  - fallback 优先级继续高于 MPPI。
  - 只有 fallback 不足以规避风险时，才允许 MPPI 接管。

后续训练方向：

- 若 teacher 数据显示 MPPI 候选在少数危险场景确实优于 SAC，可以再考虑小规模重训。
- 重训时可考虑：
  - 将 teacher 信号作为辅助数据。
  - 学 residual policy。
  - 把 shield/fallback 介入事件纳入风险感知训练。

