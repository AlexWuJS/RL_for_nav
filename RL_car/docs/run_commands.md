# 项目运行指令

默认从工作区内层目录运行：

```bash
cd ~/RL_car/RL_car
```

## 单元测试

```bash
python -m unittest discover -s tests
```

## 编译检查

```bash
python -m py_compile \
  dsac_mppi/algorithms/dsac.py \
  dsac_mppi/controllers/mppi_dbas.py \
  dsac_mppi/controllers/rl_driven_mppi.py \
  dsac_mppi/envs/ros_env.py \
  scripts/train/train_dsac.py \
  scripts/test/compare_dsac_mppi.py
```

## 验证项目是否能运行

建议按下面顺序验证，先确认 Python 代码，再确认 ROS/Gazebo。

### 1. 确认当前分支和目录

```bash
cd ~/RL_car/RL_car
git branch --show-current
pwd
```

正常情况：

- 分支应为 `project/dsac-mppi`。
- 当前目录应为内层工程目录 `~/RL_car/RL_car`。

### 2. Python 代码级检查

```bash
python -m py_compile \
  dsac_mppi/algorithms/dsac.py \
  dsac_mppi/controllers/mppi_dbas.py \
  dsac_mppi/controllers/rl_driven_mppi.py \
  dsac_mppi/envs/ros_env.py \
  scripts/train/train_dsac.py \
  scripts/test/compare_dsac_mppi.py \
  scripts/test/run_dsac_mppi.py

python -m unittest discover -s tests
```

正常情况：

- `py_compile` 没有输出。
- 单元测试显示 `OK`，部分依赖缺失时出现 `skipped` 是可以接受的。

### 3. ROS 包结构检查

```bash
catkin_make --source src/ros
```

正常情况：

- `nav_demo` 和 `urdf02_gazebo` 能被 catkin 识别。
- 编译结束没有 CMake 或 package 找不到的错误。

如果机器上的 CMake 是 4.x，ROS Noetic 自带的部分 catkin/gtest 文件会触发旧版 CMake 兼容警告。当前项目已经在 `src/ros/CMakeLists.txt` 设置了兼容策略；如果仍遇到类似 `Compatibility with CMake < 3.5 has been removed` 的报错，先删除失败生成的目录再重试：

```bash
rm -rf build devel
catkin_make --source src/ros
```

编译完成后加载环境：

```bash
source devel/setup.bash
```

### 4. 启动 Gazebo 仿真

另开一个终端：

```bash
cd ~/RL_car/RL_car
source devel/setup.bash
roslaunch urdf02_gazebo RL_car.launch
```

启动后检查关键接口：

```bash
rostopic list | grep -E '/scan|/cmd_vel|/gazebo/model_states'
rosservice list | grep -E '/gazebo/set_model_state|/gazebo/get_model_state|/gazebo/spawn_sdf_model'
```

正常情况：

- 能看到 `/scan`、`/cmd_vel`、`/gazebo/model_states`。
- 能看到 Gazebo 的 set/get/spawn model service。

### 5. 运行 DSAC+MPPI 单次测试

在 Gazebo 已启动的情况下运行：

```bash
python -m scripts.test.run_dsac_mppi \
  --model data/dsac_high_level/models/best_model \
  --mode dsac_rl_driven_mppi \
  --max-steps 200 \
  --log-every 10
```

正常情况：

- 终端持续输出 step/reward/info。
- 默认输出为精简调试信息。
- Gazebo 中机器人能移动。
- 如果模型路径不存在，先确认 `data/dsac_high_level/models/best_model/config.json` 和 `model.pt` 是否存在。

### 6. 运行短评估

```bash
python -m scripts.test.compare_dsac_mppi \
  --model data/dsac_high_level/models/best_model \
  --mode dsac \
  --episodes 1 \
  --max-steps 200 \
  --run-name verify_dsac
```

正常情况：

- 输出 JSON summary。
- 生成目录 `data/dsac_high_level/eval/verify_dsac/`。

### 7. 训练 smoke test

训练需要 Gazebo 环境正常运行：

```bash
python -m scripts.train.train_dsac \
  --model-name dsac_smoke_verify \
  --control-mode high_level_frenet \
  --dynamics-model ideal \
  --curriculum auto \
  --total-timesteps 1000 \
  --learning-starts 100
```

正常情况：

- 训练能开始采样并打印 DSAC 状态。
- 输出目录为 `data/dsac_smoke_verify/`。

## 训练 DSAC

```bash
python -m scripts.train.train_dsac \
  --model-name dsac_high_level \
  --control-mode high_level_frenet \
  --dynamics-model ideal \
  --curriculum auto \
  --total-timesteps 300000
```

输出路径：

```text
data/dsac_high_level/models/best_model
data/dsac_high_level/models/final_model_dsac
data/dsac_high_level/logs/
```

更多训练参数见 `docs/train_dsac.md`。

## 评估 DSAC+MPPI

```bash
python -m scripts.test.compare_dsac_mppi \
  --model data/dsac_high_level/models/best_model \
  --mode dsac_rl_driven_mppi \
  --run-name dsac_rlmppi_eval \
  --plot
```

输出路径：

```text
data/dsac_high_level/eval/dsac_rlmppi_eval/
```

更多评估模式见 `docs/evaluate_dsac_mppi.md`。

## Gazebo 单次运行

```bash
python -m scripts.test.run_dsac_mppi \
  --model data/dsac_high_level/models/best_model \
  --mode dsac_rl_driven_mppi
```
