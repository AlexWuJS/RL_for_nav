# CHANGELOG_MINIMAX

## 2026-04-09 14:00 真实地图+轨迹接入 - 地图语义和坐标修复

### 改动文件
- tools/convert_ros_map_to_npy.py
- tools/visualize_processed_scenario.py
- configs/train_grid.yaml
- grid_env.py
- grid_map_loader.py

### 修改内容

#### 1. ROS地图语义修复 (convert_ros_map_to_npy.py)

**问题**: 原代码 `occupancy[img_array > occupied_thresh] = 1` 逻辑错误。
- negate=0时，白色(高像素值)应该是free，黑色(低像素值)才是occupied
- 原代码把高像素值当作occupied，语义反了

**修复**:
```python
# 修复前 (错误):
occupancy[img_array > occupied_thresh] = 1  # 高像素=occupied，语义反

# 修复后 (正确):
if negate == 0:
    occ_prob = 1.0 - img_array  # 白=0(自由), 黑=1(障碍)
else:
    occ_prob = img_array
is_free = occ_prob < free_thresh
is_occupied = occ_prob > occupied_thresh
is_unknown = ~(is_free | is_occupied)
occupancy[is_occupied] = 1
occupancy[is_unknown] = 1  # 保守: unknown当障碍物
```

**新增**:
- 保存debug PNG (`navigation_map_debug.png`) 用于肉眼确认语义

**修复后统计**:
- 自由像素: 6792 (41.9%) - 白色区域=水域
- 障碍物像素: 9408 (58.1%) - 黑色区域=陆地
- 未知像素: 0 (0.0%)

#### 2. 坐标系统一 (visualize_processed_scenario.py)

**问题**: 地图用了`np.flipud`+`origin='lower'`，但轨迹坐标是世界y向下坐标系，导致上下镜像错位。

**修复**:
- 移除 `np.flipud(map_data)`
- 改用 `origin='upper'` (array顶部=world_y=0)
- 统一 y-down 坐标约定: world_y 向下增加，对应图像row向下增加

**修复前** (错误):
```python
display_map = np.flipud(map_data)
ax.imshow(display_map, ..., origin='lower')
# world_y=0在底部，导致轨迹上下镜像
```

**修复后** (正确):
```python
ax.imshow(map_data, ..., origin='upper',
          extent=[0, width*res, height*res, 0])
# world_y=0在顶部，与AIS世界坐标一致
```

**同步修复 `count_obstacles_in_water`**:
- 移除 `map_row = height - 1 - row` (原为补偿flip)
- 直接使用 `map_data[row, col]`

#### 3. grid_map_loader.py 更新

**新增**:
- `load_grid_map()` 统一加载接口
- 自动检测 `.npy` vs `.pgm`
- `.pgm` 直接加载报错并提示用户先运行转换脚本

#### 4. train_grid.yaml 参数调整

**改动**:
- `map_path`: `example_data/sample_map.npy` → `data/processed/maps/navigation_map.npy`
- `trajectory_path`: `example_data/sample_trajectories.json` → `data/processed/trajectories/multi_obstacles.json`
- `robot_radius`: 0.15 → 50.0 (配合123m粗分辨率)
- `v_max`: 1.0 → 20.0
- `w_max`: 1.0 → 0.1
- `dt`: 0.1 → 5.0
- `max_episode_steps`: 500 → 500

### 修改原因
1. 地图语义反转导致AIS船只轨迹64%落在"陆地"上(实际是水域)，需要修正occupied判断逻辑
2. 坐标系统一避免显示链与数据链错位

### 验证命令
```bash
# 1. 地图转换
python tools/convert_ros_map_to_npy.py \
  --map example_data/maps/navigation_map_0.50m.pgm \
  --yaml example_data/maps/navigation_map_0.50m.yaml \
  --out-map data/processed/maps/navigation_map.npy \
  --out-meta data/processed/maps/navigation_map_meta.yaml

# 2. 轨迹转换
python tools/convert_ais_xls_to_obstacles.py \
  --input-dir data/raw/trajectories \
  --map-meta data/processed/maps/navigation_map_meta.yaml \
  --out-json data/processed/trajectories/multi_obstacles.json

# 3. 可视化
python tools/visualize_processed_scenario.py \
  --map data/processed/maps/navigation_map.npy \
  --meta data/processed/maps/navigation_map_meta.yaml \
  --traj data/processed/trajectories/multi_obstacles.json \
  --save-png overlay_debug.png \
  --save-gif overlay_debug.gif \
  --no-show
```

### 验证结果

**地图转换**:
- ✅ 成功
- 自由像素: 41.9%, 障碍物: 58.1%
- 分辨率: 123.1276 m/pixel (非yaml中的0.5)

**轨迹转换**:
- ✅ 成功处理 38/41 个文件
- 失败3个: 数据点不足
- 总轨迹点: 9872

**可视化统计**:
- 地图: 180x90, 123.13 m/pixel
- 地理: lon=[120.4147,120.6447], lat=[31.9803,32.0819]
- 轨迹点统计:
  - 水域(free): 36.0%
  - 陆地(obstacle): 64.0%
  - 地图外: 0.0%

### 仍存在的问题

1. **64%轨迹点在陆地**: 可能是GPS误差、港口停靠、或坐标映射仍需微调
2. **分辨率过粗**: 123m/pixel，船舶(~20m)仅占0.16像素
3. **坐标系约定**: 采用y-down (world_y向下增加)，需在文档中明确

---

## 2026-04-09 11:27 Grid环境Bug修复 (前置工作)

### 改动文件
- dynamic_obstacle_manager.py
- grid_env.py
- train.py
- grid_test.py
- configs/train_grid.yaml

### 修改内容

#### 1. dynamic_obstacle_manager.py

**Bug修复**:
- `current_position` 是property不能赋值，移除update中的赋值语句
- `get_position_at_time()` 插值循环逻辑错误，修复为标准分段线性插值
- `load_trajectory_from_json()` return语句位置验证无误

**新增**:
- 6个单元测试用例覆盖reset/update/get_obstacle_positions/loop

#### 2. grid_env.py

**Bug修复**:
- 新增 `_check_out_of_bounds_xy(x, y)` 使用传入参数而非self.robot_pos
- `reset()` 中 `position_history` 在 `robot_pos` 采样之后初始化
- `step()` 逻辑：非法位置不更新robot_pos，info明确区分collision/goal/out_of_bounds/timeout

#### 3. train.py

**Bug修复**:
- CLI `--seed` 真正传入环境
- `save_dir/log_dir/tensorboard_log` 从config读取
- `--total_timesteps` CLI优先级高于config
- `final_model` 正常结束和KeyboardInterrupt都会保存

#### 4. grid_test.py

**Bug修复**:
- 移除 `gym.make('custom', ...)` 改用 `DummyVecEnv`
- 添加 `_make_batched_action()` 处理VecEnv action格式
- `--config` 参数真正读取并传递

### 验证命令
```bash
# 单元测试
python dynamic_obstacle_manager.py

# 随机Agent测试
python grid_test.py --random --no-render

# 快速训练
python train.py --env_type grid --total_timesteps 1000

# 模型测试
python grid_test.py --model ./training_grid_results/final_model.zip --episodes 1 --no-render
```

### 验证结果

**单元测试**: ✅ 6/6通过
**随机Agent**: ✅ 3个episode各500步(timeout)
**训练**: ✅ 成功完成
**模型测试**: ✅ 成功加载和运行

---

## 输出文件清单

### 地图处理
- `data/processed/maps/navigation_map.npy` - 二值占据栅格
- `data/processed/maps/navigation_map_meta.yaml` - 元数据
- `data/processed/maps/navigation_map_debug.png` - 语义确认图

### 轨迹处理
- `data/processed/trajectories/multi_obstacles.json` - 38个船舶轨迹
- `data/processed/trajectories/single_obstacle_example.json` - 单船样例
- `data/processed/trajectories/cleaned_preview.csv` - CSV预览

### 可视化
- `overlay_debug.png` - 地图+轨迹叠加图
- `overlay_debug.gif` - 动态障碍物动画 (生成中)

### 训练
- `training_grid_results/best_model.zip`
- `training_grid_results/final_model.zip`
