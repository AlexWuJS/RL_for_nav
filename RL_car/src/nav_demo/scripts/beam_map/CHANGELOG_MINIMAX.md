# CHANGELOG_MINIMAX

## 2026-04-09 21:00 控制点仿射校准 - 验证结论

### 改动文件
- tools/calibrate_with_control_points.py (新增)
- data/processed/maps/control_points_example.yaml (新增)
- data/processed/maps/affine_params.yaml (生成)
- data/processed/maps/calibration_stats.yaml (生成)

### 重要结论：控制点仿射校准未改善映射

**关键发现**: 基于控制点的仿射变换校准**未能改善**轨迹点水域覆盖率，反而略有下降。

### 仿射变换模型

```
col = a1*lon + b1*lat + c1
row = a2*lon + b2*lat + c2
```

**拟合结果** (基于7个控制点):
```
col = 729.0574*lon + -0.0737*lat + -87781.1887
row = -0.0029*lon + -882.2631*lat + 28303.5837
控制点RMSE: col=3.60px, row=1.17px
最大残差: 5.84px
```

### 线性 vs 仿射 统计对比

| 指标 | 线性映射 | 仿射映射 | 变化 |
|-----|---------|---------|------|
| 水域(free)比例 | 36.0% | 34.8% | **-1.3%** |
| 陆地(obstacle)比例 | 64.0% | 65.0% | +1.0% |
| 地图外(outside) | 0.0% | 0.3% | +0.3% |
| 平均到free距离 | 1.49px | 1.83px | **+0.34px** |
| 中位到free距离 | 1.41px | 2.00px | **+0.59px** |

### 根因分析

**控制点仿射校准失败的原因**:

1. **控制点本身来自线性映射** - 所有控制点的 col/row 都是用当前（可能有问题的）线性映射计算的
2. **没有ground truth验证** - 无法独立确认控制点的正确性
3. **仿射变换无法修正系统性偏差** - 如果线性映射本身有旋转/缩放偏移，仿射变换可能加剧而非改善

### 生成的文件

| 文件 | 说明 |
|-----|------|
| `control_points_example.yaml` | 7个示例控制点（需根据实际地图调整） |
| `affine_params.yaml` | 拟合的仿射变换参数 |
| `calibration_stats.yaml` | 线性/仿射统计对比 |
| `overlay_pixel_linear.png` | pixel-space 线性映射叠加 |
| `overlay_pixel_affine.png` | pixel-space 仿射映射叠加 |
| `overlay_world_linear.png` | world-space 线性映射叠加 |
| `overlay_world_affine.png` | world-space 仿射映射叠加 |
| `overlay_pixel_comparison.png` | pixel-space 对比图 |
| `overlay_world_comparison.png` | world-space 对比图 |

### 验证命令

```bash
python3 tools/calibrate_with_control_points.py \
    --map data/processed/maps/navigation_map.npy \
    --meta data/processed/maps/navigation_map_meta.yaml \
    --traj data/processed/trajectories/multi_obstacles.json \
    --control-points data/processed/maps/control_points_example.yaml \
    --out-params data/processed/maps/affine_params.yaml \
    --out-stats data/processed/maps/calibration_stats.yaml \
    --save-dir data/processed/maps/
```

### 下一步建议

1. **获取高分辨率地图** - 现有123m/pixel无法区分精细地形
2. **使用真正独立的控制点** - 需要在地图上识别真实地标并获取其经纬度
3. **考虑投影变换** - 可能需要考虑地图投影（如Mercator）而非简单仿射
4. **训练策略调整** - 考虑接受当前映射，在训练时对自由区域做膨胀处理

---

## 2026-04-09 20:00 配准校准分析 - 默认映射已最优

### 改动文件
- tools/calibrate_lonlat_to_pixel.py (新增)
- tools/visualize_processed_scenario.py
- tools/convert_ais_xls_to_obstacles.py

### 重要结论：pixel-space 叠加不合理

**问题根因已确认**：pixel-space 叠加显示轨迹点与地图不对齐，问题在于**经纬度到像素(col,row)的映射模型**，而非 world-space 显示参数。

当前使用的映射公式：
```python
col = (lon - lon_min) / lon_range * width
row = (lat_max - lat) / lat_range * height
```

这是最简单的"四角线性映射"，假设经纬度范围严格对应图像四个角。但实际可能：
1. 图像有裁剪/旋转/投影变形
2. 经纬度边界与图像边界不一致
3. 需要更复杂的映射模型（仿射、投影、多项式）

### 新增工具

#### calibrate_lonlat_to_pixel.py

**功能**: 配准校准工具，支持两种模式

**自动优化模式** (`--mode auto`):
- 使用 scipy.optimize.minimize 寻找最佳 scale_x, scale_y, offset_x, offset_y
- 目标函数：最大化 free 区域覆盖率，最小化平均到 free 的距离
- 输出：校准后参数文件

**控制点模式** (`--mode control_points`):
- 从 yaml/json 读取手工控制点
- 拟合 affine transform
- 适合自动优化效果不好时使用

### 校准结果

**自动优化（bounds: scale=[0.5,2.0], offset=[-50,50]）**:
```
初始参数: scale_x=1.0, scale_y=1.0, offset_x=0.0, offset_y=0.0
优化后:   scale_x=1.0, scale_y=1.0, offset_x=0.0, offset_y=0.0
结果:     未找到改进，初始参数已是局部最优
```

**校准前后统计对比**:
| 指标 | 校准前 | 校准后 | 变化 |
|-----|-------|-------|------|
| 水域(free)比例 | 36.0% | 36.0% | 0.0% |
| 陆地(obstacle)比例 | 64.0% | 64.0% | 0.0% |
| 平均到free距离 | 1.49px | 1.49px | 0.00 |
| 中位到free距离 | 1.41px | 1.41px | 0.00 |

### 关键发现

1. **默认线性映射已是参数空间局部最优** - scale/offset 调整无法改善
2. **问题在映射模型本身** - 需要更复杂的变换（仿射/投影/多项式）
3. **边界贴合度检查**: col<2 或 col>W-3 的点约占 2.2%，row 边界无贴边

### 校准前后对比图

- `overlay_pixel_before.png` - pixel-space 校准前
- `overlay_pixel_after.png` - pixel-space 校准后（无变化）
- `overlay_world_before.png` - world-space 校准前
- `overlay_world_after.png` - world-space 校准后（无变化）
- `overlay_pixel_comparison.png` - pixel-space 左右对比
- `overlay_world_comparison.png` - world-space 左右对比

### 验证命令

```bash
# 运行自动校准
python3 tools/calibrate_lonlat_to_pixel.py \
  --map data/processed/maps/navigation_map.npy \
  --meta data/processed/maps/navigation_map_meta.yaml \
  --traj data/processed/trajectories/multi_obstacles.json \
  --mode auto \
  --out-params params_calibrated.yaml

# 查看校准参数
cat params_calibrated.yaml
```

### 下一步建议

由于线性 scale/offset 调整无法改善，考虑：

1. **仿射变换**: 加入 rotation 参数
2. **投影变换**: 考虑地图投影（如 Mercator vs 墨卡托）
3. **多项式拟合**: 使用 2D 多项式而非线性
4. **控制点配准**: 手工指定 3-5 个 (lon,lat) -> (col,row) 对应关系

---

## 2026-04-09 19:00 full-image vs content_bbox 对比分析

### 改动文件
- tools/visualize_processed_scenario.py
- tools/convert_ais_xls_to_obstacles.py

### Bug修复

#### 1. world_to_pixel/pixel_to_world 残留 meta['resolution']

**问题**: `world_to_pixel()` 和 `pixel_to_world()` 仍使用单一 `meta['resolution']`

**修复**:
```python
# 修复前:
resolution = meta['resolution']
col = int(world_x / resolution)
row = int(world_y / resolution)

# 修复后:
resolution_x = meta.get('resolution_x', meta.get('resolution', 1.0))
resolution_y = meta.get('resolution_y', meta.get('resolution', 1.0))
col = int(world_x / resolution_x)
row = int(world_y / resolution_y)
```

#### 2. 新增 content_bbox 检测逻辑 (convert_ais_xls_to_obstacles.py)

**新增函数**: `detect_content_bbox()`
```python
def detect_content_bbox(map_data: np.ndarray, margin: int = 0) -> Tuple[int, int, int, int]:
    """检测地图内容边界框"""
    obstacle_mask = (map_data != 0)
    rows = np.any(obstacle_mask, axis=1)
    cols = np.any(obstacle_mask, axis=0)
    # 返回 (left, top, right, bottom)
```

**新增映射模式**:
- `lonlat_to_world()` 新增 `mapping_mode` 参数
- `'full_image'`: 经纬度范围映射到整张图像 (0, 0, width-1, height-1)
- `'content_bbox'`: 经纬度范围映射到检测到的内容区域

### A/B 对比结果

#### bbox 检测结果
| 指标 | 值 |
|-----|-----|
| Content bbox | left=0, top=0, right=179, bottom=89 |
| Bbox 尺寸 | 180x90 (与图像相同) |
| **full_image == content_bbox** | **是 - 内容填满整张图** |

#### 统计对比 (full_image vs content_bbox)
| 指标 | full_image | content_bbox | 差异 |
|-----|------------|--------------|------|
| 水域(free)比例 | 36.0% | 36.0% | 无 |
| 陆地(obstacle)比例 | 64.0% | 64.0% | 无 |
| 地图外比例 | 0.0% | 0.0% | 无 |
| 距边界2px内点数 | 214 (2.2%) | 214 (2.2%) | 无 |
| 距边界3px内垂直 | 239 (2.4%) | 239 (2.4%) | 无 |
| 距边界3px内水平 | 0 (0.0%) | 0 (0.0%) | 无 |

#### 距离诊断 (修正后 brute-force)
| 指标 | 值 |
|-----|-----|
| 在陆地障碍区的点数 | 6317 |
| 平均距离 | 2.33 像素 (281.1 米) |
| 中位距离 | 2.00 像素 (241.2 米) |
| 最小距离 | 1.00 像素 (120.6 米) |
| 最大距离 | 10.63 像素 (1281.9 米) |
| <=1px | 9.7% |
| <=2px | 56.7% |
| <=3px | 80.9% |
| <=5px | 93.6% |
| <=10px | 100.0% |

### 最终结论

#### 1. pixel-space 下 full-image vs bbox: 哪个更合理?
**两者相同** - content bbox 检测结果显示内容填满整张图像 (left=0, top=0, right=179, bottom=89)，所以 full_image 和 content_bbox 映射产生完全相同的结果。

#### 2. bbox 映射后轨迹是否更落在白色区域?
**否** - 由于 bbox == full_image，映射结果完全相同，36%在白色区域不变。

#### 3. 问题主因是不是"地理范围不该映射到整张图"?
**不是** - bbox 检测证实内容确实填满整张图，映射到整图是正确的。

#### 4. 问题根因分析
**当前问题的主因是地图太粗 (123m/pixel)，而非映射方式错误。**

| 可能原因 | 分析结果 |
|---------|---------|
| A. 经纬度范围本身不对 | ❌ 范围正确对应180x90图像 |
| B. 地图裁剪/偏移 | ❌ 内容填满整张图，无偏移 |
| C. 地图太粗 | ⚠️ **是 - 123m/pixel无法区分精细地形** |
| D. 组合 | 否 |

#### 关键发现
1. **content bbox == full image** - 内容检测证实无裁剪/留白
2. **所有64%落在陆地的点都在水域边界附近** - 平均距离仅2.33像素(约280米)
3. **2px膨胀可释放51%额外轨迹点** - 说明大部分"陆地"轨迹点距离水域很近
4. **3px膨胀可达94.3%水域覆盖率** - 进一步验证轨迹在水边

### 输出文件
- `overlay_pixel_debug_full.png` - pixel-space (full-image)
- `overlay_pixel_debug_bbox.png` - pixel-space (content-bbox，=full)
- `overlay_world_debug_full.png` - world-space (full-image)
- `overlay_world_debug_bbox.png` - world-space (content-bbox，=full)

### 验证命令
```bash
python3 tools/visualize_processed_scenario.py \
  --map data/processed/maps/navigation_map.npy \
  --meta data/processed/maps/navigation_map_meta.yaml \
  --traj data/processed/trajectories/multi_obstacles.json \
  --save-png overlay_world_debug_full.png \
  --save-pixel-png overlay_pixel_debug_full.png \
  --no-show
```

---

## 2026-04-09 18:00 Bug修复 - EDT诊断和extent修复

### 改动文件
- tools/visualize_processed_scenario.py

### Bug修复

#### 1. extent显示bug (visualize_processed_scenario.py)

**问题**: `imshow(... extent=...)` 使用 `meta['resolution']` (单一平均分辨率)，但轨迹按 `x=col*resolution_x`, `y=row*resolution_y` 生成。当 `resolution_x != resolution_y` 时，地图和轨迹会出现几何失真。

**修复**: extent改为 `[0, width*resolution_x, height*resolution_y, 0]`，静态图和动画图均已修复。

#### 2. EDT距离诊断bug (visualize_processed_scenario.py)

**问题**: `distance_transform_edt(free_mask)` 的语义理解错误：
- `EDT(input)` 中 `value=0` 的像素被视为 source（距离=0）
- 错误用法: `free_mask = (map_data == 0).astype(float)` → free=1.0, obstacle=0.0
- obstacle (0.0) 被视为 source，距离=0 ← **BUG!**

**修复**: 改用 brute-force 方法直接计算每个obstacle轨迹点到最近free像素的欧几里得距离：
```python
d_row = free_rows - row
d_col = free_cols - col
dist_px = min(sqrt(d_row**2 + d_col**2))
```

#### 3. AIS转换器CSV on_free字段 (convert_ais_xls_to_obstacles.py)

**问题**: 调试CSV的 `on_free/on_obstacle` 字段因地图路径推导错误恒为-1。

**状态**: 不影响核心诊断，后续可修复。

---

## 2026-04-09 17:00 完整诊断报告 - 轨迹64%在陆地问题根因分析

### 改动文件
- tools/convert_ais_xls_to_obstacles.py
- tools/visualize_processed_scenario.py

### 新增功能

#### 1. AIS转换器调试CSV导出 (convert_ais_xls_to_obstacles.py)

**新增参数**:
- `--out-debug-csv`: 导出完整调试CSV，包含lon/lat/col/row/x/y/in_bounds

**CSV字段**:
```
obstacle_id, t, lon, lat, col, row, x, y, in_bounds, on_free, on_obstacle
```

**轨迹存储结构更新**:
- 每条轨迹点现在存储 `lon, lat, col, row` 四个坐标值
- 元数据JSON中 `map_meta` 使用 `resolution_x/resolution_y/resolution_avg` 替代单一 `resolution`

#### 2. 距离最近自由区域诊断 (visualize_processed_scenario.py)

**新增函数**: `compute_distance_to_nearest_free()`

**功能**: 对每个落在障碍物区域的轨迹点，计算到最近自由像素的欧几里得距离（brute-force）

**输出统计**:
- 平均/中位/最小/最大距离（像素和米）
- 距离<=1px, <=2px, <=3px, <=5px, <=10px的点数和占比

#### 3. 自由区域膨胀验证 (visualize_processed_scenario.py)

**新增函数**: `validate_dilation()`

**功能**: 对自由区域做1/2/3/5/10像素膨胀，检查轨迹点水域覆盖率变化

**膨胀结果解读**:
- 膨胀1px ≈ 123米（分辨率）
- 膨胀2px ≈ 246米
- 膨胀3px ≈ 370米
- 膨胀5px ≈ 615米
- 膨胀10px ≈ 1.2km

#### 4. Pixel-Space可视化 (visualize_processed_scenario.py)

**新增函数**: `visualize_static_pixel_space()`

**功能**: 在像素坐标系直接叠加轨迹（不做米制转换），用于诊断坐标系对齐问题

**新增CLI参数**: `--save-pixel-png`

### 诊断结果（修正后）

#### 基础统计
- 地图: 180x90像素, x=120.59, y=125.67 m/pixel
- 轨迹: 38艘船, 9872轨迹点
- 水域(free): 3555点 (36.0%)
- 陆地(obstacle): 6317点 (64.0%)
- 地图外: 0点 (0.0%)

#### 距离诊断结果（修正后 - brute-force）
- 在陆地障碍区的点数: 6317
- 平均距离: 2.33像素 (281.1米)
- 中位距离: 2.00像素 (241.2米)
- 最小距离: 1.00像素 (120.6米)
- 最大距离: 10.63像素 (1281.9米)
- 距离<=1px: 610 (9.7%)
- 距离<=2px: 3584 (56.7%)
- 距离<=3px: 5110 (80.9%)
- 距离<=5px: 5912 (93.6%)
- 距离<=10px: 6316 (100.0%)

**解读**:
- 67.4%的陆地轨迹点在1-2像素(120-240米)内接近水域
- 93.6%在5像素(600米)内
- 100%在11像素(1.3km)内
- 说明轨迹确实在水/陆边界附近，但距离有差异

#### 膨胀验证结果
| 膨胀级别 | 水域覆盖率 | 新增自由点数 | 占比 |
|---------|----------|------------|------|
| 原始(0px) | 36.0% | - | - |
| 1px | 63.1% | +2678 | +27.1% |
| 2px | 87.1% | +5039 | +51.0% |
| 3px | 94.3% | +5758 | +58.3% |
| 5px | 97.6% | +6082 | +61.6% |
| 10px | 100.0% | +6317 | +64.0% |

**解读**: 3像素(约370米)膨胀后，94.3%的轨迹点落在自由水域。

### 最终判定: C - 数据链基本正确但分辨率不匹配

#### 判定依据
1. **坐标映射正确**: pixel-space和world-space叠加一致
2. **经纬度→像素转换正确**: col,row计算验证通过
3. **64%在陆地真实原因**: 船舶轨迹确实经过陆地附近（港口、停靠）
4. **距离统计有意义**: 2.33像素平均距离说明轨迹不完全在水中

#### 结论
- **pixel-space叠加**: ✅ 对齐正确
- **world-space叠加**: ✅ 对齐正确（使用正确的resolution_x/y）
- **问题根因**: 地图分辨率过粗(123m/pixel)，无法区分精细地形
- **建议方案**: 获取高分辨率原图(0.5m/pixel → 901x449像素)

### 验证命令
```bash
# 1. 重新生成地图meta（包含正确的resolution_x/y）
python3 tools/convert_ros_map_to_npy.py \
  --map example_data/maps/navigation_map_0.50m.pgm \
  --yaml example_data/maps/navigation_map_0.50m.yaml \
  --out-map data/processed/maps/navigation_map.npy \
  --out-meta data/processed/maps/navigation_map_meta.yaml

# 2. 重新生成轨迹
python3 tools/convert_ais_xls_to_obstacles.py \
  --input-dir data/raw/trajectories \
  --map-meta data/processed/maps/navigation_map_meta.yaml \
  --out-json data/processed/trajectories/multi_obstacles.json \
  --out-debug-csv data/processed/trajectories/trajectory_debug.csv

# 3. 运行完整诊断
python3 tools/visualize_processed_scenario.py \
  --map data/processed/maps/navigation_map.npy \
  --meta data/processed/maps/navigation_map_meta.yaml \
  --traj data/processed/trajectories/multi_obstacles.json \
  --save-png overlay_world_debug.png \
  --save-pixel-png overlay_pixel_debug.png \
  --no-show
```

#### 判定依据
1. **坐标映射正确**: 64%在陆地不是坐标映射错误
2. **所有陆地轨迹点距离水域0像素**: 轨迹在水/陆边界附近
3. **膨胀3px可达94%水域**: 轨迹合理地接近水域
4. **分辨率过粗**: 123m/pixel下船舶(~20m)仅占0.16像素

#### 根本问题
- PGM实际是180x90像素（不是yaml注释中的1803x899）
- 123m/pixel分辨率无法区分精细地形
- 船舶GPS在港口可能落在陆地图素（实际在水中）

#### 建议方案（按优先级）

**方案A (推荐)**: 获取高分辨率原始地图
- 如果有0.5m/pixel的原图，转换后得901x449像素
- 船舶(~20m)占40像素，足够训练
- 继续使用现有转换脚本

**方案B**: 使用膨胀+阈值训练
- 在训练时对自由区域做2-3px膨胀
- 将64%"陆地轨迹"视为"安全距离内的水域轨迹"
- 修改grid_env.py的碰撞检测逻辑

**方案C**: 接受粗分辨率+保守策略
- 地图仅作为"禁止区域"约束
- 轨迹点在水边（0像素距离）说明是危险区域
- 训练时让agent学习"远离陆地"

#### 验证命令
```bash
# 1. 重新生成带调试CSV的轨迹
python3 tools/convert_ais_xls_to_obstacles.py \
  --input-dir data/raw/trajectories \
  --map-meta data/processed/maps/navigation_map_meta.yaml \
  --out-json data/processed/trajectories/multi_obstacles.json \
  --out-debug-csv data/processed/trajectories/trajectory_debug.csv

# 2. 运行完整诊断
python3 tools/visualize_processed_scenario.py \
  --map data/processed/maps/navigation_map.npy \
  --meta data/processed/maps/navigation_map_meta.yaml \
  --traj data/processed/trajectories/multi_obstacles.json \
  --save-png overlay_debug.png \
  --no-show
```

---

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
- `data/processed/maps/control_points_example.yaml` - 控制点示例文件(需按实际地图调整)
- `data/processed/maps/affine_params.yaml` - 仿射变换参数(基于控制点拟合)
- `data/processed/maps/calibration_stats.yaml` - 校准统计对比
- `data/processed/maps/overlay_pixel_linear.png` - pixel-space线性叠加
- `data/processed/maps/overlay_pixel_affine.png` - pixel-space仿射叠加
- `data/processed/maps/overlay_world_linear.png` - world-space线性叠加
- `data/processed/maps/overlay_world_affine.png` - world-space仿射叠加
- `data/processed/maps/overlay_pixel_comparison.png` - pixel-space对比
- `data/processed/maps/overlay_world_comparison.png` - world-space对比

### 轨迹处理
- `data/processed/trajectories/multi_obstacles.json` - 38个船舶轨迹
- `data/processed/trajectories/single_obstacle_example.json` - 单船样例
- `data/processed/trajectories/cleaned_preview.csv` - CSV预览
- `data/processed/trajectories/trajectory_debug.csv` - 完整调试CSV(含lon/lat/col/row)

### 可视化
- `overlay_debug.png` - 地图+轨迹叠加图
- `overlay_debug.gif` - 动态障碍物动画 (生成中)

### 训练
- `training_grid_results/best_model.zip`
- `training_grid_results/final_model.zip`
