# Frenet坐标系奖励函数功能文档

## 概述

本文档描述了基于Frenet坐标系的导航奖励函数实现。该功能为小车导航任务添加了全局直线路径，并使用Frenet坐标系设计奖励函数，引导机器人沿着全局路径前进。

## 分支信息

- **分支名称**: `frenet_reward`
- **基础分支**: `avoid_obstables_improvement`

## 文件结构

```
RL_car/src/nav_demo/scripts/beam_map/
├── frenet_utils.py      # Frenet坐标系工具函数（新增）
└── ros_env.py           # 修改后的环境文件（包含Frenet奖励和路径可视化）
```

## 核心功能

### 1. Frenet坐标系转换 ([frenet_utils.py](frenet_utils.py))

#### `FrenetTransform` 类

Frenet坐标系将2D平面的点表示为两个分量：
- **s (弧长)**: 点沿路径的纵向距离
- **d (横向偏移)**: 点垂直于路径的横向距离

**主要方法**:

| 方法 | 说明 |
|------|------|
| `cartesian_to_frenet(point)` | 将笛卡尔坐标 [x,y] 转换为 Frenet坐标 [s,d] |
| `frenet_to_cartesian(s, d)` | 将 Frenet坐标 [s,d] 转换为笛卡尔坐标 [x,y] |
| `get_closest_point_on_path(point)` | 获取点在路径上的最近点 |
| `get_heading_error(robot_yaw)` | 计算机器人朝向与路径方向的角度误差 |
| `generate_path_points(num_points)` | 生成路径上的采样点（用于可视化） |

#### `frenet_reward()` 函数

基于Frenet坐标的奖励函数，包含以下分量：

| 奖励分量 | 说明 | 公式 |
|----------|------|------|
| `s_progress` | 纵向进度奖励 | `s / path_length * 10.0` |
| `lateral_deviation` | 横向偏离惩罚 | `-2.0 * \|d\|` |
| `heading_alignment` | 朝向对齐奖励 | `1.0 - \|heading_error\| / π` |
| `velocity` | 速度奖励 | 根据朝向给予正向或负向奖励 |
| `center_attract` | 路径中心吸引 | 当距离路径中心 < 1米时额外奖励 |

### 2. 环境修改 ([ros_env.py](ros_env.py))

#### 观测空间变化

**修改前**:
- 1000个雷达数据 + [归一化距离, 归一化朝向] = 1002维

**修改后**:
- 1000个雷达数据 + [s_norm, d_norm, heading_norm, remaining_norm] = 1004维

| 维度 | 名称 | 说明 | 范围 |
|------|------|------|------|
| 0-999 | 雷达数据 | 激光雷达扫描数据 | [0, 1] |
| 1000 | s_norm | 纵向进度（归一化） | [-1, 1] |
| 1001 | d_norm | 横向偏离（归一化） | [-1, 1] |
| 1002 | heading_norm | 朝向误差（归一化） | [-1, 1] |
| 1003 | remaining_norm | 剩余距离（归一化） | [-1, 1] |

#### 路径可视化

在RViz中可视化全局路径：
- **绿色线条**: 全局直线路径
- **蓝色箭头**: 起点位置和方向
- **红色箭头**: 终点位置和方向

发布话题: `/global_path_marker` (类型: `visualization_msgs/MarkerArray`)

## 使用方法

### 1. 环境初始化

环境会在每次 `reset()` 时自动：
1. 随机生成起点和终点
2. 初始化Frenet坐标系（直线从起点到终点）
3. 在RViz中可视化路径
4. 发布路径标记

### 2. 训练

使用修改后的环境进行训练，无需额外修改训练代码：

```bash
cd RL_car/src/nav_demo/scripts/beam_map
python train.py
```

### 3. 测试

使用加载的模型进行测试：

```bash
cd RL_car/src/nav_demo/scripts/beam_map
python test.py
```

## RViz配置

为了正确可视化路径，需要在RViz中添加：

1. **MarkerArray** 显示器
   - Topic: `/global_path_marker`
   - 将此话题添加到RViz的Displays面板

## 奖励函数设计原理

### 设计思路

1. **纵向奖励 (s_progress)**
   - 鼓励机器人沿路径前进
   - s值越大，说明越接近终点，奖励越高

2. **横向惩罚 (lateral_deviation)**
   - 惩罚偏离路径的行为
   - 使用线性惩罚，d的绝对值越大惩罚越强

3. **朝向奖励 (heading_alignment)**
   - 鼓励机器人朝向路径方向前进
   - 朝向误差越小，奖励越高

4. **速度奖励 (velocity)**
   - 朝向正确时鼓励前进
   - 朝向错误时惩罚以防止盲目前进

5. **中心吸引 (center_attract)**
   - 在路径中心附近给予额外奖励
   - 增强机器人沿路径中心行驶的倾向

## 代码位置索引

### 关键函数位置

| 文件 | 函数/类 | 行号 |
|------|---------|------|
| frenet_utils.py | FrenetTransform.__init__ | 16-35 |
| frenet_utils.py | cartesian_to_frenet | 37-48 |
| frenet_utils.py | frenet_reward | 98-135 |
| ros_env.py | __init__ (Frenet相关) | 40-51 |
| ros_env.py | step (Frenet观测和奖励) | 89-200 |
| ros_env.py | reset (Frenet初始化) | 202-280 |
| ros_env.py | _visualize_path | 354-408 |
| ros_env.py | _build_obs | 492-514 |

## 与原版本的对比

| 特性 | 原版本 | Frenet版本 |
|------|--------|------------|
| 观测维度 | 1002 | 1004 |
| 距离表示 | 欧几里得距离 | Frenet坐标 |
| 朝向计算 | 到目标的直接角度 | 到路径方向的角度 |
| 路径可视化 | 仅起点终点标记 | 完整路径+方向箭头 |
| 奖励函数 | 基于距离进展 | 基于Frenet多维奖励 |

## 注意事项

1. **观测空间变化**: 观测空间从1002维变为1004维，训练新模型时注意兼容性
2. **路径可视化**: 确保RViz已添加MarkerArray显示器才能看到路径
3. **d坐标归一化**: 横向偏离d_norm按照3米范围归一化，可根据实际需要调整
4. **奖励权重**: 当前Frenet奖励权重为0.5，可在 `step()` 函数中调整

## 扩展建议

1. **曲线路径支持**: 修改 `FrenetTransform` 类以支持曲线（如Bezier曲线或样条曲线）
2. **动态障碍物**: 在Frenet坐标系中考虑动态障碍物对路径的影响
3. **速度规划**: 在Frenet坐标系中添加速度约束奖励
4. **多路径选择**: 支持多条备选路径和路径切换

## 参考资料

- Frenet坐标系: https://en.wikipedia.org/wiki/Frenet%E2%80%93Serret_formulas
- 路径跟踪算法: Stanley Controller, Pure Pursuit

---

**文档版本**: 1.0
**创建日期**: 2026-02-12
**分支**: frenet_reward
