#!/usr/bin/env python
"""
可视化验证脚本

加载地图和轨迹数据，在地图上显示动态障碍物轨迹，用于验证数据处理链正确性

用法:
    python tools/visualize_processed_scenario.py \
        --map data/processed/maps/navigation_map.npy \
        --meta data/processed/maps/navigation_map_meta.yaml \
        --traj data/processed/trajectories/multi_obstacles.json \
        --save-png overlay_debug.png \
        --save-gif overlay_debug.gif
"""

import argparse
import os
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.animation as animation
from typing import Dict, List, Tuple, Optional


def load_map(map_path: str) -> np.ndarray:
    """加载.npy地图"""
    return np.load(map_path)


def load_meta(meta_path: str) -> dict:
    """加载元数据"""
    with open(meta_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_trajectories(traj_path: str) -> dict:
    """加载轨迹JSON"""
    with open(traj_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def world_to_pixel(world_x: float, world_y: float, meta: dict) -> Tuple[int, int]:
    """
    世界坐标转像素坐标（y-down约定）

    world_y 向下增加，对应图像row向下增加
    world_x 向右增加，对应图像col向右增加
    """
    resolution_x = meta.get('resolution_x', meta.get('resolution', 1.0))
    resolution_y = meta.get('resolution_y', meta.get('resolution', 1.0))
    col = int(world_x / resolution_x)
    row = int(world_y / resolution_y)
    return col, row


def pixel_to_world(pixel_x: int, pixel_y: int, meta: dict) -> Tuple[float, float]:
    """像素坐标转世界坐标"""
    resolution_x = meta.get('resolution_x', meta.get('resolution', 1.0))
    resolution_y = meta.get('resolution_y', meta.get('resolution', 1.0))
    return pixel_x * resolution_x, pixel_y * resolution_y


def visualize_static(map_data: np.ndarray, meta: dict, trajectories: dict,
                   obstacles_to_show: List[int] = None,
                   title: str = "Map with Trajectories") -> plt.Figure:
    """
    创建静态可视化：地图 + 所有轨迹

    坐标约定: world_y 向下增加 (y-down)
    - array row=0 对应 world_y=0 (顶部)
    - array row=height-1 对应 world_y=height*resolution (底部)

    Args:
        map_data: 占据栅格 (H, W)
        meta: 地图元数据
        trajectories: 轨迹JSON
        obstacles_to_show: 要显示的障碍物索引列表，None表示全部
        title: 图表标题
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    height, width = map_data.shape

    # 显示地图 - 使用origin='upper'使array顶部对应plot顶部(world_y=0)
    # 0=free显示为白色，1=obstacle显示为深灰色
    # 使用resolution_x和resolution_y分别计算extent
    resolution_x = meta.get('resolution_x', meta.get('resolution', 1.0))
    resolution_y = meta.get('resolution_y', meta.get('resolution', 1.0))
    cmap = plt.cm.colors.ListedColormap(['white', 'darkgray'])
    ax.imshow(map_data, cmap=cmap,
             extent=[0, width*resolution_x, height*resolution_y, 0],
             origin='upper', aspect='auto')

    # 获取障碍物列表
    obstacles = trajectories.get('obstacles', [])

    # 选择要显示的障碍物
    if obstacles_to_show is not None:
        obstacles = [obstacles[i] for i in obstacles_to_show if i < len(obstacles)]

    # 为每个障碍物绘制轨迹
    colors = plt.cm.tab20(np.linspace(0, 1, len(obstacles)))

    for i, obs in enumerate(obstacles):
        traj = obs.get('trajectory', [])
        if not traj:
            continue

        xs = [pt['x'] for pt in traj]
        ys = [pt['y'] for pt in traj]

        # 绘制轨迹线
        ax.plot(xs, ys, '-', color=colors[i], alpha=0.6, linewidth=1.5)

        # 绘制起点和终点
        ax.scatter(xs[0], ys[0], marker='o', s=80, c='green', zorder=5, edgecolors='black')
        ax.scatter(xs[-1], ys[-1], marker='s', s=80, c='red', zorder=5, edgecolors='black')

        # 标注障碍物ID
        mid_idx = len(xs) // 2
        ax.annotate(f"{obs['id']}", (xs[mid_idx], ys[mid_idx]),
                   fontsize=7, ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))

    # 图例
    legend_elements = [
        mpatches.Patch(color='white', label='Free'),
        mpatches.Patch(color='darkgray', label='Obstacle'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Start'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=8, label='End'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # 标签
    ax.set_xlabel('World X (m)')
    ax.set_ylabel('World Y (m)')
    ax.set_title(title)

    # 添加元数据信息
    res_x = meta.get('resolution_x', meta.get('resolution'))
    res_y = meta.get('resolution_y', meta.get('resolution'))
    info_text = (f"Map: {meta['width']}x{meta['height']} @ x={res_x:.1f}, y={res_y:.1f} m/pixel\n"
                f"Trajectories: {len(obstacles)} obstacles\n"
                f"Geographic: lon=[{meta['lon_min']:.4f}, {meta['lon_max']:.4f}], "
                f"lat=[{meta['lat_min']:.4f}, {meta['lat_max']:.4f}]")
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=8,
           verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    return fig


def visualize_static_pixel_space(map_data: np.ndarray, meta: dict, trajectories: dict,
                                  obstacles_to_show: List[int] = None,
                                  title: str = "Pixel-Space Overlay") -> plt.Figure:
    """
    在像素坐标系直接叠加轨迹（不做米制转换）

    背景: 原始occupancy raster，imshow用默认像素坐标
    轨迹: 直接用col,row叠加，不经过world坐标转换

    目的: 如果pixel-space对齐正确而world-space不对，说明问题在resolution/extent
         如果pixel-space也不对，说明问题在经纬度→像素映射

    Args:
        map_data: 占据栅格 (H, W)
        meta: 地图元数据
        trajectories: 轨迹JSON
        obstacles_to_show: 要显示的障碍物索引列表
        title: 图表标题
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    height, width = map_data.shape

    # 显示地图 - 直接用像素坐标，origin='upper'使array顶部对应plot顶部
    cmap = plt.cm.colors.ListedColormap(['white', 'darkgray'])
    ax.imshow(map_data, cmap=cmap, origin='upper')

    # 获取障碍物列表
    obstacles = trajectories.get('obstacles', [])

    # 选择要显示的障碍物
    if obstacles_to_show is not None:
        obstacles = [obstacles[i] for i in obstacles_to_show if i < len(obstacles)]

    # 为每个障碍物绘制轨迹（使用col,row像素坐标）
    colors = plt.cm.tab20(np.linspace(0, 1, len(obstacles)))

    for i, obs in enumerate(obstacles):
        traj = obs.get('trajectory', [])
        if not traj:
            continue

        # 检查轨迹点是否有col,row
        if 'col' in traj[0] and 'row' in traj[0]:
            cols = [pt['col'] for pt in traj]
            rows = [pt['row'] for pt in traj]
        else:
            # 如果没有col,row，用world坐标转换
            resolution_x = meta.get('resolution_x', meta.get('resolution', 1.0))
            resolution_y = meta.get('resolution_y', meta.get('resolution', 1.0))
            cols = [pt['x'] / resolution_x for pt in traj]
            rows = [pt['y'] / resolution_y for pt in traj]

        # 绘制轨迹线
        ax.plot(cols, rows, '-', color=colors[i], alpha=0.6, linewidth=1.5)

        # 绘制起点和终点
        ax.scatter(cols[0], rows[0], marker='o', s=80, c='green', zorder=5, edgecolors='black')
        ax.scatter(cols[-1], rows[-1], marker='s', s=80, c='red', zorder=5, edgecolors='black')

        # 标注障碍物ID
        mid_idx = len(cols) // 2
        ax.annotate(f"{obs['id']}", (cols[mid_idx], rows[mid_idx]),
                   fontsize=7, ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))

    # 图例
    legend_elements = [
        mpatches.Patch(color='white', label='Free'),
        mpatches.Patch(color='darkgray', label='Obstacle'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Start'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=8, label='End'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # 标签
    ax.set_xlabel('Pixel Column')
    ax.set_ylabel('Pixel Row')
    ax.set_title(title)

    # 设置坐标范围
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # origin='upper'时y轴向下

    # 添加元数据信息
    res_x = meta.get('resolution_x', meta.get('resolution'))
    res_y = meta.get('resolution_y', meta.get('resolution'))
    info_text = (f"PIXEL-SPACE OVERLAY (no world conversion)\n"
                f"Map: {width}x{height} pixels\n"
                f"Trajectories: {len(obstacles)} obstacles\n"
                f"If this looks correct but world-space doesn't => problem in extent/resolution\n"
                f"If this also looks wrong => problem in lon/lat -> col/row mapping")
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=8,
           verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    return fig


def visualize_animated(map_data: np.ndarray, meta: dict, trajectories: dict,
                      save_path: str = None, interval: int = 200) -> animation.FuncAnimation:
    """
    创建动画可视化：动态障碍物在地图上移动

    Args:
        map_data: 占据栅格
        meta: 地图元数据
        trajectories: 轨迹JSON
        save_path: GIF保存路径，None则不保存
        interval: 帧间隔（毫秒）
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    height, width = map_data.shape
    obstacles = trajectories.get('obstacles', [])

    # 显示地图 - origin='upper'使array顶部(world_y=0)在plot顶部
    resolution_x = meta.get('resolution_x', meta.get('resolution', 1.0))
    resolution_y = meta.get('resolution_y', meta.get('resolution', 1.0))
    cmap = plt.cm.colors.ListedColormap(['white', 'darkgray'])
    ax.imshow(map_data, cmap=cmap,
             extent=[0, width*resolution_x, height*resolution_y, 0],
             origin='upper', aspect='auto')

    # 计算最大时间
    max_t = 0
    for obs in obstacles:
        traj = obs.get('trajectory', [])
        if traj:
            max_t = max(max_t, traj[-1]['t'])

    # 预计算所有障碍物在各时间点的位置
    def get_position_at_time(obs, t):
        traj = obs.get('trajectory', [])
        if not traj:
            return None

        if t <= traj[0]['t']:
            return traj[0]['x'], traj[0]['y']
        if t >= traj[-1]['t']:
            return None  # 轨迹结束

        for i in range(len(traj) - 1):
            t0, t1 = traj[i]['t'], traj[i+1]['t']
            if t0 <= t <= t1:
                alpha = (t - t0) / (t1 - t0) if (t1 - t0) > 0 else 0
                x = traj[i]['x'] + alpha * (traj[i+1]['x'] - traj[i]['x'])
                y = traj[i]['y'] + alpha * (traj[i+1]['y'] - traj[i]['y'])
                return x, y
        return None

    # 为每个障碍物创建轨迹线和散点
    colors = plt.cm.tab20(np.linspace(0, 1, len(obstacles)))
    traj_lines = []
    pos_scatters = []

    for i, obs in enumerate(obstacles):
        traj = obs.get('trajectory', [])
        if not traj:
            continue

        xs = [pt['x'] for pt in traj]
        ys = [pt['y'] for pt in traj]

        # 轨迹线
        line, = ax.plot([], [], '-', color=colors[i], alpha=0.4, linewidth=1)
        traj_lines.append(line)

        # 当前位置散点
        scatter = ax.scatter([], [], marker='o', s=100, c=[colors[i]], edgecolors='black', zorder=5)
        pos_scatters.append(scatter)

        # 标注ID
        mid_idx = len(xs) // 2
        ax.annotate(f"{obs['id']}", (xs[mid_idx], ys[mid_idx]),
                   fontsize=6, ha='center', va='bottom', alpha=0.5)

    # 时间和状态文本
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 图例
    legend_elements = [
        mpatches.Patch(color='white', label='Free'),
        mpatches.Patch(color='darkgray', label='Obstacle'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_xlabel('World X (m)')
    ax.set_ylabel('World Y (m)')
    ax.set_title("Dynamic Obstacles Animation")

    def init():
        for line in traj_lines:
            line.set_data([], [])
        for scatter in pos_scatters:
            scatter.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        return traj_lines + pos_scatters + [time_text]

    def animate(frame):
        t = frame * 10  # 每帧代表10秒
        active_count = 0

        for i, obs in enumerate(obstacles):
            pos = get_position_at_time(obs, t)
            if pos is not None:
                x, y = pos
                # 更新轨迹线（显示历史）
                traj = obs.get('trajectory', [])
                hist_x = [pt['x'] for pt in traj if pt['t'] <= t]
                hist_y = [pt['y'] for pt in traj if pt['t'] <= t]
                traj_lines[i].set_data(hist_x, hist_y)

                # 更新当前位置
                pos_scatters[i].set_offsets([[x, y]])
                active_count += 1
            else:
                traj_lines[i].set_data([], [])
                pos_scatters[i].set_offsets(np.empty((0, 2)))

        time_text.set_text(f'Time: {t:.0f}s ({t/60:.1f}min) | Active: {active_count}/{len(obstacles)}')
        return traj_lines + pos_scatters + [time_text]

    # 计算帧数（每10秒一帧，覆盖整个轨迹时间）
    num_frames = int(max_t / 10) + 1

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=num_frames, interval=interval, blit=True)

    if save_path:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer='pillow', fps=5)
        print(f"Animation saved!")

    return anim


def count_obstacles_in_water(map_data: np.ndarray, meta: dict, trajectories: dict,
                             radius: float = 10.0) -> Dict:
    """
    统计障碍物在水上/陆地上的情况

    坐标约定: world_y 向下增加 (y-down)
    array row=0 对应 world_y=0 (顶部)

    Args:
        map_data: 占据栅格
        meta: 地图元数据
        trajectories: 轨迹JSON
        radius: 障碍物半径（米）

    Returns:
        统计字典
    """
    resolution_x = meta.get('resolution_x', meta.get('resolution', 1.0))
    resolution_y = meta.get('resolution_y', meta.get('resolution', 1.0))

    stats = {
        'total_points': 0,
        'in_water': 0,
        'on_land': 0,
        'outside_map': 0,
    }

    obstacles = trajectories.get('obstacles', [])

    for obs in obstacles:
        traj = obs.get('trajectory', [])
        for pt in traj:
            stats['total_points'] += 1
            x, y = pt['x'], pt['y']

            # 转像素坐标 (y-down: world_y -> row)
            col = int(x / resolution_x)
            row = int(y / resolution_y)

            # 检查是否在地图范围内
            if not (0 <= col < meta['width'] and 0 <= row < meta['height']):
                stats['outside_map'] += 1
                continue

            # 检查是否是水域（自由区域）
            # 不再flip，因为display和world坐标一致
            if map_data[row, col] == 0:  # 0 = free
                stats['in_water'] += 1
            else:
                stats['on_land'] += 1

    return stats


def compute_distance_to_nearest_free(map_data: np.ndarray, meta: dict,
                                     trajectories: dict) -> Dict:
    """
    计算每个落在障碍物区域的轨迹点到最近自由像素的距离（单位：像素）

    Bug修复: 使用brute force计算代替有问题的EDT方法

    EDT的正确用法:
    - distance_transform_edt(input) 计算到最近值为0的像素的距离
    - 值为0的像素 = source (自身距离=0)
    - 值为>0的像素 = 需计算距离

    错误用法:
    - free_mask = (map_data == 0).astype(float) → free=1.0, obstacle=0.0
    - EDT(0.0) = 0 (source), EDT(1.0) = positive
    - 所以obstacle(False=0)被当作source，距离=0 ← BUG!

    正确用法:
    - 我们想计算"obstacle到最近free的距离"
    - 也就是说: free是目的地，obstacle是起点
    - EDT(free_mask) where free=0, obstacle>0 → obstacle到free的距离
    - 即: input = (map_data != 0).astype(float) → obstacle=1.0, free=0.0
    - 但EDT(1.0)=positive (到最近0的距离), EDT(0.0)=0
    - 所以EDT(obstacle_mask)[obstacle] = positive (到最近0的距离)
    - 而obstacle_mask中0的位置是free!
    - 所以EDT(obstacle_mask)[obstacle] = distance to nearest free ← 正确!

    但实际测试表明这还是有问题，所以用brute force方法直接验证。

    Returns:
        统计字典
    """
    resolution_x = meta.get('resolution_x', meta.get('resolution', 1.0))
    resolution_y = meta.get('resolution_y', meta.get('resolution', 1.0))

    stats = {
        'total_on_land': 0,
        'total_on_land_with_dist': 0,
        'avg_dist_pixels': 0.0,
        'median_dist_pixels': 0.0,
        'min_dist_pixels': float('inf'),
        'max_dist_pixels': 0.0,
        'dist_1px_count': 0,
        'dist_2px_count': 0,
        'dist_3px_count': 0,
        'dist_5px_count': 0,
        'dist_10px_count': 0,
    }

    # 获取所有free像素的坐标
    free_rows, free_cols = np.where(map_data == 0)
    if len(free_rows) == 0:
        return stats

    distances = []
    obstacles = trajectories.get('obstacles', [])

    for obs in obstacles:
        obs_id = obs['id']
        traj = obs.get('trajectory', [])
        for pt in traj:
            x, y = pt['x'], pt['y']
            col = int(x / resolution_x)
            row = int(y / resolution_y)

            # 检查是否在地图范围内
            if not (0 <= col < meta['width'] and 0 <= row < meta['height']):
                continue

            # 只分析在陆地（障碍区）的点
            if map_data[row, col] != 0:
                stats['total_on_land'] += 1

                # Brute force: 计算到最近free像素的欧几里得距离
                # 使用vectorized操作加速
                d_row = free_rows - row
                d_col = free_cols - col
                dists = np.sqrt(d_row**2 + d_col**2)
                dist_px = float(np.min(dists))

                distances.append(dist_px)

                if dist_px <= 1:
                    stats['dist_1px_count'] += 1
                if dist_px <= 2:
                    stats['dist_2px_count'] += 1
                if dist_px <= 3:
                    stats['dist_3px_count'] += 1
                if dist_px <= 5:
                    stats['dist_5px_count'] += 1
                if dist_px <= 10:
                    stats['dist_10px_count'] += 1

                stats['min_dist_pixels'] = min(stats['min_dist_pixels'], dist_px)
                stats['max_dist_pixels'] = max(stats['max_dist_pixels'], dist_px)

    if distances:
        distances = np.array(distances)
        stats['total_on_land_with_dist'] = len(distances)
        stats['avg_dist_pixels'] = float(np.mean(distances))
        stats['median_dist_pixels'] = float(np.median(distances))

    return stats


def validate_dilation(map_data: np.ndarray, meta: dict, trajectories: dict,
                      dilations: List[int] = [1, 2, 3]) -> Dict:
    """
    验证自由区域膨胀后的轨迹点覆盖率

    对自由区域做膨胀(dilation)，然后检查轨迹点在水域的比例变化

    Args:
        map_data: 占据栅格 (H, W), 0=free, 1=obstacle
        meta: 地图元数据
        trajectories: 轨迹JSON
        dilations: 膨胀像素数列表

    Returns:
        每种膨胀级别下的统计字典
    """
    from scipy.ndimage import binary_dilation, generate_binary_structure

    resolution_x = meta.get('resolution_x', meta.get('resolution', 1.0))
    resolution_y = meta.get('resolution_y', meta.get('resolution', 1.0))

    results = {}
    obstacles = trajectories.get('obstacles', [])

    # 统计所有轨迹点
    all_points = []
    for obs in obstacles:
        obs_id = obs['id']
        traj = obs.get('trajectory', [])
        for pt in traj:
            x, y = pt['x'], pt['y']
            col = int(x / resolution_x)
            row = int(y / resolution_y)

            # 检查是否在地图范围内
            if not (0 <= col < meta['width'] and 0 <= row < meta['height']):
                in_bounds = False
            else:
                in_bounds = True

            all_points.append({
                'obs_id': obs_id,
                't': pt['t'],
                'col': col,
                'row': row,
                'in_bounds': in_bounds,
                'on_free': map_data[row, col] == 0 if in_bounds else None,
            })

    total = len(all_points)
    in_bounds_count = sum(1 for p in all_points if p['in_bounds'])

    for dilate_px in dilations:
        # 创建膨胀结构元素（圆形）
        struct = generate_binary_structure(2, 2)  # 近似圆形
        # 自定义膨胀半径
        if dilate_px > 1:
            # 多次膨胀
            dilated_free = map_data == 0
            for _ in range(dilate_px):
                dilated_free = binary_dilation(dilated_free, structure=struct).astype(np.uint8)
        else:
            # 一次性膨胀
            dilated_free = binary_dilation(map_data == 0, structure=struct).astype(np.uint8)

        # 统计膨胀后在自由区的点
        in_water_dilated = 0
        for p in all_points:
            if not p['in_bounds']:
                continue
            if dilated_free[p['row'], p['col']] == 1:
                in_water_dilated += 1

        on_land_original = sum(1 for p in all_points if p['in_bounds'] and not p['on_free'])

        results[dilate_px] = {
            'total_points': total,
            'in_bounds': in_bounds_count,
            'original_on_land': on_land_original,
            'original_on_water': in_bounds_count - on_land_original,
            'after_dilation_on_water': in_water_dilated,
            'water_coverage_original': in_bounds_count - on_land_original,
            'water_coverage_dilated': in_water_dilated,
            'newly_freed': in_water_dilated - (in_bounds_count - on_land_original),
            'pct_on_water_original': 100 * (in_bounds_count - on_land_original) / in_bounds_count if in_bounds_count > 0 else 0,
            'pct_on_water_dilated': 100 * in_water_dilated / in_bounds_count if in_bounds_count > 0 else 0,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description='Visualize processed map and trajectories')
    parser.add_argument('--map', type=str, required=True, help='Path to .npy map file')
    parser.add_argument('--meta', type=str, required=True, help='Path to map metadata yaml')
    parser.add_argument('--traj', type=str, required=True, help='Path to trajectory JSON')
    parser.add_argument('--save-png', type=str, default=None, help='Save world-space static visualization as PNG')
    parser.add_argument('--save-pixel-png', type=str, default=None, help='Save pixel-space static visualization as PNG')
    parser.add_argument('--save-gif', type=str, default=None, help='Save animation as GIF')
    parser.add_argument('--no-show', action='store_true', help='Do not display plots')
    parser.add_argument('--interval', type=int, default=200, help='Animation frame interval in ms')

    args = parser.parse_args()

    print("=" * 60)
    print("可视化验证")
    print("=" * 60)

    # 加载数据
    print(f"\n[1] 加载数据...")
    map_data = load_map(args.map)
    meta = load_meta(args.meta)
    trajectories = load_trajectories(args.traj)

    print(f"  地图: {map_data.shape} ({meta['width']}x{meta['height']})")
    res_x = meta.get('resolution_x', meta.get('resolution'))
    res_y = meta.get('resolution_y', meta.get('resolution'))
    res_avg = meta.get('resolution_avg', meta.get('resolution'))
    print(f"  分辨率: x={res_x:.2f}, y={res_y:.2f} m/pixel (avg={res_avg:.2f})")
    print(f"  地理范围: lon=[{meta['lon_min']:.4f}, {meta['lon_max']:.4f}], "
          f"lat=[{meta['lat_min']:.4f}, {meta['lat_max']:.4f}]")
    print(f"  障碍物数量: {len(trajectories.get('obstacles', []))}")

    # 统计障碍物在水/陆地情况
    print(f"\n[2] 统计障碍物位置...")
    water_stats = count_obstacles_in_water(map_data, meta, trajectories, radius=10.0)
    total = water_stats['total_points']
    if total > 0:
        print(f"  总轨迹点数: {total}")
        print(f"  在水域(自由区): {water_stats['in_water']} ({100*water_stats['in_water']/total:.1f}%)")
        print(f"  在陆地(障碍区): {water_stats['on_land']} ({100*water_stats['on_land']/total:.1f}%)")
        print(f"  地图范围外: {water_stats['outside_map']} ({100*water_stats['outside_map']/total:.1f}%)")

    # 距离最近自由区域诊断
    print(f"\n[2.5] 距离最近自由区域诊断...")
    dist_stats = compute_distance_to_nearest_free(map_data, meta, trajectories)
    if dist_stats['total_on_land'] > 0:
        print(f"  在陆地(障碍区)的点数: {dist_stats['total_on_land']}")
        print(f"  平均距离: {dist_stats['avg_dist_pixels']:.2f} 像素 ({dist_stats['avg_dist_pixels']*res_x:.1f} 米)")
        print(f"  中位距离: {dist_stats['median_dist_pixels']:.2f} 像素 ({dist_stats['median_dist_pixels']*res_x:.1f} 米)")
        print(f"  最小距离: {dist_stats['min_dist_pixels']:.2f} 像素 ({dist_stats['min_dist_pixels']*res_x:.1f} 米)")
        print(f"  最大距离: {dist_stats['max_dist_pixels']:.2f} 像素 ({dist_stats['max_dist_pixels']*res_x:.1f} 米)")
        print(f"  距离<=1px: {dist_stats['dist_1px_count']} ({100*dist_stats['dist_1px_count']/dist_stats['total_on_land']:.1f}%)")
        print(f"  距离<=2px: {dist_stats['dist_2px_count']} ({100*dist_stats['dist_2px_count']/dist_stats['total_on_land']:.1f}%)")
        print(f"  距离<=3px: {dist_stats['dist_3px_count']} ({100*dist_stats['dist_3px_count']/dist_stats['total_on_land']:.1f}%)")
        print(f"  距离<=5px: {dist_stats['dist_5px_count']} ({100*dist_stats['dist_5px_count']/dist_stats['total_on_land']:.1f}%)")
        print(f"  距离<=10px: {dist_stats['dist_10px_count']} ({100*dist_stats['dist_10px_count']/dist_stats['total_on_land']:.1f}%)")
    else:
        print(f"  没有在陆地(障碍区)的点")

    # 自由区域膨胀验证
    print(f"\n[2.6] 自由区域膨胀验证...")
    dilation_results = validate_dilation(map_data, meta, trajectories, dilations=[1, 2, 3, 5, 10])
    for dilate_px, res in dilation_results.items():
        print(f"  {dilate_px}px膨胀: 原水域={res['pct_on_water_original']:.1f}% -> 膨胀后={res['pct_on_water_dilated']:.1f}% "
              f"(+{res['newly_freed']}点, {res['newly_freed']/max(1,res['in_bounds'])*100:.1f}%)")

    # 静态可视化 - world-space
    print(f"\n[3] 生成world-space静态可视化...")
    fig_world = visualize_static(map_data, meta, trajectories,
                          title="Map with Dynamic Obstacles (World-Space)")
    if args.save_png:
        fig_world.savefig(args.save_png, dpi=150, bbox_inches='tight')
        print(f"  已保存world-space PNG: {args.save_png}")

    # 静态可视化 - pixel-space
    if args.save_pixel_png:
        print(f"\n[3.5] 生成pixel-space静态可视化...")
        fig_pixel = visualize_static_pixel_space(map_data, meta, trajectories,
                                      title="Pixel-Space Overlay (col=row)")
        fig_pixel.savefig(args.save_pixel_png, dpi=150, bbox_inches='tight')
        print(f"  已保存pixel-space PNG: {args.save_pixel_png}")

    # 动画可视化
    if args.save_gif:
        print(f"\n[5] 生成动画可视化...")
        anim = visualize_animated(map_data, meta, trajectories,
                                save_path=args.save_gif, interval=args.interval)

    # 显示
    if not args.no_show:
        print("\n[6] 显示图像...")
        plt.show()

    print("\n" + "=" * 60)
    print("可视化验证完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()