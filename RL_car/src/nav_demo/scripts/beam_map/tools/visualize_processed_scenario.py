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
    """世界坐标转像素坐标"""
    resolution = meta['resolution']
    col = int(world_x / resolution)
    row = int(world_y / resolution)
    # 注意: 图像y轴向下，所以row方向需要翻转
    # 实际上世界坐标的y对应图像的row，但图像row向下增加
    # 在我们的坐标系中，world_y=0 对应图像顶部
    return col, row


def pixel_to_world(pixel_x: int, pixel_y: int, meta: dict) -> Tuple[float, float]:
    """像素坐标转世界坐标"""
    resolution = meta['resolution']
    return pixel_x * resolution, pixel_y * resolution


def visualize_static(map_data: np.ndarray, meta: dict, trajectories: dict,
                   obstacles_to_show: List[int] = None,
                   title: str = "Map with Trajectories") -> plt.Figure:
    """
    创建静态可视化：地图 + 所有轨迹

    Args:
        map_data: 占据栅格 (H, W)
        meta: 地图元数据
        trajectories: 轨迹JSON
        obstacles_to_show: 要显示的障碍物索引列表，None表示全部
        title: 图表标题
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    height, width = map_data.shape

    # 显示地图（反转使y轴向上）
    # 0=free显示为浅色，1=obstacle显示为深色
    display_map = np.flipud(map_data)  # 翻转使y轴向上
    cmap = plt.cm.colors.ListedColormap(['white', 'darkgray'])
    ax.imshow(display_map, cmap=cmap, extent=[0, width*meta['resolution'], 0, height*meta['resolution']],
             origin='lower', aspect='auto')

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
    info_text = (f"Map: {meta['width']}x{meta['height']} @ {meta['resolution']:.1f}m/pixel\n"
                f"Trajectories: {len(obstacles)} obstacles\n"
                f"Geographic: lon=[{meta['lon_min']:.4f}, {meta['lon_max']:.4f}], "
                f"lat=[{meta['lat_min']:.4f}, {meta['lat_max']:.4f}]")
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=8,
           verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

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

    # 显示地图
    display_map = np.flipud(map_data)
    cmap = plt.cm.colors.ListedColormap(['white', 'darkgray'])
    ax.imshow(display_map, cmap=cmap, extent=[0, width*meta['resolution'], 0, height*meta['resolution']],
             origin='lower', aspect='auto')

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

    Args:
        map_data: 占据栅格
        meta: 地图元数据
        trajectories: 轨迹JSON
        radius: 障碍物半径（米）

    Returns:
        统计字典
    """
    resolution = meta['resolution']
    radius_in_pixels = int(radius / resolution) + 1

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

            # 转像素坐标
            col = int(x / resolution)
            row = int(y / resolution)

            # 检查是否在地图范围内
            if not (0 <= col < meta['width'] and 0 <= row < meta['height']):
                stats['outside_map'] += 1
                continue

            # 检查是否是水域（自由区域）
            # 注意图像已经翻转，所以row需要映射回来
            map_row = meta['height'] - 1 - row
            if map_data[map_row, col] == 0:  # 0 = free
                stats['in_water'] += 1
            else:
                stats['on_land'] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description='Visualize processed map and trajectories')
    parser.add_argument('--map', type=str, required=True, help='Path to .npy map file')
    parser.add_argument('--meta', type=str, required=True, help='Path to map metadata yaml')
    parser.add_argument('--traj', type=str, required=True, help='Path to trajectory JSON')
    parser.add_argument('--save-png', type=str, default=None, help='Save static visualization as PNG')
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
    print(f"  分辨率: {meta['resolution']:.2f} m/pixel")
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

    # 静态可视化
    print(f"\n[3] 生成静态可视化...")
    fig = visualize_static(map_data, meta, trajectories,
                          title="Map with Dynamic Obstacles (Static)")
    if args.save_png:
        fig.savefig(args.save_png, dpi=150, bbox_inches='tight')
        print(f"  已保存PNG: {args.save_png}")

    # 动画可视化
    if args.save_gif:
        print(f"\n[4] 生成动画可视化...")
        anim = visualize_animated(map_data, meta, trajectories,
                                save_path=args.save_gif, interval=args.interval)

    # 显示
    if not args.no_show:
        print("\n[5] 显示图像...")
        plt.show()

    print("\n" + "=" * 60)
    print("可视化验证完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()