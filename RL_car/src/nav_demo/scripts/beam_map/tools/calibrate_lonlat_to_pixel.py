#!/usr/bin/env python3
"""
经纬度到像素坐标的配准校准脚本

目标：从默认线性映射出发，拟合更合理的 lon/lat -> pixel 变换，
     使 AIS 轨迹点尽可能落在 free (水域) 区域。

支持两种模式：
1. 自动优化模式 (auto): 最大化 free 区域覆盖率
2. 控制点模式 (control_points): 从 yaml/json 读取手工控制点，拟合 affine transform

用法:
    # 自动优化
    python tools/calibrate_lonlat_to_pixel.py \
        --map data/processed/maps/navigation_map.npy \
        --meta data/processed/maps/navigation_map_meta.yaml \
        --traj data/processed/trajectories/multi_obstacles.json \
        --mode auto \
        --out-params params_calibrated.yaml

    # 控制点模式
    python tools/calibrate_lonlat_to_pixel.py \
        --map data/processed/maps/navigation_map.npy \
        --meta data/processed/maps/navigation_map_meta.yaml \
        --traj data/processed/trajectories/multi_obstacles.json \
        --mode control_points \
        --control-points my_control_points.yaml \
        --out-params params_calibrated.yaml
"""

import argparse
import os
import sys
import json
import yaml
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
from scipy.ndimage import distance_transform_edt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def load_map(map_path: str) -> np.ndarray:
    return np.load(map_path)


def load_meta(meta_path: str) -> dict:
    with open(meta_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_trajectories(traj_path: str) -> dict:
    with open(traj_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def lonlat_to_pixel_with_params(lon: float, lat: float, meta: dict,
                                scale_x: float = 1.0,
                                scale_y: float = 1.0,
                                offset_x: float = 0.0,
                                offset_y: float = 0.0,
                                flip_y: bool = False,
                                rotation_deg: float = 0.0) -> Tuple[int, int]:
    """
    使用校准参数将经纬度转换为像素坐标

    默认映射 (scale=1, offset=0):
        col = (lon - lon_min) / lon_range * width
        row = (lat_max - lat) / lat_range * height

    校准后:
        col = scale_x * (lon - lon_min) / lon_range * width + offset_x
        row = scale_y * (lat_max - lat) / lat_range * height + offset_y

    可选 flip_y: 翻转 y 方向
    可选 rotation_deg: 绕图像中心的小角度旋转
    """
    lon_min = meta['lon_min']
    lon_max = meta['lon_max']
    lat_min = meta['lat_min']
    lat_max = meta['lat_max']
    width = meta['width']
    height = meta['height']

    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min

    # 默认映射 (0-1 范围)
    col_default = (lon - lon_min) / lon_range if lon_range > 0 else 0.0
    row_default = (lat_max - lat) / lat_range if lat_range > 0 else 0.0

    # 应用 scale
    col = col_default * scale_x * width
    row = row_default * scale_y * height

    # 应用 offset
    col = col + offset_x
    row = row + offset_y

    # 可选: flip y
    if flip_y:
        row = height - row

    # 可选: rotation (simplified, around image center)
    if rotation_deg != 0.0:
        angle = np.radians(rotation_deg)
        cx, cy = width / 2.0, height / 2.0
        # Rotate around center
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dx = col - cx
        dy = row - cy
        col_rot = cos_a * dx - sin_a * dy + cx
        row_rot = sin_a * dx + cos_a * dy + cy
        col = col_rot
        row = row_rot

    # 裁剪到有效范围
    col = max(0, min(width - 1, col))
    row = max(0, min(height - 1, row))

    return int(col), int(row)


def compute_stats(map_data: np.ndarray, trajectories: dict,
                  transform_func) -> Dict:
    """
    计算给定转换函数下轨迹点的统计信息

    Returns:
        dict with keys: free_count, obstacle_count, outside_count,
                        total, free_pct, obstacle_pct, outside_pct,
                        avg_dist_to_free, median_dist_to_free
    """
    width = map_data.shape[1]
    height = map_data.shape[0]

    # Precompute free pixel coordinates for distance calculation
    free_rows, free_cols = np.where(map_data == 0)

    free_count = 0
    obstacle_count = 0
    outside_count = 0
    distances = []

    for obs in trajectories.get('obstacles', []):
        for pt in obs.get('trajectory', []):
            lon = pt.get('lon')
            lat = pt.get('lat')
            if lon is None or lat is None:
                continue

            col, row = transform_func(lon, lat)

            # Check bounds
            if not (0 <= col < width and 0 <= row < height):
                outside_count += 1
                continue

            # Check if free or obstacle
            if map_data[row, col] == 0:
                free_count += 1
                distances.append(0.0)  # On free pixel, distance = 0
            else:
                obstacle_count += 1
                # Compute distance to nearest free pixel
                d_row = free_rows - row
                d_col = free_cols - col
                dist = float(np.min(np.sqrt(d_row**2 + d_col**2)))
                distances.append(dist)

    total = free_count + obstacle_count + outside_count
    distances = np.array(distances) if distances else np.array([0.0])

    return {
        'free_count': free_count,
        'obstacle_count': obstacle_count,
        'outside_count': outside_count,
        'total': total,
        'free_pct': 100 * free_count / total if total > 0 else 0.0,
        'obstacle_pct': 100 * obstacle_count / total if total > 0 else 0.0,
        'outside_pct': 100 * outside_count / total if total > 0 else 0.0,
        'avg_dist_to_free': float(np.mean(distances)) if len(distances) > 0 else 0.0,
        'median_dist_to_free': float(np.median(distances)) if len(distances) > 0 else 0.0,
        'edge_2px_count': sum(1 for _ in range(total) if False),  # Placeholder
    }


def compute_boundary_proximity(map_data: np.ndarray, trajectories: dict,
                               transform_func, thresholds=[2, 3, 5]) -> Dict:
    """
    计算轨迹点贴近边界的比例
    """
    width = map_data.shape[1]
    height = map_data.shape[0]

    edge_stats = {t: 0 for t in thresholds}
    total = 0

    for obs in trajectories.get('obstacles', []):
        for pt in obs.get('trajectory', []):
            lon = pt.get('lon')
            lat = pt.get('lat')
            if lon is None or lat is None:
                continue

            col, row = transform_func(lon, lat)
            total += 1

            for t in thresholds:
                if col < t or col > width - 1 - t or row < t or row > height - 1 - t:
                    edge_stats[t] += 1

    return {t: (edge_stats[t], 100 * edge_stats[t] / total if total > 0 else 0.0)
            for t in thresholds}


def objective_auto(params, meta, trajectories, map_data):
    """
    自动优化的目标函数: 最小化 (1 - free_pct) + alpha * avg_dist

    We want to maximize free_pct and minimize avg_dist_to_free
    """
    scale_x, scale_y, offset_x, offset_y = params

    def transform_func(lon, lat):
        return lonlat_to_pixel_with_params(lon, lat, meta,
                                          scale_x=scale_x, scale_y=scale_y,
                                          offset_x=offset_x, offset_y=offset_y)

    stats = compute_stats(map_data, trajectories, transform_func)

    # Objective: minimize (1 - free_pct) + 0.1 * avg_dist
    # (1 - free_pct/100) gives 0 when free_pct=100, 1 when free_pct=0
    alpha = 0.05  # Weight for distance term
    obj = (1 - stats['free_pct'] / 100) + alpha * stats['avg_dist_to_free']

    return obj


def calibrate_auto(meta, trajectories, map_data, initial_params=None):
    """
    自动优化模式: 使用 scipy.optimize.minimize 寻找最佳参数
    """
    if initial_params is None:
        # Default: scale_x=1, scale_y=1, offset_x=0, offset_y=0
        initial_params = [1.0, 1.0, 0.0, 0.0]

    # Bounds: scale in [0.5, 2.0], offsets in [-50, 50] pixels
    bounds = [(0.5, 2.0), (0.5, 2.0), (-50, 50), (-50, 50)]

    print("  Running automatic optimization...")
    result = minimize(
        objective_auto,
        initial_params,
        args=(meta, trajectories, map_data),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 100, 'disp': False}
    )

    scale_x, scale_y, offset_x, offset_y = result.x

    return {
        'scale_x': float(scale_x),
        'scale_y': float(scale_y),
        'offset_x': float(offset_x),
        'offset_y': float(offset_y),
        'flip_y': False,
        'rotation_deg': 0.0,
        'optimization_success': result.success,
        'optimization_message': result.message,
        'initial_objective': float(objective_auto(initial_params, meta, trajectories, map_data)),
        'final_objective': float(result.fun),
    }


def calibrate_with_control_points(meta, map_data, control_points):
    """
    控制点模式: 根据手工控制点拟合 affine transform

    control_points: list of dicts with 'lon', 'lat', 'col', 'row'
    """
    import numpy as np

    n = len(control_points)
    if n < 3:
        print(f"  Warning: only {n} control points, need at least 3")
        return None

    # Build matrices for least squares: col = a*lon + b*lat + c
    #                                row = d*lon + e*lat + f
    A = np.zeros((n, 3))
    b_col = np.zeros(n)
    b_row = np.zeros(n)

    lon_min = meta['lon_min']
    lon_max = meta['lon_max']
    lat_min = meta['lat_min']
    lat_max = meta['lat_max']

    for i, cp in enumerate(control_points):
        lon = cp['lon']
        lat = cp['lat']
        col = cp['col']
        row = cp['row']

        # Normalize lon/lat to 0-1 range
        lon_norm = (lon - lon_min) / (lon_max - lon_min) if lon_max > lon_min else 0.0
        lat_norm = (lat - lat_min) / (lat_max - lat_min) if lat_max > lat_min else 0.0

        A[i] = [lon_norm, lat_norm, 1]
        b_col[i] = col
        b_row[i] = row

    # Solve least squares
    # col = A * [a, b, c]^T
    # row = A * [d, e, f]^T
    x_col, residuals_col, rank_col, s_col = np.linalg.lstsq(A, b_col, rcond=None)
    x_row, residuals_row, rank_row, s_row = np.linalg.lstsq(A, b_row, rcond=None)

    # Extract parameters
    # col = a*lon_norm + b*lat_norm + c
    # row = d*lon_norm + e*lat_norm + f
    a, b, c = x_col
    d, e, f = x_row

    # Convert to scale and offset in pixel space
    width = meta['width']
    height = meta['height']

    # For normalized lon/lat (0-1):
    # col = a * lon_norm * width + c  => scale_x = a, offset_x = c
    # row = d * lat_norm * height + f => scale_y = d, offset_y = f

    scale_x = a
    scale_y = d
    offset_x = c
    offset_y = f

    # Compute residuals for validation
    col_pred = A @ x_col
    row_pred = A @ x_row
    col_rmse = np.sqrt(np.mean((b_col - col_pred)**2))
    row_rmse = np.sqrt(np.mean((b_row - row_pred)**2))

    return {
        'scale_x': float(scale_x),
        'scale_y': float(scale_y),
        'offset_x': float(offset_x),
        'offset_y': float(offset_y),
        'flip_y': False,
        'rotation_deg': 0.0,
        'control_points_used': n,
        'col_rmse': float(col_rmse),
        'row_rmse': float(row_rmse),
    }


def create_comparison_plots(map_data, meta, trajectories, params_before, params_after,
                           save_dir='.'):
    """
    生成校准前后对比图
    """
    width = meta['width']
    height = meta['height']
    resolution_x = meta.get('resolution_x', meta.get('resolution', 1.0))
    resolution_y = meta.get('resolution_y', meta.get('resolution', 1.0))

    def make_transform(params):
        def transform(lon, lat):
            return lonlat_to_pixel_with_params(lon, lat, meta, **params)
        return transform

    # Collect all points for both transforms
    points_before = []
    points_after = []

    transform_keys = ['scale_x', 'scale_y', 'offset_x', 'offset_y', 'flip_y', 'rotation_deg']
    params_before_clean = {k: v for k, v in params_before.items() if k in transform_keys}
    params_after_clean = {k: v for k, v in params_after.items() if k in transform_keys}

    for obs in trajectories.get('obstacles', []):
        for pt in obs.get('trajectory', []):
            lon = pt.get('lon')
            lat = pt.get('lat')
            if lon is None or lat is None:
                continue
            col_b, row_b = lonlat_to_pixel_with_params(lon, lat, meta, **params_before_clean)
            col_a, row_a = lonlat_to_pixel_with_params(lon, lat, meta, **params_after_clean)
            points_before.append((col_b, row_b))
            points_after.append((col_a, row_a))

    # Pixel-space overlay
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for ax, (points, title) in zip(axes, [
        (points_before, 'BEFORE (Default Mapping)'),
        (points_after, 'AFTER (Calibrated)')
    ]):
        ax.imshow(map_data, cmap=plt.cm.colors.ListedColormap(['white', 'darkgray']),
                 origin='upper')
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)

        # Plot trajectory points as scatter
        if points:
            cols, rows = zip(*points)
            ax.scatter(cols, rows, s=1, alpha=0.3, c='blue')

        ax.set_xlabel('Pixel Column')
        ax.set_ylabel('Pixel Row')
        ax.set_title(f'Pixel-Space: {title}')

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'overlay_pixel_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: overlay_pixel_comparison.png")

    # World-space overlay
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for ax, (points, title) in zip(axes, [
        (points_before, 'BEFORE (Default Mapping)'),
        (points_after, 'AFTER (Calibrated)')
    ]):
        ax.imshow(map_data, cmap=plt.cm.colors.ListedColormap(['white', 'darkgray']),
                 extent=[0, width*resolution_x, height*resolution_y, 0],
                 origin='upper', aspect='auto')

        if points:
            # Convert col,row to world_x,world_y
            xs = [p[0] * resolution_x for p in points]
            ys = [p[1] * resolution_y for p in points]
            ax.scatter(xs, ys, s=1, alpha=0.3, c='blue')

        ax.set_xlabel('World X (m)')
        ax.set_ylabel('World Y (m)')
        ax.set_title(f'World-Space: {title}')

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'overlay_world_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: overlay_world_comparison.png")

    # Also save individual images
    for name, points in [('before', points_before), ('after', points_after)]:
        # Pixel-space
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(map_data, cmap=plt.cm.colors.ListedColormap(['white', 'darkgray']),
                 origin='upper')
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        if points:
            cols, rows = zip(*points)
            ax.scatter(cols, rows, s=1, alpha=0.3, c='blue')
        ax.set_xlabel('Pixel Column')
        ax.set_ylabel('Pixel Row')
        ax.set_title(f'Pixel-Space Overlay - {name.upper()}')
        fig.savefig(os.path.join(save_dir, f'overlay_pixel_{name}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: overlay_pixel_{name}.png")

        # World-space
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(map_data, cmap=plt.cm.colors.ListedColormap(['white', 'darkgray']),
                 extent=[0, width*resolution_x, height*resolution_y, 0],
                 origin='upper', aspect='auto')
        if points:
            xs = [p[0] * resolution_x for p in points]
            ys = [p[1] * resolution_y for p in points]
            ax.scatter(xs, ys, s=1, alpha=0.3, c='blue')
        ax.set_xlabel('World X (m)')
        ax.set_ylabel('World Y (m)')
        ax.set_title(f'World-Space Overlay - {name.upper()}')
        fig.savefig(os.path.join(save_dir, f'overlay_world_{name}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: overlay_world_{name}.png")


def main():
    parser = argparse.ArgumentParser(description='Calibrate lon/lat to pixel transformation')
    parser.add_argument('--map', type=str, required=True, help='Path to .npy map file')
    parser.add_argument('--meta', type=str, required=True, help='Path to map metadata yaml')
    parser.add_argument('--traj', type=str, required=True, help='Path to trajectory JSON')
    parser.add_argument('--mode', type=str, default='auto',
                       choices=['auto', 'control_points'],
                       help='Calibration mode')
    parser.add_argument('--control-points', type=str, default=None,
                       help='Path to control points yaml/json (for control_points mode)')
    parser.add_argument('--out-params', type=str, required=True,
                       help='Output calibrated parameters yaml')
    parser.add_argument('--out-stats', type=str, default=None,
                       help='Output statistics yaml')
    parser.add_argument('--save-dir', type=str, default='.',
                       help='Directory to save comparison plots')

    args = parser.parse_args()

    print("=" * 60)
    print("经纬度到像素坐标配准校准")
    print("=" * 60)

    # Load data
    print(f"\n[1] 加载数据...")
    map_data = load_map(args.map)
    meta = load_meta(args.meta)
    trajectories = load_trajectories(args.traj)

    print(f"  地图: {map_data.shape} ({meta['width']}x{meta['height']})")
    print(f"  分辨率: x={meta.get('resolution_x', meta.get('resolution')):.2f}, "
          f"y={meta.get('resolution_y', meta.get('resolution')):.2f} m/pixel")
    print(f"  地理范围: lon=[{meta['lon_min']:.4f}, {meta['lon_max']:.4f}], "
          f"lat=[{meta['lat_min']:.4f}, {meta['lat_max']:.4f}]")
    print(f"  轨迹: {len(trajectories.get('obstacles', []))} 个障碍物")

    # Default parameters (before calibration)
    params_before = {
        'scale_x': 1.0,
        'scale_y': 1.0,
        'offset_x': 0.0,
        'offset_y': 0.0,
        'flip_y': False,
        'rotation_deg': 0.0,
    }

    # Compute before stats
    print(f"\n[2] 计算校准前统计...")

    def transform_before(lon, lat):
        return lonlat_to_pixel_with_params(lon, lat, meta, **params_before)

    stats_before = compute_stats(map_data, trajectories, transform_before)
    boundary_before = compute_boundary_proximity(map_data, trajectories, transform_before)

    print(f"  水域(free)比例: {stats_before['free_pct']:.1f}%")
    print(f"  陆地(obstacle)比例: {stats_before['obstacle_pct']:.1f}%")
    print(f"  地图外比例: {stats_before['outside_pct']:.1f}%")
    print(f"  平均到free距离: {stats_before['avg_dist_to_free']:.2f} 像素")
    print(f"  中位到free距离: {stats_before['median_dist_to_free']:.2f} 像素")

    # Calibrate
    print(f"\n[3] 校准 ({args.mode} 模式)...")

    if args.mode == 'auto':
        params_after = calibrate_auto(meta, trajectories, map_data)
    elif args.mode == 'control_points':
        if not args.control_points:
            print("  Error: --control-points required for control_points mode")
            sys.exit(1)
        with open(args.control_points, 'r') as f:
            if args.control_points.endswith('.json'):
                control_points = json.load(f)
            else:
                control_points = yaml.safe_load(f)
        params_after = calibrate_with_control_points(meta, map_data, control_points)
    else:
        print(f"  Unknown mode: {args.mode}")
        sys.exit(1)

    print(f"  校准后参数:")
    for k, v in params_after.items():
        if not k.startswith('optimization') and not k.startswith('control') and not k.startswith('col_') and not k.startswith('row_'):
            print(f"    {k}: {v}")

    # Compute after stats
    print(f"\n[4] 计算校准后统计...")

    # Extract only the transform parameters (exclude optimization metadata)
    transform_keys = ['scale_x', 'scale_y', 'offset_x', 'offset_y', 'flip_y', 'rotation_deg']
    params_transform = {k: v for k, v in params_after.items() if k in transform_keys}

    def transform_after(lon, lat):
        return lonlat_to_pixel_with_params(lon, lat, meta, **params_transform)

    stats_after = compute_stats(map_data, trajectories, transform_after)
    boundary_after = compute_boundary_proximity(map_data, trajectories, transform_after)

    print(f"  水域(free)比例: {stats_after['free_pct']:.1f}%")
    print(f"  陆地(obstacle)比例: {stats_after['obstacle_pct']:.1f}%")
    print(f"  地图外比例: {stats_after['outside_pct']:.1f}%")
    print(f"  平均到free距离: {stats_after['avg_dist_to_free']:.2f} 像素")
    print(f"  中位到free距离: {stats_after['median_dist_to_free']:.2f} 像素")

    # Compare
    print(f"\n[5] 校准前后对比...")
    print(f"  free比例变化: {stats_before['free_pct']:.1f}% -> {stats_after['free_pct']:.1f}% "
          f"({stats_after['free_pct'] - stats_before['free_pct']:+.1f}%)")
    print(f"  obstacle比例变化: {stats_before['obstacle_pct']:.1f}% -> {stats_after['obstacle_pct']:.1f}% "
          f"({stats_after['obstacle_pct'] - stats_before['obstacle_pct']:+.1f}%)")
    print(f"  平均距离变化: {stats_before['avg_dist_to_free']:.2f} -> {stats_after['avg_dist_to_free']:.2f} "
          f"({stats_after['avg_dist_to_free'] - stats_before['avg_dist_to_free']:+.2f})")

    # Generate comparison plots
    print(f"\n[6] 生成对比图...")
    os.makedirs(args.save_dir, exist_ok=True)
    create_comparison_plots(map_data, meta, trajectories, params_before, params_after, args.save_dir)

    # Save parameters
    print(f"\n[7] 保存参数...")
    with open(args.out_params, 'w') as f:
        yaml.dump({
            'before': params_before,
            'after': params_after,
            'stats_before': {k: float(v) if isinstance(v, (np.floating, float)) else v
                            for k, v in stats_before.items()},
            'stats_after': {k: float(v) if isinstance(v, (np.floating, float)) else v
                           for k, v in stats_after.items()},
            'boundary_before': {k: (int(v[0]), float(v[1])) for k, v in boundary_before.items()},
            'boundary_after': {k: (int(v[0]), float(v[1])) for k, v in boundary_after.items()},
        }, f, default_flow_style=False)
    print(f"  已保存: {args.out_params}")

    if args.out_stats:
        with open(args.out_stats, 'w') as f:
            yaml.dump({
                'before': stats_before,
                'after': stats_after,
            }, f, default_flow_style=False)

    print("\n" + "=" * 60)
    print("校准完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
