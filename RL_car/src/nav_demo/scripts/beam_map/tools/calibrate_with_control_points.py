#!/usr/bin/env python3
"""
控制点配准脚本 - 基于手工控制点的仿射变换拟合

从 yaml/json 文件读取控制点 (lon, lat) -> (col, row)，
拟合二维仿射变换: col = a1*lon + b1*lat + c1, row = a2*lon + b2*lat + c2

用法:
    python tools/calibrate_with_control_points.py \
        --map data/processed/maps/navigation_map.npy \
        --meta data/processed/maps/navigation_map_meta.yaml \
        --traj data/processed/trajectories/multi_obstacles.json \
        --control-points data/processed/maps/control_points.yaml \
        --out-params affine_params.yaml
"""

import argparse
import os
import sys
import json
import yaml
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_map(map_path: str) -> np.ndarray:
    return np.load(map_path)


def load_meta(meta_path: str) -> dict:
    with open(meta_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_trajectories(traj_path: str) -> dict:
    with open(traj_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_control_points(path: str) -> List[Dict]:
    """加载控制点"""
    with open(path, 'r', encoding='utf-8') as f:
        if path.endswith('.json'):
            data = json.load(f)
        else:
            data = yaml.safe_load(f)

    if isinstance(data, dict) and 'control_points' in data:
        return data['control_points']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unknown control points format in {path}")


def fit_affine_transform(control_points: List[Dict], meta: dict) -> Dict:
    """
    基于控制点拟合仿射变换

    仿射变换模型:
        col = a1 * lon + b1 * lat + c1
        row = a2 * lon + b2 * lat + c2

    使用最小二乘法求解
    """
    lon_min = meta['lon_min']
    lon_max = meta['lon_max']
    lat_min = meta['lat_min']
    lat_max = meta['lat_max']

    n = len(control_points)
    if n < 3:
        return None

    # 构建矩阵 A = [lon, lat, 1]，求解 col = A @ [a1, b1, c1]^T
    A = np.zeros((n, 3))
    b_col = np.zeros(n)
    b_row = np.zeros(n)

    for i, cp in enumerate(control_points):
        lon = cp['lon']
        lat = cp['lat']
        col = cp['col']
        row = cp['row']

        A[i] = [lon, lat, 1]
        b_col[i] = col
        b_row[i] = row

    # 最小二乘求解
    x_col, residuals_col, rank_col, s_col = np.linalg.lstsq(A, b_col, rcond=None)
    x_row, residuals_row, rank_row, s_row = np.linalg.lstsq(A, b_row, rcond=None)

    a1, b1, c1 = x_col
    a2, b2, c2 = x_row

    # 计算拟合残差
    col_pred = A @ x_col
    row_pred = A @ x_row
    col_residuals = b_col - col_pred
    row_residuals = b_row - row_pred
    col_rmse = np.sqrt(np.mean(col_residuals**2))
    row_rmse = np.sqrt(np.mean(row_residuals**2))
    max_residual = np.max(np.sqrt(col_residuals**2 + row_residuals**2))

    return {
        'a1': float(a1),
        'b1': float(b1),
        'c1': float(c1),
        'a2': float(a2),
        'b2': float(b2),
        'c2': float(c2),
        'col_rmse': float(col_rmse),
        'row_rmse': float(row_rmse),
        'max_residual': float(max_residual),
        'control_points_used': n,
    }


def lonlat_to_pixel_linear(lon: float, lat: float, meta: dict) -> Tuple[int, int]:
    """默认四角线性映射"""
    lon_min = meta['lon_min']
    lon_max = meta['lon_max']
    lat_min = meta['lat_min']
    lat_max = meta['lat_max']
    width = meta['width']
    height = meta['height']

    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min

    col = (lon - lon_min) / lon_range * width if lon_range > 0 else 0
    row = (lat_max - lat) / lat_range * height if lat_range > 0 else 0

    col = max(0, min(width - 1, col))
    row = max(0, min(height - 1, row))

    return int(col), int(row)


def lonlat_to_pixel_affine(lon: float, lat: float, params: Dict) -> Tuple[int, int]:
    """仿射变换映射"""
    a1, b1, c1 = params['a1'], params['b1'], params['c1']
    a2, b2, c2 = params['a2'], params['b2'], params['c2']

    col = a1 * lon + b1 * lat + c1
    row = a2 * lon + b2 * lat + c2

    return int(col), int(row)


def compute_stats(map_data: np.ndarray, trajectories: dict,
                 transform_func) -> Dict:
    """计算统计信息"""
    width = map_data.shape[1]
    height = map_data.shape[0]

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

            if not (0 <= col < width and 0 <= row < height):
                outside_count += 1
                continue

            if map_data[row, col] == 0:
                free_count += 1
                distances.append(0.0)
            else:
                obstacle_count += 1
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
        'avg_dist': float(np.mean(distances)) if len(distances) > 0 else 0.0,
        'median_dist': float(np.median(distances)) if len(distances) > 0 else 0.0,
        'max_dist': float(np.max(distances)) if len(distances) > 0 else 0.0,
    }


def compute_boundary_proximity(map_data: np.ndarray, trajectories: dict,
                               transform_func, thresholds=[2, 3, 5]) -> Dict:
    """计算贴近边界的点数"""
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


def create_comparison_plots(map_data, meta, trajectories,
                           params_linear, params_affine,
                           save_dir='.'):
    """生成线性 vs 仿射对比图"""
    width = meta['width']
    height = meta['height']
    resolution_x = meta.get('resolution_x', meta.get('resolution', 1.0))
    resolution_y = meta.get('resolution_y', meta.get('resolution', 1.0))

    # 收集所有轨迹点
    points_linear = []
    points_affine = []

    for obs in trajectories.get('obstacles', []):
        for pt in obs.get('trajectory', []):
            lon = pt.get('lon')
            lat = pt.get('lat')
            if lon is None or lat is None:
                continue

            col_lin, row_lin = lonlat_to_pixel_linear(lon, lat, meta)
            col_aff, row_aff = lonlat_to_pixel_affine(lon, lat, params_affine)

            points_linear.append((col_lin, row_lin))
            points_affine.append((col_aff, row_aff))

    # Pixel-space 对比
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for ax, (points, title) in zip(axes, [
        (points_linear, 'LINEAR (Default)'),
        (points_affine, 'AFFINE (Control Points)')
    ]):
        ax.imshow(map_data, cmap=plt.cm.colors.ListedColormap(['white', 'darkgray']),
                 origin='upper')
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)

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

    # World-space 对比
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for ax, (points, title) in zip(axes, [
        (points_linear, 'LINEAR (Default)'),
        (points_affine, 'AFFINE (Control Points)')
    ]):
        ax.imshow(map_data, cmap=plt.cm.colors.ListedColormap(['white', 'darkgray']),
                 extent=[0, width*resolution_x, height*resolution_y, 0],
                 origin='upper', aspect='auto')

        if points:
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

    # 单独保存4张图
    for name, points in [('linear', points_linear), ('affine', points_affine)]:
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
    parser = argparse.ArgumentParser(description='Control point based affine calibration')
    parser.add_argument('--map', type=str, required=True, help='Path to .npy map file')
    parser.add_argument('--meta', type=str, required=True, help='Path to map metadata yaml')
    parser.add_argument('--traj', type=str, required=True, help='Path to trajectory JSON')
    parser.add_argument('--control-points', type=str, required=True,
                       help='Path to control points yaml/json')
    parser.add_argument('--out-params', type=str, required=True,
                       help='Output affine parameters yaml')
    parser.add_argument('--out-stats', type=str, default=None,
                       help='Output statistics yaml')
    parser.add_argument('--save-dir', type=str, default='.',
                       help='Directory to save comparison plots')

    args = parser.parse_args()

    print("=" * 60)
    print("控制点配准 - 仿射变换拟合")
    print("=" * 60)

    # 加载数据
    print(f"\n[1] 加载数据...")
    map_data = load_map(args.map)
    meta = load_meta(args.meta)
    trajectories = load_trajectories(args.traj)
    control_points = load_control_points(args.control_points)

    print(f"  地图: {map_data.shape} ({meta['width']}x{meta['height']})")
    print(f"  地理范围: lon=[{meta['lon_min']:.4f}, {meta['lon_max']:.4f}], "
          f"lat=[{meta['lat_min']:.4f}, {meta['lat_max']:.4f}]")
    print(f"  轨迹: {len(trajectories.get('obstacles', []))} 个障碍物")
    print(f"  控制点: {len(control_points)} 个")

    # 打印控制点
    print(f"\n[2] 控制点列表:")
    for i, cp in enumerate(control_points):
        print(f"  [{i+1}] lon={cp.get('lon', 'N/A'):.5f}, lat={cp.get('lat', 'N/A'):.5f} "
              f"-> col={cp.get('col', 'N/A')}, row={cp.get('row', 'N/A')}")

    # 拟合仿射变换
    print(f"\n[3] 拟合仿射变换...")
    affine_params = fit_affine_transform(control_points, meta)

    if affine_params is None:
        print(f"  错误: 控制点数量不足 (需要至少3个)")
        sys.exit(1)

    print(f"  拟合结果:")
    print(f"    col = {affine_params['a1']:.4f}*lon + {affine_params['b1']:.4f}*lat + {affine_params['c1']:.4f}")
    print(f"    row = {affine_params['a2']:.4f}*lon + {affine_params['b2']:.4f}*lat + {affine_params['c2']:.4f}")
    print(f"    控制点RMSE: col={affine_params['col_rmse']:.2f}px, row={affine_params['row_rmse']:.2f}px")
    print(f"    最大残差: {affine_params['max_residual']:.2f}px")

    # 计算线性映射统计
    print(f"\n[4] 线性映射统计...")
    stats_linear = compute_stats(map_data, trajectories, lambda lon, lat: lonlat_to_pixel_linear(lon, lat, meta))
    boundary_linear = compute_boundary_proximity(map_data, trajectories, lambda lon, lat: lonlat_to_pixel_linear(lon, lat, meta))

    print(f"  水域(free): {stats_linear['free_count']} ({stats_linear['free_pct']:.1f}%)")
    print(f"  陆地(obstacle): {stats_linear['obstacle_count']} ({stats_linear['obstacle_pct']:.1f}%)")
    print(f"  地图外: {stats_linear['outside_count']} ({stats_linear['outside_pct']:.1f}%)")
    print(f"  平均到free距离: {stats_linear['avg_dist']:.2f}px")
    print(f"  中位到free距离: {stats_linear['median_dist']:.2f}px")

    # 计算仿射映射统计
    print(f"\n[5] 仿射映射统计...")
    stats_affine = compute_stats(map_data, trajectories, lambda lon, lat: lonlat_to_pixel_affine(lon, lat, affine_params))
    boundary_affine = compute_boundary_proximity(map_data, trajectories, lambda lon, lat: lonlat_to_pixel_affine(lon, lat, affine_params))

    print(f"  水域(free): {stats_affine['free_count']} ({stats_affine['free_pct']:.1f}%)")
    print(f"  陆地(obstacle): {stats_affine['obstacle_count']} ({stats_affine['obstacle_pct']:.1f}%)")
    print(f"  地图外: {stats_affine['outside_count']} ({stats_affine['outside_pct']:.1f}%)")
    print(f"  平均到free距离: {stats_affine['avg_dist']:.2f}px")
    print(f"  中位到free距离: {stats_affine['median_dist']:.2f}px")

    # 对比
    print(f"\n[6] 线性 vs 仿射 对比:")
    print(f"  free比例: {stats_linear['free_pct']:.1f}% -> {stats_affine['free_pct']:.1f}% "
          f"({stats_affine['free_pct'] - stats_linear['free_pct']:+.1f}%)")
    print(f"  obstacle比例: {stats_linear['obstacle_pct']:.1f}% -> {stats_affine['obstacle_pct']:.1f}% "
          f"({stats_affine['obstacle_pct'] - stats_linear['obstacle_pct']:+.1f}%)")
    print(f"  平均距离: {stats_linear['avg_dist']:.2f}px -> {stats_affine['avg_dist']:.2f}px "
          f"({stats_affine['avg_dist'] - stats_linear['avg_dist']:+.2f}px)")
    print(f"  中位距离: {stats_linear['median_dist']:.2f}px -> {stats_affine['median_dist']:.2f}px "
          f"({stats_affine['median_dist'] - stats_linear['median_dist']:+.2f}px)")

    # 生成对比图
    print(f"\n[7] 生成对比图...")
    os.makedirs(args.save_dir, exist_ok=True)
    create_comparison_plots(map_data, meta, trajectories,
                          None, affine_params, args.save_dir)

    # 保存参数
    print(f"\n[8] 保存参数...")
    output = {
        'affine_params': affine_params,
        'control_points': control_points,
        'meta_used': {
            'lon_min': meta['lon_min'],
            'lon_max': meta['lon_max'],
            'lat_min': meta['lat_min'],
            'lat_max': meta['lat_max'],
            'width': meta['width'],
            'height': meta['height'],
        },
        'stats_linear': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                        for k, v in stats_linear.items()},
        'stats_affine': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                        for k, v in stats_affine.items()},
        'boundary_linear': {k: (int(v[0]), float(v[1])) for k, v in boundary_linear.items()},
        'boundary_affine': {k: (int(v[0]), float(v[1])) for k, v in boundary_affine.items()},
    }

    with open(args.out_params, 'w') as f:
        yaml.dump(output, f, default_flow_style=False, allow_unicode=True)
    print(f"  已保存: {args.out_params}")

    if args.out_stats:
        with open(args.out_stats, 'w') as f:
            yaml.dump({
                'linear': stats_linear,
                'affine': stats_affine,
            }, f, default_flow_style=False)
        print(f"  已保存: {args.out_stats}")

    print("\n" + "=" * 60)
    print("配准完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
