#!/usr/bin/env python
"""
AIS XLS轨迹转动态障碍物JSON脚本

批量读取AIS轨迹文件，映射到地图坐标系，生成训练可直接使用的JSON格式

用法:
    python tools/convert_ais_xls_to_obstacles.py \
        --input-dir data/raw/trajectories \
        --map-meta data/processed/maps/navigation_map_meta.yaml \
        --out-json data/processed/trajectories/multi_obstacles.json
"""

import argparse
import os
import sys
import json
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any


def load_map_metadata(meta_path: str) -> dict:
    """加载地图元数据"""
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = yaml.safe_load(f)
    return meta


def detect_content_bbox(map_data: np.ndarray, margin: int = 0) -> Tuple[int, int, int, int]:
    """
    检测地图内容边界框

    找到所有非零像素(障碍物)的边界，如果没有障碍物则找非255像素

    Args:
        map_data: 占据栅格 numpy 数组 (H, W), 0=free, 1=obstacle
        margin: 边界外扩像素数

    Returns:
        (left, top, right, bottom) 内容边界框像素坐标
    """
    from scipy.ndimage import binary_opening, binary_closing

    # 方法1: 直接找obstacle像素边界
    obstacle_mask = (map_data != 0)

    # 检查是否有任何obstacle像素
    if not np.any(obstacle_mask):
        # 如果没有obstacle，找非255像素（用于原始PGM）
        non_uniform = (map_data != 255)
        if np.any(non_uniform):
            rows = np.any(non_uniform, axis=1)
            cols = np.any(non_uniform, axis=0)
        else:
            rows = cols = np.ones(map_data.shape[0], dtype=bool)
    else:
        rows = np.any(obstacle_mask, axis=1)
        cols = np.any(obstacle_mask, axis=0)

    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]

    if len(row_indices) == 0 or len(col_indices) == 0:
        # 全图空白，返回全图
        return (0, 0, map_data.shape[1]-1, map_data.shape[0]-1)

    top = max(0, row_indices[0] - margin)
    bottom = min(map_data.shape[0] - 1, row_indices[-1] + margin)
    left = max(0, col_indices[0] - margin)
    right = min(map_data.shape[1] - 1, col_indices[-1] + margin)

    return (left, top, right, bottom)


def lonlat_to_world(lon: float, lat: float, meta: dict,
                   mapping_mode: str = 'full_image',
                   content_bbox: Tuple[int, int, int, int] = None) -> Tuple[float, float, int, int]:
    """
    将经纬度转换为地图世界坐标

    坐标映射 (y-down约定):
    - lon -> world_x: lon范围映射到图像x方向 (col)
    - lat -> row: lat范围映射到图像y方向 (y-down: lat_max -> row=0)

    支持两种映射模式:
    - 'full_image': 将经纬度范围映射到整张图像 (0, 0, width-1, height-1)
    - 'content_bbox': 只映射到检测到的内容区域

    Args:
        lon: 经度
        lat: 纬度
        meta: 地图元数据字典 (支持 resolution 或 resolution_x/resolution_y)
        mapping_mode: 'full_image' 或 'content_bbox'
        content_bbox: 内容边界框 (left, top, right, bottom)，仅在 mapping_mode='content_bbox' 时使用

    Returns:
        (world_x, world_y, col, row) 世界坐标和像素坐标
    """
    lon_min = meta['lon_min']
    lon_max = meta['lon_max']
    lat_min = meta['lat_min']
    lat_max = meta['lat_max']
    width = meta['width']
    height = meta['height']

    # 使用 resolution_x/resolution_y（如果存在）或 fallback 到 resolution
    resolution_x = meta.get('resolution_x', meta.get('resolution', 1.0))
    resolution_y = meta.get('resolution_y', meta.get('resolution', 1.0))

    # 经纬度范围
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min

    # 根据映射模式确定目标范围
    if mapping_mode == 'content_bbox' and content_bbox is not None:
        left, top, right, bottom = content_bbox
        target_width = right - left + 1
        target_height = bottom - top + 1
    else:
        # full_image 模式
        left, top = 0, 0
        right, bottom = width - 1, height - 1
        target_width = width
        target_height = height

    # 转换到像素坐标 (col, row)
    # y-down: lat_max (北) -> row=0 (顶部), lat_min (南) -> row=height-1 (底部)
    if lon_range > 0:
        col = (lon - lon_min) / lon_range * target_width + left
    else:
        col = left

    if lat_range > 0:
        row = (lat_max - lat) / lat_range * target_height + top
    else:
        row = top

    # 裁剪到有效范围
    col = max(0, min(width - 1, col))
    row = max(0, min(height - 1, row))

    # 像素坐标转世界坐标 (y-down: world_y向下增加)
    world_x = col * resolution_x
    world_y = row * resolution_y

    return world_x, world_y, int(col), int(row)
    """
    将经纬度转换为地图世界坐标

    坐标映射 (y-down约定):
    - lon -> world_x: lon范围映射到图像x方向 (col)
    - lat -> row: lat范围映射到图像y方向 (y-down: lat_max -> row=0)

    Args:
        lon: 经度
        lat: 纬度
        meta: 地图元数据字典 (支持 resolution 或 resolution_x/resolution_y)

    Returns:
        (world_x, world_y, col, row) 世界坐标和像素坐标
    """
    lon_min = meta['lon_min']
    lon_max = meta['lon_max']
    lat_min = meta['lat_min']
    lat_max = meta['lat_max']
    width = meta['width']
    height = meta['height']

    # 使用 resolution_x/resolution_y（如果存在）或 fallback 到 resolution
    resolution_x = meta.get('resolution_x', meta.get('resolution', 1.0))
    resolution_y = meta.get('resolution_y', meta.get('resolution', 1.0))

    # 经纬度范围
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min

    # 转换到像素坐标 (col, row)
    # y-down: lat_max (北) -> row=0 (顶部), lat_min (南) -> row=height-1 (底部)
    if lon_range > 0:
        col = (lon - lon_min) / lon_range * width
    else:
        col = 0

    if lat_range > 0:
        row = (lat_max - lat) / lat_range * height
    else:
        row = 0

    # 裁剪到有效范围
    col = max(0, min(width - 1, col))
    row = max(0, min(height - 1, row))

    # 像素坐标转世界坐标 (y-down: world_y向下增加)
    world_x = col * resolution_x
    world_y = row * resolution_y

    return world_x, world_y, int(col), int(row)


def is_in_map_bounds(lon: float, lat: float, meta: dict, margin: float = 0.05) -> bool:
    """检查经纬度是否在地图范围内（带margin）"""
    lon_min = meta['lon_min']
    lon_max = meta['lon_max']
    lat_min = meta['lat_min']
    lat_max = meta['lat_max']

    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min

    return (lon_min - margin * lon_range <= lon <= lon_max + margin * lon_range and
            lat_min - margin * lat_range <= lat <= lat_max + margin * lat_range)


def detect_header_row(df: pd.DataFrame) -> int:
    """检测表头行（包含mmsi/lon/lat/time等关键词）"""
    header_keywords = ['mmsi', '经度', 'longitude', 'lon', '纬度', 'latitude', 'lat',
                      '时间', 'time', '速度', 'speed', 'heading', 'cog', '航向']

    for idx, row in df.iterrows():
        row_str = ' '.join([str(v).lower() for v in row.values if pd.notna(v)])
        matches = sum(1 for kw in header_keywords if kw.lower() in row_str)
        if matches >= 2:  # 至少匹配2个关键词
            return idx
    return 1  # 默认第2行（index=1）


def standardize_columns(df: pd.DataFrame, header_row: int) -> pd.DataFrame:
    """标准化列名"""
    # 使用header_row作为列名
    df.columns = df.iloc[header_row].values
    df = df.iloc[header_row + 1:].reset_index(drop=True)

    # 列名映射
    column_mapping = {
        'mmsi': 'mmsi',
        '经度': 'lon',
        'longitude': 'lon',
        'lon': 'lon',
        'lng': 'lon',
        '纬度': 'lat',
        'latitude': 'lat',
        'lat': 'lat',
        '时间': 'time',
        'time': 'time',
        'timestamp': 'time',
        '速度': 'speed',
        'speed': 'speed',
        'sog': 'speed',
        '船首向': 'heading',
        'heading': 'heading',
        '对地航向': 'cog',
        'cog': 'cog',
        '航行状态': 'nav_status',
        'nav_status': 'nav_status',
        'status': 'nav_status',
    }

    # 标准化列名
    new_columns = {}
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if col_lower in column_mapping:
            new_columns[col] = column_mapping[col_lower]
        else:
            # 尝试模糊匹配
            for kw, standard_name in column_mapping.items():
                if kw in col_lower:
                    new_columns[col] = standard_name
                    break

    df = df.rename(columns=new_columns)

    return df


def parse_time(time_str: Any) -> Optional[datetime]:
    """解析时间字符串"""
    if pd.isna(time_str):
        return None

    time_str = str(time_str).strip()

    # 尝试多种格式
    formats = [
        '%Y-%m-%d %H:%M(UTC+8)',
        '%Y-%m-%d %H:%M:%S(UTC+8)',
        '%Y-%m-%d %H:%M',
        '%Y-%m-%d %H:%M:%S',
        '%Y/%m/%d %H:%M:%S',
        '%Y/%m/%d %H:%M',
        '%m/%d %H:%M:%S',
    ]

    for fmt in formats:
        try:
            return datetime.strptime(time_str, fmt)
        except ValueError:
            continue

    # 尝试去除时区信息
    time_str = time_str.replace('(UTC+8)', '').replace('(UTC)', '').strip()
    for fmt in formats:
        try:
            return datetime.strptime(time_str, fmt)
        except ValueError:
            continue

    return None


def clean_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    """清洗轨迹数据"""
    initial_count = len(df)

    # 确保必要的列存在
    required_cols = ['mmsi', 'lon', 'lat', 'time']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")

    # 删除空行
    df = df.dropna(subset=['lon', 'lat', 'time'])

    # 转换为数值类型
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')

    # 删除转换失败的行
    df = df.dropna(subset=['lon', 'lat'])

    # 删除无效经纬度
    df = df[(df['lon'] >= 0) & (df['lon'] <= 180) &
            (df['lat'] >= 0) & (df['lat'] <= 90)]

    # 解析时间
    df['time_dt'] = df['time'].apply(parse_time)
    df = df.dropna(subset=['time_dt'])

    # 按时间排序
    df = df.sort_values('time_dt').reset_index(drop=True)

    # 删除重复时间点
    df = df.drop_duplicates(subset=['time_dt'], keep='first')

    # 过滤异常跳变（相邻点距离过大）
    if len(df) > 1:
        lons = df['lon'].values.astype(float)
        lats = df['lat'].values.astype(float)
        distances = np.sqrt(np.diff(lons)**2 + np.cos(np.radians(lats[:-1])) * np.diff(lons)**2 +
                          np.diff(lats)**2)

        # 大于1度为异常（约100km）
        max_dist = 1.0
        valid = np.concatenate([[True], distances < max_dist])
        df = df[valid].reset_index(drop=True)

    cleaned_count = len(df)

    print(f"    清洗: {initial_count} -> {cleaned_count} (删除 {initial_count - cleaned_count})")

    return df


def extract_mmsi_from_filename(filename: str) -> str:
    """从文件名提取mmsi"""
    # 文件名格式: 413872274_Sat+Nov+02+00_04_00+CST+2024-Sat+Nov+02+00_04_00+CST+2024.xls
    basename = os.path.basename(filename)
    name_part = basename.replace('.xls', '')
    parts = name_part.split('_')
    if parts:
        # 尝试第一个下划线前的部分作为mmsi
        mmsi_candidate = parts[0]
        if mmsi_candidate.isdigit() and len(mmsi_candidate) == 9:
            return mmsi_candidate
        # 否则返回第一部分
        return mmsi_candidate
    return basename


def resample_trajectory(df: pd.DataFrame, dt: float) -> List[Dict]:
    """
    对轨迹进行重采样

    Args:
        df: 清洗后的轨迹DataFrame
        dt: 重采样时间间隔（秒）

    Returns:
        轨迹点列表 [{t, x, y, lon, lat}, ...]
    """
    if len(df) < 2:
        return []

    # 计算时间范围
    t_start = df['time_dt'].iloc[0]
    t_end = df['time_dt'].iloc[-1]
    total_seconds = (t_end - t_start).total_seconds()

    # 计算需要的点数
    num_points = int(total_seconds / dt) + 1

    # 重采样时刻点
    resampled_times = [t_start + timedelta(seconds=i * dt) for i in range(num_points)]

    result = []
    for t in resampled_times:
        # 找到前后两个原始点
        idx_after = df['time_dt'].searchsorted(t)
        if idx_after == 0:
            # 在第一个点之前
            row = df.iloc[0]
            lon, lat = row['lon'], row['lat']
        elif idx_after >= len(df):
            # 在最后一个点之后
            row = df.iloc[-1]
            lon, lat = row['lon'], row['lat']
        else:
            # 在两个点之间，线性插值
            t0 = df['time_dt'].iloc[idx_after - 1]
            t1 = df['time_dt'].iloc[idx_after]
            row0 = df.iloc[idx_after - 1]
            row1 = df.iloc[idx_after]

            alpha = (t - t0).total_seconds() / (t1 - t0).total_seconds()
            lon = row0['lon'] + alpha * (row1['lon'] - row0['lon'])
            lat = row0['lat'] + alpha * (row1['lat'] - row0['lat'])

        result.append({
            't': (t - t_start).total_seconds(),
            'lon': float(lon),
            'lat': float(lat)
        })

    return result


def process_single_file(filepath: str, meta: dict, trajectory_dt: float,
                       obstacle_radius: float, world_origin: Tuple[float, float] = (0.0, 0.0),
                       mapping_mode: str = 'full_image',
                       content_bbox: Tuple[int, int, int, int] = None) -> Optional[Dict]:
    """处理单个xls文件，返回障碍物字典"""
    try:
        # 读取xls
        df_raw = pd.read_excel(filepath, header=None)

        # 检测表头行
        header_row = detect_header_row(df_raw)

        # 标准化列名
        df = standardize_columns(df_raw, header_row)

        # 确保必要列存在
        if 'lon' not in df.columns or 'lat' not in df.columns or 'time' not in df.columns:
            print(f"    跳过: 缺少必要列")
            return None

        # 清洗
        df = clean_trajectory(df)

        if len(df) < 2:
            print(f"    跳过: 数据点不足")
            return None

        # 提取mmsi
        if 'mmsi' not in df.columns or df['mmsi'].isna().all():
            mmsi = extract_mmsi_from_filename(filepath)
            df['mmsi'] = mmsi
        else:
            mmsi = str(df['mmsi'].iloc[0])

        # 过滤地图外的点
        df['in_bounds'] = df.apply(lambda r: is_in_map_bounds(r['lon'], r['lat'], meta), axis=1)
        out_of_bounds_count = len(df) - df['in_bounds'].sum()
        if out_of_bounds_count > 0:
            print(f"    过滤掉 {out_of_bounds_count} 个地图范围外的点")
        df = df[df['in_bounds']].reset_index(drop=True)

        if len(df) < 2:
            print(f"    跳过: 地图内数据点不足")
            return None

        # 重采样
        trajectory = resample_trajectory(df, trajectory_dt)

        if len(trajectory) < 2:
            print(f"    跳过: 重采样后数据点不足")
            return None

        # 转换坐标到世界坐标
        world_trajectory = []
        for pt in trajectory:
            wx, wy, col, row = lonlat_to_world(
                pt['lon'], pt['lat'], meta,
                mapping_mode=mapping_mode,
                content_bbox=content_bbox
            )
            # 应用origin偏移
            wx += world_origin[0]
            wy += world_origin[1]
            world_trajectory.append({
                't': round(pt['t'], 2),
                'x': round(wx, 2),
                'y': round(wy, 2),
                'col': col,
                'row': row,
                'lon': round(pt['lon'], 7),
                'lat': round(pt['lat'], 7),
            })

        return {
            'id': mmsi,
            'radius': obstacle_radius,
            'trajectory': world_trajectory,
            'original_points': len(df),
            'resampled_points': len(world_trajectory),
        }

    except Exception as e:
        print(f"    错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_time_intervals(input_dir: str) -> float:
    """分析所有轨迹文件的时间间隔，选择合适的dt"""
    all_intervals = []

    xls_files = [f for f in os.listdir(input_dir) if f.endswith('.xls')]

    for filename in xls_files[:5]:  # 只分析前5个文件
        filepath = os.path.join(input_dir, filename)
        try:
            df_raw = pd.read_excel(filepath, header=None)
            header_row = detect_header_row(df_raw)
            df = standardize_columns(df_raw, header_row)

            if 'time' not in df.columns:
                continue

            df['time_dt'] = df['time'].apply(parse_time)
            df = df.dropna(subset=['time_dt'])
            df = df.sort_values('time_dt')

            if len(df) > 1:
                intervals = df['time_dt'].diff().dt.total_seconds().dropna().values
                intervals = intervals[intervals > 0]  # 只考虑正向间隔
                if len(intervals) > 0:
                    all_intervals.extend(intervals.tolist())
        except:
            continue

    if not all_intervals:
        return 5.0  # 默认5秒

    # 统计间隔分布
    intervals_arr = np.array(all_intervals)
    median_interval = np.median(intervals_arr)
    mean_interval = np.mean(intervals_arr)

    print(f"\n时间间隔分析 (基于前5个文件):")
    print(f"  样本数: {len(intervals_arr)}")
    print(f"  中位数: {median_interval:.1f} 秒")
    print(f"  平均值: {mean_interval:.1f} 秒")
    print(f"  最小值: {intervals_arr.min():.1f} 秒")
    print(f"  最大值: {intervals_arr.max():.1f} 秒")

    # 选择合适的dt: 优先5秒，如果原始间隔太大则退到10秒
    if median_interval <= 3:
        chosen_dt = 5.0
    elif median_interval <= 8:
        chosen_dt = 10.0
    else:
        chosen_dt = max(median_interval, 10.0)

    print(f"  选择的重采样dt: {chosen_dt} 秒")
    return chosen_dt


def main():
    parser = argparse.ArgumentParser(description='Convert AIS XLS trajectories to obstacle JSON')
    parser.add_argument('--input-dir', type=str, default='data/raw/trajectories',
                       help='Input directory containing .xls files')
    parser.add_argument('--map-meta', type=str, required=True,
                       help='Path to map metadata yaml')
    parser.add_argument('--out-json', type=str, required=True,
                       help='Output JSON file path')
    parser.add_argument('--out-single', type=str, default=None,
                       help='Output single ship JSON for debugging')
    parser.add_argument('--out-csv', type=str, default=None,
                       help='Output cleaned CSV for debugging')
    parser.add_argument('--out-debug-csv', type=str, default=None,
                       help='Output debug CSV with all trajectory points (lon, lat, col, row, x, y, in_bounds, on_free, on_obstacle)')
    parser.add_argument('--dt', type=float, default=None,
                       help='Fixed time step for resampling (seconds). Auto-detect if not specified.')
    parser.add_argument('--radius', type=float, default=10.0,
                       help='Default obstacle radius in meters')
    parser.add_argument('--world-origin-x', type=float, default=0.0,
                       help='World origin x offset')
    parser.add_argument('--world-origin-y', type=float, default=0.0,
                       help='World origin y offset')

    args = parser.parse_args()

    print("=" * 60)
    print("AIS XLS轨迹转动态障碍物JSON")
    print("=" * 60)

    # 加载地图元数据
    print(f"\n[步骤1] 加载地图元数据: {args.map_meta}")
    meta = load_map_metadata(args.map_meta)
    print(f"  地图尺寸: {meta['width']} x {meta['height']}")
    res_x = meta.get('resolution_x', meta.get('resolution'))
    res_y = meta.get('resolution_y', meta.get('resolution'))
    res_avg = meta.get('resolution_avg', meta.get('resolution'))
    print(f"  分辨率: x={res_x:.2f}, y={res_y:.2f} m/pixel (avg={res_avg:.2f})")
    print(f"  地理范围: lon=[{meta['lon_min']:.4f}, {meta['lon_max']:.4f}], lat=[{meta['lat_min']:.4f}, {meta['lat_max']:.4f}]")

    # 分析时间间隔
    if args.dt is None:
        print(f"\n[步骤2] 分析时间间隔...")
        trajectory_dt = analyze_time_intervals(args.input_dir)
    else:
        trajectory_dt = args.dt
        print(f"\n[步骤2] 使用指定时间间隔: {trajectory_dt} 秒")

    # 获取xls文件列表
    xls_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.xls')])
    print(f"\n[步骤3] 找到 {len(xls_files)} 个XLS文件")

    # 处理每个文件
    obstacles = []
    failed_count = 0
    total_original_points = 0
    total_resampled_points = 0
    filtered_points = 0

    print(f"\n[步骤4] 处理轨迹文件...")
    for i, filename in enumerate(xls_files):
        filepath = os.path.join(input_dir, filename) if 'input_dir' in dir() else os.path.join(args.input_dir, filename)
        filepath = os.path.join(args.input_dir, filename)
        print(f"  [{i+1}/{len(xls_files)}] {filename}")

        obstacle = process_single_file(
            filepath, meta, trajectory_dt, args.radius,
            world_origin=(args.world_origin_x, args.world_origin_y)
        )

        if obstacle:
            obstacles.append(obstacle)
            total_original_points += obstacle['original_points']
            total_resampled_points += obstacle['resampled_points']
        else:
            failed_count += 1

    print(f"\n处理完成:")
    print(f"  成功: {len(obstacles)} 个障碍物")
    print(f"  失败: {failed_count} 个")
    print(f"  总原始点数: {total_original_points}")
    print(f"  总重采样点数: {total_resampled_points}")

    # 构建输出JSON
    output = {
        'dt': trajectory_dt,
        'loop': False,  # AIS轨迹不循环
        'obstacles': obstacles,
        'metadata': {
            'source_files': len(xls_files),
            'successful_files': len(obstacles),
            'failed_files': failed_count,
            'total_original_points': total_original_points,
            'total_resampled_points': total_resampled_points,
            'filtered_points': filtered_points,
            'obstacle_radius': args.radius,
            'processed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'map_meta': {
                'width': meta['width'],
                'height': meta['height'],
                'resolution_x': meta.get('resolution_x', meta.get('resolution')),
                'resolution_y': meta.get('resolution_y', meta.get('resolution')),
                'resolution_avg': meta.get('resolution_avg', meta.get('resolution')),
                'lon_range': [meta['lon_min'], meta['lon_max']],
                'lat_range': [meta['lat_min'], meta['lat_max']],
            }
        }
    }

    # 保存主JSON
    print(f"\n[步骤5] 保存JSON: {args.out_json}")
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  已保存 {len(obstacles)} 个障碍物")

    # 保存单个障碍物样例（用于调试）
    if args.out_single and obstacles:
        single_example = {
            'dt': trajectory_dt,
            'loop': False,
            'obstacles': [obstacles[0]]
        }
        with open(args.out_single, 'w', encoding='utf-8') as f:
            json.dump(single_example, f, ensure_ascii=False, indent=2)
        print(f"  已保存单障碍物样例: {args.out_single}")

    # 保存清洗后的CSV样例
    if args.out_csv and obstacles:
        # 保存第一个障碍物的轨迹为CSV
        first_obs = obstacles[0]
        import csv
        with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['t', 'x', 'y'])
            for pt in first_obs['trajectory']:
                writer.writerow([pt['t'], pt['x'], pt['y']])
        print(f"  已保存CSV样例: {args.out_csv}")

    # 保存调试CSV（所有轨迹点）
    if args.out_debug_csv and obstacles:
        print(f"\n[步骤6] 保存调试CSV: {args.out_debug_csv}")
        # 加载地图数据以判断on_free/on_obstacle
        map_npy_path = args.out_json.replace('/trajectories/', '/maps/').replace('.json', '.npy')
        if os.path.exists(map_npy_path):
            map_data = np.load(map_npy_path)
            print(f"  加载地图: {map_npy_path}")
        else:
            print(f"  警告: 找不到地图文件 {map_npy_path}，on_free/on_obstacle将无法判断")
            map_data = None

        import csv
        with open(args.out_debug_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['obstacle_id', 't', 'lon', 'lat', 'col', 'row', 'x', 'y', 'in_bounds', 'on_free', 'on_obstacle'])
            for obs in obstacles:
                obs_id = obs['id']
                for pt in obs['trajectory']:
                    col = pt['col']
                    row = pt['row']
                    in_bounds = 0 <= col < meta['width'] and 0 <= row < meta['height']
                    if map_data is not None and in_bounds:
                        cell_value = map_data[row, col]
                        on_free = 1 if cell_value == 0 else 0
                        on_obstacle = 1 if cell_value == 1 else 0
                    else:
                        on_free = -1
                        on_obstacle = -1
                    writer.writerow([
                        obs_id,
                        pt['t'],
                        pt.get('lon', ''),
                        pt.get('lat', ''),
                        col,
                        row,
                        pt['x'],
                        pt['y'],
                        1 if in_bounds else 0,
                        on_free,
                        on_obstacle
                    ])
        print(f"  已保存调试CSV，包含 {sum(len(o['trajectory']) for o in obstacles)} 个轨迹点")

    # 打印摘要
    print("\n" + "=" * 60)
    print("轨迹转换完成!")
    print("=" * 60)
    print(f"  成功处理: {len(obstacles)} / {len(xls_files)} 个文件")
    print(f"  重采样dt: {trajectory_dt} 秒")
    print(f"  障碍物半径: {args.radius} 米")
    print(f"  总轨迹点数: {total_resampled_points}")
    print("=" * 60)


if __name__ == '__main__':
    main()