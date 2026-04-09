#!/usr/bin/env python
"""
ROS地图转npy格式转换脚本

读取 .pgm + .yaml 地图，转换为训练可直接使用的 .npy 二值占据栅格

用法:
    python tools/convert_ros_map_to_npy.py \
        --map example_data/maps/navigation_map_0.50m.pgm \
        --yaml example_data/maps/navigation_map_0.50m.yaml \
        --out-map data/processed/maps/navigation_map.npy \
        --out-meta data/processed/maps/navigation_map_meta.yaml
"""

import argparse
import os
import sys
import yaml
import numpy as np
from PIL import Image
from datetime import datetime


def load_yaml_metadata(yaml_path):
    """读取yaml文件并提取元数据"""
    # 尝试多种编码
    for encoding in ['utf-8', 'gbk', 'gb2312', 'latin1']:
        try:
            with open(yaml_path, 'r', encoding=encoding) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue
    else:
        # 如果都失败，使用errors='replace'
        with open(yaml_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

    # 尝试用yaml读取，如果失败则手动解析
    try:
        # 处理可能的编码问题
        data = yaml.safe_load(content)
    except:
        # 手动解析关键字段
        data = {}
        for line in content.split('\n'):
            if line.startswith('image:'):
                data['image'] = line.split(':')[1].strip()
            elif line.startswith('resolution:'):
                data['resolution'] = float(line.split(':')[1].strip())
            elif line.startswith('origin:'):
                parts = line.split(':')[1].strip()
                data['origin'] = [float(x.strip()) for x in parts.strip('[]').split(',')]
            elif line.startswith('negate:'):
                data['negate'] = int(line.split(':')[1].strip())
            elif line.startswith('occupied_thresh:'):
                data['occupied_thresh'] = float(line.split(':')[1].strip())
            elif line.startswith('free_thresh:'):
                data['free_thresh'] = float(line.split(':')[1].strip())

    # 提取注释中的地理范围信息
    geo_bounds = {}
    for line in content.split('\n'):
        if '经纬度范围' in line or 'longitude' in line.lower() or 'latitude' in line.lower():
            # 提取 [lon_min, lat_min] 到 [lon_max, lat_max]
            import re
            matches = re.findall(r'[\d.]+', line)
            if len(matches) >= 4:
                geo_bounds['lon_min'] = float(matches[0])
                geo_bounds['lat_min'] = float(matches[1])
                geo_bounds['lon_max'] = float(matches[2])
                geo_bounds['lat_max'] = float(matches[3])

    # 提取物理尺寸
    physical_size = {}
    for line in content.split('\n'):
        if '物理尺寸' in line or 'physical' in line.lower():
            import re
            matches = re.findall(r'[\d.]+', line)
            if len(matches) >= 2:
                physical_size['width_m'] = float(matches[0])
                physical_size['height_m'] = float(matches[1])

    return data, geo_bounds, physical_size


def estimate_resolution_from_geo(img_width, img_height, geo_bounds, physical_size):
    """
    根据地理范围/物理尺寸估算真实分辨率

    Returns:
        resolution_x, resolution_y, resolution_avg, confidence
    """
    resolutions = []

    # 方法1: 根据地理经纬度范围
    if geo_bounds:
        lon_range = geo_bounds.get('lon_max', 0) - geo_bounds.get('lon_min', 0)
        lat_range = geo_bounds.get('lat_max', 0) - geo_bounds.get('lat_min', 0)

        # 经纬度转米（近似，在中纬度地区）
        # 1度经度 ≈ 111320 * cos(lat) 米
        # 1度纬度 ≈ 111320 米
        lat_center = (geo_bounds.get('lat_min', 0) + geo_bounds.get('lat_max', 0)) / 2
        meters_per_lon_degree = 111320 * np.cos(np.radians(lat_center))
        meters_per_lat_degree = 111320

        physical_width = lon_range * meters_per_lon_degree
        physical_height = lat_range * meters_per_lat_degree

        res_x = physical_width / img_width
        res_y = physical_height / img_height
        resolutions.append(('geo', res_x, res_y, physical_width, physical_height))

        print(f"  [方法1] 根据地理范围估算:")
        print(f"    经纬度范围: lon=[{geo_bounds.get('lon_min'):.4f}, {geo_bounds.get('lon_max'):.4f}], "
              f"lat=[{geo_bounds.get('lat_min'):.4f}, {geo_bounds.get('lat_max'):.4f}]")
        print(f"    对应物理范围: {physical_width:.1f} x {physical_height:.1f} 米")
        print(f"    估算分辨率: x={res_x:.2f} m/pixel, y={res_y:.2f} m/pixel")

    # 方法2: 根据yaml注释中的物理尺寸
    if physical_size:
        res_x = physical_size['width_m'] / img_width
        res_y = physical_size['height_m'] / img_height

        resolutions.append(('physical_size', res_x, res_y,
                          physical_size['width_m'], physical_size['height_m']))

        print(f"  [方法2] 根据yaml物理尺寸估算:")
        print(f"    物理尺寸: {physical_size['width_m']:.1f} x {physical_size['height_m']:.1f} 米")
        print(f"    估算分辨率: x={res_x:.2f} m/pixel, y={res_y:.2f} m/pixel")

    # 选择最佳估算
    if not resolutions:
        return None, None, None, None

    # 如果有地理范围方法，优先使用
    if geo_bounds:
        best = resolutions[0]  # 地理方法是第一个
    else:
        best = resolutions[-1]

    _, res_x, res_y, phys_w, phys_h = best

    # 检查x和y分辨率是否一致（偏差应该小于20%）
    avg_res = (res_x + res_y) / 2
    deviation = abs(res_x - res_y) / avg_res * 100

    print(f"\n  估算结果:")
    print(f"    x分辨率: {res_x:.4f} m/pixel")
    print(f"    y分辨率: {res_y:.4f} m/pixel")
    print(f"    平均分辨率: {avg_res:.4f} m/pixel")
    print(f"    x/y偏差: {deviation:.1f}%")

    if deviation > 20:
        print(f"  ⚠️ 警告: x和y方向分辨率偏差超过20%，可能地图不是正方形像素")

    return res_x, res_y, avg_res, phys_w, phys_h, deviation


def convert_pgm_to_occupancy_grid(pgm_path, yaml_data):
    """
    将PGM图像转换为二值占据栅格

    ROS地图语义:
    - negate: 0表示白色=自由(occupied_prob低)，黑色=障碍物(occupied_prob高)
              1表示相反
    - occupied_thresh: 占据概率阈值，超过此值视为障碍物
    - free_thresh: 自由概率阈值，低于此值视为自由
    - 中间区域视为unknown，第一版保守处理为障碍物
    """
    # 读取PGM
    img = Image.open(pgm_path)
    img_array = np.array(img).astype(np.float32) / 255.0

    height, width = img_array.shape
    print(f"\n图像尺寸: {width} x {height} (W x H)")

    negate = yaml_data.get('negate', 0)
    occupied_thresh = yaml_data.get('occupied_thresh', 0.65)
    free_thresh = yaml_data.get('free_thresh', 0.196)

    print(f"ROS地图参数:")
    print(f"  negate={negate}, occupied_thresh={occupied_thresh}, free_thresh={free_thresh}")

    # 计算占据概率 occ_prob
    # occ_prob = 1.0 - pixel_value when negate=0 (白=0=自由, 黑=1=障碍)
    # occ_prob = pixel_value when negate=1 (白=1=障碍, 黑=0=自由)
    if negate == 0:
        # 标准ROS地图: 白色=自由, 黑色=障碍
        # pixel=1.0(白)->occ_prob=0.0(自由), pixel=0.0(黑)->occ_prob=1.0(障碍)
        occ_prob = 1.0 - img_array
    else:
        # 反转语义
        occ_prob = img_array

    # 三分类: free / occupied / unknown
    # free: occ_prob < free_thresh (非常低概率占据 = 高概率自由)
    # occupied: occ_prob > occupied_thresh (非常高概率占据)
    # unknown: free_thresh <= occ_prob <= occupied_thresh (中间区域)
    is_free = occ_prob < free_thresh
    is_occupied = occ_prob > occupied_thresh
    is_unknown = ~(is_free | is_occupied)

    # 构建二值占据栅格: 0=free, 1=obstacle
    # unknown区域保守处理为obstacle
    occupancy = np.zeros_like(img_array, dtype=np.uint8)
    occupancy[is_occupied] = 1
    occupancy[is_unknown] = 1  # 保守: unknown当作障碍物

    free_count = np.sum(is_free)
    occupied_count = np.sum(is_occupied)
    unknown_count = np.sum(is_unknown)

    print(f"栅格统计:")
    print(f"  自由像素: {free_count} ({100*free_count/(width*height):.1f}%)")
    print(f"  障碍物像素: {occupied_count} ({100*occupied_count/(width*height):.1f}%)")
    print(f"  未知像素(按障碍物处理): {unknown_count} ({100*unknown_count/(width*height):.1f}%)")

    return occupancy


def main():
    parser = argparse.ArgumentParser(description='Convert ROS map (pgm+yaml) to npy format')
    parser.add_argument('--map', type=str, required=True, help='Path to .pgm file')
    parser.add_argument('--yaml', type=str, required=True, help='Path to .yaml file')
    parser.add_argument('--out-map', type=str, required=True, help='Output .npy file path')
    parser.add_argument('--out-meta', type=str, required=True, help='Output metadata .yaml path')
    parser.add_argument('--origin', type=float, nargs=3, default=[0.0, 0.0, 0.0],
                       help='Origin x, y, yaw (default: 0 0 0)')

    args = parser.parse_args()

    print("=" * 60)
    print("ROS地图转npy格式")
    print("=" * 60)
    print(f"输入PGM: {args.map}")
    print(f"输入YAML: {args.yaml}")
    print(f"输出地图: {args.out_map}")
    print(f"输出元数据: {args.out_meta}")

    # 1. 读取YAML元数据
    print("\n[步骤1] 读取YAML元数据...")
    yaml_data, geo_bounds, physical_size = load_yaml_metadata(args.yaml)
    print(f"  YAML中的resolution字段: {yaml_data.get('resolution', 'N/A')} m/pixel")
    print(f"  YAML中的origin字段: {yaml_data.get('origin', 'N/A')}")

    if geo_bounds:
        print(f"  地理范围: lon=[{geo_bounds.get('lon_min', 'N/A'):.4f}, {geo_bounds.get('lon_max', 'N/A'):.4f}]")
        print(f"           lat=[{geo_bounds.get('lat_min', 'N/A'):.4f}, {geo_bounds.get('lat_max', 'N/A'):.4f}]")

    if physical_size:
        print(f"  物理尺寸: {physical_size.get('width_m', 'N/A')} x {physical_size.get('height_m', 'N/A')} 米")

    # 2. 读取PGM图像
    print("\n[步骤2] 读取PGM图像...")
    img = Image.open(args.map)
    img_array = np.array(img)
    img_height, img_width = img_array.shape[:2]
    print(f"  实际图像尺寸: {img_width} x {img_height} (W x H)")

    # 3. 估算真实分辨率
    print("\n[步骤3] 估算真实分辨率...")
    yaml_resolution = yaml_data.get('resolution', 0.5)
    res_x, res_y, res_avg, phys_w, phys_h, deviation = estimate_resolution_from_geo(
        img_width, img_height, geo_bounds, physical_size
    )

    if res_avg is not None:
        # 判断yaml分辨率是否正确
        if yaml_resolution > 0:
            yaml_phys_w = yaml_resolution * img_width
            yaml_phys_h = yaml_resolution * img_height
            print(f"\n  YAML resolution={yaml_resolution} 对应物理尺寸: {yaml_phys_w:.1f} x {yaml_phys_h:.1f} 米")

            if physical_size:
                diff_w = abs(yaml_phys_w - physical_size['width_m']) / physical_size['width_m'] * 100
                diff_h = abs(yaml_phys_h - physical_size['height_m']) / physical_size['height_m'] * 100
                print(f"  与yaml物理尺寸偏差: W={diff_w:.1f}%, H={diff_h:.1f}%")

                if diff_w > 50 or diff_h > 50:
                    print(f"  ⚠️ 严重偏差! yaml中的resolution={yaml_resolution}可能是错误的")
                    print(f"  ✅ 采用基于物理尺寸/地理范围的估算分辨率: x={res_x:.4f}, y={res_y:.4f} m/pixel")

        # 采用估算分辨率（分开x/y）
        final_resolution_x = res_x
        final_resolution_y = res_y
        final_resolution_avg = res_avg
        final_phys_w = phys_w if phys_w else yaml_resolution * img_width
        final_phys_h = phys_h if phys_h else yaml_resolution * img_height
    else:
        # 无法估算，使用yaml中的分辨率
        print(f"\n  ⚠️ 无法从数据估算分辨率，使用yaml中的值: {yaml_resolution}")
        final_resolution_x = yaml_resolution
        final_resolution_y = yaml_resolution
        final_resolution_avg = yaml_resolution
        final_phys_w = yaml_resolution * img_width
        final_phys_h = yaml_resolution * img_height

    print(f"\n  最终采用的训练分辨率:")
    print(f"    resolution_x: {final_resolution_x:.4f} m/pixel")
    print(f"    resolution_y: {final_resolution_y:.4f} m/pixel")
    print(f"    resolution_avg: {final_resolution_avg:.4f} m/pixel")
    print(f"  最终物理尺寸: {final_phys_w:.1f} x {final_phys_h:.1f} 米")

    # 4. 转换为占据栅格
    print("\n[步骤4] 转换为二值占据栅格...")
    occupancy = convert_pgm_to_occupancy_grid(args.map, yaml_data)

    # 4.5 保存debug PNG（用于肉眼确认语义）
    print("\n[步骤4.5] 保存debug PNG...")
    debug_png_path = args.out_map.replace('.npy', '_debug.png')
    # 0=free(白), 1=obstacle(灰)
    debug_display = (1 - occupancy) * 255  # 反转: free白, obstacle暗
    debug_img = Image.fromarray(debug_display.astype(np.uint8), mode='L')
    debug_img.save(debug_png_path)
    print(f"  已保存debug PNG: {debug_png_path}")

    # 5. 保存npy文件
    print("\n[步骤5] 保存npy文件...")
    os.makedirs(os.path.dirname(args.out_map), exist_ok=True)
    np.save(args.out_map, occupancy)
    print(f"  已保存: {args.out_map}")

    # 6. 生成并保存元数据yaml
    print("\n[步骤6] 生成元数据yaml...")

    # 确定地理范围（如果可用）
    if geo_bounds:
        lon_min = geo_bounds.get('lon_min', 0)
        lat_min = geo_bounds.get('lat_min', 0)
        lon_max = geo_bounds.get('lon_max', 0)
        lat_max = geo_bounds.get('lat_max', 0)
    else:
        # 根据原点和分辨率计算（使用x分辨率作为主要参考）
        lon_min = args.origin[0]
        lat_min = args.origin[1]
        lon_max = lon_min + img_width * final_resolution_x
        lat_max = lat_min + img_height * final_resolution_y

    meta_data = {
        'image': os.path.basename(args.out_map),
        'resolution_x': float(round(final_resolution_x, 6)),
        'resolution_y': float(round(final_resolution_y, 6)),
        'resolution_avg': float(round(final_resolution_avg, 6)),
        'origin': [float(x) for x in args.origin],
        'width': int(img_width),
        'height': int(img_height),
        'physical_width': float(round(final_phys_w, 2)),
        'physical_height': float(round(final_phys_h, 2)),
        'lon_min': float(round(lon_min, 7)),
        'lat_min': float(round(lat_min, 7)),
        'lon_max': float(round(lon_max, 7)),
        'lat_max': float(round(lat_max, 7)),
        'negate': int(yaml_data.get('negate', 0)),
        'occupied_thresh': float(yaml_data.get('occupied_thresh', 0.65)),
        'free_thresh': float(yaml_data.get('free_thresh', 0.196)),
        'processed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source_pgm': os.path.basename(args.map),
        'source_yaml': os.path.basename(args.yaml),
    }

    with open(args.out_meta, 'w', encoding='utf-8') as f:
        yaml.dump(meta_data, f, allow_unicode=True, default_flow_style=False)

    print(f"  已保存: {args.out_meta}")

    # 7. 打印摘要
    print("\n" + "=" * 60)
    print("地图转换完成!")
    print("=" * 60)
    print(f"  图像尺寸: {img_width} x {img_height}")
    print(f"  训练分辨率: x={final_resolution_x:.4f}, y={final_resolution_y:.4f} m/pixel (avg={final_resolution_avg:.4f})")
    print(f"  物理尺寸: {final_phys_w:.1f} x {final_phys_h:.1f} 米")
    print(f"  地理范围: lon=[{lon_min:.4f}, {lon_max:.4f}], lat=[{lat_min:.4f}, {lat_max:.4f}]")
    print(f"  障碍物占比: {100*np.sum(occupancy)/occupancy.size:.1f}%")
    print(f"  自由区域占比: {100*(1-np.sum(occupancy)/occupancy.size):.1f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()