"""
Grid Map Loader
负责加载栅格地图，支持多种格式，提供世界坐标与栅格坐标互转
"""

import numpy as np
import yaml
import os
from typing import Tuple, Optional


class GridMap:
    """栅格地图类"""

    def __init__(self, occupancy: np.ndarray, resolution: float, origin: Tuple[float, float] = (0.0, 0.0)):
        """
        Args:
            occupancy: 2D numpy array, 0=free, 1=obstacle
            resolution: 米/格
            origin: 地图原点在世界坐标系中的位置 (x, y)
        """
        self.occupancy = occupancy.astype(np.uint8)
        self.resolution = resolution
        self.origin = np.array(origin)
        self.height, self.width = occupancy.shape

    def is_free(self, grid_x: int, grid_y: int) -> bool:
        """检查栅格坐标是否可通行"""
        if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
            return self.occupancy[grid_y, grid_x] == 0
        return False

    def is_free_world(self, world_x: float, world_y: float) -> bool:
        """检查世界坐标是否可通行"""
        gx, gy = self.world_to_grid(world_x, world_y)
        return self.is_free(gx, gy)

    def world_to_grid(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """世界坐标转栅格坐标"""
        dx = (world_x - self.origin[0]) / self.resolution
        dy = (world_y - self.origin[1]) / self.resolution
        return int(round(dx)), int(round(dy))

    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """栅格坐标转世界坐标"""
        wx = grid_x * self.resolution + self.origin[0]
        wy = grid_y * self.resolution + self.origin[1]
        return wx, wy

    def get_free_cells(self) -> np.ndarray:
        """获取所有自由栅格坐标，形状为 (N, 2)"""
        free_y, free_x = np.where(self.occupancy == 0)
        return np.stack([free_x, free_y], axis=1)

    def get_free_world_points(self, num_samples: Optional[int] = None) -> np.ndarray:
        """获取随机采样的自由空间世界坐标点"""
        free_cells = self.get_free_cells()
        if num_samples is not None and num_samples < len(free_cells):
            indices = np.random.choice(len(free_cells), num_samples, replace=False)
            free_cells = free_cells[indices]
        world_points = np.array([self.grid_to_world(gx, gy) for gx, gy in free_cells])
        return world_points

    def check_collision_circle(self, world_x: float, world_y: float, radius: float) -> bool:
        """检查圆形是否与障碍物碰撞"""
        # 获取圆形覆盖的栅格范围
        gx_center, gy_center = self.world_to_grid(world_x, world_y)
        radius_in_cells = int(np.ceil(radius / self.resolution))

        for dx in range(-radius_in_cells, radius_in_cells + 1):
            for dy in range(-radius_in_cells, radius_in_cells + 1):
                # 检查圆形内接矩形
                if dx * dx + dy * dy <= radius_in_cells * radius_in_cells:
                    gx, gy = gx_center + dx, gy_center + dy
                    if not self.is_free(gx, gy):
                        return True
        return False

    def get_sub_patch(self, center_world_x: float, center_world_y: float,
                      patch_size: int) -> Tuple[np.ndarray, int, int]:
        """
        获取以世界坐标为中心的局部栅格patch

        Returns:
            patch: 2D array, 0=free, 1=obstacle, 2=dynamic_obstacle
            start_grid_x, start_grid_y: patch左上角对应的栅格坐标
        """
        center_gx, center_gy = self.world_to_grid(center_world_x, center_world_y)
        half_size = patch_size // 2

        start_gx = center_gx - half_size
        start_gy = center_gy - half_size

        patch = np.zeros((patch_size, patch_size), dtype=np.uint8)

        for dy in range(patch_size):
            for dx in range(patch_size):
                gx = start_gx + dx
                gy = start_gy + dy
                if 0 <= gx < self.width and 0 <= gy < self.height:
                    patch[dy, dx] = self.occupancy[gy, gx]
                else:
                    # 超出地图范围视为障碍
                    patch[dy, dx] = 1

        return patch, start_gx, start_gy

    def __repr__(self) -> str:
        return f"GridMap(shape={self.occupancy.shape}, resolution={self.resolution}, origin={self.origin})"


def load_npy_map(path: str) -> GridMap:
    """从 .npy 文件加载地图"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Map file not found: {path}")

    data = np.load(path)
    if len(data.shape) != 2:
        raise ValueError(f"Map must be 2D array, got shape {data.shape}")

    return GridMap(occupancy=data, resolution=0.1, origin=(0.0, 0.0))


def load_npy_map_with_meta(map_path: str, yaml_path: Optional[str] = None) -> GridMap:
    """
    从 npy + yaml 元数据加载地图
    yaml 格式示例:
        resolution: 0.1
        origin: [0.0, 0.0]
        negate: 0
        occupied_thresh: 0.65
        free_thresh: 0.196
    """
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"Map file not found: {map_path}")

    occupancy = np.load(map_path)

    if yaml_path and os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            meta = yaml.safe_load(f)
        resolution = meta.get('resolution', 0.1)
        origin = meta.get('origin', [0.0, 0.0])
        if isinstance(origin, list):
            origin = tuple(origin)
        else:
            origin = (0.0, 0.0)
    else:
        # 默认参数
        resolution = 0.1
        origin = (0.0, 0.0)

    return GridMap(occupancy=occupancy, resolution=resolution, origin=origin)


def create_simple_map(width: int, height: int, resolution: float = 0.1) -> GridMap:
    """
    创建一个简单的空地图（用于测试）
    四周有边界墙
    """
    occupancy = np.zeros((height, width), dtype=np.uint8)

    # 边界墙
    occupancy[0, :] = 1
    occupancy[-1, :] = 1
    occupancy[:, 0] = 1
    occupancy[:, -1] = 1

    origin = (-width * resolution / 2, -height * resolution / 2)
    return GridMap(occupancy=occupancy, resolution=resolution, origin=origin)


def create_map_with_obstacles(width: int, height: int, resolution: float = 0.1,
                               obstacle_positions: list = None) -> GridMap:
    """
    创建带静态障碍物的地图

    Args:
        width, height: 栅格尺寸
        resolution: 米/格
        obstacle_positions: List of (x, y, radius) in world coordinates
    """
    occupancy = np.zeros((height, width), dtype=np.uint8)

    # 边界墙
    occupancy[0, :] = 1
    occupancy[-1, :] = 1
    occupancy[:, 0] = 1
    occupancy[:, -1] = 1

    origin = (-width * resolution / 2, -height * resolution / 2)
    grid_map = GridMap(occupancy=occupancy, resolution=resolution, origin=origin)

    # 添加障碍物
    if obstacle_positions:
        for obs_x, obs_y, obs_radius in obstacle_positions:
            # 将圆形障碍物填充到栅格
            gx_center, gy_center = grid_map.world_to_grid(obs_x, obs_y)
            radius_cells = int(np.ceil(obs_radius / resolution))

            for dy in range(-radius_cells, radius_cells + 1):
                for dx in range(-radius_cells, radius_cells + 1):
                    if dx * dx + dy * dy <= radius_cells * radius_cells:
                        gx, gy = gx_center + dx, gy_center + dy
                        if 0 <= gx < width and 0 <= gy < height:
                            occupancy[gy, gx] = 1

    grid_map.occupancy = occupancy
    return grid_map


if __name__ == "__main__":
    # 测试代码
    print("Testing GridMap...")

    # 创建简单测试地图
    test_map = create_simple_map(100, 100, resolution=0.1)
    print(f"Created test map: {test_map}")

    # 测试坐标转换
    wx, wy = test_map.grid_to_world(50, 50)
    gx, gy = test_map.world_to_grid(wx, wy)
    print(f"World (50,50) -> Grid ({gx},{gy}) -> World ({wx},{wy})")

    # 创建带障碍物地图
    obstacles = [(0, 0, 0.5), (2, 0, 0.3)]
    obs_map = create_map_with_obstacles(100, 100, 0.1, obstacles)
    print(f"Created map with obstacles: {obs_map}")

    # 保存测试地图
    os.makedirs("example_data", exist_ok=True)
    np.save("example_data/sample_map.npy", obs_map.occupancy)
    print("Saved sample map to example_data/sample_map.npy")
