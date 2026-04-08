"""
Dynamic Obstacle Manager
负责加载和管理动态障碍物轨迹
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
import os


class DynamicObstacle:
    """单个动态障碍物"""

    def __init__(self, obstacle_id: int, radius: float, trajectory: np.ndarray):
        """
        Args:
            obstacle_id: 障碍物ID
            radius: 障碍物半径（米）
            trajectory: 形状为 (T, 3) 的数组，每行 [t, x, y]，t是时间戳
        """
        self.id = obstacle_id
        self.radius = radius
        self.trajectory = trajectory
        self.t_min = trajectory[0, 0]
        self.t_max = trajectory[-1, 0]
        self.current_index = 0

    def get_position_at_time(self, t: float) -> Tuple[float, float]:
        """
        获取指定时间的平滑插值位置
        使用线性插值
        """
        if t <= self.t_min:
            return self.trajectory[0, 1], self.trajectory[0, 2]
        if t >= self.t_max:
            return self.trajectory[-1, 1], self.trajectory[-1, 2]

        # 线性插值
        for i in range(len(self.trajectory) - 1):
            if self.t_min <= t <= self.t_max:
                break
            t0, x0, y0 = self.trajectory[i]
            t1, x1, y1 = self.trajectory[i + 1]
            if t0 <= t <= t1:
                alpha = (t - t0) / (t1 - t0) if (t1 - t0) > 0 else 0
                return x0 + alpha * (x1 - x0), y0 + alpha * (y1 - y0)

        # Fallback
        return self.trajectory[-1, 1], self.trajectory[-1, 2]

    def get_velocity_at_time(self, t: float) -> Tuple[float, float]:
        """获取指定时间的近似速度"""
        dt = 0.01
        x1, y1 = self.get_position_at_time(t + dt)
        x0, y0 = self.get_position_at_time(t - dt)
        vx = (x1 - x0) / (2 * dt)
        vy = (y1 - y0) / (2 * dt)
        return vx, vy

    def check_collision_circle(self, px: float, py: float, pradius: float) -> bool:
        """检查与圆形是否碰撞"""
        if self.current_position is None:
            return False
        ox, oy = self.current_position
        dist = np.sqrt((px - ox) ** 2 + (py - oy) ** 2)
        return dist < (pradius + self.radius)

    @property
    def current_position(self) -> Optional[Tuple[float, float]]:
        """获取当前位置"""
        if not hasattr(self, '_current_time'):
            return None
        return self.get_position_at_time(self._current_time)

    def update(self, t: float):
        """更新当前位置"""
        self._current_time = t
        self.current_position = self.get_position_at_time(t)


class DynamicObstacleManager:
    """动态障碍物管理器"""

    def __init__(self, dt: float = 0.1, loop: bool = True):
        """
        Args:
            dt: 时间步长（秒）
            loop: 是否循环播放轨迹
        """
        self.dt = dt
        self.loop = loop
        self.obstacles: Dict[int, DynamicObstacle] = {}
        self.current_time = 0.0
        self.max_time = 0.0

    def add_obstacle(self, obstacle: DynamicObstacle):
        """添加障碍物"""
        self.obstacles[obstacle.id] = obstacle
        self.max_time = max(self.max_time, obstacle.t_max)

    def update(self, dt: Optional[float] = None):
        """更新所有障碍物位置"""
        if dt is not None:
            self.current_time += dt
        else:
            self.current_time += self.dt

        if self.loop and self.current_time > self.max_time:
            self.current_time = self.current_time % self.max_time

        for obstacle in self.obstacles.values():
            obstacle.update(self.current_time)

    def get_obstacle_positions(self) -> Dict[int, Tuple[float, float]]:
        """获取所有障碍物当前位置"""
        return {oid: obs.current_position for oid, obs in self.obstacles.items()}

    def get_nearest_obstacles(self, px: float, py: float, k: int = 3) -> List[Tuple[int, float, float, float]]:
        """
        获取最近的k个障碍物

        Returns:
            List of (id, distance, relative_x, relative_y)
        """
        distances = []
        for oid, obs in self.obstacles.items():
            if obs.current_position:
                ox, oy = obs.current_position
                dist = np.sqrt((px - ox) ** 2 + (py - oy) ** 2)
                distances.append((oid, dist, ox - px, oy - py))

        distances.sort(key=lambda x: x[1])
        return distances[:k]

    def check_collision_circle(self, px: float, py: float, radius: float) -> bool:
        """检查与任何障碍物是否碰撞"""
        for obs in self.obstacles.values():
            if obs.check_collision_circle(px, py, radius):
                return True
        return False

    def reset(self):
        """重置到初始状态"""
        self.current_time = 0.0
        for obs in self.obstacles.values():
            obs.update(0.0)


def load_trajectory_from_json(path: str) -> DynamicObstacleManager:
    """
    从JSON文件加载轨迹

    JSON格式示例:
    {
        "dt": 0.1,
        "loop": true,
        "obstacles": [
            {
                "id": 0,
                "radius": 0.25,
                "trajectory": [
                    {"t": 0, "x": 1.0, "y": 2.0},
                    {"t": 0.1, "x": 1.1, "y": 2.0},
                    ...
                ]
            },
            {
                "id": 1,
                "radius": 0.3,
                "trajectory": [
                    {"t": 0, "x": 3.0, "y": 1.0},
                    ...
                ]
            }
        ]
    }
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trajectory file not found: {path}")

    with open(path, 'r') as f:
        data = json.load(f)

    dt = data.get('dt', 0.1)
    loop = data.get('loop', True)

    manager = DynamicObstacleManager(dt=dt, loop=loop)

    for obs_data in data.get('obstacles', []):
        obs_id = obs_data['id']
        radius = obs_data.get('radius', 0.25)
        traj_list = obs_data['trajectory']

        # 转换为numpy数组
        trajectory = np.array([[p['t'], p['x'], p['y']] for p in traj_list])

        obstacle = DynamicObstacle(obstacle_id=obs_id, radius=radius, trajectory=trajectory)
        manager.add_obstacle(obstacle)

    return manager


def create_circular_trajectory(center_x: float, center_y: float, radius: float,
                                num_points: int, period: float, dt: float = 0.1) -> List[Dict]:
    """
    创建圆形轨迹（用于测试）

    Args:
        center_x, center_y: 圆心
        radius: 半径
        num_points: 轨迹点数
        period: 周期（秒）
        dt: 时间步长
    """
    trajectory = []
    for i in range(num_points):
        t = i * dt
        angle = 2 * np.pi * t / period
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        trajectory.append({'t': t, 'x': x, 'y': y})

    return trajectory


def create_linear_trajectory(start_x: float, start_y: float, end_x: float, end_y: float,
                              duration: float, dt: float = 0.1) -> List[Dict]:
    """创建线性轨迹"""
    num_points = int(duration / dt) + 1
    trajectory = []
    for i in range(num_points):
        t = i * dt
        alpha = min(t / duration, 1.0)
        x = start_x + alpha * (end_x - start_x)
        y = start_y + alpha * (end_y - start_y)
        trajectory.append({'t': t, 'x': x, 'y': y})

    return trajectory


def create_figure8_trajectory(center_x: float, center_y: float, a: float,
                               period: float, dt: float = 0.1) -> List[Dict]:
    """创建8字形轨迹"""
    trajectory = []
    t = 0.0
    while t < period:
        x = center_x + a * np.sin(2 * np.pi * t / period)
        y = center_y + a * np.sin(4 * np.pi * t / period) / 2
        trajectory.append({'t': t, 'x': x, 'y': y})
        t += dt
    return trajectory


def create_sample_trajectory_file(output_path: str):
    """创建样例轨迹文件"""
    data = {
        "dt": 0.1,
        "loop": True,
        "obstacles": [
            {
                "id": 0,
                "radius": 0.25,
                "trajectory": create_circular_trajectory(0, 0, 2, 100, 10, 0.1)
            },
            {
                "id": 1,
                "radius": 0.3,
                "trajectory": create_linear_trajectory(-3, -3, 3, 3, 10, 0.1)
            },
            {
                "id": 2,
                "radius": 0.2,
                "trajectory": create_figure8_trajectory(0, 0, 1.5, 8, 0.1)
            }
        ]
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Created sample trajectory: {output_path}")


if __name__ == "__main__":
    print("Testing DynamicObstacleManager...")

    # 创建样例轨迹
    os.makedirs("example_data", exist_ok=True)
    create_sample_trajectory_file("example_data/sample_trajectories.json")

    # 加载并测试
    manager = load_trajectory_from_json("example_data/sample_trajectories.json")
    print(f"Loaded manager with {len(manager.obstacles)} obstacles")

    # 模拟更新
    for i in range(50):
        manager.update()
        positions = manager.get_obstacle_positions()
        if i % 10 == 0:
            print(f"t={manager.current_time:.1f}: positions={positions}")

    # 测试最近障碍物
    nearest = manager.get_nearest_obstacles(0, 0, k=3)
    print(f"Nearest to (0,0): {nearest}")
