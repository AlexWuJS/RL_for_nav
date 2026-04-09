"""
Grid Dynamic Obstacle Environment
纯Python实现的Gymnasium环境，基于栅格地图和动态障碍物轨迹进行训练
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from grid_map_loader import GridMap, load_npy_map_with_meta, create_simple_map
from dynamic_obstacle_manager import DynamicObstacleManager, load_trajectory_from_json


class GridDynamicObstacleEnv(gym.Env):
    """
    基于栅格地图和动态障碍物轨迹的强化学习环境
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}

    def __init__(
        self,
        map_path: Optional[str] = None,
        trajectory_path: Optional[str] = None,
        # 地图参数
        map_resolution: float = 0.1,
        # 机器人参数
        robot_radius: float = 0.15,
        v_max: float = 1.0,
        w_max: float = 1.0,
        # 动态障碍物参数
        dynamic_obstacle_radius: float = 0.25,
        # 观测参数
        use_local_patch: bool = True,
        patch_size: int = 21,
        include_dynamic_in_patch: bool = True,
        include_nearest_dynamic: bool = True,
        nearest_dynamic_k: int = 3,
        # 仿真参数
        dt: float = 0.1,
        max_episode_steps: int = 500,
        # 奖励参数
        goal_reward: float = 100.0,
        collision_penalty: float = -100.0,
        step_penalty: float = -0.1,
        progress_weight: float = 1.0,
        safe_distance: float = 0.5,
        safe_distance_penalty: float = -5.0,
        # 其他
        render_mode: str = 'human',
        seed: Optional[int] = None,
    ):
        super().__init__()

        # 存储参数
        self.map_path = map_path
        self.trajectory_path = trajectory_path
        self.map_resolution = map_resolution
        self.robot_radius = robot_radius
        self.v_max = v_max
        self.w_max = w_max
        self.dynamic_obstacle_radius = dynamic_obstacle_radius
        self.use_local_patch = use_local_patch
        self.patch_size = patch_size
        self.include_dynamic_in_patch = include_dynamic_in_patch
        self.include_nearest_dynamic = include_nearest_dynamic
        self.nearest_dynamic_k = nearest_dynamic_k
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.goal_reward = goal_reward
        self.collision_penalty = collision_penalty
        self.step_penalty = step_penalty
        self.progress_weight = progress_weight
        self.safe_distance = safe_distance
        self.safe_distance_penalty = safe_distance_penalty
        self.render_mode = render_mode

        # 加载地图
        if map_path:
            self.grid_map = load_npy_map_with_meta(map_path)
        else:
            # 创建默认测试地图
            self.grid_map = create_simple_map(100, 100, resolution=map_resolution)

        # 加载动态障碍物轨迹
        if trajectory_path:
            self.obstacle_manager = load_trajectory_from_json(trajectory_path)
        else:
            self.obstacle_manager = None

        # 机器人状态
        self.robot_pos = np.zeros(2)  # [x, y]
        self.robot_yaw = 0.0
        self.robot_v = 0.0
        self.robot_w = 0.0

        # 目标点
        self.goal_pos = np.zeros(2)

        # 步数计数
        self.step_count = 0
        self.last_progress = 0.0

        # 轨迹记录（用于可视化）
        self.position_history = []

        # 设置随机种子
        self._np_random = np.random.RandomState(seed)

        # 定义动作空间: [v, w]
        self.action_space = spaces.Box(
            low=np.array([0.0, -self.w_max]),
            high=np.array([self.v_max, self.w_max]),
            dtype=np.float32
        )

        # 计算观测空间维度
        obs_dim = self._compute_observation_dim()

        # 观测空间
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # 渲染器
        self._renderer = None

    def _compute_observation_dim(self) -> int:
        """计算观测向量维度"""
        dim = 0

        # 1. 局部栅格patch
        if self.use_local_patch:
            dim += self.patch_size * self.patch_size

        # 2. 目标相对信息: [distance, relative_heading]
        dim += 2

        # 3. 机器人自身状态: [v, w]
        dim += 2

        # 4. 最近动态障碍物信息: k * [rel_x, rel_y, rel_vx, rel_vy]
        if self.include_nearest_dynamic:
            dim += self.nearest_dynamic_k * 4

        return dim

    def _get_observation(self) -> np.ndarray:
        """构建观测向量"""
        obs_list = []

        # 1. 局部栅格patch
        if self.use_local_patch:
            patch, _, _ = self.grid_map.get_sub_patch(
                self.robot_pos[0], self.robot_pos[1], self.patch_size
            )

            # 将动态障碍物位置标记到patch上
            if self.include_dynamic_in_patch and self.obstacle_manager:
                for obs_id, (ox, oy) in self.obstacle_manager.get_obstacle_positions().items():
                    gx, gy = self.grid_map.world_to_grid(ox, oy)
                    # 转换为patch局部坐标
                    patch_gx, patch_gy = self.grid_map.world_to_grid(self.robot_pos[0], self.robot_pos[1])
                    local_x = gx - (patch_gx - self.patch_size // 2)
                    local_y = gy - (patch_gy - self.patch_size // 2)
                    if 0 <= local_x < self.patch_size and 0 <= local_y < self.patch_size:
                        patch[local_y, local_x] = 2  # 标记为动态障碍物

            obs_list.append(patch.flatten() / 2.0)  # 归一化到 [-0.5, 0.5] 或 [0, 0.5]
        else:
            # 全局地图作为观测（展平）
            obs_list.append(self.grid_map.occupancy.flatten() / 2.0)

        # 2. 目标相对信息
        dx = self.goal_pos[0] - self.robot_pos[0]
        dy = self.goal_pos[1] - self.robot_pos[1]
        goal_distance = np.sqrt(dx * dx + dy * dy)
        goal_heading = np.arctan2(dy, dx) - self.robot_yaw
        # 归一化到 [-1, 1]
        goal_distance_norm = np.clip(goal_distance / 20.0, -1, 1)
        goal_heading_norm = np.clip(goal_heading / np.pi, -1, 1)
        obs_list.append(np.array([goal_distance_norm, goal_heading_norm], dtype=np.float32))

        # 3. 机器人自身状态
        v_norm = self.robot_v / self.v_max
        w_norm = self.robot_w / self.w_max
        obs_list.append(np.array([v_norm, w_norm], dtype=np.float32))

        # 4. 最近动态障碍物信息
        if self.include_nearest_dynamic and self.obstacle_manager:
            nearest = self.obstacle_manager.get_nearest_obstacles(
                self.robot_pos[0], self.robot_pos[1], self.nearest_dynamic_k
            )
            nearest_features = []
            for i in range(self.nearest_dynamic_k):
                if i < len(nearest):
                    _, dist, rel_x, rel_y = nearest[i]
                    # 近似速度
                    vx, vy = 0, 0  # 简化：暂不提供速度
                    nearest_features.extend([
                        np.clip(rel_x / 10.0, -1, 1),
                        np.clip(rel_y / 10.0, -1, 1),
                        np.clip(dist / 10.0, -1, 1),
                        0.0  # 占位
                    ])
                else:
                    nearest_features.extend([0.0, 0.0, 0.0, 0.0])
            obs_list.append(np.array(nearest_features, dtype=np.float32))

        return np.concatenate(obs_list).astype(np.float32)

    def _check_static_collision(self) -> bool:
        """检查与静态障碍物的碰撞"""
        return self.grid_map.check_collision_circle(
            self.robot_pos[0], self.robot_pos[1], self.robot_radius
        )

    def _check_dynamic_collision(self) -> bool:
        """检查与动态障碍物的碰撞"""
        if self.obstacle_manager is None:
            return False
        return self.obstacle_manager.check_collision_circle(
            self.robot_pos[0], self.robot_pos[1], self.robot_radius
        )

    def _check_goal_reached(self) -> bool:
        """检查是否到达目标"""
        dist = np.linalg.norm(self.robot_pos - self.goal_pos)
        return dist < (self.robot_radius + 0.2)  # 目标到达阈值

    def _check_out_of_bounds_xy(self, x: float, y: float) -> bool:
        """检查指定位置是否超出边界"""
        gx, gy = self.grid_map.world_to_grid(x, y)
        return not (0 <= gx < self.grid_map.width and 0 <= gy < self.grid_map.height)

    def _check_out_of_bounds(self) -> bool:
        """检查机器人当前位置是否超出边界"""
        return self._check_out_of_bounds_xy(self.robot_pos[0], self.robot_pos[1])

    def _is_valid_position(self, x: float, y: float) -> bool:
        """检查位置是否有效（可通行且在地图内）"""
        if self._check_out_of_bounds_xy(x, y):
            return False
        return not self.grid_map.check_collision_circle(x, y, self.robot_radius)

    def _sample_free_position(self) -> np.ndarray:
        """在自由空间中随机采样位置"""
        while True:
            free_points = self.grid_map.get_free_world_points(num_samples=1000)
            if len(free_points) == 0:
                # fallback: 返回地图中心
                return np.array([0.0, 0.0])
            idx = self._np_random.randint(len(free_points))
            pos = free_points[idx]
            # 确保与动态障碍物初始位置不冲突
            if self.obstacle_manager:
                if not self.obstacle_manager.check_collision_circle(pos[0], pos[1], self.robot_radius + self.dynamic_obstacle_radius):
                    return pos
            else:
                return pos

    def _sample_goal_position(self, min_dist: float = 3.0, max_attempts: int = 100) -> np.ndarray:
        """采样目标位置"""
        for _ in range(max_attempts):
            goal = self._sample_free_position()
            if np.linalg.norm(goal - self.robot_pos) >= min_dist:
                return goal
        # fallback: 返回远离机器人的随机位置
        direction = self._np_random.rand(2) - 0.5
        direction = direction / np.linalg.norm(direction) * 5.0
        return self.robot_pos + direction

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        if seed is not None:
            self._np_random = np.random.RandomState(seed)

        self.step_count = 0
        self.last_progress = 0.0

        # 重置机器人位置（先采样，再记录历史）
        self.robot_pos = self._sample_free_position()
        self.position_history = [self.robot_pos.copy()]
        self.robot_yaw = self._np_random.uniform(-np.pi, np.pi)
        self.robot_v = 0.0
        self.robot_w = 0.0

        # 重置目标位置
        self.goal_pos = self._sample_goal_position()

        # 重置动态障碍物
        if self.obstacle_manager:
            self.obstacle_manager.reset()

        obs = self._get_observation()
        info = {
            'goal_pos': self.goal_pos.copy(),
            'robot_pos': self.robot_pos.copy(),
            'map_size': (self.grid_map.width, self.grid_map.height)
        }

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步"""
        # 解析动作
        v_desired = float(np.clip(action[0], 0, self.v_max))
        w_desired = float(np.clip(action[1], -self.w_max, self.w_max))

        # 简单一阶跟踪
        self.robot_v = v_desired
        self.robot_w = w_desired

        # 运动学积分计算新位置
        dx = self.robot_v * np.cos(self.robot_yaw) * self.dt
        dy = self.robot_v * np.sin(self.robot_yaw) * self.dt
        new_pos = self.robot_pos + np.array([dx, dy])

        # 检查新位置是否有效（基于new_pos判断，不是self.robot_pos）
        out_of_bounds = self._check_out_of_bounds_xy(new_pos[0], new_pos[1])

        # 临时更新位置来检查碰撞
        old_pos = self.robot_pos.copy()
        self.robot_pos = new_pos
        collision = self._check_static_collision() or self._check_dynamic_collision()
        self.robot_pos = old_pos  # 恢复，等会再决定是否真的更新

        # 如果新位置有效（在边界内且无碰撞）才更新
        if not out_of_bounds and not collision:
            self.robot_pos = new_pos
        # 否则停在原地（不更新position_history，保持原位置）

        # 更新朝向
        self.robot_yaw += self.robot_w * self.dt
        # 归一化朝向到 [-pi, pi]
        while self.robot_yaw > np.pi:
            self.robot_yaw -= 2 * np.pi
        while self.robot_yaw < -np.pi:
            self.robot_yaw += 2 * np.pi

        # 更新动态障碍物
        if self.obstacle_manager:
            self.obstacle_manager.update(self.dt)

        # 记录轨迹
        self.position_history.append(self.robot_pos.copy())

        # 更新步数
        self.step_count += 1

        # 计算奖励
        reward = self._compute_reward()

        # 检查终止条件
        terminated = False
        truncated = False
        collision_occurred = False
        goal_reached = False
        out_of_bounds_occurred = False

        # 碰撞终止（用new_pos判断后的实际位置）
        collision_occurred = self._check_static_collision() or self._check_dynamic_collision()
        if collision_occurred:
            terminated = True

        # 目标到达终止
        goal_reached = self._check_goal_reached()
        if goal_reached:
            terminated = True

        # 超出边界终止（基于new_pos判断）
        if out_of_bounds:
            out_of_bounds_occurred = True
            terminated = True

        # 超时截断
        if self.step_count >= self.max_episode_steps:
            truncated = True

        obs = self._get_observation()
        info = {
            'collision': collision_occurred,
            'goal_reached': goal_reached,
            'out_of_bounds': out_of_bounds_occurred,
            'timeout': self.step_count >= self.max_episode_steps,
            'step_count': self.step_count,
            'robot_pos': self.robot_pos.copy(),
            'goal_pos': self.goal_pos.copy(),
        }

        return obs, reward, terminated, truncated, info

    def _compute_reward(self) -> float:
        """计算奖励"""
        reward = 0.0

        # 1. 步长时间惩罚
        reward += self.step_penalty

        # 2. 目标进度奖励
        current_progress = np.linalg.norm(self.robot_pos - self.goal_pos)
        if self.last_progress > 0:
            progress_delta = self.last_progress - current_progress
            reward += progress_delta * self.progress_weight
        self.last_progress = current_progress

        # 3. 目标到达奖励
        if self._check_goal_reached():
            reward += self.goal_reward

        # 4. 碰撞惩罚
        if self._check_static_collision() or self._check_dynamic_collision():
            reward += self.collision_penalty

        # 5. 安全距离惩罚（接近动态障碍物）
        if self.obstacle_manager:
            for obs_id, (ox, oy) in self.obstacle_manager.get_obstacle_positions().items():
                dist = np.linalg.norm(self.robot_pos - np.array([ox, oy]))
                if dist < self.safe_distance:
                    penalty = (self.safe_distance - dist) / self.safe_distance * self.safe_distance_penalty
                    reward += penalty

        return reward

    def render(self):
        """渲染环境"""
        if self.render_mode == 'human':
            try:
                from grid_render import SimpleRenderer
                if self._renderer is None:
                    self._renderer = SimpleRenderer(self)
                self._renderer.render()
            except ImportError:
                print("Warning: grid_render not available. Using numpy render.")
                self._render_numpy()

    def _render_numpy(self):
        """简单的numpy方式渲染"""
        occ = self.grid_map.occupancy.copy().astype(float)
        # 缩放显示
        scale = 5
        large_occ = np.kron(occ, np.ones((scale, scale)))

        # 叠加机器人位置
        rx, ry = self.grid_map.world_to_grid(self.robot_pos[0], self.robot_pos[1])
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if 0 <= rx + dx < large_occ.shape[1] and 0 <= ry + dy < large_occ.shape[0]:
                    if dx * dx + dy * dy <= 4:
                        large_occ[ry + dy, rx + dx] = 0.7  # 机器人

        # 叠加目标
        gx, gy = self.grid_map.world_to_grid(self.goal_pos[0], self.goal_pos[1])
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if 0 <= gx + dx < large_occ.shape[1] and 0 <= gy + dy < large_occ.shape[0]:
                    if dx * dx + dy * dy <= 4:
                        large_occ[gy + dy, gx + dx] = 0.3  # 目标

        print(large_occ)

    def close(self):
        """关闭环境"""
        if self._renderer:
            self._renderer.close()
            self._renderer = None

    def get_state_dict(self) -> Dict[str, Any]:
        """获取当前状态（用于保存回放）"""
        return {
            'robot_pos': self.robot_pos.copy(),
            'robot_yaw': self.robot_yaw,
            'robot_v': self.robot_v,
            'robot_w': self.robot_w,
            'goal_pos': self.goal_pos.copy(),
            'step_count': self.step_count,
            'position_history': [p.copy() for p in self.position_history],
            'obstacle_positions': self.obstacle_manager.get_obstacle_positions() if self.obstacle_manager else {}
        }


def make_grid_env(env_id: str = "GridDynamicObstacle-v0", **kwargs) -> GridDynamicObstacleEnv:
    """创建环境的工厂函数"""
    return GridDynamicObstacleEnv(**kwargs)


if __name__ == "__main__":
    # 测试代码
    print("Testing GridDynamicObstacleEnv...")

    # 创建环境
    env = GridDynamicObstacleEnv(
        map_path=None,  # 使用默认测试地图
        trajectory_path=None,  # 使用无动态障碍
        use_local_patch=True,
        patch_size=21,
        render_mode='human',
        seed=42
    )

    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # 测试reset/step
    obs, info = env.reset()
    print(f"Reset obs shape: {obs.shape}")

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.2f}, terminated={terminated}, truncated={truncated}")
        if terminated or truncated:
            break

    print("Environment test passed!")
    env.close()
