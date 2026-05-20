import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class SharedRewardConfig:
    """Unified reward parameters shared between env and MPPI rollout."""
    lateral_deadband: float = 0.5
    lateral_penalty_slope: float = 1.5
    lateral_penalty_cap: float = 5.0
    heading_penalty_scale: float = 1.5
    progress_reward_scale: float = 30.0
    frenet_amplification: float = 2.5
    safe_distance: float = 0.45
    obstacle_penalty_k: float = 5.0
    obstacle_penalty_cap: float = 10.0
    smoothness_surge_weight: float = 0.2
    smoothness_yaw_weight: float = 0.1
    alive_penalty: float = 0.1
    collision_reward: float = -100.0
    out_of_bounds_reward: float = -100.0
    success_reward: float = 1000.0
    goal_reach_threshold: float = 0.4
    max_frenet_d: float = 3.0


def compute_shared_reward(
    delta_s: float,
    frenet_d: float,
    heading_error: float,
    min_laser_dist: float,
    prev_action: np.ndarray,
    current_action: np.ndarray,
    config: Optional[SharedRewardConfig] = None,
) -> float:
    """Unified per-step reward shared by env and MPPI rollout.

    Returns a single float representing the instant reward for the current
    control step.  Terminal rewards (collision / success / out-of-bounds)
    are NOT included — callers handle those separately.
    """
    cfg = config or SharedRewardConfig()

    # --- Frenet path-tracking ---
    abs_d = abs(frenet_d)
    if abs_d <= cfg.lateral_deadband:
        lateral_penalty = 0.0
    else:
        lateral_penalty = max(-(abs_d - cfg.lateral_deadband) * cfg.lateral_penalty_slope,
                              -cfg.lateral_penalty_cap)
    heading_penalty = -(abs(heading_error) / math.pi) * cfg.heading_penalty_scale
    frenet_total = (delta_s * cfg.progress_reward_scale
                    + lateral_penalty
                    + heading_penalty) * cfg.frenet_amplification

    # --- Obstacle avoidance ---
    obstacle_penalty = 0.0
    if min_laser_dist < cfg.safe_distance:
        penalty = math.exp(cfg.obstacle_penalty_k * (cfg.safe_distance - min_laser_dist)) - 1.0
        obstacle_penalty = -min(penalty, cfg.obstacle_penalty_cap)

    # --- Action smoothness ---
    prev = np.asarray(prev_action, dtype=float).reshape(-1)[:2]
    curr = np.asarray(current_action, dtype=float).reshape(-1)[:2]
    surge_change = abs(curr[0] - prev[0])
    yaw_change = abs(curr[1] - prev[1])
    smoothness_penalty = -(surge_change * cfg.smoothness_surge_weight
                           + yaw_change * cfg.smoothness_yaw_weight)

    # --- Alive step ---
    alive_penalty = -cfg.alive_penalty

    return float(frenet_total + obstacle_penalty + smoothness_penalty + alive_penalty)


def frenet_reward(delta_s: float, frenet_d: float, heading_error: float) -> dict:
    """Legacy Frenet-only reward (kept for backward compat)."""
    components = {}
    components['s_progress'] = delta_s * 30.0
    abs_d = abs(frenet_d)
    if abs_d <= 0.5:
        raw_lateral_penalty = 0.0
    else:
        raw_lateral_penalty = -(abs_d - 0.5) * 1.5
    components['lateral_deviation'] = max(raw_lateral_penalty, -5.0)
    heading_normalized = abs(heading_error) / math.pi
    components['heading_penalty'] = -heading_normalized * 1.5
    reward = sum(components.values())
    return {'total': reward, 'components': components}


class FrenetTransform:
    """
    改进的曲线 Frenet 坐标系转换类
    内部使用二次贝塞尔曲线将起点和终点连接为平滑曲线，并进行离散化处理。
    """

    def __init__(self, start_point: np.ndarray, end_point: np.ndarray, curve_offset: float = 2.0, num_waypoints: int = 500):
        """
        初始化曲线 Frenet 坐标系

        Args:
            start_point: 路径起点 [x, y]
            end_point: 路径终点 [x, y]
            curve_offset: 曲线偏离直线的程度（正数向左弯，负数向右弯）
            num_waypoints: 离散化路点的数量
        """
        self.start_point = np.array(start_point, dtype=float)
        self.end_point = np.array(end_point, dtype=float)
        self.num_waypoints = num_waypoints

        mid_point = (self.start_point + self.end_point) / 2.0
        line_vec = self.end_point - self.start_point
        normal_vec = np.array([-line_vec[1], line_vec[0]])
        normal_vec = normal_vec / (np.linalg.norm(normal_vec) + 1e-6)
        control_point = mid_point + normal_vec * curve_offset

        t = np.linspace(0, 1, num_waypoints)[:, np.newaxis]
        self.waypoints = (1 - t)**2 * self.start_point + 2 * (1 - t) * t * control_point + t**2 * self.end_point

        self.s_values = np.zeros(num_waypoints)
        self.tangents = np.zeros_like(self.waypoints)
        self.normals = np.zeros_like(self.waypoints)
        self.path_angles = np.zeros(num_waypoints)

        diffs = np.diff(self.waypoints, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        self.s_values[1:] = np.cumsum(segment_lengths)
        self.path_length = self.s_values[-1]

        for i in range(num_waypoints - 1):
            self.tangents[i] = diffs[i] / (segment_lengths[i] + 1e-6)
            self.normals[i] = np.array([-self.tangents[i][1], self.tangents[i][0]])
            self.path_angles[i] = math.atan2(self.tangents[i][1], self.tangents[i][0])
        self.tangents[-1] = self.tangents[-2]
        self.normals[-1] = self.normals[-2]
        self.path_angles[-1] = self.path_angles[-2]

    def cartesian_to_frenet(self, point: np.ndarray) -> Tuple[float, float]:
        point = np.array(point)
        dists = np.linalg.norm(self.waypoints - point, axis=1)
        closest_idx = np.argmin(dists)
        closest_wp = self.waypoints[closest_idx]
        local_tangent = self.tangents[closest_idx]
        local_normal = self.normals[closest_idx]
        local_s = self.s_values[closest_idx]
        local_vec = point - closest_wp
        s = local_s + np.dot(local_vec, local_tangent)
        d = np.dot(local_vec, local_normal)
        return s, d

    def frenet_to_cartesian(self, s: float, d: float) -> np.ndarray:
        s = np.clip(s, 0, self.path_length)
        idx = np.searchsorted(self.s_values, s)
        if idx == 0:
            idx = 1
        elif idx == self.num_waypoints:
            idx = self.num_waypoints - 1
        s0, s1 = self.s_values[idx-1], self.s_values[idx]
        ratio = (s - s0) / (s1 - s0 + 1e-6)
        wp = self.waypoints[idx-1] + ratio * (self.waypoints[idx] - self.waypoints[idx-1])
        normal = self.normals[idx-1] + ratio * (self.normals[idx] - self.normals[idx-1])
        normal = normal / (np.linalg.norm(normal) + 1e-6)
        cartesian_point = wp + d * normal
        return cartesian_point

    def generate_path_points(self, num_points: int = None) -> np.ndarray:
        if num_points is None or num_points == self.num_waypoints:
            return self.waypoints
        indices = np.linspace(0, self.num_waypoints - 1, num_points, dtype=int)
        return self.waypoints[indices]

    def get_heading_error(self, robot_yaw: float, s: float) -> float:
        s = np.clip(s, 0, self.path_length)
        idx = np.searchsorted(self.s_values, s)
        idx = min(idx, self.num_waypoints - 1)
        local_path_angle = self.path_angles[idx]
        error = local_path_angle - robot_yaw
        while error > math.pi: error -= 2 * math.pi
        while error < -math.pi: error += 2 * math.pi
        return error
