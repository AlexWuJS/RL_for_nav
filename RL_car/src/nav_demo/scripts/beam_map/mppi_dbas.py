import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class MPPIDBaSConfig:
    """Configuration for the MPPI-DBaS action optimizer."""

    num_samples: int = 128
    horizon: int = 15
    lambda_: float = 1.0
    safe_distance: float = 0.55
    collision_distance: float = 0.25
    base_noise_std: Tuple[float, float] = (0.25, 0.25)
    min_noise_scale: float = 0.4
    max_noise_scale: float = 2.5
    action_low: Tuple[float, float] = (-1.0, -1.0)
    action_high: Tuple[float, float] = (2.0, 1.0)
    goal_weight: float = 1.2
    progress_weight: float = 2.0
    lateral_weight: float = 2.0
    heading_weight: float = 0.8
    obstacle_weight: float = 8.0
    dbas_weight: float = 12.0
    control_weight: float = 0.08
    smoothness_weight: float = 0.15
    seed: Optional[int] = None


class MPPIDBaSOptimizer:
    """MPPI optimizer with a DBaS-style safety cost and adaptive exploration."""

    def __init__(self, config: Optional[MPPIDBaSConfig] = None):
        self.config = config or MPPIDBaSConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.last_action = np.zeros(2, dtype=np.float32)
        self.last_noise_scale = 1.0

    def reset(self) -> None:
        self.last_action = np.zeros(2, dtype=np.float32)
        self.last_noise_scale = 1.0

    def optimize(self, base_action: Any, planner_state: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, float]]:
        cfg = self.config
        base_action = self._clip_action(base_action)
        obstacles = self._scan_to_obstacle_points(planner_state)
        current_min_dist = self._min_obstacle_distance(np.asarray(planner_state["position"], dtype=float), obstacles, planner_state)
        noise_scale = self._adaptive_noise_scale(current_min_dist)
        noise_std = np.asarray(cfg.base_noise_std, dtype=float) * noise_scale

        noise = self.rng.normal(
            loc=0.0,
            scale=noise_std,
            size=(cfg.num_samples, cfg.horizon, 2),
        )
        action_sequences = np.clip(
            base_action.reshape(1, 1, 2) + noise,
            np.asarray(cfg.action_low, dtype=float),
            np.asarray(cfg.action_high, dtype=float),
        )

        costs = np.zeros(cfg.num_samples, dtype=float)
        min_distances = np.full(cfg.num_samples, float(planner_state.get("max_laser_range", 10.0)), dtype=float)
        dbas_costs = np.zeros(cfg.num_samples, dtype=float)

        for sample_idx in range(cfg.num_samples):
            sample_cost, sample_min_dist, sample_dbas_cost = self._rollout_cost(
                action_sequences[sample_idx],
                base_action,
                planner_state,
                obstacles,
            )
            costs[sample_idx] = sample_cost
            min_distances[sample_idx] = sample_min_dist
            dbas_costs[sample_idx] = sample_dbas_cost

        weights = self._mppi_weights(costs)
        correction = np.sum(weights[:, None] * noise[:, 0, :], axis=0)
        optimized_action = self._clip_action(base_action + correction)

        self.last_action = optimized_action.astype(np.float32)
        self.last_noise_scale = float(noise_scale)

        best_idx = int(np.argmin(costs))
        debug = {
            "raw_action_surge": float(base_action[0]),
            "raw_action_yaw": float(base_action[1]),
            "optimized_action_surge": float(optimized_action[0]),
            "optimized_action_yaw": float(optimized_action[1]),
            "mppi_cost": float(costs[best_idx]),
            "mppi_mean_cost": float(np.mean(costs)),
            "dbas_cost": float(dbas_costs[best_idx]),
            "dbas_mean_cost": float(np.mean(dbas_costs)),
            "min_predicted_obstacle_distance": float(min_distances[best_idx]),
            "current_obstacle_distance": float(current_min_dist),
            "exploration_noise_scale": float(noise_scale),
        }
        return optimized_action.astype(np.float32), debug

    def _rollout_cost(
        self,
        action_sequence: np.ndarray,
        base_action: np.ndarray,
        planner_state: Dict[str, Any],
        obstacles: Optional[np.ndarray],
    ) -> Tuple[float, float, float]:
        cfg = self.config
        dt = float(planner_state.get("dt", 0.1))
        mass = float(planner_state.get("mass", 2.0))
        damping = float(planner_state.get("damping", 0.5))
        position = np.asarray(planner_state["position"], dtype=float).copy()
        yaw = float(planner_state.get("yaw", 0.0))
        velocity = np.asarray(planner_state.get("velocity", [0.0, 0.0, 0.0]), dtype=float).copy()
        target = np.asarray(planner_state.get("target_position", position), dtype=float)
        frenet_transform = planner_state.get("frenet_transform")
        max_laser_range = float(planner_state.get("max_laser_range", 10.0))

        start_dist_to_goal = float(np.linalg.norm(target - position))
        prev_action = self.last_action
        total_cost = 0.0
        total_dbas_cost = 0.0
        min_distance = max_laser_range

        for step_idx, action in enumerate(action_sequence):
            control_input = np.array([float(action[0]), 0.0, float(action[1])], dtype=float)
            acceleration = (control_input / mass) - (damping * velocity)
            velocity = velocity + acceleration * dt

            yaw = self._wrap_angle(yaw + velocity[2] * dt)
            position = position + np.array([math.cos(yaw), math.sin(yaw)], dtype=float) * velocity[0] * dt

            dist_to_goal = float(np.linalg.norm(target - position))
            progress = start_dist_to_goal - dist_to_goal
            heading_error = 0.0
            lateral_error = 0.0
            if frenet_transform is not None:
                frenet_s, frenet_d = frenet_transform.cartesian_to_frenet(position)
                lateral_error = abs(float(frenet_d))
                heading_error = abs(float(frenet_transform.get_heading_error(yaw, frenet_s)))

            obstacle_dist = self._min_obstacle_distance(position, obstacles, planner_state)
            min_distance = min(min_distance, obstacle_dist)
            dbas_cost = self._dbas_cost(obstacle_dist)
            total_dbas_cost += dbas_cost

            obstacle_cost = 0.0
            if obstacle_dist < cfg.safe_distance:
                obstacle_cost = (cfg.safe_distance - obstacle_dist) ** 2
            if obstacle_dist < cfg.collision_distance:
                obstacle_cost += 100.0 * (cfg.collision_distance - obstacle_dist + 1e-3)

            action_delta = action - prev_action
            base_delta = action - base_action

            total_cost += (
                cfg.goal_weight * dist_to_goal
                - cfg.progress_weight * progress
                + cfg.lateral_weight * lateral_error
                + cfg.heading_weight * (heading_error / math.pi)
                + cfg.obstacle_weight * obstacle_cost
                + cfg.dbas_weight * dbas_cost
                + cfg.control_weight * float(np.dot(base_delta, base_delta))
                + cfg.smoothness_weight * float(np.dot(action_delta, action_delta))
            )
            prev_action = action

            # Give earlier unsafe states slightly more influence.
            total_cost += 0.02 * step_idx * dbas_cost

        return total_cost, min_distance, total_dbas_cost

    def _mppi_weights(self, costs: np.ndarray) -> np.ndarray:
        shifted = costs - np.min(costs)
        weights = np.exp(-shifted / max(self.config.lambda_, 1e-6))
        normalizer = np.sum(weights)
        if not np.isfinite(normalizer) or normalizer <= 1e-12:
            return np.full_like(costs, 1.0 / len(costs), dtype=float)
        return weights / normalizer

    def _adaptive_noise_scale(self, min_obstacle_distance: float) -> float:
        cfg = self.config
        if not np.isfinite(min_obstacle_distance):
            return cfg.min_noise_scale
        risk = np.clip((cfg.safe_distance * 2.0 - min_obstacle_distance) / (cfg.safe_distance * 2.0), 0.0, 1.0)
        return float(cfg.min_noise_scale + risk * (cfg.max_noise_scale - cfg.min_noise_scale))

    def _dbas_cost(self, obstacle_distance: float) -> float:
        cfg = self.config
        barrier = obstacle_distance - cfg.safe_distance
        if barrier >= 0.0:
            return 1.0 / (barrier + 1.0)
        return 1.0 + ((-barrier + 1e-3) / max(cfg.safe_distance, 1e-6)) ** 2 * 10.0

    def _scan_to_obstacle_points(self, planner_state: Dict[str, Any]) -> Optional[np.ndarray]:
        scan = planner_state.get("scan")
        if scan is None or not hasattr(scan, "ranges"):
            return None

        ranges = np.asarray(scan.ranges, dtype=float)
        max_range = float(planner_state.get("max_laser_range", 10.0))
        ranges = np.nan_to_num(ranges, nan=max_range, posinf=max_range, neginf=0.0)
        ranges = np.clip(ranges, 0.0, max_range)

        valid = (ranges > 0.02) & (ranges < max_range * 0.995)
        if not np.any(valid):
            return None

        angle_min = float(getattr(scan, "angle_min", -math.pi))
        angle_increment = float(getattr(scan, "angle_increment", (2.0 * math.pi) / max(len(ranges), 1)))
        scan_angles = angle_min + np.arange(len(ranges), dtype=float) * angle_increment
        world_angles = float(planner_state.get("yaw", 0.0)) + scan_angles[valid]
        position = np.asarray(planner_state["position"], dtype=float)

        return position + np.column_stack((np.cos(world_angles), np.sin(world_angles))) * ranges[valid, None]

    def _min_obstacle_distance(
        self,
        position: np.ndarray,
        obstacles: Optional[np.ndarray],
        planner_state: Dict[str, Any],
    ) -> float:
        if obstacles is None or len(obstacles) == 0:
            scan = planner_state.get("scan")
            if scan is None or not hasattr(scan, "ranges"):
                return float(planner_state.get("max_laser_range", 10.0))
            ranges = np.asarray(scan.ranges, dtype=float)
            max_range = float(planner_state.get("max_laser_range", 10.0))
            ranges = np.nan_to_num(ranges, nan=max_range, posinf=max_range, neginf=max_range)
            return float(np.min(np.clip(ranges, 0.0, max_range)))

        distances = np.linalg.norm(obstacles - position.reshape(1, 2), axis=1)
        return float(np.min(distances))

    def _clip_action(self, action: Any) -> np.ndarray:
        action_arr = np.asarray(action, dtype=float).reshape(-1)
        if action_arr.size < 2:
            raise ValueError("MPPI-DBaS expects a 2D action: [surge, yaw].")
        action_arr = action_arr[:2]
        return np.clip(action_arr, np.asarray(self.config.action_low), np.asarray(self.config.action_high))

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
