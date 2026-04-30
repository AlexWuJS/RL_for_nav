import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class MPPIDBaSConfig:
    """Configuration for the conservative MPPI-DBaS safety filter."""

    num_samples: int = 128
    horizon: int = 12
    lambda_: float = 1.5
    safe_distance: float = 0.55
    collision_distance: float = 0.25
    risk_activation_distance: float = 1.2
    base_noise_std: Tuple[float, float] = (0.12, 0.14)
    min_noise_scale: float = 0.3
    max_noise_scale: float = 1.5
    max_action_delta: Tuple[float, float] = (0.25, 0.25)
    action_low: Tuple[float, float] = (-1.0, -1.0)
    action_high: Tuple[float, float] = (2.0, 1.0)
    goal_weight: float = 0.8
    progress_weight: float = 3.0
    lateral_weight: float = 2.5
    heading_weight: float = 1.5
    obstacle_weight: float = 4.0
    dbas_weight: float = 3.0
    trust_region_weight: float = 18.0
    control_weight: float = 0.04
    smoothness_weight: float = 0.5
    ttc_weight: float = 2.0
    ttc_horizon: float = 2.0
    out_of_bounds_limit: float = 2.5
    out_of_bounds_weight: float = 80.0
    front_sector_half_angle: float = 1.35
    seed: Optional[int] = None


class MPPIDBaSOptimizer:
    """A conservative MPPI-DBaS action filter around the SAC action."""

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
        position = np.asarray(planner_state["position"], dtype=float)
        current_min_dist = self._min_obstacle_distance(position, obstacles, planner_state)

        if current_min_dist >= cfg.risk_activation_distance:
            self.last_action = base_action.astype(np.float32)
            return base_action.astype(np.float32), self._make_passthrough_debug(base_action, current_min_dist)

        noise_scale = self._adaptive_noise_scale(current_min_dist)
        noise_std = np.asarray(cfg.base_noise_std, dtype=float) * noise_scale
        max_delta = np.asarray(cfg.max_action_delta, dtype=float)
        lower = np.maximum(np.asarray(cfg.action_low, dtype=float), base_action - max_delta)
        upper = np.minimum(np.asarray(cfg.action_high, dtype=float), base_action + max_delta)

        noise = self.rng.normal(
            loc=0.0,
            scale=noise_std,
            size=(cfg.num_samples, cfg.horizon, 2),
        )
        noise[:, 0, :] = np.clip(noise[:, 0, :], -max_delta, max_delta)
        action_sequences = np.clip(base_action.reshape(1, 1, 2) + noise, lower, upper)

        costs = np.zeros(cfg.num_samples, dtype=float)
        min_distances = np.full(cfg.num_samples, float(planner_state.get("max_laser_range", 10.0)), dtype=float)
        dbas_costs = np.zeros(cfg.num_samples, dtype=float)
        ttc_costs = np.zeros(cfg.num_samples, dtype=float)
        out_of_bounds_costs = np.zeros(cfg.num_samples, dtype=float)

        for sample_idx in range(cfg.num_samples):
            sample_cost, sample_min_dist, sample_dbas_cost, sample_ttc_cost, sample_oob_cost = self._rollout_cost(
                action_sequences[sample_idx],
                base_action,
                planner_state,
                obstacles,
            )
            costs[sample_idx] = sample_cost
            min_distances[sample_idx] = sample_min_dist
            dbas_costs[sample_idx] = sample_dbas_cost
            ttc_costs[sample_idx] = sample_ttc_cost
            out_of_bounds_costs[sample_idx] = sample_oob_cost

        weights = self._mppi_weights(costs)
        correction = np.sum(weights[:, None] * noise[:, 0, :], axis=0)
        correction = np.clip(correction, -max_delta, max_delta)
        optimized_action = np.clip(base_action + correction, lower, upper)
        optimized_action = self._clip_action(optimized_action)

        self.last_action = optimized_action.astype(np.float32)
        self.last_noise_scale = float(noise_scale)

        best_idx = int(np.argmin(costs))
        action_delta = optimized_action - base_action
        debug = {
            "raw_action_surge": float(base_action[0]),
            "raw_action_yaw": float(base_action[1]),
            "optimized_action_surge": float(optimized_action[0]),
            "optimized_action_yaw": float(optimized_action[1]),
            "action_delta_surge": float(action_delta[0]),
            "action_delta_yaw": float(action_delta[1]),
            "action_delta_norm": float(np.linalg.norm(action_delta)),
            "mppi_cost": float(costs[best_idx]),
            "mppi_mean_cost": float(np.mean(costs)),
            "dbas_cost": float(dbas_costs[best_idx]),
            "dbas_mean_cost": float(np.mean(dbas_costs)),
            "ttc_cost": float(ttc_costs[best_idx]),
            "out_of_bounds_cost": float(out_of_bounds_costs[best_idx]),
            "min_predicted_obstacle_distance": float(min_distances[best_idx]),
            "current_obstacle_distance": float(current_min_dist),
            "exploration_noise_scale": float(noise_scale),
            "mppi_active": True,
        }
        return optimized_action.astype(np.float32), debug

    def _rollout_cost(
        self,
        action_sequence: np.ndarray,
        base_action: np.ndarray,
        planner_state: Dict[str, Any],
        obstacles: Optional[np.ndarray],
    ) -> Tuple[float, float, float, float, float]:
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
        prev_action = np.asarray(planner_state.get("last_action", self.last_action), dtype=float).reshape(-1)[:2]

        start_dist_to_goal = float(np.linalg.norm(target - position))
        total_cost = 0.0
        total_dbas_cost = 0.0
        total_ttc_cost = 0.0
        total_oob_cost = 0.0
        min_distance = max_laser_range

        for action in action_sequence:
            control_input = np.array([float(action[0]), 0.0, float(action[1])], dtype=float)
            acceleration = (control_input / mass) - (damping * velocity)
            velocity = velocity + acceleration * dt

            yaw = self._wrap_angle(yaw + velocity[2] * dt)
            heading_vec = np.array([math.cos(yaw), math.sin(yaw)], dtype=float)
            position = position + heading_vec * velocity[0] * dt

            dist_to_goal = float(np.linalg.norm(target - position))
            progress = start_dist_to_goal - dist_to_goal
            heading_error = 0.0
            lateral_error = 0.0
            out_of_bounds_cost = 0.0
            if frenet_transform is not None:
                frenet_s, frenet_d = frenet_transform.cartesian_to_frenet(position)
                lateral_error = abs(float(frenet_d))
                heading_error = abs(float(frenet_transform.get_heading_error(yaw, frenet_s)))
                if lateral_error > cfg.out_of_bounds_limit:
                    out_of_bounds_cost = (lateral_error - cfg.out_of_bounds_limit) ** 2 + 1.0

            obstacle_dist = self._min_obstacle_distance(position, obstacles, planner_state)
            min_distance = min(min_distance, obstacle_dist)
            dbas_cost = self._dbas_cost(obstacle_dist)
            ttc_cost = self._ttc_cost(position, heading_vec * velocity[0], obstacles)

            obstacle_cost = 0.0
            if obstacle_dist < cfg.safe_distance:
                obstacle_cost = ((cfg.safe_distance - obstacle_dist) / cfg.safe_distance) ** 2
            if obstacle_dist < cfg.collision_distance:
                obstacle_cost += 60.0 * (cfg.collision_distance - obstacle_dist + 1e-3)

            action_delta = action - prev_action
            sac_delta = action - base_action

            total_cost += (
                cfg.goal_weight * dist_to_goal
                - cfg.progress_weight * progress
                + cfg.lateral_weight * lateral_error
                + cfg.heading_weight * (heading_error / math.pi)
                + cfg.obstacle_weight * obstacle_cost
                + cfg.dbas_weight * dbas_cost
                + cfg.ttc_weight * ttc_cost
                + cfg.out_of_bounds_weight * out_of_bounds_cost
                + cfg.trust_region_weight * float(np.dot(sac_delta, sac_delta))
                + cfg.control_weight * float(np.dot(action, action))
                + cfg.smoothness_weight * float(np.dot(action_delta, action_delta))
            )
            total_dbas_cost += dbas_cost
            total_ttc_cost += ttc_cost
            total_oob_cost += out_of_bounds_cost
            prev_action = action

        return total_cost, min_distance, total_dbas_cost, total_ttc_cost, total_oob_cost

    def _make_passthrough_debug(self, base_action: np.ndarray, current_min_dist: float) -> Dict[str, float]:
        return {
            "raw_action_surge": float(base_action[0]),
            "raw_action_yaw": float(base_action[1]),
            "optimized_action_surge": float(base_action[0]),
            "optimized_action_yaw": float(base_action[1]),
            "action_delta_surge": 0.0,
            "action_delta_yaw": 0.0,
            "action_delta_norm": 0.0,
            "mppi_cost": 0.0,
            "mppi_mean_cost": 0.0,
            "dbas_cost": 0.0,
            "dbas_mean_cost": 0.0,
            "ttc_cost": 0.0,
            "out_of_bounds_cost": 0.0,
            "min_predicted_obstacle_distance": float(current_min_dist),
            "current_obstacle_distance": float(current_min_dist),
            "exploration_noise_scale": 0.0,
            "mppi_active": False,
        }

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
        risk = np.clip((cfg.risk_activation_distance - min_obstacle_distance) / cfg.risk_activation_distance, 0.0, 1.0)
        return float(cfg.min_noise_scale + risk * (cfg.max_noise_scale - cfg.min_noise_scale))

    def _dbas_cost(self, obstacle_distance: float) -> float:
        cfg = self.config
        if obstacle_distance >= cfg.safe_distance:
            return 0.0
        normalized_violation = (cfg.safe_distance - obstacle_distance) / max(cfg.safe_distance - cfg.collision_distance, 1e-6)
        return float(normalized_violation ** 2)

    def _ttc_cost(self, position: np.ndarray, velocity_vec: np.ndarray, obstacles: Optional[np.ndarray]) -> float:
        if obstacles is None or len(obstacles) == 0:
            return 0.0
        speed = float(np.linalg.norm(velocity_vec))
        if speed < 1e-4:
            return 0.0
        rel = obstacles - position.reshape(1, 2)
        distances = np.linalg.norm(rel, axis=1)
        unit_rel = rel / (distances.reshape(-1, 1) + 1e-6)
        closing_speed = unit_rel @ velocity_vec
        valid = closing_speed > 1e-4
        if not np.any(valid):
            return 0.0
        ttc = distances[valid] / closing_speed[valid]
        min_ttc = float(np.min(ttc))
        if min_ttc >= self.config.ttc_horizon:
            return 0.0
        return ((self.config.ttc_horizon - min_ttc) / self.config.ttc_horizon) ** 2

    def _scan_to_obstacle_points(self, planner_state: Dict[str, Any]) -> Optional[np.ndarray]:
        scan = planner_state.get("scan")
        if scan is None or not hasattr(scan, "ranges"):
            return None

        ranges = np.asarray(scan.ranges, dtype=float)
        max_range = float(planner_state.get("max_laser_range", 10.0))
        ranges = np.nan_to_num(ranges, nan=max_range, posinf=max_range, neginf=0.0)
        ranges = np.clip(ranges, 0.0, max_range)

        angle_min = float(getattr(scan, "angle_min", -math.pi))
        angle_increment = float(getattr(scan, "angle_increment", (2.0 * math.pi) / max(len(ranges), 1)))
        scan_angles = angle_min + np.arange(len(ranges), dtype=float) * angle_increment
        forward_mask = np.abs(np.array([self._wrap_angle(a) for a in scan_angles])) <= self.config.front_sector_half_angle
        valid = (ranges > 0.02) & (ranges < self.config.risk_activation_distance) & forward_mask
        if not np.any(valid):
            return None

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
            return float(planner_state.get("max_laser_range", 10.0))

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
