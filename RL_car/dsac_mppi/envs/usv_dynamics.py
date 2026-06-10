import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class USVDynamicsConfig:
    """Planar USV first-order surge/yaw dynamics.

    State: x, y, psi, u, r
    Command: u_cmd, r_cmd
    """

    dt: float = 0.1
    surge_time_constant: float = 0.6
    yaw_time_constant: float = 0.4
    max_du: float = 0.15
    max_dr: float = 0.12
    action_low: Tuple[float, float] = (0.0, -0.6)
    action_high: Tuple[float, float] = (1.5, 0.6)


def wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return float(angle)


def velocity_to_ur(velocity: Any) -> np.ndarray:
    arr = np.asarray(velocity, dtype=float).reshape(-1)
    if arr.size >= 3:
        return np.array([arr[0], arr[2]], dtype=float)
    if arr.size >= 2:
        return arr[:2].astype(float)
    if arr.size == 1:
        return np.array([arr[0], 0.0], dtype=float)
    return np.zeros(2, dtype=float)


def ur_to_twist_vector(velocity_ur: Any) -> np.ndarray:
    velocity_ur = np.asarray(velocity_ur, dtype=float).reshape(-1)
    u = float(velocity_ur[0]) if velocity_ur.size > 0 else 0.0
    r = float(velocity_ur[1]) if velocity_ur.size > 1 else 0.0
    return np.array([u, 0.0, r], dtype=float)


def config_from_planner_state(planner_state: Dict[str, Any], defaults: Optional[USVDynamicsConfig] = None) -> USVDynamicsConfig:
    defaults = defaults or USVDynamicsConfig()
    action_low = planner_state.get("action_low", defaults.action_low)
    action_high = planner_state.get("action_high", defaults.action_high)
    return USVDynamicsConfig(
        dt=float(planner_state.get("dt", defaults.dt)),
        surge_time_constant=float(planner_state.get("surge_time_constant", defaults.surge_time_constant)),
        yaw_time_constant=float(planner_state.get("yaw_time_constant", defaults.yaw_time_constant)),
        max_du=float(planner_state.get("max_du", defaults.max_du)),
        max_dr=float(planner_state.get("max_dr", defaults.max_dr)),
        action_low=tuple(float(x) for x in np.asarray(action_low, dtype=float).reshape(-1)[:2]),
        action_high=tuple(float(x) for x in np.asarray(action_high, dtype=float).reshape(-1)[:2]),
    )


def clip_command(command: Any, config: USVDynamicsConfig) -> np.ndarray:
    command = np.asarray(command, dtype=float).reshape(-1)
    if command.size < 2:
        raise ValueError("USV command must be [u_cmd, r_cmd].")
    return np.clip(command[:2], np.asarray(config.action_low), np.asarray(config.action_high))


def rate_limit_command(command: Any, previous_command: Any, config: USVDynamicsConfig) -> np.ndarray:
    command = clip_command(command, config)
    previous = clip_command(previous_command, config)
    max_delta = np.array([float(config.max_du), float(config.max_dr)], dtype=float)
    return np.clip(command, previous - max_delta, previous + max_delta)


def step_velocity(
    velocity: Any,
    command: Any,
    previous_command: Any,
    config: USVDynamicsConfig,
    dynamics_model: str = "first_order",
) -> Tuple[np.ndarray, np.ndarray]:
    """Advance [u, r] and return (new_velocity_ur, applied_command)."""
    model = "first_order" if str(dynamics_model) == "inertia" else str(dynamics_model)
    if model == "ideal":
        applied = clip_command(command, config)
        return applied.copy(), applied

    applied = rate_limit_command(command, previous_command, config)
    current = velocity_to_ur(velocity)
    dt = float(config.dt)
    tu = max(float(config.surge_time_constant), 1e-6)
    tr = max(float(config.yaw_time_constant), 1e-6)
    u = current[0] + dt * (applied[0] - current[0]) / tu
    r = current[1] + dt * (applied[1] - current[1]) / tr
    new_velocity = np.array([u, r], dtype=float)
    return new_velocity, applied


def step_pose(
    position: Any,
    yaw: float,
    velocity: Any,
    command: Any,
    previous_command: Any,
    config: USVDynamicsConfig,
    dynamics_model: str = "first_order",
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Euler-step x, y, psi, u, r using the configured USV dynamics."""
    position = np.asarray(position, dtype=float).reshape(-1)[:2].copy()
    velocity_ur, applied = step_velocity(velocity, command, previous_command, config, dynamics_model)
    dt = float(config.dt)
    yaw0 = float(yaw)
    position = position + np.array([math.cos(yaw0), math.sin(yaw0)], dtype=float) * float(velocity_ur[0]) * dt
    yaw1 = wrap_angle(yaw0 + float(velocity_ur[1]) * dt)
    return position, yaw1, velocity_ur, applied


def frenet_outputs(
    position: Any,
    yaw: float,
    target_position: Any,
    frenet_transform: Any,
) -> Dict[str, float]:
    position = np.asarray(position, dtype=float).reshape(-1)[:2]
    target = np.asarray(target_position, dtype=float).reshape(-1)[:2]
    dist_to_goal = float(np.linalg.norm(target - position))
    if frenet_transform is not None:
        frenet_s, frenet_d = frenet_transform.cartesian_to_frenet(position)
        heading_error = float(frenet_transform.get_heading_error(float(yaw), frenet_s))
        path_length = float(frenet_transform.path_length)
        remaining_path = max(path_length - float(frenet_s), 0.0)
    else:
        frenet_s = 0.0
        frenet_d = 0.0
        heading_error = 0.0
        path_length = dist_to_goal + 1.0
        remaining_path = dist_to_goal
    return {
        "distance_to_goal": dist_to_goal,
        "frenet_s": float(frenet_s),
        "frenet_d": float(frenet_d),
        "heading_error": float(heading_error),
        "remaining_path": float(remaining_path),
        "path_length": float(path_length),
    }
