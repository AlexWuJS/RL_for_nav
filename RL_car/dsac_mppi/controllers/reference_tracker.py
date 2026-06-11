import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from dsac_mppi.envs.usv_dynamics import USVDynamicsConfig, clip_command, frenet_outputs, wrap_angle


@dataclass
class ReferenceTrackerConfig:
    lookahead_distance: float = 3.0
    target_speed: float = 0.8
    min_heading_speed_scale: float = 0.25
    slowdown_distance: float = 1.0
    goal_tolerance: float = 0.4
    heading_gain: float = 1.2
    lateral_gain: float = 0.25


class ReferenceLineTracker:
    """Simple pure-pursuit style tracker for straight/curved Frenet reference lines."""

    def __init__(
        self,
        config: Optional[ReferenceTrackerConfig] = None,
        dynamics_config: Optional[USVDynamicsConfig] = None,
    ):
        self.config = config or ReferenceTrackerConfig()
        self.dynamics_config = dynamics_config or USVDynamicsConfig()

    def compute_action(
        self,
        position: Any,
        yaw: float,
        target_position: Any,
        frenet_transform: Any,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        metrics = frenet_outputs(position, yaw, target_position, frenet_transform)
        cfg = self.config

        if frenet_transform is None:
            target = np.asarray(target_position, dtype=float).reshape(-1)[:2]
            target_point = target
        else:
            lookahead_s, target_point = frenet_transform.get_lookahead_point(
                metrics["frenet_s"],
                cfg.lookahead_distance,
            )

        position = np.asarray(position, dtype=float).reshape(-1)[:2]
        target_vec = np.asarray(target_point, dtype=float).reshape(2) - position
        target_heading = math.atan2(float(target_vec[1]), float(target_vec[0]))
        pursuit_heading_error = wrap_angle(target_heading - float(yaw))
        path_heading_error = float(metrics["heading_error"])
        remaining = float(metrics["remaining_path"])
        slowdown_scale = min(1.0, max(0.0, remaining / max(float(cfg.slowdown_distance), 1e-6)))
        heading_speed_scale = max(
            float(cfg.min_heading_speed_scale),
            math.cos(float(np.clip(pursuit_heading_error, -math.pi / 2.0, math.pi / 2.0))),
        )
        u_cmd = float(cfg.target_speed) * heading_speed_scale * slowdown_scale
        if remaining <= float(cfg.goal_tolerance):
            u_cmd = 0.0
        r_cmd = float(cfg.heading_gain) * pursuit_heading_error - float(cfg.lateral_gain) * float(metrics["frenet_d"])
        action = clip_command(np.array([u_cmd, r_cmd], dtype=float), self.dynamics_config)

        debug = {
            **metrics,
            "lookahead_s": float(lookahead_s) if frenet_transform is not None else float(metrics["frenet_s"]),
            "lookahead_x": float(target_point[0]),
            "lookahead_y": float(target_point[1]),
            "target_heading": float(target_heading),
            "pursuit_heading_error": float(pursuit_heading_error),
            "path_heading_error": path_heading_error,
            "u_cmd": float(action[0]),
            "r_cmd": float(action[1]),
        }
        return action.astype(np.float32), debug
