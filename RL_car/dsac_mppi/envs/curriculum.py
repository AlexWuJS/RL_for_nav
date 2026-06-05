from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

import numpy as np


@dataclass
class CurriculumConfig:
    window_size: int = 50
    success_threshold: float = 0.8
    collision_threshold: float = 0.1
    mean_abs_frenet_d_threshold: float = 1.5
    max_stage: int = 4


class CurriculumManager:
    """Automatic stage scheduler for Gazebo DSAC training."""

    def __init__(self, mode: str = "off", config: Optional[CurriculumConfig] = None):
        self.mode = str(mode)
        self.config = config or CurriculumConfig()
        self.stage = 0 if self.mode == "auto" else 4
        self.history: Deque[Dict[str, float]] = deque(maxlen=self.config.window_size)

    def record_episode(self, info: Dict[str, float]) -> Dict[str, float]:
        metrics = {
            "success": float(bool(info.get("is_success", False))),
            "collision": float(bool(info.get("is_collision", False))),
            "abs_frenet_d": abs(float(info.get("frenet_d", 0.0))),
            "min_laser_dist": float(info.get("min_laser_dist", 0.0)),
        }
        self.history.append(metrics)
        if self.mode == "auto":
            self._maybe_advance()
        return self.stats()

    def stats(self) -> Dict[str, float]:
        if not self.history:
            return {
                "stage": float(self.stage),
                "success_rate": 0.0,
                "collision_rate": 0.0,
                "mean_abs_frenet_d": 0.0,
                "mean_min_laser_dist": 0.0,
                "ready_to_advance": 0.0,
            }
        success_rate = float(np.mean([row["success"] for row in self.history]))
        collision_rate = float(np.mean([row["collision"] for row in self.history]))
        mean_abs_d = float(np.mean([row["abs_frenet_d"] for row in self.history]))
        mean_min_laser = float(np.mean([row["min_laser_dist"] for row in self.history]))
        return {
            "stage": float(self.stage),
            "success_rate": success_rate,
            "collision_rate": collision_rate,
            "mean_abs_frenet_d": mean_abs_d,
            "mean_min_laser_dist": mean_min_laser,
            "ready_to_advance": float(self._ready_to_advance(success_rate, collision_rate, mean_abs_d)),
        }

    def _maybe_advance(self) -> None:
        if self.stage >= self.config.max_stage or len(self.history) < self.config.window_size:
            return
        stats = self.stats()
        if self._ready_to_advance(stats["success_rate"], stats["collision_rate"], stats["mean_abs_frenet_d"]):
            self.stage += 1
            self.history.clear()

    def _ready_to_advance(self, success_rate: float, collision_rate: float, mean_abs_frenet_d: float) -> bool:
        return (
            success_rate >= self.config.success_threshold
            and collision_rate <= self.config.collision_threshold
            and mean_abs_frenet_d <= self.config.mean_abs_frenet_d_threshold
        )
