from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Tuple

import numpy as np


@dataclass
class DynamicObstacle:
    id: int
    x: float
    y: float
    vx: float
    vy: float
    radius: float = 0.4
    type: str = "crossing"

    @property
    def position(self) -> np.ndarray:
        return np.array([float(self.x), float(self.y)], dtype=float)

    @property
    def velocity(self) -> np.ndarray:
        return np.array([float(self.vx), float(self.vy)], dtype=float)

    def step(self, dt: float) -> "DynamicObstacle":
        self.x = float(self.x + self.vx * float(dt))
        self.y = float(self.y + self.vy * float(dt))
        return self

    def copy(self) -> "DynamicObstacle":
        return DynamicObstacle(
            id=int(self.id),
            x=float(self.x),
            y=float(self.y),
            vx=float(self.vx),
            vy=float(self.vy),
            radius=float(self.radius),
            type=str(self.type),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": int(self.id),
            "x": float(self.x),
            "y": float(self.y),
            "vx": float(self.vx),
            "vy": float(self.vy),
            "radius": float(self.radius),
            "type": str(self.type),
        }


class DynamicObstacleScenarioFactory:
    @staticmethod
    def crossing(obstacle_id: int = 0, x: float = 5.0, y: float = 3.0, vy: float = -0.5, radius: float = 0.4) -> List[DynamicObstacle]:
        return [DynamicObstacle(obstacle_id, x, y, 0.0, vy, radius, "crossing")]

    @staticmethod
    def oncoming(obstacle_id: int = 0, x: float = 8.0, y: float = 0.0, vx: float = -0.6, radius: float = 0.4) -> List[DynamicObstacle]:
        return [DynamicObstacle(obstacle_id, x, y, vx, 0.0, radius, "oncoming")]

    @staticmethod
    def overtaking_slow_ahead(obstacle_id: int = 0, x: float = 4.0, y: float = 0.0, vx: float = 0.25, radius: float = 0.4) -> List[DynamicObstacle]:
        return [DynamicObstacle(obstacle_id, x, y, vx, 0.0, radius, "overtaking_slow_ahead")]

    @staticmethod
    def overtaking_fast_behind(obstacle_id: int = 0, x: float = -2.0, y: float = 0.0, vx: float = 1.0, radius: float = 0.4) -> List[DynamicObstacle]:
        return [DynamicObstacle(obstacle_id, x, y, vx, 0.0, radius, "overtaking_fast_behind")]

    @staticmethod
    def mixed() -> List[DynamicObstacle]:
        obstacles: List[DynamicObstacle] = []
        obstacles.extend(DynamicObstacleScenarioFactory.crossing(0, x=5.0, y=3.0, vy=-0.5))
        obstacles.extend(DynamicObstacleScenarioFactory.oncoming(1, x=9.0, y=0.7, vx=-0.55))
        obstacles.extend(DynamicObstacleScenarioFactory.overtaking_slow_ahead(2, x=3.0, y=-0.8, vx=0.25))
        obstacles.extend(DynamicObstacleScenarioFactory.overtaking_fast_behind(3, x=-2.0, y=0.9, vx=1.0))
        return obstacles

    @staticmethod
    def create(scenario_type: str) -> List[DynamicObstacle]:
        scenario = str(scenario_type)
        if scenario == "crossing":
            return DynamicObstacleScenarioFactory.crossing()
        if scenario == "oncoming":
            return DynamicObstacleScenarioFactory.oncoming()
        if scenario == "overtaking_slow_ahead":
            return DynamicObstacleScenarioFactory.overtaking_slow_ahead()
        if scenario == "overtaking_fast_behind":
            return DynamicObstacleScenarioFactory.overtaking_fast_behind()
        if scenario == "mixed":
            return DynamicObstacleScenarioFactory.mixed()
        raise ValueError(f"Unsupported dynamic obstacle scenario: {scenario_type}")


def step_obstacles(obstacles: List[DynamicObstacle], dt: float) -> List[DynamicObstacle]:
    return [obstacle.step(dt) for obstacle in obstacles]


def obstacle_from_any(raw: Any) -> DynamicObstacle:
    if isinstance(raw, DynamicObstacle):
        return raw
    if isinstance(raw, dict):
        position = np.asarray(raw.get("position", [raw.get("x", 0.0), raw.get("y", 0.0)]), dtype=float).reshape(-1)
        velocity = np.asarray(raw.get("velocity", [raw.get("vx", 0.0), raw.get("vy", 0.0)]), dtype=float).reshape(-1)
        return DynamicObstacle(
            id=_parse_obstacle_id(raw.get("id", raw.get("name", 0))),
            x=float(position[0]) if position.size >= 1 else 0.0,
            y=float(position[1]) if position.size >= 2 else 0.0,
            vx=float(velocity[0]) if velocity.size >= 1 else 0.0,
            vy=float(velocity[1]) if velocity.size >= 2 else 0.0,
            radius=float(raw.get("radius", 0.4)),
            type=str(raw.get("type", raw.get("scenario_type", "unknown"))),
        )
    raise TypeError(f"Unsupported obstacle type: {type(raw)!r}")


def _parse_obstacle_id(value: Any) -> int:
    text = str(value)
    if text.lstrip("-").isdigit():
        return int(text)
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else 0


@dataclass
class CVPredictionConfig:
    history_len: int = 4
    prediction_horizon: int = 12
    dt: float = 0.1
    t_max: float = 3.0
    safe_distance: float = 1.2
    collision_distance: float = 0.25


class CVObstaclePredictor:
    def __init__(self, config: CVPredictionConfig = None):
        self.config = config or CVPredictionConfig()
        self.history: Dict[int, Deque[np.ndarray]] = {}

    def reset(self) -> None:
        self.history.clear()

    def update(self, obstacles: List[Any]) -> None:
        for raw in obstacles:
            obstacle = obstacle_from_any(raw)
            history = self.history.setdefault(int(obstacle.id), deque(maxlen=int(self.config.history_len)))
            history.append(obstacle.position.copy())

    def estimate_velocity(self, obstacle: Any) -> np.ndarray:
        obs = obstacle_from_any(obstacle)
        history = self.history.get(int(obs.id))
        if history is None or len(history) < 2:
            return obs.velocity.copy()
        k = len(history) - 1
        if k <= 0:
            return obs.velocity.copy()
        velocity = (history[-1] - history[0]) / (float(k) * float(self.config.dt))
        return velocity.astype(float)

    def predict(
        self,
        obstacles: List[Any],
        usv_position: Any,
        usv_velocity: Any,
    ) -> List[Dict[str, Any]]:
        usv_position = np.asarray(usv_position, dtype=float).reshape(2)
        usv_velocity = np.asarray(usv_velocity, dtype=float).reshape(2)
        predictions = []
        for raw in obstacles:
            obstacle = obstacle_from_any(raw)
            velocity = self.estimate_velocity(obstacle)
            positions = []
            for step in range(1, int(self.config.prediction_horizon) + 1):
                point = obstacle.position + velocity * float(step) * float(self.config.dt)
                positions.append((float(point[0]), float(point[1])))
            tcpa, dcpa, risk = compute_tcpa_dcpa_risk(
                obstacle.position,
                velocity,
                usv_position,
                usv_velocity,
                radius=float(obstacle.radius),
                t_max=float(self.config.t_max),
                safe_distance=float(self.config.safe_distance),
                collision_distance=float(self.config.collision_distance),
            )
            predictions.append(
                {
                    "id": int(obstacle.id),
                    "positions": positions,
                    "velocity": (float(velocity[0]), float(velocity[1])),
                    "radius": float(obstacle.radius),
                    "tcpa": float(tcpa),
                    "dcpa": float(dcpa),
                    "risk": float(risk),
                    "type": str(obstacle.type),
                }
            )
        return predictions


def compute_tcpa_dcpa_risk(
    obstacle_position: Any,
    obstacle_velocity: Any,
    usv_position: Any,
    usv_velocity: Any,
    radius: float = 0.4,
    t_max: float = 3.0,
    safe_distance: float = 1.2,
    collision_distance: float = 0.25,
) -> Tuple[float, float, float]:
    p_obs = np.asarray(obstacle_position, dtype=float).reshape(2)
    v_obs = np.asarray(obstacle_velocity, dtype=float).reshape(2)
    p_usv = np.asarray(usv_position, dtype=float).reshape(2)
    v_usv = np.asarray(usv_velocity, dtype=float).reshape(2)
    p_rel = p_obs - p_usv
    v_rel = v_obs - v_usv
    speed_sq = float(np.dot(v_rel, v_rel))
    if speed_sq < 1e-12:
        return float("inf"), float(np.linalg.norm(p_rel)), 0.0

    tcpa = -float(np.dot(p_rel, v_rel)) / speed_sq
    dcpa_time = max(tcpa, 0.0)
    dcpa = float(np.linalg.norm(p_rel + v_rel * dcpa_time))
    if tcpa < 0.0 or tcpa > float(t_max):
        return float(tcpa), dcpa, 0.0

    effective_collision = float(collision_distance) + float(radius)
    effective_safe = max(float(safe_distance), effective_collision + 1e-6)
    if dcpa >= effective_safe:
        return float(tcpa), dcpa, 0.0

    distance_risk = (effective_safe - dcpa) / max(effective_safe - effective_collision, 1e-6)
    time_risk = (float(t_max) - tcpa) / max(float(t_max), 1e-6)
    if dcpa <= effective_collision:
        distance_risk = 1.0
    risk = float(np.clip(distance_risk, 0.0, 1.0) * np.clip(time_risk, 0.0, 1.0))
    return float(tcpa), dcpa, risk
