import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


V4_MODE_NAMES = (
    "cruise",
    "cautious_cruise",
    "avoid_left",
    "avoid_right",
    "recover_center",
    "brake",
)


def intent_to_params(intent: Any) -> Dict[str, float]:
    """Map SAC's normalized high-level intent to MPPI priors and cost weights."""
    values = np.asarray(intent, dtype=float).reshape(-1)
    padded = np.zeros(4, dtype=float)
    padded[: min(4, values.size)] = values[:4]
    clipped = np.clip(padded, -1.0, 1.0)
    return {
        "target_speed": float(0.75 * (clipped[0] + 1.0)),
        "turn_bias": float(0.8 * clipped[1]),
        "path_weight": float(0.5 + 1.25 * (clipped[2] + 1.0)),
        "safety_weight": float(0.5 + 1.75 * (clipped[3] + 1.0)),
        "raw_target_speed": float(clipped[0]),
        "raw_turn_bias": float(clipped[1]),
        "raw_path_weight": float(clipped[2]),
        "raw_safety_weight": float(clipped[3]),
    }


def intent_to_params_v2(intent: Any) -> Dict[str, float]:
    """Conservative intent mapping for trigger-based hierarchical MPPI."""
    values = np.asarray(intent, dtype=float).reshape(-1)
    padded = np.zeros(4, dtype=float)
    padded[: min(4, values.size)] = values[:4]
    clipped = np.clip(padded, -1.0, 1.0)
    return {
        "target_speed": float(0.75 * (clipped[0] + 1.0)),
        "turn_bias": float(0.55 * clipped[1]),
        "path_weight": float(0.8 + 0.7 * (clipped[2] + 1.0)),
        "safety_weight": float(1.2 + 1.0 * (clipped[3] + 1.0)),
        "raw_target_speed": float(clipped[0]),
        "raw_turn_bias": float(clipped[1]),
        "raw_path_weight": float(clipped[2]),
        "raw_safety_weight": float(clipped[3]),
    }


def decode_mode_intent_v4(intent: Any) -> Dict[str, Any]:
    """Decode an 8D continuous policy output into a mode choice plus 2 continuous parameters."""
    values = np.asarray(intent, dtype=float).reshape(-1)
    padded = np.zeros(8, dtype=float)
    padded[: min(8, values.size)] = values[:8]
    clipped = np.clip(padded, -1.0, 1.0)
    mode_scores = clipped[: len(V4_MODE_NAMES)]
    mode_index = int(np.argmax(mode_scores))
    sorted_scores = np.sort(mode_scores)
    if mode_scores.size >= 2:
        mode_margin = float(sorted_scores[-1] - sorted_scores[-2])
    else:
        mode_margin = 0.0
    speed_scale = float(0.5 * (clipped[6] + 1.0))
    avoid_strength = float(0.5 * (clipped[7] + 1.0))
    return {
        "mode_scores": mode_scores.astype(float),
        "mode_index": mode_index,
        "mode_name": V4_MODE_NAMES[mode_index],
        "mode_margin": mode_margin,
        "speed_scale": speed_scale,
        "avoid_strength": avoid_strength,
        "raw_speed_scale": float(clipped[6]),
        "raw_avoid_strength": float(clipped[7]),
        "raw_vector": clipped.astype(float),
    }


def intent_to_mode_params_v4(
    intent: Any,
    planner_state: Optional[Dict[str, Any]] = None,
    config: Optional["MPPIDBaSConfig"] = None,
) -> Dict[str, Any]:
    """Map the semi-discrete v4 policy output to interpretable control priors and cost weights."""
    cfg = config or MPPIDBaSConfig()
    decoded = decode_mode_intent_v4(intent)
    mode_name = str(decoded["mode_name"])
    speed_scale = float(decoded["speed_scale"])
    avoid_strength = float(decoded["avoid_strength"])
    current_d = 0.0
    frenet_transform = None if not planner_state else planner_state.get("frenet_transform")
    if planner_state and frenet_transform is not None:
        try:
            _, current_d = frenet_transform.cartesian_to_frenet(np.asarray(planner_state["position"], dtype=float))
        except Exception:
            current_d = 0.0
    center_correction = float(np.clip(-0.35 * current_d, -0.35, 0.35))
    target_speed = 0.55 + 0.45 * speed_scale
    turn_bias = center_correction
    path_weight = 1.05
    safety_weight = 1.6
    desired_lateral_offset = float(np.clip(-0.4 * current_d, -0.75, 0.75))

    if mode_name == "cautious_cruise":
        target_speed = min(target_speed, 0.55 + 0.20 * speed_scale)
        safety_weight = 2.4
        path_weight = 1.1
    elif mode_name == "avoid_left":
        target_speed = 0.35 + 0.35 * speed_scale
        turn_bias = float(np.clip(0.18 + 0.55 * avoid_strength + 0.25 * center_correction, -0.8, 0.8))
        path_weight = 1.25
        safety_weight = 2.6
        desired_lateral_offset = float(np.clip(0.35 + 0.55 * avoid_strength, -0.85, 0.85))
    elif mode_name == "avoid_right":
        target_speed = 0.35 + 0.35 * speed_scale
        turn_bias = float(np.clip(-0.18 - 0.55 * avoid_strength + 0.25 * center_correction, -0.8, 0.8))
        path_weight = 1.25
        safety_weight = 2.6
        desired_lateral_offset = float(np.clip(-0.35 - 0.55 * avoid_strength, -0.85, 0.85))
    elif mode_name == "recover_center":
        target_speed = 0.25 + 0.35 * speed_scale
        recovery_gain = 0.45 + 0.40 * avoid_strength
        turn_bias = float(np.clip(-recovery_gain * current_d, -0.85, 0.85))
        path_weight = 1.7
        safety_weight = 2.3
        desired_lateral_offset = 0.0
    elif mode_name == "brake":
        target_speed = 0.02 + 0.15 * speed_scale
        turn_bias = float(np.clip(-0.25 * current_d, -0.45, 0.45))
        path_weight = 1.3
        safety_weight = 3.1
        desired_lateral_offset = 0.0

    safe_distance = float(cfg.safe_distance)
    if mode_name == "cautious_cruise":
        safe_distance += 0.20
    elif mode_name in ("avoid_left", "avoid_right"):
        safe_distance += 0.25 + 0.15 * avoid_strength
    elif mode_name == "recover_center":
        safe_distance += 0.10
    elif mode_name == "brake":
        safe_distance += 0.35

    decoded.update(
        {
            "target_speed": float(np.clip(target_speed, 0.0, cfg.action_high[0])),
            "turn_bias": float(np.clip(turn_bias, cfg.action_low[1], cfg.action_high[1])),
            "path_weight": float(path_weight),
            "safety_weight": float(safety_weight),
            "safe_distance": float(safe_distance),
            "desired_lateral_offset": float(desired_lateral_offset),
            "current_lateral_offset": float(current_d),
        }
    )
    return decoded


def intent_to_structured_params_v41(
    intent: Any,
    planner_state: Optional[Dict[str, Any]] = None,
    config: Optional["MPPIDBaSConfig"] = None,
) -> Dict[str, float]:
    """Map v4.1's continuous high-level intent to MPPI targets and constraints."""
    cfg = config or MPPIDBaSConfig()
    values = np.asarray(intent, dtype=float).reshape(-1)
    padded = np.zeros(4, dtype=float)
    padded[: min(4, values.size)] = values[:4]
    clipped = np.clip(padded, -1.0, 1.0)

    progress_level = float(0.5 * (clipped[0] + 1.0))
    lateral_intent = float(clipped[1])
    safety_level = float(0.5 * (clipped[2] + 1.0))
    mppi_gate = float(0.5 * (clipped[3] + 1.0))

    current_d = 0.0
    dynamic_offset_limit = 1.0
    if planner_state:
        frenet_transform = planner_state.get("frenet_transform")
        if frenet_transform is not None:
            try:
                _, current_d = frenet_transform.cartesian_to_frenet(np.asarray(planner_state["position"], dtype=float))
            except Exception:
                current_d = 0.0
        boundary_margin = float(cfg.env_out_of_bounds_limit) - abs(float(current_d)) - float(cfg.hierarchical_v41_boundary_buffer)
        dynamic_offset_limit = float(np.clip(boundary_margin, 0.05, cfg.hierarchical_v41_max_lateral_target))

    target_lateral_offset = float(lateral_intent * dynamic_offset_limit)
    safe_distance = float(cfg.safe_distance + safety_level * cfg.hierarchical_v41_safety_margin_range)
    target_progress_speed = float(
        cfg.hierarchical_v41_min_progress_speed
        + progress_level * (cfg.hierarchical_v41_max_progress_speed - cfg.hierarchical_v41_min_progress_speed)
    )

    return {
        "target_progress_speed": target_progress_speed,
        "target_lateral_offset": target_lateral_offset,
        "safe_distance": safe_distance,
        "mppi_gate": mppi_gate,
        "dynamic_offset_limit": dynamic_offset_limit,
        "current_lateral_offset": float(current_d),
        "progress_level": progress_level,
        "lateral_intent": lateral_intent,
        "safety_level": safety_level,
        "raw_progress_intent": float(clipped[0]),
        "raw_lateral_target_intent": float(clipped[1]),
        "raw_safety_margin_intent": float(clipped[2]),
        "raw_mppi_gate_intent": float(clipped[3]),
        "path_weight": float(1.0 + 1.25 * safety_level),
        "safety_weight": float(1.2 + 1.8 * safety_level),
        "obstacle_weight": float(cfg.obstacle_weight * (1.0 + 1.5 * safety_level)),
        "ttc_weight": float(cfg.ttc_weight * (1.0 + 2.0 * safety_level)),
        "oob_weight": float(cfg.out_of_bounds_weight * (1.0 + 1.2 * safety_level)),
    }


def intent_to_frenet_params_v3(
    intent: Any,
    planner_state: Optional[Dict[str, Any]] = None,
    config: Optional["MPPIDBaSConfig"] = None,
) -> Dict[str, float]:
    """Map SAC's high-level Frenet intent to local MPPI trajectory targets."""
    cfg = config or MPPIDBaSConfig()
    values = np.asarray(intent, dtype=float).reshape(-1)
    padded = np.zeros(4, dtype=float)
    padded[: min(4, values.size)] = values[:4]
    clipped = np.clip(padded, -1.0, 1.0)
    caution_level = float(0.5 * (clipped[2] + 1.0))
    recovery_level = float(0.5 * (clipped[3] + 1.0))
    dynamic_offset_limit = _dynamic_lateral_offset_limit_v3(planner_state, cfg)
    base_oob_weight = float(cfg.out_of_bounds_weight)
    base_lateral_weight = float(cfg.lateral_weight)
    base_ttc_weight = float(cfg.ttc_weight)
    base_obstacle_weight = float(cfg.obstacle_weight)
    return {
        "target_progress_speed": float(0.25 + 0.5 * (clipped[0] + 1.0)),
        "target_lateral_offset": float(clipped[1] * dynamic_offset_limit),
        "dynamic_offset_limit": float(dynamic_offset_limit),
        "caution_level": caution_level,
        "recovery_level": recovery_level,
        "safe_distance": float(cfg.safe_distance + caution_level * 0.55),
        "ttc_weight": float(base_ttc_weight * (1.0 + 2.5 * caution_level)),
        "obstacle_weight": float(base_obstacle_weight * (1.0 + 2.0 * caution_level)),
        "oob_weight": float(base_oob_weight * (1.0 + 1.5 * recovery_level)),
        "lateral_target_weight": float(base_lateral_weight * (0.5 + 1.5 * recovery_level)),
        "path_relaxation": float(1.0 - recovery_level),
        "base_oob_weight": base_oob_weight,
        "base_lateral_weight": base_lateral_weight,
        "raw_progress_intent": float(clipped[0]),
        "raw_lateral_offset_intent": float(clipped[1]),
        "raw_caution_intent": float(clipped[2]),
        "raw_recovery_intent": float(clipped[3]),
    }


def _dynamic_lateral_offset_limit_v3(
    planner_state: Optional[Dict[str, Any]],
    config: "MPPIDBaSConfig",
) -> float:
    max_offset = 1.0
    if not planner_state:
        return max_offset
    frenet_transform = planner_state.get("frenet_transform")
    if frenet_transform is None:
        return max_offset
    try:
        _, frenet_d = frenet_transform.cartesian_to_frenet(np.asarray(planner_state["position"], dtype=float))
    except Exception:
        return max_offset
    boundary_margin = max(float(config.env_out_of_bounds_limit) - abs(float(frenet_d)) - 0.25, 0.05)
    return float(np.clip(boundary_margin, 0.05, max_offset))


try:
    from frenet_utils import compute_tracking_reward, piecewise_lateral_penalty
except ImportError:
    def piecewise_lateral_penalty(frenet_d: float) -> float:
        abs_d = abs(float(frenet_d))
        if abs_d <= 0.35:
            return 0.0
        if abs_d <= 1.2:
            return float(5.0 * (abs_d - 0.35) ** 2)
        stage_1 = 5.0 * (1.2 - 0.35) ** 2
        if abs_d <= 2.4:
            return float(stage_1 + 14.0 * (abs_d - 1.2) ** 2)
        stage_2 = stage_1 + 14.0 * (2.4 - 1.2) ** 2
        excess = abs_d - 2.4
        return float(stage_2 + 80.0 * excess * excess + 45.0 * excess)

    def compute_tracking_reward(
        delta_s: float,
        frenet_d: float,
        heading_error: float,
        min_obstacle_dist: float = None,
        previous_abs_frenet_d: float = None,
        action=None,
        previous_action=None,
        **kwargs,
    ) -> dict:
        abs_d = abs(float(frenet_d))
        track_factor = max(0.0, min(1.0, 1.0 - abs_d / 3.0))
        lateral_penalty = -piecewise_lateral_penalty(abs_d)
        heading_penalty = -(abs(float(heading_error)) / math.pi) * 1.5
        total = (float(delta_s) * 30.0 * track_factor + lateral_penalty + heading_penalty) * 2.5
        return {"total": float(total), "components": {}}


@dataclass
class MPPIDBaSConfig:
    """Configuration for the conservative MPPI-DBaS safety filter."""

    num_samples: int = 128
    horizon: int = 12
    lambda_: float = 1.5
    safe_distance: float = 0.55
    collision_distance: float = 0.25
    risk_activation_distance: float = 1.8
    fallback_trigger_distance: float = 0.8
    base_noise_std: Tuple[float, float] = (0.10, 0.10)
    min_noise_scale: float = 0.3
    max_noise_scale: float = 1.5
    max_action_delta: Tuple[float, float] = (0.25, 0.25)
    mppi_max_action_delta: Tuple[float, float] = (0.25, 0.30)
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
    safety_distance_margin: float = 0.03
    max_progress_loss: float = 0.02
    max_lateral_worsening: float = 0.03
    fallback_max_lateral_worsening: float = 0.05
    out_of_bounds_limit: float = 2.5
    out_of_bounds_weight: float = 80.0
    front_sector_half_angle: float = 1.75
    obstacle_sector_half_angle: float = 2.35
    fallback_surge: float = 0.15
    hard_brake_surge: float = 0.0
    fallback_yaw: float = 0.65
    fallback_min_clearance_delta: float = 0.08
    residual_low: Tuple[float, float] = (-0.18, -0.22)
    residual_high: Tuple[float, float] = (0.08, 0.22)
    reward_aligned_residual_low: Tuple[float, float] = (-0.15, -0.18)
    reward_aligned_residual_high: Tuple[float, float] = (0.06, 0.18)
    action_lag_alpha: float = 0.35
    mppi_min_score_gain: float = 0.05
    fallback_score_margin: float = 0.02
    use_reward_aligned_cost: bool = False
    always_run_mppi: bool = False
    execute_mppi: bool = True
    final_safety_check: bool = True
    reward_improvement_threshold: float = 0.05
    emergency_brake_distance: float = 0.35
    env_safe_distance: float = 0.45
    env_out_of_bounds_limit: float = 3.0
    env_goal_threshold: float = 0.4
    env_terminal_reward: float = 100.0
    env_success_reward: float = 1000.0
    env_frenet_reward_scale: float = 2.5
    hard_safety_penalty: float = 1000.0
    lateral_deadband: float = 0.5
    lateral_soft_limit: float = 2.0
    teacher_only: bool = False
    enable_mppi: bool = True
    enable_fallback: bool = True
    seed: Optional[int] = None
    hierarchical_front_trigger_distance: float = 1.2
    hierarchical_global_trigger_distance: float = 0.9
    hierarchical_lateral_trigger: float = 0.8
    hierarchical_heading_trigger: float = 0.6
    hierarchical_min_risk_gain: float = 0.05
    hierarchical_accept_score_margin: float = 0.01
    hierarchical_lateral_recovery_gain: float = 0.08
    hierarchical_v4_cruise_max_yaw_delta: float = 0.18
    hierarchical_v4_mode_switch_margin: float = 0.02
    hierarchical_v4_fallback_risk_margin: float = 0.08
    hierarchical_v4_direction_consistency_weight: float = 0.5
    hierarchical_v41_min_progress_speed: float = 0.25
    hierarchical_v41_max_progress_speed: float = 1.25
    hierarchical_v41_max_lateral_target: float = 1.0
    hierarchical_v41_boundary_buffer: float = 0.35
    hierarchical_v41_safety_margin_range: float = 0.55
    hierarchical_v41_lateral_gain: float = 0.65
    hierarchical_v41_gate_threshold: float = 0.65
    hierarchical_v41_gate_risk_distance: float = 1.8
    hierarchical_v41_target_limit_warning: float = 0.92


class MPPIDBaSOptimizer:
    """A conservative MPPI-DBaS action filter around the SAC action."""

    def __init__(self, config: Optional[MPPIDBaSConfig] = None):
        self.config = config or MPPIDBaSConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.last_action = np.zeros(2, dtype=np.float32)
        self.last_noise_scale = 1.0
        self.best_sequence: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.last_action = np.zeros(2, dtype=np.float32)
        self.last_noise_scale = 1.0
        self.best_sequence = None

    def optimize(self, base_action: Any, planner_state: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, float]]:
        cfg = self.config
        base_action = self._clip_action(base_action)
        if cfg.use_reward_aligned_cost:
            return self._optimize_reward_aligned(base_action, planner_state)

        radar = self._scan_risk_summary(planner_state)
        obstacles = self._scan_to_obstacle_points(planner_state)
        current_min_dist = radar["global_min"]
        base_sequence = np.tile(base_action.reshape(1, 2), (cfg.horizon, 1))
        base_metrics = self._rollout_metrics(base_sequence, base_action, planner_state, obstacles)

        if not self._base_action_needs_filter(base_metrics, radar):
            self.last_action = base_action.astype(np.float32)
            return base_action.astype(np.float32), self._make_passthrough_debug(
                base_action,
                current_min_dist,
                base_metrics,
                radar,
                "base_safe",
            )

        fallback_action = base_action
        fallback_metrics = base_metrics
        fallback_active = False
        fallback_accept = False
        fallback_reason = "fallback_not_needed"
        if cfg.enable_fallback and self._fallback_should_run(base_metrics, radar):
            fallback_active = True
            fallback_action = self._fallback_action(base_action, planner_state, radar)
            fallback_sequence = np.tile(fallback_action.reshape(1, 2), (cfg.horizon, 1))
            fallback_metrics = self._rollout_metrics(fallback_sequence, base_action, planner_state, obstacles)
            fallback_accept, fallback_reason = self._accept_fallback(base_metrics, fallback_metrics)

        if not cfg.enable_mppi:
            executed = fallback_action if fallback_accept else base_action
            source = "fallback" if fallback_accept else "sac"
            self.last_action = executed.astype(np.float32)
            return executed.astype(np.float32), self._make_decision_debug(
                base_action=base_action,
                executed_action=executed,
                candidate_action=base_action,
                base_metrics=base_metrics,
                candidate_metrics=base_metrics,
                fallback_action=fallback_action,
                fallback_metrics=fallback_metrics,
                radar=radar,
                costs=np.array([base_metrics["total_cost"]]),
                min_distances=np.array([base_metrics["min_distance"]]),
                dbas_costs=np.array([base_metrics["dbas_cost"]]),
                ttc_costs=np.array([base_metrics["ttc_cost"]]),
                out_of_bounds_costs=np.array([base_metrics["out_of_bounds_cost"]]),
                noise_scale=0.0,
                mppi_active=False,
                mppi_accept=False,
                action_source=source,
                decision_reason=f"mppi_disabled|{fallback_reason}",
                fallback_active=fallback_active,
                fallback_accept=fallback_accept,
                selected_reason=f"select_{source}",
                reject_reason="mppi_disabled",
                prior_type="none",
                warm_start_used=False,
                teacher_mppi_would_accept=False,
            )

        noise_scale = self._adaptive_noise_scale(current_min_dist)
        action_sequences, prior_names, warm_start_used = self._sample_action_sequences(
            base_action,
            fallback_action,
            noise_scale,
        )
        costs, min_distances, dbas_costs, ttc_costs, out_of_bounds_costs, rollout_metrics = self._evaluate_sequences(
            action_sequences,
            base_action,
            planner_state,
            obstacles,
        )
        best_idx = self._best_sequence_index(rollout_metrics)
        best_sequence = action_sequences[best_idx]
        optimized_action = self._clip_action(best_sequence[0])
        candidate_metrics = rollout_metrics[best_idx]
        prior_type = prior_names[best_idx]
        self.best_sequence = best_sequence.copy()

        accept, decision_reason = self._accept_candidate_against_baselines(
            base_metrics,
            fallback_metrics,
            candidate_metrics,
            optimized_action,
            base_action,
            fallback_accept,
        )
        teacher_mppi_would_accept = bool(accept)

        if accept and not cfg.teacher_only:
            executed_action = optimized_action
            action_source = "mppi"
            selected_reason = "select_mppi"
            reject_reason = "none"
        elif fallback_accept:
            executed_action = fallback_action
            action_source = "fallback"
            selected_reason = "select_fallback"
            reject_reason = "teacher_record_only" if accept and cfg.teacher_only else decision_reason
        else:
            executed_action = base_action
            action_source = "sac"
            selected_reason = "select_sac"
            reject_reason = "teacher_record_only" if accept and cfg.teacher_only else decision_reason

        self.last_action = executed_action.astype(np.float32)
        self.last_noise_scale = float(noise_scale)

        debug = self._make_decision_debug(
            base_action=base_action,
            executed_action=executed_action,
            candidate_action=optimized_action,
            base_metrics=base_metrics,
            candidate_metrics=candidate_metrics,
            fallback_action=fallback_action,
            fallback_metrics=fallback_metrics,
            radar=radar,
            costs=costs,
            min_distances=min_distances,
            dbas_costs=dbas_costs,
            ttc_costs=ttc_costs,
            out_of_bounds_costs=out_of_bounds_costs,
            noise_scale=noise_scale,
            mppi_active=True,
            mppi_accept=accept and not cfg.teacher_only,
            action_source=action_source,
            decision_reason=decision_reason,
            fallback_active=fallback_active,
            fallback_accept=fallback_accept,
            selected_reason=selected_reason,
            reject_reason=reject_reason,
            prior_type=prior_type,
            warm_start_used=warm_start_used,
            teacher_mppi_would_accept=teacher_mppi_would_accept,
        )
        return executed_action.astype(np.float32), debug

    def _optimize_reward_aligned(
        self,
        base_action: np.ndarray,
        planner_state: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        cfg = self.config
        radar = self._scan_risk_summary(planner_state)
        obstacles = self._scan_to_obstacle_points(planner_state)
        base_sequence = np.tile(base_action.reshape(1, 2), (cfg.horizon, 1))
        base_metrics = self._reward_aligned_rollout(base_sequence, base_action, planner_state, obstacles)
        fallback_action = base_action
        fallback_metrics = base_metrics
        fallback_active = False
        fallback_accept = False
        fallback_reason = "fallback_not_needed"

        if not cfg.always_run_mppi and not self._base_action_needs_filter(base_metrics, radar):
            self.last_action = base_action.astype(np.float32)
            return base_action.astype(np.float32), self._make_reward_aligned_debug(
                base_action=base_action,
                executed_action=base_action,
                candidate_action=base_action,
                base_metrics=base_metrics,
                candidate_metrics=base_metrics,
                radar=radar,
                mppi_active=False,
                mppi_selected=False,
                selected_reason="select_sac",
                reject_reason="base_safe",
                prior_type="none",
                warm_start_used=False,
                noise_scale=0.0,
                fallback_metrics=fallback_metrics,
                fallback_active=False,
                fallback_accept=False,
                teacher_mppi_would_accept=False,
            )

        if cfg.enable_fallback and self._fallback_should_run(base_metrics, radar):
            fallback_active = True
            fallback_action = self._fallback_action(base_action, planner_state, radar)
            fallback_sequence = np.tile(fallback_action.reshape(1, 2), (cfg.horizon, 1))
            fallback_metrics = self._reward_aligned_rollout(fallback_sequence, base_action, planner_state, obstacles)
            fallback_accept, fallback_reason = self._accept_fallback(base_metrics, fallback_metrics)

        if not cfg.enable_mppi:
            executed = fallback_action if fallback_accept else base_action
            if self._emergency_brake_needed(base_metrics, base_metrics, radar):
                executed = self._emergency_brake_action(base_action)
            source = "fallback" if not np.allclose(executed, base_action) else "sac"
            self.last_action = executed.astype(np.float32)
            return executed.astype(np.float32), self._make_reward_aligned_debug(
                base_action=base_action,
                executed_action=executed,
                candidate_action=base_action,
                base_metrics=base_metrics,
                candidate_metrics=base_metrics,
                radar=radar,
                mppi_active=False,
                mppi_selected=False,
                selected_reason=f"select_{source}",
                reject_reason=f"mppi_disabled|{fallback_reason}",
                prior_type="none",
                warm_start_used=False,
                noise_scale=0.0,
                fallback_metrics=fallback_metrics,
                fallback_active=fallback_active,
                fallback_accept=fallback_accept,
                teacher_mppi_would_accept=False,
            )

        noise_scale = self._adaptive_noise_scale(radar["global_min"])
        action_sequences, prior_names, warm_start_used = self._sample_sac_centered_sequences(base_action, noise_scale)
        metrics = [
            self._reward_aligned_rollout(sequence, base_action, planner_state, obstacles)
            for sequence in action_sequences
        ]
        best_idx = int(np.argmin([metric["total_cost"] for metric in metrics]))
        best_sequence = action_sequences[best_idx]
        candidate_action = self._clip_action(best_sequence[0])
        candidate_metrics = metrics[best_idx]
        self.best_sequence = best_sequence.copy()

        mppi_would_select, reject_reason = self._accept_reward_aligned_candidate(
            base_metrics,
            fallback_metrics,
            candidate_metrics,
            candidate_action,
            base_action,
            fallback_accept,
        )

        if cfg.teacher_only:
            executed_action = base_action
            action_source = "sac"
            selected_reason = "select_sac"
            reject_reason = "teacher_record_only" if mppi_would_select else reject_reason
            mppi_selected = False
        elif cfg.execute_mppi and mppi_would_select:
            executed_action = candidate_action
            action_source = "mppi"
            selected_reason = "select_mppi"
            reject_reason = "none"
            mppi_selected = True
        elif fallback_accept:
            executed_action = fallback_action
            action_source = "fallback"
            selected_reason = "select_fallback"
            mppi_selected = False
        else:
            executed_action = base_action
            action_source = "sac"
            selected_reason = "select_sac"
            mppi_selected = False

        if cfg.final_safety_check and self._emergency_brake_needed(base_metrics, candidate_metrics, radar):
            executed_action = self._emergency_brake_action(base_action)
            action_source = "fallback"
            selected_reason = "select_emergency_brake"
            reject_reason = "emergency_brake"
            mppi_selected = False

        self.last_action = executed_action.astype(np.float32)
        self.last_noise_scale = float(noise_scale)
        debug = self._make_reward_aligned_debug(
            base_action=base_action,
            executed_action=executed_action,
            candidate_action=candidate_action,
            base_metrics=base_metrics,
            candidate_metrics=candidate_metrics,
            radar=radar,
            mppi_active=True,
            mppi_selected=mppi_selected,
            selected_reason=selected_reason,
            reject_reason=reject_reason,
            prior_type=prior_names[best_idx],
            warm_start_used=warm_start_used,
            noise_scale=noise_scale,
            fallback_metrics=fallback_metrics,
            fallback_active=fallback_active,
            fallback_accept=fallback_accept,
            teacher_mppi_would_accept=mppi_would_select,
        )
        debug["action_source"] = action_source
        debug["terminal_source"] = action_source
        return executed_action.astype(np.float32), debug

    def optimize_from_intent(self, intent: Any, planner_state: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, float]]:
        """Run MPPI every step using SAC's high-level intent as the local prior."""
        cfg = self.config
        params = intent_to_params(intent)
        prior_action = self._clip_action([params["target_speed"], params["turn_bias"]])
        radar = self._scan_risk_summary(planner_state)
        obstacles = self._scan_to_obstacle_points(planner_state)
        noise_scale = self._adaptive_noise_scale(radar["global_min"])
        action_sequences, prior_names, warm_start_used = self._sample_sac_centered_sequences(prior_action, noise_scale)

        metrics = [self._rollout_metrics(sequence, prior_action, planner_state, obstacles) for sequence in action_sequences]
        costs = np.asarray(
            [
                self._intent_conditioned_cost(metric, sequence, prior_action, params)
                for metric, sequence in zip(metrics, action_sequences)
            ],
            dtype=float,
        )
        best_idx = int(np.argmin(costs))
        best_sequence = action_sequences[best_idx]
        executed_action = self._clip_action(best_sequence[0])
        best_metrics = metrics[best_idx]
        self.best_sequence = best_sequence.copy()
        self.last_action = executed_action.astype(np.float32)
        self.last_noise_scale = float(noise_scale)

        debug = self._make_intent_debug(
            intent=np.asarray(intent, dtype=float).reshape(-1),
            params=params,
            executed_action=executed_action,
            prior_action=prior_action,
            metrics=best_metrics,
            costs=costs,
            radar=radar,
            prior_type=prior_names[best_idx],
            warm_start_used=warm_start_used,
            noise_scale=noise_scale,
        )
        return executed_action.astype(np.float32), debug

    def optimize_from_intent_v2(self, intent: Any, planner_state: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, float]]:
        """Use SAC intent as the default action and run MPPI only when local risk warrants it."""
        cfg = self.config
        params = intent_to_params_v2(intent)
        prior_action = self._clip_action([params["target_speed"], params["turn_bias"]])
        radar = self._scan_risk_summary(planner_state)
        obstacles = self._scan_to_obstacle_points(planner_state)
        prior_sequence = np.tile(prior_action.reshape(1, 2), (cfg.horizon, 1))
        prior_metrics = self._reward_aligned_rollout(prior_sequence, prior_action, planner_state, obstacles)
        should_trigger, trigger_reason = self._intent_v2_should_trigger(prior_metrics, radar)

        if not should_trigger or not cfg.enable_mppi:
            executed_action = prior_action
            self.last_action = executed_action.astype(np.float32)
            debug = self._make_intent_v2_debug(
                intent=np.asarray(intent, dtype=float).reshape(-1),
                params=params,
                executed_action=executed_action,
                prior_action=prior_action,
                candidate_action=prior_action,
                prior_metrics=prior_metrics,
                candidate_metrics=prior_metrics,
                costs=np.array([self._intent_conditioned_cost_v2(prior_metrics, prior_sequence, prior_action, params)]),
                radar=radar,
                prior_type="intent_prior",
                warm_start_used=False,
                noise_scale=0.0,
                mppi_triggered=False,
                trigger_reason=trigger_reason,
                candidate_accepted=False,
                reject_reason="base_safe" if cfg.enable_mppi else "mppi_disabled",
                action_source="intent_prior",
            )
            return executed_action.astype(np.float32), debug

        noise_scale = self._adaptive_noise_scale(radar["global_min"])
        action_sequences, prior_names, warm_start_used = self._sample_sac_centered_sequences(prior_action, noise_scale)
        metrics = [
            self._reward_aligned_rollout(sequence, prior_action, planner_state, obstacles)
            for sequence in action_sequences
        ]
        costs = np.asarray(
            [
                self._intent_conditioned_cost_v2(metric, sequence, prior_action, params)
                for metric, sequence in zip(metrics, action_sequences)
            ],
            dtype=float,
        )
        best_idx = int(np.argmin(costs))
        best_sequence = action_sequences[best_idx]
        candidate_action = self._clip_action(best_sequence[0])
        candidate_metrics = metrics[best_idx]
        self.best_sequence = best_sequence.copy()

        accepted, reject_reason = self._accept_intent_v2_candidate(
            prior_metrics=prior_metrics,
            candidate_metrics=candidate_metrics,
            candidate_action=candidate_action,
            prior_action=prior_action,
            prior_score=self._intent_conditioned_cost_v2(prior_metrics, prior_sequence, prior_action, params),
            candidate_score=float(costs[best_idx]),
        )
        if accepted:
            executed_action = candidate_action
            action_source = "hierarchical_mppi"
        elif cfg.enable_fallback and self._fallback_should_run(prior_metrics, radar):
            fallback_action = self._fallback_action(prior_action, planner_state, radar)
            fallback_sequence = np.tile(fallback_action.reshape(1, 2), (cfg.horizon, 1))
            fallback_metrics = self._reward_aligned_rollout(fallback_sequence, prior_action, planner_state, obstacles)
            fallback_accepted, _ = self._accept_fallback(prior_metrics, fallback_metrics)
            if fallback_accepted:
                executed_action = fallback_action
                action_source = "fallback"
            else:
                executed_action = prior_action
                action_source = "intent_prior"
        else:
            executed_action = prior_action
            action_source = "intent_prior"

        if cfg.final_safety_check and self._emergency_brake_needed(prior_metrics, candidate_metrics, radar):
            executed_action = self._emergency_brake_action(prior_action)
            action_source = "fallback"
            reject_reason = "emergency_brake"
            accepted = False

        self.last_action = executed_action.astype(np.float32)
        self.last_noise_scale = float(noise_scale)
        debug = self._make_intent_v2_debug(
            intent=np.asarray(intent, dtype=float).reshape(-1),
            params=params,
            executed_action=executed_action,
            prior_action=prior_action,
            candidate_action=candidate_action,
            prior_metrics=prior_metrics,
            candidate_metrics=candidate_metrics,
            costs=costs,
            radar=radar,
            prior_type=prior_names[best_idx],
            warm_start_used=warm_start_used,
            noise_scale=noise_scale,
            mppi_triggered=True,
            trigger_reason=trigger_reason,
            candidate_accepted=accepted and action_source == "hierarchical_mppi",
            reject_reason=reject_reason,
            action_source=action_source,
        )
        return executed_action.astype(np.float32), debug

    def optimize_from_mode_intent_v4(self, intent: Any, planner_state: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, float]]:
        """Use semi-discrete v4 intent semantics and keep MPPI as a trigger-based local corrector."""
        cfg = self.config
        params = intent_to_mode_params_v4(intent, planner_state, cfg)
        mode_name = str(params["mode_name"])
        prior_action = self._v4_prior_action(params, planner_state)
        radar = self._scan_risk_summary(planner_state)
        obstacles = self._scan_to_obstacle_points(planner_state)
        prior_sequence = np.tile(prior_action.reshape(1, 2), (cfg.horizon, 1))
        prior_metrics = self._reward_aligned_rollout(prior_sequence, prior_action, planner_state, obstacles)
        should_trigger, trigger_reason = self._intent_v4_should_trigger(mode_name, params, prior_metrics, radar)

        fallback_active = False
        fallback_accept = False
        fallback_action = prior_action
        fallback_metrics = prior_metrics
        if cfg.enable_fallback and should_trigger and self._fallback_should_run(prior_metrics, radar):
            fallback_active = True
            fallback_action = self._fallback_action(prior_action, planner_state, radar)
            fallback_sequence = np.tile(fallback_action.reshape(1, 2), (cfg.horizon, 1))
            fallback_metrics = self._reward_aligned_rollout(fallback_sequence, prior_action, planner_state, obstacles)
            fallback_accept, _ = self._accept_fallback(prior_metrics, fallback_metrics)

        if not should_trigger or not cfg.enable_mppi:
            executed_action = fallback_action if fallback_accept else prior_action
            action_source = "fallback" if fallback_accept else "intent_prior"
            self.last_action = executed_action.astype(np.float32)
            debug = self._make_intent_v4_debug(
                intent=np.asarray(intent, dtype=float).reshape(-1),
                params=params,
                executed_action=executed_action,
                prior_action=prior_action,
                candidate_action=prior_action,
                prior_metrics=prior_metrics,
                candidate_metrics=prior_metrics,
                fallback_metrics=fallback_metrics,
                costs=np.array([self._intent_conditioned_cost_v4(prior_metrics, prior_sequence, prior_action, params)]),
                radar=radar,
                prior_type="mode_prior",
                warm_start_used=False,
                noise_scale=0.0,
                mppi_triggered=False,
                trigger_reason=trigger_reason,
                candidate_accepted=False,
                reject_reason="base_safe" if cfg.enable_mppi else "mppi_disabled",
                action_source=action_source,
                fallback_active=fallback_active,
                fallback_accept=fallback_accept,
            )
            return executed_action.astype(np.float32), debug

        noise_scale = self._adaptive_noise_scale(radar["global_min"])
        action_sequences, prior_names, warm_start_used = self._sample_sac_centered_sequences(prior_action, noise_scale)
        metrics = [
            self._reward_aligned_rollout(sequence, prior_action, planner_state, obstacles)
            for sequence in action_sequences
        ]
        costs = np.asarray(
            [
                self._intent_conditioned_cost_v4(metric, sequence, prior_action, params)
                for metric, sequence in zip(metrics, action_sequences)
            ],
            dtype=float,
        )
        best_idx = int(np.argmin(costs))
        candidate_action = self._clip_action(action_sequences[best_idx][0])
        candidate_metrics = metrics[best_idx]
        self.best_sequence = action_sequences[best_idx].copy()

        accepted, reject_reason = self._accept_intent_v4_candidate(
            mode_name=mode_name,
            params=params,
            prior_metrics=prior_metrics,
            candidate_metrics=candidate_metrics,
            candidate_action=candidate_action,
            prior_action=prior_action,
            prior_score=self._intent_conditioned_cost_v4(prior_metrics, prior_sequence, prior_action, params),
            candidate_score=float(costs[best_idx]),
        )

        if fallback_accept:
            prefer_mppi = accepted and self._intent_v4_prefer_mppi_over_fallback(candidate_metrics, fallback_metrics)
            if prefer_mppi:
                executed_action = candidate_action
                action_source = "hierarchical_mppi_v4"
            else:
                executed_action = fallback_action
                action_source = "fallback"
                if accepted:
                    reject_reason = "prefer_fallback"
                    accepted = False
        elif accepted:
            executed_action = candidate_action
            action_source = "hierarchical_mppi_v4"
        else:
            executed_action = prior_action
            action_source = "intent_prior"

        if cfg.final_safety_check and self._emergency_brake_needed(prior_metrics, candidate_metrics, radar):
            executed_action = self._emergency_brake_action(prior_action)
            action_source = "fallback"
            fallback_active = True
            fallback_accept = True
            reject_reason = "emergency_brake"
            accepted = False

        self.last_action = executed_action.astype(np.float32)
        self.last_noise_scale = float(noise_scale)
        debug = self._make_intent_v4_debug(
            intent=np.asarray(intent, dtype=float).reshape(-1),
            params=params,
            executed_action=executed_action,
            prior_action=prior_action,
            candidate_action=candidate_action,
            prior_metrics=prior_metrics,
            candidate_metrics=candidate_metrics,
            fallback_metrics=fallback_metrics,
            costs=costs,
            radar=radar,
            prior_type=prior_names[best_idx],
            warm_start_used=warm_start_used,
            noise_scale=noise_scale,
            mppi_triggered=True,
            trigger_reason=trigger_reason,
            candidate_accepted=accepted and action_source == "hierarchical_mppi_v4",
            reject_reason=reject_reason,
            action_source=action_source,
            fallback_active=fallback_active,
            fallback_accept=fallback_accept,
        )
        return executed_action.astype(np.float32), debug

    def optimize_from_structured_intent_v41(self, intent: Any, planner_state: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, float]]:
        """Use v4.1 continuous structured targets and trigger MPPI only when risk warrants it."""
        cfg = self.config
        params = intent_to_structured_params_v41(intent, planner_state, cfg)
        prior_action = self._structured_prior_action_v41(params, planner_state)
        radar = self._scan_risk_summary(planner_state)
        obstacles = self._scan_to_obstacle_points(planner_state)
        prior_sequence = np.tile(prior_action.reshape(1, 2), (cfg.horizon, 1))
        prior_metrics = self._reward_aligned_rollout(prior_sequence, prior_action, planner_state, obstacles)
        should_trigger, trigger_reason = self._structured_should_trigger_v41(params, prior_metrics, radar)

        fallback_active = False
        fallback_accept = False
        fallback_action = prior_action
        fallback_metrics = prior_metrics
        if cfg.enable_fallback and should_trigger and self._fallback_should_run(prior_metrics, radar):
            fallback_active = True
            fallback_action = self._fallback_action(prior_action, planner_state, radar)
            fallback_sequence = np.tile(fallback_action.reshape(1, 2), (cfg.horizon, 1))
            fallback_metrics = self._reward_aligned_rollout(fallback_sequence, prior_action, planner_state, obstacles)
            fallback_accept, _ = self._accept_fallback(prior_metrics, fallback_metrics)

        if not should_trigger or not cfg.enable_mppi:
            executed_action = fallback_action if fallback_accept else prior_action
            action_source = "fallback" if fallback_accept else "intent_prior"
            self.last_action = executed_action.astype(np.float32)
            debug = self._make_structured_v41_debug(
                intent=np.asarray(intent, dtype=float).reshape(-1),
                params=params,
                executed_action=executed_action,
                prior_action=prior_action,
                candidate_action=prior_action,
                prior_metrics=prior_metrics,
                candidate_metrics=prior_metrics,
                fallback_metrics=fallback_metrics,
                costs=np.array([self._structured_intent_cost_v41(prior_metrics, prior_sequence, prior_action, params)]),
                radar=radar,
                prior_type="structured_prior",
                warm_start_used=False,
                noise_scale=0.0,
                mppi_triggered=False,
                trigger_reason=trigger_reason,
                candidate_accepted=False,
                reject_reason="base_safe" if cfg.enable_mppi else "mppi_disabled",
                action_source=action_source,
                fallback_active=fallback_active,
                fallback_accept=fallback_accept,
            )
            return executed_action.astype(np.float32), debug

        noise_scale = self._adaptive_noise_scale(radar["global_min"])
        action_sequences, prior_names, warm_start_used = self._sample_sac_centered_sequences(prior_action, noise_scale)
        metrics = [
            self._reward_aligned_rollout(sequence, prior_action, planner_state, obstacles)
            for sequence in action_sequences
        ]
        costs = np.asarray(
            [
                self._structured_intent_cost_v41(metric, sequence, prior_action, params)
                for metric, sequence in zip(metrics, action_sequences)
            ],
            dtype=float,
        )
        best_idx = int(np.argmin(costs))
        candidate_action = self._clip_action(action_sequences[best_idx][0])
        candidate_metrics = metrics[best_idx]
        self.best_sequence = action_sequences[best_idx].copy()

        accepted, reject_reason = self._accept_structured_candidate_v41(
            params=params,
            prior_metrics=prior_metrics,
            candidate_metrics=candidate_metrics,
            candidate_action=candidate_action,
            prior_action=prior_action,
            prior_score=self._structured_intent_cost_v41(prior_metrics, prior_sequence, prior_action, params),
            candidate_score=float(costs[best_idx]),
        )

        if fallback_accept and not self._structured_prefer_mppi_over_fallback_v41(candidate_metrics, fallback_metrics, accepted):
            executed_action = fallback_action
            action_source = "fallback"
            if accepted:
                reject_reason = "prefer_fallback"
            accepted = False
        elif accepted:
            executed_action = candidate_action
            action_source = "hierarchical_mppi_v41"
        else:
            executed_action = prior_action
            action_source = "intent_prior"

        if cfg.final_safety_check and self._emergency_brake_needed(prior_metrics, candidate_metrics, radar):
            executed_action = self._emergency_brake_action(prior_action)
            action_source = "fallback"
            fallback_active = True
            fallback_accept = True
            reject_reason = "emergency_brake"
            accepted = False

        self.last_action = executed_action.astype(np.float32)
        self.last_noise_scale = float(noise_scale)
        debug = self._make_structured_v41_debug(
            intent=np.asarray(intent, dtype=float).reshape(-1),
            params=params,
            executed_action=executed_action,
            prior_action=prior_action,
            candidate_action=candidate_action,
            prior_metrics=prior_metrics,
            candidate_metrics=candidate_metrics,
            fallback_metrics=fallback_metrics,
            costs=costs,
            radar=radar,
            prior_type=prior_names[best_idx],
            warm_start_used=warm_start_used,
            noise_scale=noise_scale,
            mppi_triggered=True,
            trigger_reason=trigger_reason,
            candidate_accepted=accepted and action_source == "hierarchical_mppi_v41",
            reject_reason=reject_reason,
            action_source=action_source,
            fallback_active=fallback_active,
            fallback_accept=fallback_accept,
        )
        return executed_action.astype(np.float32), debug

    def optimize_from_frenet_intent_v3(self, intent: Any, planner_state: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, float]]:
        """Use SAC intent as a local Frenet target and let MPPI execute the short-horizon plan."""
        cfg = self.config
        params = intent_to_frenet_params_v3(intent, planner_state, cfg)
        radar = self._scan_risk_summary(planner_state)
        obstacles = self._scan_to_obstacle_points(planner_state)
        prior_action = self._v3_prior_action(params, planner_state)
        noise_scale = self._adaptive_noise_scale(radar["global_min"])
        action_sequences, prior_names, warm_start_used = self._sample_sac_centered_sequences(prior_action, noise_scale)
        metrics = [
            self._frenet_rollout_metrics_v3(sequence, prior_action, planner_state, obstacles, params)
            for sequence in action_sequences
        ]
        costs = np.asarray(
            [
                self._frenet_intent_cost_v3(metric, sequence, params)
                for metric, sequence in zip(metrics, action_sequences)
            ],
            dtype=float,
        )
        best_idx = int(np.argmin(costs))
        best_sequence = action_sequences[best_idx]
        candidate_action = self._clip_action(best_sequence[0])
        candidate_metrics = metrics[best_idx]
        self.best_sequence = best_sequence.copy()

        fallback_active = False
        fallback_accept = False
        reject_reason = "none"
        executed_action = candidate_action
        action_source = "hierarchical_mppi_v3"
        fallback_action = candidate_action
        fallback_metrics = candidate_metrics

        candidate_high_risk = bool(
            candidate_metrics["collision_risk"] > 0.0
            or candidate_metrics["out_of_bounds_risk"] > 0.0
            or candidate_metrics["ttc_cost"] > 0.0
            or candidate_metrics["min_distance"] < cfg.safe_distance
        )
        if cfg.enable_fallback and candidate_high_risk:
            fallback_active = True
            fallback_action = self._fallback_action(prior_action, planner_state, radar)
            fallback_sequence = np.tile(fallback_action.reshape(1, 2), (cfg.horizon, 1))
            fallback_metrics = self._frenet_rollout_metrics_v3(fallback_sequence, prior_action, planner_state, obstacles, params)
            fallback_accept = bool(fallback_metrics["risk_score"] <= candidate_metrics["risk_score"] + 1e-6)
            if fallback_accept:
                executed_action = fallback_action
                action_source = "fallback"
                reject_reason = "fallback_high_risk_mppi"

        if cfg.final_safety_check and self._emergency_brake_needed(candidate_metrics, candidate_metrics, radar):
            executed_action = self._emergency_brake_action(prior_action)
            action_source = "fallback"
            fallback_active = True
            fallback_accept = True
            reject_reason = "emergency_brake"

        self.last_action = executed_action.astype(np.float32)
        self.last_noise_scale = float(noise_scale)
        debug = self._make_frenet_v3_debug(
            intent=np.asarray(intent, dtype=float).reshape(-1),
            params=params,
            executed_action=executed_action,
            prior_action=prior_action,
            candidate_action=candidate_action,
            candidate_metrics=candidate_metrics,
            fallback_metrics=fallback_metrics,
            costs=costs,
            radar=radar,
            prior_type=prior_names[best_idx],
            warm_start_used=warm_start_used,
            noise_scale=noise_scale,
            action_source=action_source,
            fallback_active=fallback_active,
            fallback_accept=fallback_accept,
            reject_reason=reject_reason,
        )
        return executed_action.astype(np.float32), debug

    def _v3_prior_action(self, params: Dict[str, float], planner_state: Dict[str, Any]) -> np.ndarray:
        target_speed = float(params["target_progress_speed"])
        target_offset = float(params["target_lateral_offset"])
        current_d = 0.0
        frenet_transform = planner_state.get("frenet_transform")
        if frenet_transform is not None:
            try:
                _, current_d = frenet_transform.cartesian_to_frenet(np.asarray(planner_state["position"], dtype=float))
            except Exception:
                current_d = 0.0
        lateral_error = target_offset - float(current_d)
        yaw_bias = float(np.clip(0.65 * lateral_error, -0.65, 0.65))
        if params["caution_level"] > 0.75:
            target_speed = min(target_speed, 0.85)
        return self._clip_action([target_speed, yaw_bias])

    def _frenet_rollout_metrics_v3(
        self,
        action_sequence: np.ndarray,
        base_action: np.ndarray,
        planner_state: Dict[str, Any],
        obstacles: Optional[np.ndarray],
        params: Dict[str, float],
    ) -> Dict[str, float]:
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

        if frenet_transform is not None:
            start_s, start_d = frenet_transform.cartesian_to_frenet(position)
            prev_s = float(start_s)
        else:
            start_d = 0.0
            prev_s = 0.0

        min_distance = max_laser_range
        total_ttc_cost = 0.0
        total_oob_cost = 0.0
        max_lateral_error = 0.0
        max_heading_error = 0.0
        lateral_target_error_sum = 0.0
        final_d = float(start_d)
        terminal = "running"

        for action in action_sequence:
            effective_action = cfg.action_lag_alpha * prev_action + (1.0 - cfg.action_lag_alpha) * action
            control_input = np.array([float(effective_action[0]), 0.0, float(effective_action[1])], dtype=float)
            acceleration = (control_input / mass) - (damping * velocity)
            velocity = velocity + acceleration * dt
            yaw = self._wrap_angle(yaw + velocity[2] * dt)
            heading_vec = np.array([math.cos(yaw), math.sin(yaw)], dtype=float)
            position = position + heading_vec * velocity[0] * dt

            if frenet_transform is not None:
                frenet_s, frenet_d = frenet_transform.cartesian_to_frenet(position)
                heading_error = float(frenet_transform.get_heading_error(yaw, frenet_s))
                final_d = float(frenet_d)
            else:
                frenet_s = prev_s + max(0.0, float(velocity[0] * dt))
                frenet_d = 0.0
                heading_error = 0.0
                final_d = 0.0

            lateral_error = abs(float(frenet_d))
            max_lateral_error = max(max_lateral_error, lateral_error)
            max_heading_error = max(max_heading_error, abs(float(heading_error)))
            lateral_target_error_sum += (float(frenet_d) - float(params["target_lateral_offset"])) ** 2

            obstacle_dist = self._min_obstacle_distance(position, obstacles, planner_state)
            min_distance = min(min_distance, obstacle_dist)
            total_ttc_cost += self._ttc_cost(position, heading_vec * velocity[0], obstacles)
            if lateral_error > cfg.env_out_of_bounds_limit:
                total_oob_cost += (lateral_error - cfg.env_out_of_bounds_limit) ** 2 + 1.0
                terminal = "out_of_bounds"
                break
            if obstacle_dist < cfg.collision_distance:
                terminal = "collision"
                break
            prev_s = float(frenet_s)
            prev_action = effective_action

        progress = float(prev_s - float(start_s)) if frenet_transform is not None else float(
            np.linalg.norm(target - np.asarray(planner_state["position"], dtype=float)) - np.linalg.norm(target - position)
        )
        collision_risk = 1.0 if terminal == "collision" or min_distance < cfg.collision_distance else 0.0
        out_of_bounds_risk = 1.0 if terminal == "out_of_bounds" or max_lateral_error > cfg.env_out_of_bounds_limit else 0.0
        safety_violation = max(0.0, float(params["safe_distance"]) - min_distance)
        risk_score = 10.0 * collision_risk + 4.0 * out_of_bounds_risk + total_ttc_cost + 2.0 * safety_violation
        return {
            "min_distance": float(min_distance),
            "ttc_cost": float(total_ttc_cost),
            "out_of_bounds_cost": float(total_oob_cost),
            "max_lateral_error": float(max_lateral_error),
            "max_heading_error": float(max_heading_error),
            "progress": float(progress),
            "target_lateral_error": float(lateral_target_error_sum / max(len(action_sequence), 1)),
            "final_lateral_offset": float(final_d),
            "risk_score": float(risk_score),
            "collision_risk": float(collision_risk),
            "out_of_bounds_risk": float(out_of_bounds_risk),
            "rollout_terminal": terminal,
        }

    def _frenet_intent_cost_v3(
        self,
        metrics: Dict[str, float],
        action_sequence: np.ndarray,
        params: Dict[str, float],
    ) -> float:
        cfg = self.config
        sequence = np.asarray(action_sequence, dtype=float)
        target_progress = float(params["target_progress_speed"]) * cfg.horizon * 0.1
        progress_shortfall = max(0.0, target_progress - float(metrics["progress"]))
        progress_overshoot = max(0.0, float(metrics["progress"]) - 1.25 * target_progress)
        progress_tracking_cost = 12.0 * progress_shortfall ** 2 + 1.5 * progress_overshoot ** 2
        lateral_target_cost = float(params["lateral_target_weight"]) * float(metrics["target_lateral_error"])
        obstacle_barrier = max(0.0, float(params["safe_distance"]) - float(metrics["min_distance"])) ** 2
        safety_cost = (
            float(params["obstacle_weight"]) * obstacle_barrier
            + float(params["ttc_weight"]) * float(metrics["ttc_cost"])
            + 1000.0 * float(metrics["collision_risk"])
        )
        oob_cost = float(params["oob_weight"]) * (
            float(metrics["out_of_bounds_cost"]) + 4.0 * float(metrics["out_of_bounds_risk"])
        )
        smoothness_cost = 0.0
        if len(sequence) > 1:
            smoothness_cost = 3.0 * float(np.mean(np.sum(np.diff(sequence, axis=0) ** 2, axis=1)))
        action_effort_cost = 0.04 * float(np.mean(np.sum(sequence * sequence, axis=1)))
        intent_feasibility_cost = self._intent_feasibility_cost_v3(params, metrics)
        return float(
            progress_tracking_cost
            + lateral_target_cost
            + safety_cost
            + oob_cost
            + smoothness_cost
            + action_effort_cost
            + intent_feasibility_cost
        )

    def _intent_feasibility_cost_v3(self, params: Dict[str, float], metrics: Dict[str, float]) -> float:
        target_offset = abs(float(params["target_lateral_offset"]))
        dynamic_limit = max(float(params["dynamic_offset_limit"]), 1e-6)
        boundary_cost = max(0.0, target_offset - dynamic_limit) ** 2 * 50.0
        obstacle_cost = 20.0 * max(0.0, float(params["safe_distance"]) - float(metrics["min_distance"])) ** 2
        return float(boundary_cost + obstacle_cost)

    def _make_frenet_v3_debug(
        self,
        intent: np.ndarray,
        params: Dict[str, float],
        executed_action: np.ndarray,
        prior_action: np.ndarray,
        candidate_action: np.ndarray,
        candidate_metrics: Dict[str, float],
        fallback_metrics: Dict[str, float],
        costs: np.ndarray,
        radar: Dict[str, float],
        prior_type: str,
        warm_start_used: bool,
        noise_scale: float,
        action_source: str,
        fallback_active: bool,
        fallback_accept: bool,
        reject_reason: str,
    ) -> Dict[str, float]:
        padded = np.zeros(4, dtype=float)
        padded[: min(4, intent.size)] = intent[:4]
        feasibility_cost = self._intent_feasibility_cost_v3(params, candidate_metrics)
        intent_feasible = bool(feasibility_cost < 1e-6 and abs(params["target_lateral_offset"]) <= params["dynamic_offset_limit"] + 1e-6)
        action_delta = np.asarray(executed_action, dtype=float) - np.asarray(prior_action, dtype=float)
        return {
            "action_source": action_source,
            "terminal_source": action_source,
            "mppi_dbas_enabled": True,
            "hierarchical_mppi_v3_enabled": True,
            "mppi_active": True,
            "mppi_accept": action_source == "hierarchical_mppi_v3",
            "mppi_reject": action_source != "hierarchical_mppi_v3",
            "mppi_fallback_active": bool(fallback_active),
            "fallback_active": bool(fallback_active),
            "fallback_accept": bool(fallback_accept),
            "mppi_decision_reason": "select_frenet_mppi_v3",
            "selected_reason": f"select_{action_source}",
            "reject_reason": reject_reason,
            "mppi_triggered": True,
            "mppi_trigger_reason": "always_frenet_v3",
            "candidate_accepted": action_source == "hierarchical_mppi_v3",
            "candidate_reject_reason": reject_reason,
            "mppi_prior_type": prior_type,
            "mppi_warm_start_used": bool(warm_start_used),
            "exploration_noise_scale": float(noise_scale),
            "sac_intent_progress": float(padded[0]),
            "sac_intent_lateral_offset": float(padded[1]),
            "sac_intent_caution": float(padded[2]),
            "sac_intent_recovery": float(padded[3]),
            "target_progress_speed": float(params["target_progress_speed"]),
            "target_lateral_offset": float(params["target_lateral_offset"]),
            "dynamic_offset_limit": float(params["dynamic_offset_limit"]),
            "caution_level": float(params["caution_level"]),
            "recovery_level": float(params["recovery_level"]),
            "path_relaxation": float(params["path_relaxation"]),
            "intent_feasible": bool(intent_feasible),
            "intent_feasibility_cost": float(feasibility_cost),
            "intent_prior_surge": float(prior_action[0]),
            "intent_prior_yaw": float(prior_action[1]),
            "raw_action_surge": float(prior_action[0]),
            "raw_action_yaw": float(prior_action[1]),
            "candidate_action_surge": float(candidate_action[0]),
            "candidate_action_yaw": float(candidate_action[1]),
            "optimized_action_surge": float(executed_action[0]),
            "optimized_action_yaw": float(executed_action[1]),
            "mppi_executed_surge": float(executed_action[0]),
            "mppi_executed_yaw": float(executed_action[1]),
            "action_delta_surge": float(action_delta[0]),
            "action_delta_yaw": float(action_delta[1]),
            "action_delta_norm": float(np.linalg.norm(action_delta)),
            "mppi_best_cost": float(np.min(costs)) if costs.size else 0.0,
            "mppi_predicted_progress": float(candidate_metrics["progress"]),
            "mppi_predicted_lateral_error": float(candidate_metrics["target_lateral_error"]),
            "mppi_predicted_min_obstacle_distance": float(candidate_metrics["min_distance"]),
            "mppi_predicted_oob_risk": float(candidate_metrics["out_of_bounds_risk"]),
            "current_obstacle_distance": float(radar["global_min"]),
            "front_obstacle_distance": float(radar["front_min"]),
            "left_clearance": float(radar["left_min"]),
            "right_clearance": float(radar["right_min"]),
            "min_predicted_obstacle_distance": float(candidate_metrics["min_distance"]),
            "dbas_cost": 0.0,
            "ttc_cost": float(candidate_metrics["ttc_cost"]),
            "out_of_bounds_cost": float(candidate_metrics["out_of_bounds_cost"]),
            "candidate_risk_score": float(candidate_metrics["risk_score"]),
            "candidate_progress": float(candidate_metrics["progress"]),
            "candidate_max_lateral_error": float(candidate_metrics["max_lateral_error"]),
            "candidate_ttc_cost": float(candidate_metrics["ttc_cost"]),
            "candidate_min_distance": float(candidate_metrics["min_distance"]),
            "candidate_pred_collision": bool(candidate_metrics["collision_risk"] > 0.0),
            "candidate_pred_out_of_bounds": bool(candidate_metrics["out_of_bounds_risk"] > 0.0),
            "fallback_risk": float(fallback_metrics["risk_score"]),
            "fallback_progress": float(fallback_metrics["progress"]),
            "fallback_max_lateral_error": float(fallback_metrics["max_lateral_error"]),
            "fallback_ttc_cost": float(fallback_metrics["ttc_cost"]),
        }

    def _intent_v2_should_trigger(self, prior_metrics: Dict[str, float], radar: Dict[str, float]) -> Tuple[bool, str]:
        cfg = self.config
        if radar["front_min"] < cfg.hierarchical_front_trigger_distance:
            return True, "trigger_front_obstacle"
        if radar["global_min"] < cfg.hierarchical_global_trigger_distance:
            return True, "trigger_near_obstacle"
        if prior_metrics.get("ttc_cost", 0.0) > 0.0:
            return True, "trigger_ttc"
        if prior_metrics.get("collision_risk", 0.0) > 0.0:
            return True, "trigger_collision_risk"
        if prior_metrics.get("out_of_bounds_risk", 0.0) > 0.0:
            return True, "trigger_out_of_bounds"
        if prior_metrics.get("max_lateral_error", 0.0) > cfg.hierarchical_lateral_trigger:
            return True, "trigger_lateral_error"
        if prior_metrics.get("max_heading_error", 0.0) > cfg.hierarchical_heading_trigger:
            return True, "trigger_heading_error"
        return False, "base_safe"

    def _accept_intent_v2_candidate(
        self,
        prior_metrics: Dict[str, float],
        candidate_metrics: Dict[str, float],
        candidate_action: np.ndarray,
        prior_action: np.ndarray,
        prior_score: float,
        candidate_score: float,
    ) -> Tuple[bool, str]:
        cfg = self.config
        if not self._trust_region_ok(candidate_action, prior_action):
            return False, "reject_trust_region"
        if candidate_metrics["collision_risk"] > prior_metrics["collision_risk"] + 1e-6:
            return False, "reject_collision_risk"
        if candidate_metrics["out_of_bounds_risk"] > prior_metrics["out_of_bounds_risk"] + 1e-6:
            return False, "reject_out_of_bounds"
        if candidate_metrics["progress"] < prior_metrics["progress"] - cfg.max_progress_loss:
            return False, "reject_progress_loss"
        if candidate_metrics["max_lateral_error"] > prior_metrics["max_lateral_error"] + cfg.max_lateral_worsening:
            return False, "reject_lateral_worsening"
        if (
            prior_metrics["max_lateral_error"] > cfg.hierarchical_lateral_trigger
            and candidate_metrics["max_lateral_error"]
            > prior_metrics["max_lateral_error"] - cfg.hierarchical_lateral_recovery_gain
        ):
            return False, "reject_no_lateral_recovery"
        risk_gain = prior_metrics["risk_score"] - candidate_metrics["risk_score"]
        lateral_gain = prior_metrics["max_lateral_error"] - candidate_metrics["max_lateral_error"]
        ttc_gain = float(prior_metrics.get("ttc_cost", 0.0)) - float(candidate_metrics.get("ttc_cost", 0.0))
        if (
            risk_gain < cfg.hierarchical_min_risk_gain
            and lateral_gain < 0.5 * cfg.hierarchical_lateral_recovery_gain
            and ttc_gain <= 1e-6
        ):
            return False, "reject_no_safety_or_path_gain"
        score_gain = prior_score - candidate_score
        if risk_gain < cfg.hierarchical_min_risk_gain and score_gain < cfg.hierarchical_accept_score_margin:
            return False, "reject_no_score_gain"
        return True, "none"

    def _intent_conditioned_cost_v2(
        self,
        metrics: Dict[str, float],
        action_sequence: np.ndarray,
        prior_action: np.ndarray,
        params: Dict[str, float],
    ) -> float:
        cfg = self.config
        sequence = np.asarray(action_sequence, dtype=float)
        prior = np.asarray(prior_action, dtype=float).reshape(1, 2)
        min_distance = float(metrics.get("min_distance", cfg.safe_distance))
        obstacle_barrier = max(0.0, cfg.safe_distance - min_distance) ** 2
        lateral_error = float(metrics.get("max_lateral_error", 0.0))
        heading_error = float(metrics.get("max_heading_error", 0.0))
        progress = float(metrics.get("progress", 0.0))
        local_path_cost = self._mppi_lateral_cost(lateral_error) + 2.0 * heading_error ** 2
        safety_cost = (
            55.0 * obstacle_barrier
            + 8.0 * float(metrics.get("ttc_cost", 0.0))
            + 90.0 * float(metrics.get("out_of_bounds_cost", 0.0))
            + 1000.0 * float(metrics.get("collision_risk", 0.0))
            + 700.0 * float(metrics.get("out_of_bounds_risk", 0.0))
        )
        prior_tracking = float(np.mean(np.sum((sequence - prior) ** 2, axis=1)))
        smoothness = 0.0
        if len(sequence) > 1:
            smoothness = float(np.mean(np.sum(np.diff(sequence, axis=0) ** 2, axis=1)))
        action_effort = float(np.mean(np.sum(sequence * sequence, axis=1)))
        return float(
            params["path_weight"] * local_path_cost
            + params["safety_weight"] * safety_cost
            + 14.0 * prior_tracking
            + 3.0 * smoothness
            + 0.04 * action_effort
            - 1.25 * progress
        )

    def _mppi_lateral_cost(self, lateral_error: float) -> float:
        lateral_error = abs(float(lateral_error))
        if lateral_error <= 0.4:
            return 0.0
        if lateral_error <= 1.2:
            return float(5.0 * (lateral_error - 0.4) ** 2)
        return float(3.2 + 22.0 * (lateral_error - 1.2) ** 2)

    def _make_intent_v2_debug(
        self,
        intent: np.ndarray,
        params: Dict[str, float],
        executed_action: np.ndarray,
        prior_action: np.ndarray,
        candidate_action: np.ndarray,
        prior_metrics: Dict[str, float],
        candidate_metrics: Dict[str, float],
        costs: np.ndarray,
        radar: Dict[str, float],
        prior_type: str,
        warm_start_used: bool,
        noise_scale: float,
        mppi_triggered: bool,
        trigger_reason: str,
        candidate_accepted: bool,
        reject_reason: str,
        action_source: str,
    ) -> Dict[str, float]:
        padded = np.zeros(4, dtype=float)
        padded[: min(4, intent.size)] = intent[:4]
        return {
            "action_source": action_source,
            "terminal_source": action_source,
            "mppi_dbas_enabled": True,
            "mppi_active": bool(mppi_triggered),
            "mppi_accept": bool(candidate_accepted),
            "mppi_reject": bool(mppi_triggered and not candidate_accepted),
            "mppi_decision_reason": trigger_reason if mppi_triggered else "select_intent_prior",
            "selected_reason": f"select_{action_source}",
            "reject_reason": reject_reason,
            "mppi_triggered": bool(mppi_triggered),
            "mppi_trigger_reason": trigger_reason,
            "candidate_accepted": bool(candidate_accepted),
            "candidate_reject_reason": reject_reason,
            "mppi_prior_type": prior_type,
            "mppi_warm_start_used": bool(warm_start_used),
            "exploration_noise_scale": float(noise_scale),
            "sac_intent_target_speed": float(padded[0]),
            "sac_intent_turn_bias": float(padded[1]),
            "sac_intent_path_weight": float(padded[2]),
            "sac_intent_safety_weight": float(padded[3]),
            "intent_target_speed": float(params["target_speed"]),
            "intent_turn_bias": float(params["turn_bias"]),
            "intent_path_weight": float(params["path_weight"]),
            "intent_safety_weight": float(params["safety_weight"]),
            "intent_prior_surge": float(prior_action[0]),
            "intent_prior_yaw": float(prior_action[1]),
            "raw_action_surge": float(prior_action[0]),
            "raw_action_yaw": float(prior_action[1]),
            "candidate_action_surge": float(candidate_action[0]),
            "candidate_action_yaw": float(candidate_action[1]),
            "optimized_action_surge": float(executed_action[0]),
            "optimized_action_yaw": float(executed_action[1]),
            "mppi_executed_surge": float(executed_action[0]),
            "mppi_executed_yaw": float(executed_action[1]),
            "action_delta_surge": float(executed_action[0] - prior_action[0]),
            "action_delta_yaw": float(executed_action[1] - prior_action[1]),
            "action_delta_norm": float(np.linalg.norm(executed_action - prior_action)),
            "mppi_cost": float(candidate_metrics["total_cost"]),
            "mppi_best_cost": float(np.min(costs)) if len(costs) else float(candidate_metrics["total_cost"]),
            "mppi_mean_cost": float(np.mean(costs)) if len(costs) else float(candidate_metrics["total_cost"]),
            "dbas_cost": float(candidate_metrics["dbas_cost"]),
            "ttc_cost": float(candidate_metrics["ttc_cost"]),
            "out_of_bounds_cost": float(candidate_metrics["out_of_bounds_cost"]),
            "min_predicted_obstacle_distance": float(candidate_metrics["min_distance"]),
            "current_obstacle_distance": float(radar["global_min"]),
            "mppi_pred_collision": bool(candidate_metrics["collision_risk"] > 0.0),
            "mppi_pred_out_of_bounds": bool(candidate_metrics["out_of_bounds_risk"] > 0.0),
            "mppi_min_obstacle_distance": float(candidate_metrics["min_distance"]),
            "candidate_risk": float(candidate_metrics["risk_score"]),
            "candidate_progress": float(candidate_metrics["progress"]),
            "candidate_max_lateral_error": float(candidate_metrics["max_lateral_error"]),
            "candidate_ttc_cost": float(candidate_metrics["ttc_cost"]),
            "candidate_min_distance": float(candidate_metrics["min_distance"]),
            "base_risk": float(prior_metrics["risk_score"]),
            "prior_risk_score": float(prior_metrics["risk_score"]),
            "candidate_risk_score": float(candidate_metrics["risk_score"]),
            "base_progress": float(prior_metrics["progress"]),
            "base_max_lateral_error": float(prior_metrics["max_lateral_error"]),
            "base_ttc_cost": float(prior_metrics["ttc_cost"]),
            "base_min_distance": float(prior_metrics["min_distance"]),
            "sac_pred_collision": bool(prior_metrics["collision_risk"] > 0.0),
            "sac_pred_out_of_bounds": bool(prior_metrics["out_of_bounds_risk"] > 0.0),
            "sac_min_obstacle_distance": float(prior_metrics["min_distance"]),
            "front_obstacle_distance": float(radar["front_min"]),
            "left_clearance": float(radar["left_min"]),
            "right_clearance": float(radar["right_min"]),
            "global_obstacle_distance": float(radar["global_min"]),
        }

    def _structured_prior_action_v41(self, params: Dict[str, float], planner_state: Dict[str, Any]) -> np.ndarray:
        lateral_error = float(params["target_lateral_offset"]) - float(params["current_lateral_offset"])
        yaw_bias = float(np.clip(self.config.hierarchical_v41_lateral_gain * lateral_error, -0.85, 0.85))
        return self._clip_action([float(params["target_progress_speed"]), yaw_bias])

    def _structured_should_trigger_v41(
        self,
        params: Dict[str, float],
        prior_metrics: Dict[str, float],
        radar: Dict[str, float],
    ) -> Tuple[bool, str]:
        base_trigger, base_reason = self._intent_v2_should_trigger(prior_metrics, radar)
        if base_trigger:
            return True, base_reason
        gate_active = (
            float(params["mppi_gate"]) > self.config.hierarchical_v41_gate_threshold
            and radar["global_min"] < self.config.hierarchical_v41_gate_risk_distance
        )
        if gate_active:
            return True, "trigger_mppi_gate"
        return False, "base_safe"

    def _structured_intent_cost_v41(
        self,
        metrics: Dict[str, float],
        action_sequence: np.ndarray,
        prior_action: np.ndarray,
        params: Dict[str, float],
    ) -> float:
        base_params = {
            "target_speed": float(params["target_progress_speed"]),
            "turn_bias": 0.0,
            "path_weight": float(params["path_weight"]),
            "safety_weight": float(params["safety_weight"]),
        }
        base_cost = self._intent_conditioned_cost_v2(metrics, action_sequence, prior_action, base_params)
        target_error = float(metrics.get("final_lateral_offset", params["current_lateral_offset"])) - float(params["target_lateral_offset"])
        target_tracking_cost = float(params["path_weight"]) * 8.0 * target_error ** 2
        safe_distance = float(params["safe_distance"])
        safety_barrier = max(0.0, safe_distance - float(metrics.get("min_distance", safe_distance))) ** 2
        safety_cost = (
            float(params["obstacle_weight"]) * safety_barrier
            + float(params["ttc_weight"]) * float(metrics.get("ttc_cost", 0.0))
            + float(params["oob_weight"]) * float(metrics.get("out_of_bounds_cost", 0.0))
        )
        return float(base_cost + target_tracking_cost + safety_cost)

    def _accept_structured_candidate_v41(
        self,
        params: Dict[str, float],
        prior_metrics: Dict[str, float],
        candidate_metrics: Dict[str, float],
        candidate_action: np.ndarray,
        prior_action: np.ndarray,
        prior_score: float,
        candidate_score: float,
    ) -> Tuple[bool, str]:
        accepted, reason = self._accept_intent_v2_candidate(
            prior_metrics=prior_metrics,
            candidate_metrics=candidate_metrics,
            candidate_action=candidate_action,
            prior_action=prior_action,
            prior_score=prior_score,
            candidate_score=candidate_score,
        )
        if not accepted:
            return False, reason
        prior_target_error = abs(float(prior_metrics.get("final_lateral_offset", params["current_lateral_offset"])) - float(params["target_lateral_offset"]))
        candidate_target_error = abs(float(candidate_metrics.get("final_lateral_offset", params["current_lateral_offset"])) - float(params["target_lateral_offset"]))
        if (
            candidate_metrics["risk_score"] >= prior_metrics["risk_score"] - self.config.hierarchical_min_risk_gain
            and candidate_target_error > prior_target_error - 0.03
        ):
            return False, "reject_no_structured_gain"
        return True, "none"

    def _structured_prefer_mppi_over_fallback_v41(
        self,
        candidate_metrics: Dict[str, float],
        fallback_metrics: Dict[str, float],
        accepted: bool,
    ) -> bool:
        if not accepted:
            return False
        if candidate_metrics["out_of_bounds_risk"] > fallback_metrics["out_of_bounds_risk"] + 1e-6:
            return False
        if candidate_metrics["collision_risk"] > fallback_metrics["collision_risk"] + 1e-6:
            return False
        risk_gain = float(fallback_metrics["risk_score"]) - float(candidate_metrics["risk_score"])
        return bool(risk_gain > self.config.hierarchical_v4_fallback_risk_margin)

    def _make_structured_v41_debug(
        self,
        intent: np.ndarray,
        params: Dict[str, float],
        executed_action: np.ndarray,
        prior_action: np.ndarray,
        candidate_action: np.ndarray,
        prior_metrics: Dict[str, float],
        candidate_metrics: Dict[str, float],
        fallback_metrics: Dict[str, float],
        costs: np.ndarray,
        radar: Dict[str, float],
        prior_type: str,
        warm_start_used: bool,
        noise_scale: float,
        mppi_triggered: bool,
        trigger_reason: str,
        candidate_accepted: bool,
        reject_reason: str,
        action_source: str,
        fallback_active: bool,
        fallback_accept: bool,
    ) -> Dict[str, float]:
        debug = self._make_intent_v2_debug(
            intent=np.asarray(
                [
                    params["raw_progress_intent"],
                    params["raw_lateral_target_intent"],
                    params["raw_safety_margin_intent"],
                    params["raw_mppi_gate_intent"],
                ],
                dtype=float,
            ),
            params={
                "target_speed": float(params["target_progress_speed"]),
                "turn_bias": float(prior_action[1]),
                "path_weight": float(params["path_weight"]),
                "safety_weight": float(params["safety_weight"]),
            },
            executed_action=executed_action,
            prior_action=prior_action,
            candidate_action=candidate_action,
            prior_metrics=prior_metrics,
            candidate_metrics=candidate_metrics,
            costs=costs,
            radar=radar,
            prior_type=prior_type,
            warm_start_used=warm_start_used,
            noise_scale=noise_scale,
            mppi_triggered=mppi_triggered,
            trigger_reason=trigger_reason,
            candidate_accepted=candidate_accepted,
            reject_reason=reject_reason,
            action_source=action_source,
        )
        target_limit_ratio = abs(float(params["target_lateral_offset"])) / max(float(params["dynamic_offset_limit"]), 1e-6)
        target_feasible = bool(target_limit_ratio <= self.config.hierarchical_v41_target_limit_warning)
        debug.update(
            {
                "terminal_source": action_source,
                "hierarchical_mppi_v41_enabled": True,
                "structured_target_progress_speed": float(params["target_progress_speed"]),
                "structured_target_lateral_offset": float(params["target_lateral_offset"]),
                "structured_safety_margin": float(params["safe_distance"] - self.config.safe_distance),
                "structured_mppi_gate": float(params["mppi_gate"]),
                "structured_dynamic_offset_limit": float(params["dynamic_offset_limit"]),
                "structured_current_lateral_offset": float(params["current_lateral_offset"]),
                "structured_target_feasible": bool(target_feasible),
                "structured_target_limit_ratio": float(target_limit_ratio),
                "structured_progress_level": float(params["progress_level"]),
                "structured_safety_level": float(params["safety_level"]),
                "structured_prior_risk_score": float(prior_metrics["risk_score"]),
                "structured_candidate_risk_score": float(candidate_metrics["risk_score"]),
                "structured_prior_target_error": float(
                    abs(float(prior_metrics.get("final_lateral_offset", params["current_lateral_offset"])) - float(params["target_lateral_offset"]))
                ),
                "structured_candidate_target_error": float(
                    abs(float(candidate_metrics.get("final_lateral_offset", params["current_lateral_offset"])) - float(params["target_lateral_offset"]))
                ),
                "fallback_active": bool(fallback_active),
                "fallback_accept": bool(fallback_accept),
                "fallback_risk": float(fallback_metrics["risk_score"]),
                "fallback_progress": float(fallback_metrics["progress"]),
            }
        )
        return debug

    def _v4_prior_action(self, params: Dict[str, Any], planner_state: Dict[str, Any]) -> np.ndarray:
        current_d = float(params.get("current_lateral_offset", 0.0))
        target_speed = float(params["target_speed"])
        mode_name = str(params["mode_name"])
        if mode_name == "recover_center":
            yaw_bias = float(np.clip(-0.6 * current_d, -0.85, 0.85))
        elif mode_name == "brake":
            yaw_bias = float(np.clip(-0.25 * current_d, -0.45, 0.45))
        else:
            yaw_bias = float(params["turn_bias"])
        return self._clip_action([target_speed, yaw_bias])

    def _intent_v4_should_trigger(
        self,
        mode_name: str,
        params: Dict[str, Any],
        prior_metrics: Dict[str, float],
        radar: Dict[str, float],
    ) -> Tuple[bool, str]:
        if mode_name in ("avoid_left", "avoid_right", "recover_center", "brake"):
            return True, f"mode_{mode_name}"
        if mode_name == "cautious_cruise" and (
            radar["front_min"] < self.config.hierarchical_front_trigger_distance * 1.25
            or prior_metrics.get("ttc_cost", 0.0) > 0.0
        ):
            return True, "mode_cautious_cruise"
        return self._intent_v2_should_trigger(prior_metrics, radar)

    def _intent_conditioned_cost_v4(
        self,
        metrics: Dict[str, float],
        action_sequence: np.ndarray,
        prior_action: np.ndarray,
        params: Dict[str, Any],
    ) -> float:
        base_cost = self._intent_conditioned_cost_v2(
            metrics=metrics,
            action_sequence=action_sequence,
            prior_action=prior_action,
            params=params,
        )
        mode_name = str(params["mode_name"])
        direction_cost = 0.0
        current_offset = float(params.get("current_lateral_offset", 0.0))
        if mode_name == "avoid_left":
            direction_cost += 8.0 * max(0.0, -float(metrics.get("progress", 0.0)))
            direction_cost += 4.0 * max(0.0, -float(prior_action[1]))
        elif mode_name == "avoid_right":
            direction_cost += 8.0 * max(0.0, -float(metrics.get("progress", 0.0)))
            direction_cost += 4.0 * max(0.0, float(prior_action[1]))
        elif mode_name == "recover_center":
            direction_cost += 10.0 * float(metrics.get("max_lateral_error", 0.0)) ** 2
            direction_cost += 6.0 * float(metrics.get("final_lateral_offset", current_offset)) ** 2
        elif mode_name == "brake":
            direction_cost += 8.0 * max(0.0, float(action_sequence[0, 0]) - 0.25) ** 2
        elif mode_name == "cruise":
            direction_cost += 5.0 * float(np.mean(action_sequence[:, 1] ** 2))
        return float(base_cost + direction_cost)

    def _accept_intent_v4_candidate(
        self,
        mode_name: str,
        params: Dict[str, Any],
        prior_metrics: Dict[str, float],
        candidate_metrics: Dict[str, float],
        candidate_action: np.ndarray,
        prior_action: np.ndarray,
        prior_score: float,
        candidate_score: float,
    ) -> Tuple[bool, str]:
        accepted, reason = self._accept_intent_v2_candidate(
            prior_metrics=prior_metrics,
            candidate_metrics=candidate_metrics,
            candidate_action=candidate_action,
            prior_action=prior_action,
            prior_score=prior_score,
            candidate_score=candidate_score,
        )
        if not accepted:
            return accepted, reason

        if mode_name == "cruise":
            if abs(float(candidate_action[1] - prior_action[1])) > self.config.hierarchical_v4_cruise_max_yaw_delta:
                return False, "reject_cruise_yaw_override"
        elif mode_name == "avoid_left":
            if float(candidate_action[1]) < -0.05:
                return False, "reject_wrong_direction"
        elif mode_name == "avoid_right":
            if float(candidate_action[1]) > 0.05:
                return False, "reject_wrong_direction"
        elif mode_name == "recover_center":
            current_offset = abs(float(params.get("current_lateral_offset", 0.0)))
            final_offset = abs(float(candidate_metrics.get("final_lateral_offset", current_offset)))
            if final_offset > current_offset - self.config.hierarchical_lateral_recovery_gain:
                return False, "reject_no_center_recovery"
        elif mode_name == "brake":
            if float(candidate_action[0]) > float(prior_action[0]) + 0.05:
                return False, "reject_brake_speedup"
        return True, "none"

    def _intent_v4_prefer_mppi_over_fallback(
        self,
        candidate_metrics: Dict[str, float],
        fallback_metrics: Dict[str, float],
    ) -> bool:
        if candidate_metrics["out_of_bounds_risk"] > fallback_metrics["out_of_bounds_risk"] + 1e-6:
            return False
        if candidate_metrics["collision_risk"] > fallback_metrics["collision_risk"] + 1e-6:
            return False
        risk_margin = float(fallback_metrics["risk_score"]) - float(candidate_metrics["risk_score"])
        if risk_margin > self.config.hierarchical_v4_fallback_risk_margin:
            return True
        return float(candidate_metrics["progress"]) > float(fallback_metrics["progress"]) + 0.03

    def _make_intent_v4_debug(
        self,
        intent: np.ndarray,
        params: Dict[str, Any],
        executed_action: np.ndarray,
        prior_action: np.ndarray,
        candidate_action: np.ndarray,
        prior_metrics: Dict[str, float],
        candidate_metrics: Dict[str, float],
        fallback_metrics: Dict[str, float],
        costs: np.ndarray,
        radar: Dict[str, float],
        prior_type: str,
        warm_start_used: bool,
        noise_scale: float,
        mppi_triggered: bool,
        trigger_reason: str,
        candidate_accepted: bool,
        reject_reason: str,
        action_source: str,
        fallback_active: bool,
        fallback_accept: bool,
    ) -> Dict[str, float]:
        decoded = decode_mode_intent_v4(intent)
        action_delta = np.asarray(executed_action, dtype=float) - np.asarray(prior_action, dtype=float)
        debug = self._make_intent_v2_debug(
            intent=np.asarray([params["target_speed"], params["turn_bias"], params["path_weight"], params["safety_weight"]], dtype=float),
            params={
                "target_speed": float(params["target_speed"]),
                "turn_bias": float(params["turn_bias"]),
                "path_weight": float(params["path_weight"]),
                "safety_weight": float(params["safety_weight"]),
            },
            executed_action=executed_action,
            prior_action=prior_action,
            candidate_action=candidate_action,
            prior_metrics=prior_metrics,
            candidate_metrics=candidate_metrics,
            costs=costs,
            radar=radar,
            prior_type=prior_type,
            warm_start_used=warm_start_used,
            noise_scale=noise_scale,
            mppi_triggered=mppi_triggered,
            trigger_reason=trigger_reason,
            candidate_accepted=candidate_accepted,
            reject_reason=reject_reason,
            action_source=action_source,
        )
        debug.update(
            {
                "action_source": action_source,
                "terminal_source": action_source,
                "hierarchical_mppi_v4_enabled": True,
                "high_level_mode": str(params["mode_name"]),
                "high_level_mode_index": int(params["mode_index"]),
                "high_level_mode_margin": float(params["mode_margin"]),
                "raw_mode": str(decoded["mode_name"]),
                "raw_mode_index": int(decoded["mode_index"]),
                "high_level_speed_scale": float(params["speed_scale"]),
                "high_level_avoid_strength": float(params["avoid_strength"]),
                "high_level_raw_speed_scale": float(params["raw_speed_scale"]),
                "high_level_raw_avoid_strength": float(params["raw_avoid_strength"]),
                "mode_score_cruise": float(params["mode_scores"][0]),
                "mode_score_cautious_cruise": float(params["mode_scores"][1]),
                "mode_score_avoid_left": float(params["mode_scores"][2]),
                "mode_score_avoid_right": float(params["mode_scores"][3]),
                "mode_score_recover_center": float(params["mode_scores"][4]),
                "mode_score_brake": float(params["mode_scores"][5]),
                "mode_target_speed": float(params["target_speed"]),
                "mode_turn_bias": float(params["turn_bias"]),
                "mode_safe_distance": float(params["safe_distance"]),
                "mode_desired_lateral_offset": float(params["desired_lateral_offset"]),
                "mode_current_lateral_offset": float(params["current_lateral_offset"]),
                "fallback_active": bool(fallback_active),
                "fallback_accept": bool(fallback_accept),
                "fallback_risk": float(fallback_metrics["risk_score"]),
                "fallback_progress": float(fallback_metrics["progress"]),
                "fallback_max_lateral_error": float(fallback_metrics["max_lateral_error"]),
                "action_delta_norm": float(np.linalg.norm(action_delta)),
                "mppi_decision_reason": trigger_reason if mppi_triggered else "select_mode_prior",
            }
        )
        return debug

    def _intent_conditioned_cost(
        self,
        metrics: Dict[str, float],
        action_sequence: np.ndarray,
        prior_action: np.ndarray,
        params: Dict[str, float],
    ) -> float:
        sequence = np.asarray(action_sequence, dtype=float)
        prior = np.asarray(prior_action, dtype=float).reshape(1, 2)
        action_error = float(np.mean(np.sum((sequence - prior) ** 2, axis=1)))
        speed_error = float(np.mean((sequence[:, 0] - params["target_speed"]) ** 2))
        yaw_error = float(np.mean((sequence[:, 1] - params["turn_bias"]) ** 2))
        path_cost = (
            8.0 * float(metrics.get("max_lateral_error", 0.0)) ** 2
            + 2.0 * float(metrics.get("max_heading_error", 0.0)) ** 2
            - 4.0 * float(metrics.get("progress", 0.0))
        )
        safety_cost = (
            6.0 * float(metrics.get("dbas_cost", 0.0))
            + 4.0 * float(metrics.get("ttc_cost", 0.0))
            + 25.0 * float(metrics.get("out_of_bounds_cost", 0.0))
            + 500.0 * float(metrics.get("collision_risk", 0.0))
            + 300.0 * float(metrics.get("out_of_bounds_risk", 0.0))
            - 0.5 * float(metrics.get("min_distance", 0.0))
        )
        return float(
            metrics.get("total_cost", 0.0)
            + params["path_weight"] * path_cost
            + params["safety_weight"] * safety_cost
            + 5.0 * action_error
            + 2.0 * speed_error
            + 1.5 * yaw_error
        )

    def _make_intent_debug(
        self,
        intent: np.ndarray,
        params: Dict[str, float],
        executed_action: np.ndarray,
        prior_action: np.ndarray,
        metrics: Dict[str, float],
        costs: np.ndarray,
        radar: Dict[str, float],
        prior_type: str,
        warm_start_used: bool,
        noise_scale: float,
    ) -> Dict[str, float]:
        padded = np.zeros(4, dtype=float)
        padded[: min(4, intent.size)] = intent[:4]
        return {
            "action_source": "hierarchical_mppi",
            "terminal_source": "hierarchical_mppi",
            "mppi_dbas_enabled": True,
            "mppi_active": True,
            "mppi_accept": True,
            "mppi_reject": False,
            "mppi_decision_reason": "select_intent_mppi",
            "selected_reason": "select_intent_mppi",
            "reject_reason": "none",
            "mppi_prior_type": prior_type,
            "mppi_warm_start_used": bool(warm_start_used),
            "exploration_noise_scale": float(noise_scale),
            "sac_intent_target_speed": float(padded[0]),
            "sac_intent_turn_bias": float(padded[1]),
            "sac_intent_path_weight": float(padded[2]),
            "sac_intent_safety_weight": float(padded[3]),
            "intent_target_speed": float(params["target_speed"]),
            "intent_turn_bias": float(params["turn_bias"]),
            "intent_path_weight": float(params["path_weight"]),
            "intent_safety_weight": float(params["safety_weight"]),
            "raw_action_surge": float(prior_action[0]),
            "raw_action_yaw": float(prior_action[1]),
            "optimized_action_surge": float(executed_action[0]),
            "optimized_action_yaw": float(executed_action[1]),
            "mppi_executed_surge": float(executed_action[0]),
            "mppi_executed_yaw": float(executed_action[1]),
            "action_delta_surge": float(executed_action[0] - prior_action[0]),
            "action_delta_yaw": float(executed_action[1] - prior_action[1]),
            "action_delta_norm": float(np.linalg.norm(executed_action - prior_action)),
            "mppi_cost": float(metrics["total_cost"]),
            "mppi_best_cost": float(np.min(costs)) if len(costs) else float(metrics["total_cost"]),
            "mppi_mean_cost": float(np.mean(costs)) if len(costs) else float(metrics["total_cost"]),
            "dbas_cost": float(metrics["dbas_cost"]),
            "ttc_cost": float(metrics["ttc_cost"]),
            "out_of_bounds_cost": float(metrics["out_of_bounds_cost"]),
            "min_predicted_obstacle_distance": float(metrics["min_distance"]),
            "current_obstacle_distance": float(radar["global_min"]),
            "mppi_pred_collision": bool(metrics["collision_risk"] > 0.0),
            "mppi_pred_out_of_bounds": bool(metrics["out_of_bounds_risk"] > 0.0),
            "mppi_min_obstacle_distance": float(metrics["min_distance"]),
            "candidate_risk": float(metrics["risk_score"]),
            "candidate_progress": float(metrics["progress"]),
            "candidate_max_lateral_error": float(metrics["max_lateral_error"]),
            "front_obstacle_distance": float(radar["front_min"]),
            "left_clearance": float(radar["left_min"]),
            "right_clearance": float(radar["right_min"]),
            "global_obstacle_distance": float(radar["global_min"]),
        }

    def _sample_sac_centered_sequences(
        self,
        base_action: np.ndarray,
        noise_scale: float,
    ) -> Tuple[np.ndarray, List[str], bool]:
        cfg = self.config
        residual_low = np.asarray(cfg.reward_aligned_residual_low, dtype=float)
        residual_high = np.asarray(cfg.reward_aligned_residual_high, dtype=float)
        noise_std = np.asarray(cfg.base_noise_std, dtype=float) * noise_scale
        sequences: List[np.ndarray] = [np.tile(base_action.reshape(1, 2), (cfg.horizon, 1))]
        names: List[str] = ["sac_prior"]
        warm_start_used = False

        if self.best_sequence is not None and self.best_sequence.shape == (cfg.horizon, 2):
            shifted = np.vstack([self.best_sequence[1:], base_action.reshape(1, 2)])
            shifted = np.clip(
                shifted,
                base_action.reshape(1, 2) + residual_low.reshape(1, 2),
                base_action.reshape(1, 2) + residual_high.reshape(1, 2),
            )
            sequences.append(self._clip_sequence(shifted))
            names.append("warm_start")
            warm_start_used = True

        while len(sequences) < cfg.num_samples:
            residual = self.rng.normal(0.0, noise_std, size=(cfg.horizon, 2))
            residual = np.clip(residual, residual_low, residual_high)
            sequences.append(self._clip_sequence(base_action.reshape(1, 1, 2) + residual)[0])
            names.append("sac_residual")

        return np.asarray(sequences[: cfg.num_samples], dtype=float), names[: cfg.num_samples], warm_start_used

    def _reward_aligned_rollout(
        self,
        action_sequence: np.ndarray,
        base_action: np.ndarray,
        planner_state: Dict[str, Any],
        obstacles: Optional[np.ndarray],
    ) -> Dict[str, float]:
        cfg = self.config
        dt = float(planner_state.get("dt", 0.1))
        mass = float(planner_state.get("mass", 2.0))
        damping = float(planner_state.get("damping", 0.5))
        position = np.asarray(planner_state["position"], dtype=float).copy()
        start_position = position.copy()
        yaw = float(planner_state.get("yaw", 0.0))
        velocity = np.asarray(planner_state.get("velocity", [0.0, 0.0, 0.0]), dtype=float).copy()
        target = np.asarray(planner_state.get("target_position", position), dtype=float)
        frenet_transform = planner_state.get("frenet_transform")
        max_laser_range = float(planner_state.get("max_laser_range", 10.0))
        prev_action = np.asarray(planner_state.get("last_action", self.last_action), dtype=float).reshape(-1)[:2]

        if frenet_transform is not None:
            prev_s, prev_d = frenet_transform.cartesian_to_frenet(position)
            start_s = float(prev_s)
            prev_abs_d = abs(float(prev_d))
            path_length = float(frenet_transform.path_length)
        else:
            prev_s = 0.0
            start_s = 0.0
            prev_abs_d = 0.0
            path_length = float(np.linalg.norm(target - position) + 1.0)

        predicted_reward = 0.0
        hard_safety_penalty = 0.0
        total_dbas_cost = 0.0
        total_ttc_cost = 0.0
        total_oob_cost = 0.0
        min_distance = max_laser_range
        max_lateral_error = 0.0
        max_heading_error = 0.0
        final_lateral_offset = 0.0
        terminal = "running"
        terminal_step = -1

        for step_idx, action in enumerate(action_sequence):
            effective_action = cfg.action_lag_alpha * prev_action + (1.0 - cfg.action_lag_alpha) * action
            control_input = np.array([float(effective_action[0]), 0.0, float(effective_action[1])], dtype=float)
            acceleration = (control_input / mass) - (damping * velocity)
            velocity = velocity + acceleration * dt
            yaw = self._wrap_angle(yaw + velocity[2] * dt)
            heading_vec = np.array([math.cos(yaw), math.sin(yaw)], dtype=float)
            position = position + heading_vec * velocity[0] * dt

            dist_to_goal = float(np.linalg.norm(target - position))
            if frenet_transform is not None:
                frenet_s, frenet_d = frenet_transform.cartesian_to_frenet(position)
                heading_error = float(frenet_transform.get_heading_error(yaw, frenet_s))
                distance_remaining = max(float(path_length - frenet_s), 0.0)
            else:
                frenet_s = prev_s + max(0.0, float(velocity[0] * dt))
                frenet_d = 0.0
                heading_error = 0.0
                distance_remaining = dist_to_goal

            delta_s = float(frenet_s - prev_s)
            lateral_error = abs(float(frenet_d))
            final_lateral_offset = float(frenet_d)
            max_lateral_error = max(max_lateral_error, lateral_error)
            max_heading_error = max(max_heading_error, abs(float(heading_error)))

            obstacle_dist = self._min_obstacle_distance(position, obstacles, planner_state)
            min_distance = min(min_distance, obstacle_dist)
            dbas_cost = self._dbas_cost(obstacle_dist)
            ttc_cost = self._ttc_cost(position, heading_vec * velocity[0], obstacles)
            total_dbas_cost += dbas_cost
            total_ttc_cost += ttc_cost

            if obstacle_dist < cfg.collision_distance:
                predicted_reward -= cfg.env_terminal_reward
                hard_safety_penalty += cfg.hard_safety_penalty
                terminal = "collision"
                terminal_step = step_idx
                break
            if lateral_error > cfg.env_out_of_bounds_limit:
                total_oob_cost += piecewise_lateral_penalty(lateral_error)
            if distance_remaining < cfg.env_goal_threshold and lateral_error <= 1.0:
                predicted_reward += cfg.env_success_reward
                terminal = "success"
                terminal_step = step_idx
                break

            reward_info = compute_tracking_reward(
                delta_s,
                frenet_d,
                heading_error,
                min_obstacle_dist=obstacle_dist,
                previous_abs_frenet_d=prev_abs_d,
                safe_distance=cfg.env_safe_distance,
                action=action,
                previous_action=prev_action,
                reward_scale=cfg.env_frenet_reward_scale,
            )
            env_reward = float(reward_info["total"])
            predicted_reward += env_reward
            prev_s = frenet_s
            prev_abs_d = lateral_error
            prev_action = effective_action

        trust_delta = action_sequence - base_action.reshape(1, 2)
        trust_region_cost = float(np.mean(np.sum(trust_delta * trust_delta, axis=1)))
        total_cost = (
            -predicted_reward
            + hard_safety_penalty
            + cfg.trust_region_weight * trust_region_cost
            + 0.25 * cfg.dbas_weight * total_dbas_cost
            + 0.25 * cfg.ttc_weight * total_ttc_cost
            + cfg.out_of_bounds_weight * total_oob_cost
        )
        if frenet_transform is not None:
            progress = float(prev_s - start_s)
        else:
            progress = float(np.linalg.norm(target - start_position) - np.linalg.norm(target - position))
        collision_risk = 1.0 if terminal == "collision" or min_distance < cfg.collision_distance else 0.0
        out_of_bounds_risk = 1.0 if terminal == "out_of_bounds" or max_lateral_error > cfg.env_out_of_bounds_limit else 0.0
        risk_score = 10.0 * collision_risk + 4.0 * out_of_bounds_risk + total_ttc_cost + 0.2 * total_dbas_cost

        return {
            "total_cost": float(total_cost),
            "predicted_reward": float(predicted_reward),
            "hard_safety_penalty": float(hard_safety_penalty),
            "trust_region_cost": float(trust_region_cost),
            "min_distance": float(min_distance),
            "dbas_cost": float(total_dbas_cost),
            "ttc_cost": float(total_ttc_cost),
            "out_of_bounds_cost": float(total_oob_cost),
            "max_lateral_error": float(max_lateral_error),
            "max_heading_error": float(max_heading_error),
            "final_lateral_offset": float(final_lateral_offset),
            "progress": float(progress),
            "final_distance_to_goal": float(np.linalg.norm(target - position)),
            "risk_score": float(risk_score),
            "collision_risk": float(collision_risk),
            "out_of_bounds_risk": float(out_of_bounds_risk),
            "rollout_terminal": terminal,
            "rollout_terminal_step": float(terminal_step),
        }

    def _trust_region_ok(self, candidate_action: np.ndarray, base_action: np.ndarray) -> bool:
        action_delta = np.abs(candidate_action - base_action)
        return bool(np.all(action_delta <= np.asarray(self.config.mppi_max_action_delta, dtype=float) + 1e-6))

    def _accept_reward_aligned_candidate(
        self,
        base_metrics: Dict[str, float],
        fallback_metrics: Dict[str, float],
        candidate_metrics: Dict[str, float],
        candidate_action: np.ndarray,
        base_action: np.ndarray,
        fallback_accept: bool,
    ) -> Tuple[bool, str]:
        cfg = self.config
        reward_delta = candidate_metrics["predicted_reward"] - base_metrics["predicted_reward"]
        if not self._trust_region_ok(candidate_action, base_action):
            return False, "reject_trust_region"
        if cfg.final_safety_check and not self._final_safety_ok(candidate_metrics):
            return False, "reject_hard_safety"
        if candidate_metrics["progress"] < base_metrics["progress"] - cfg.max_progress_loss:
            return False, "reject_progress_loss"
        if candidate_metrics["max_lateral_error"] > base_metrics["max_lateral_error"] + cfg.max_lateral_worsening:
            return False, "reject_out_of_bounds"

        safety_gain = (
            candidate_metrics["min_distance"] >= base_metrics["min_distance"] + cfg.safety_distance_margin
            or candidate_metrics["ttc_cost"] < base_metrics["ttc_cost"] - 1e-6
            or candidate_metrics["risk_score"] < base_metrics["risk_score"] - 0.05
        )
        reward_gain = reward_delta >= cfg.reward_improvement_threshold
        if not reward_gain:
            return False, "reject_no_reward_gain"
        if not safety_gain and not cfg.teacher_only:
            return False, "reject_no_reward_gain" if not reward_gain else "reject_no_safety_gain"

        if fallback_accept:
            if candidate_metrics["risk_score"] >= fallback_metrics["risk_score"] - 1e-6:
                return False, "reject_not_better_than_fallback"
            if candidate_metrics["progress"] < fallback_metrics["progress"] - cfg.max_progress_loss:
                return False, "reject_worse_than_fallback_progress"
            if candidate_metrics["max_lateral_error"] > fallback_metrics["max_lateral_error"] + cfg.max_lateral_worsening:
                return False, "reject_worse_than_fallback_oob"
            if candidate_metrics["total_cost"] > fallback_metrics["total_cost"] - cfg.fallback_score_margin:
                return False, "reject_not_better_than_fallback"

        return True, "accept_reward_aligned_candidate"

    def _final_safety_ok(self, candidate_metrics: Dict[str, float]) -> bool:
        return bool(
            candidate_metrics["collision_risk"] <= 0.0
            and candidate_metrics["out_of_bounds_risk"] <= 0.0
            and candidate_metrics["min_distance"] >= self.config.collision_distance
        )

    def _emergency_brake_needed(
        self,
        base_metrics: Dict[str, float],
        candidate_metrics: Dict[str, float],
        radar: Dict[str, float],
    ) -> bool:
        if radar["global_min"] < self.config.emergency_brake_distance:
            return True
        return bool(base_metrics["collision_risk"] > 0.0 and candidate_metrics["collision_risk"] > 0.0)

    def _emergency_brake_action(self, base_action: np.ndarray) -> np.ndarray:
        action = base_action.astype(float).copy()
        action[0] = min(action[0], self.config.hard_brake_surge)
        action[1] = 0.0
        return self._clip_action(action)

    def _piecewise_lateral_penalty(self, lateral_error: float) -> float:
        return piecewise_lateral_penalty(lateral_error)

    @staticmethod
    def _reward_aligned_reject_reason(reward_ok: bool, trust_ok: bool, safety_ok: bool) -> str:
        if not safety_ok:
            return "reject_hard_safety"
        if not trust_ok:
            return "reject_trust_region"
        if not reward_ok:
            return "reject_no_reward_gain"
        return "reject_disabled"

    def _make_reward_aligned_debug(
        self,
        base_action: np.ndarray,
        executed_action: np.ndarray,
        candidate_action: np.ndarray,
        base_metrics: Dict[str, float],
        candidate_metrics: Dict[str, float],
        radar: Dict[str, float],
        mppi_active: bool,
        mppi_selected: bool,
        selected_reason: str,
        reject_reason: str,
        prior_type: str,
        warm_start_used: bool,
        noise_scale: float,
        fallback_metrics: Optional[Dict[str, float]] = None,
        fallback_active: bool = False,
        fallback_accept: bool = False,
        teacher_mppi_would_accept: bool = False,
    ) -> Dict[str, float]:
        action_delta = executed_action - base_action
        predicted_reward_delta = candidate_metrics["predicted_reward"] - base_metrics["predicted_reward"]
        source = "mppi" if mppi_selected else ("fallback" if selected_reason == "select_emergency_brake" else "sac")
        if selected_reason == "select_fallback":
            source = "fallback"
        fallback_metrics = fallback_metrics or base_metrics
        return {
            "raw_action_surge": float(base_action[0]),
            "raw_action_yaw": float(base_action[1]),
            "optimized_action_surge": float(executed_action[0]),
            "optimized_action_yaw": float(executed_action[1]),
            "candidate_action_surge": float(candidate_action[0]),
            "candidate_action_yaw": float(candidate_action[1]),
            "action_delta_surge": float(action_delta[0]),
            "action_delta_yaw": float(action_delta[1]),
            "action_delta_norm": float(np.linalg.norm(action_delta)),
            "mppi_cost": float(candidate_metrics["total_cost"]),
            "dbas_cost": float(candidate_metrics["dbas_cost"]),
            "ttc_cost": float(candidate_metrics["ttc_cost"]),
            "out_of_bounds_cost": float(candidate_metrics["out_of_bounds_cost"]),
            "min_predicted_obstacle_distance": float(candidate_metrics["min_distance"]),
            "current_obstacle_distance": float(radar["global_min"]),
            "exploration_noise_scale": float(noise_scale),
            "mppi_active": bool(mppi_active),
            "mppi_accept": bool(mppi_selected),
            "mppi_reject": bool(mppi_active and not mppi_selected),
            "mppi_selected": bool(mppi_selected),
            "mppi_decision_reason": selected_reason if mppi_selected else reject_reason,
            "selected_reason": selected_reason,
            "reject_reason": reject_reason,
            "mppi_rejected_reason": reject_reason,
            "mppi_prior_type": prior_type,
            "mppi_warm_start_used": bool(warm_start_used),
            "teacher_mppi_would_accept": bool(teacher_mppi_would_accept),
            "fallback_active": bool(fallback_active or selected_reason == "select_emergency_brake"),
            "fallback_accept": bool(fallback_accept or selected_reason == "select_emergency_brake"),
            "action_source": source,
            "terminal_source": source,
            "predicted_reward_sac": float(base_metrics["predicted_reward"]),
            "predicted_reward_mppi": float(candidate_metrics["predicted_reward"]),
            "predicted_reward_delta": float(predicted_reward_delta),
            "reward_prediction_error": 0.0,
            "sac_rollout_terminal": base_metrics["rollout_terminal"],
            "mppi_rollout_terminal": candidate_metrics["rollout_terminal"],
            "candidate_sac_score": float(base_metrics["total_cost"]),
            "candidate_fallback_score": float(fallback_metrics["total_cost"]),
            "candidate_mppi_score": float(candidate_metrics["total_cost"]),
            "sac_pred_collision": bool(base_metrics["collision_risk"] > 0.0),
            "fallback_pred_collision": bool(fallback_metrics["collision_risk"] > 0.0),
            "mppi_pred_collision": bool(candidate_metrics["collision_risk"] > 0.0),
            "sac_pred_out_of_bounds": bool(base_metrics["out_of_bounds_risk"] > 0.0),
            "fallback_pred_out_of_bounds": bool(fallback_metrics["out_of_bounds_risk"] > 0.0),
            "mppi_pred_out_of_bounds": bool(candidate_metrics["out_of_bounds_risk"] > 0.0),
            "sac_min_obstacle_distance": float(base_metrics["min_distance"]),
            "fallback_min_obstacle_distance": float(fallback_metrics["min_distance"]),
            "mppi_min_obstacle_distance": float(candidate_metrics["min_distance"]),
            "base_risk": float(base_metrics["risk_score"]),
            "candidate_risk": float(candidate_metrics["risk_score"]),
            "fallback_risk": float(fallback_metrics["risk_score"]),
            "base_min_distance": float(base_metrics["min_distance"]),
            "candidate_min_distance": float(candidate_metrics["min_distance"]),
            "fallback_min_distance": float(fallback_metrics["min_distance"]),
            "base_ttc_cost": float(base_metrics["ttc_cost"]),
            "candidate_ttc_cost": float(candidate_metrics["ttc_cost"]),
            "fallback_ttc_cost": float(fallback_metrics["ttc_cost"]),
            "base_max_lateral_error": float(base_metrics["max_lateral_error"]),
            "candidate_max_lateral_error": float(candidate_metrics["max_lateral_error"]),
            "fallback_max_lateral_error": float(fallback_metrics["max_lateral_error"]),
            "base_progress": float(base_metrics["progress"]),
            "candidate_progress": float(candidate_metrics["progress"]),
            "fallback_progress": float(fallback_metrics["progress"]),
            "front_obstacle_distance": float(radar["front_min"]),
            "left_clearance": float(radar["left_min"]),
            "right_clearance": float(radar["right_min"]),
            "global_obstacle_distance": float(radar["global_min"]),
        }

    def _sample_action_sequences(
        self,
        base_action: np.ndarray,
        fallback_action: np.ndarray,
        noise_scale: float,
    ) -> Tuple[np.ndarray, List[str], bool]:
        cfg = self.config
        priors = self._prior_sequences(base_action, fallback_action)
        warm_start_used = False
        if self.best_sequence is not None and self.best_sequence.shape == (cfg.horizon, 2):
            shifted = np.vstack([self.best_sequence[1:], base_action.reshape(1, 2)])
            priors.append(("warm_start", shifted))
            warm_start_used = True

        per_prior = max(1, int(math.ceil(cfg.num_samples / len(priors))))
        sequences: List[np.ndarray] = []
        names: List[str] = []
        residual_low = np.asarray(cfg.residual_low, dtype=float)
        residual_high = np.asarray(cfg.residual_high, dtype=float)
        noise_std = np.asarray(cfg.base_noise_std, dtype=float) * noise_scale

        for prior_name, prior_sequence in priors:
            clipped_prior = self._clip_sequence(prior_sequence)
            sequences.append(clipped_prior)
            names.append(prior_name)
            for _ in range(per_prior - 1):
                residual = self.rng.normal(0.0, noise_std, size=(cfg.horizon, 2))
                residual = np.clip(residual, residual_low, residual_high)
                residual[0] = np.clip(residual[0], residual_low, residual_high)
                sequences.append(self._clip_sequence(clipped_prior + residual))
                names.append(prior_name)

        return np.asarray(sequences[: cfg.num_samples], dtype=float), names[: cfg.num_samples], warm_start_used

    def _prior_sequences(self, base_action: np.ndarray, fallback_action: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        cfg = self.config
        brake = self._clip_action([min(base_action[0], cfg.hard_brake_surge), 0.0])
        left_escape = self._clip_action([min(base_action[0], cfg.fallback_surge), cfg.fallback_yaw])
        right_escape = self._clip_action([min(base_action[0], cfg.fallback_surge), -cfg.fallback_yaw])
        return [
            ("sac_prior", np.tile(base_action.reshape(1, 2), (cfg.horizon, 1))),
            ("brake_prior", np.tile(brake.reshape(1, 2), (cfg.horizon, 1))),
            ("left_escape_prior", np.tile(left_escape.reshape(1, 2), (cfg.horizon, 1))),
            ("right_escape_prior", np.tile(right_escape.reshape(1, 2), (cfg.horizon, 1))),
            ("fallback_prior", np.tile(fallback_action.reshape(1, 2), (cfg.horizon, 1))),
        ]

    def _clip_sequence(self, sequence: np.ndarray) -> np.ndarray:
        return np.clip(
            np.asarray(sequence, dtype=float),
            np.asarray(self.config.action_low, dtype=float),
            np.asarray(self.config.action_high, dtype=float),
        )

    def _evaluate_sequences(
        self,
        action_sequences: np.ndarray,
        base_action: np.ndarray,
        planner_state: Dict[str, Any],
        obstacles: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, float]]]:
        count = len(action_sequences)
        costs = np.zeros(count, dtype=float)
        min_distances = np.full(count, float(planner_state.get("max_laser_range", 10.0)), dtype=float)
        dbas_costs = np.zeros(count, dtype=float)
        ttc_costs = np.zeros(count, dtype=float)
        out_of_bounds_costs = np.zeros(count, dtype=float)
        metrics: List[Dict[str, float]] = []

        for sample_idx, sequence in enumerate(action_sequences):
            sample_metrics = self._rollout_metrics(sequence, base_action, planner_state, obstacles)
            metrics.append(sample_metrics)
            costs[sample_idx] = sample_metrics["total_cost"]
            min_distances[sample_idx] = sample_metrics["min_distance"]
            dbas_costs[sample_idx] = sample_metrics["dbas_cost"]
            ttc_costs[sample_idx] = sample_metrics["ttc_cost"]
            out_of_bounds_costs[sample_idx] = sample_metrics["out_of_bounds_cost"]

        return costs, min_distances, dbas_costs, ttc_costs, out_of_bounds_costs, metrics

    def _best_sequence_index(self, metrics: List[Dict[str, float]]) -> int:
        return min(range(len(metrics)), key=lambda idx: self._selection_key(metrics[idx]))

    def _selection_key(self, metrics: Dict[str, float]) -> Tuple[float, float, float, float, float, float]:
        return (
            float(metrics["collision_risk"]),
            float(metrics["out_of_bounds_risk"]),
            float(metrics["ttc_cost"]),
            -float(metrics["min_distance"]),
            float(metrics["max_lateral_error"]),
            float(metrics["total_cost"]),
        )

    def _candidate_score(self, metrics: Dict[str, float]) -> float:
        collision_penalty = 1000.0 * float(metrics["collision_risk"])
        oob_penalty = 500.0 * float(metrics["out_of_bounds_risk"])
        return float(
            collision_penalty
            + oob_penalty
            + 25.0 * metrics["ttc_cost"]
            - 2.0 * metrics["min_distance"]
            + 2.0 * metrics["max_lateral_error"]
            - 0.5 * metrics["progress"]
            + 0.02 * metrics["total_cost"]
        )

    def _accept_candidate_against_baselines(
        self,
        base_metrics: Dict[str, float],
        fallback_metrics: Dict[str, float],
        candidate_metrics: Dict[str, float],
        optimized_action: np.ndarray,
        base_action: np.ndarray,
        fallback_accept: bool,
    ) -> Tuple[bool, str]:
        candidate_ok, candidate_reason = self._accept_candidate(
            base_metrics,
            candidate_metrics,
            optimized_action,
            base_action,
        )
        if not candidate_ok:
            return False, candidate_reason

        if fallback_accept:
            if candidate_metrics["risk_score"] >= fallback_metrics["risk_score"] - 1e-6:
                return False, "reject_not_better_than_fallback"
            if candidate_metrics["collision_risk"] > fallback_metrics["collision_risk"]:
                return False, "reject_worse_than_fallback_collision"
            if candidate_metrics["out_of_bounds_risk"] > fallback_metrics["out_of_bounds_risk"]:
                return False, "reject_worse_than_fallback_oob"
            if candidate_metrics["max_lateral_error"] > fallback_metrics["max_lateral_error"] + self.config.max_lateral_worsening:
                return False, "reject_worse_than_fallback_oob"
            if candidate_metrics["progress"] < fallback_metrics["progress"] - self.config.max_progress_loss:
                return False, "reject_worse_than_fallback_progress"
            fallback_score = self._candidate_score(fallback_metrics)
            candidate_score = self._candidate_score(candidate_metrics)
            if candidate_score > fallback_score - self.config.fallback_score_margin:
                return False, "reject_not_better_than_fallback"

        base_score = self._candidate_score(base_metrics)
        candidate_score = self._candidate_score(candidate_metrics)
        if candidate_score > base_score - self.config.mppi_min_score_gain:
            return False, "reject_no_score_gain"
        return True, "accept_candidate_selection"

    def _make_decision_debug(
        self,
        base_action: np.ndarray,
        executed_action: np.ndarray,
        candidate_action: np.ndarray,
        base_metrics: Dict[str, float],
        candidate_metrics: Dict[str, float],
        fallback_action: np.ndarray,
        fallback_metrics: Dict[str, float],
        radar: Dict[str, float],
        costs: np.ndarray,
        min_distances: np.ndarray,
        dbas_costs: np.ndarray,
        ttc_costs: np.ndarray,
        out_of_bounds_costs: np.ndarray,
        noise_scale: float,
        mppi_active: bool,
        mppi_accept: bool,
        action_source: str,
        decision_reason: str,
        fallback_active: bool,
        fallback_accept: bool,
        selected_reason: str,
        reject_reason: str,
        prior_type: str,
        warm_start_used: bool,
        teacher_mppi_would_accept: bool,
    ) -> Dict[str, float]:
        action_delta = executed_action - base_action
        fallback_delta = fallback_action - base_action
        return {
            "raw_action_surge": float(base_action[0]),
            "raw_action_yaw": float(base_action[1]),
            "optimized_action_surge": float(executed_action[0]),
            "optimized_action_yaw": float(executed_action[1]),
            "candidate_action_surge": float(candidate_action[0]),
            "candidate_action_yaw": float(candidate_action[1]),
            "fallback_action_surge": float(fallback_action[0]),
            "fallback_action_yaw": float(fallback_action[1]),
            "action_delta_surge": float(action_delta[0]),
            "action_delta_yaw": float(action_delta[1]),
            "action_delta_norm": float(np.linalg.norm(action_delta)),
            "fallback_delta_norm": float(np.linalg.norm(fallback_delta)),
            "mppi_cost": float(candidate_metrics["total_cost"]),
            "mppi_mean_cost": float(np.mean(costs)) if len(costs) else 0.0,
            "dbas_cost": float(candidate_metrics["dbas_cost"]),
            "dbas_mean_cost": float(np.mean(dbas_costs)) if len(dbas_costs) else 0.0,
            "ttc_cost": float(candidate_metrics["ttc_cost"]),
            "out_of_bounds_cost": float(candidate_metrics["out_of_bounds_cost"]),
            "min_predicted_obstacle_distance": float(candidate_metrics["min_distance"]),
            "current_obstacle_distance": float(radar["global_min"]),
            "exploration_noise_scale": float(noise_scale),
            "mppi_active": bool(mppi_active),
            "mppi_accept": bool(mppi_accept),
            "mppi_reject": bool(mppi_active and not mppi_accept),
            "mppi_decision_reason": decision_reason,
            "selected_reason": selected_reason,
            "reject_reason": reject_reason,
            "mppi_prior_type": prior_type,
            "mppi_warm_start_used": bool(warm_start_used),
            "teacher_mppi_would_accept": bool(teacher_mppi_would_accept),
            "fallback_active": bool(fallback_active),
            "fallback_accept": bool(fallback_accept),
            "action_source": action_source,
            "terminal_source": action_source,
            "candidate_sac_score": float(self._candidate_score(base_metrics)),
            "candidate_fallback_score": float(self._candidate_score(fallback_metrics)),
            "candidate_mppi_score": float(self._candidate_score(candidate_metrics)),
            "sac_pred_collision": bool(base_metrics["collision_risk"] > 0.0),
            "fallback_pred_collision": bool(fallback_metrics["collision_risk"] > 0.0),
            "mppi_pred_collision": bool(candidate_metrics["collision_risk"] > 0.0),
            "sac_pred_out_of_bounds": bool(base_metrics["out_of_bounds_risk"] > 0.0),
            "fallback_pred_out_of_bounds": bool(fallback_metrics["out_of_bounds_risk"] > 0.0),
            "mppi_pred_out_of_bounds": bool(candidate_metrics["out_of_bounds_risk"] > 0.0),
            "sac_min_obstacle_distance": float(base_metrics["min_distance"]),
            "fallback_min_obstacle_distance": float(fallback_metrics["min_distance"]),
            "mppi_min_obstacle_distance": float(candidate_metrics["min_distance"]),
            "base_risk": float(base_metrics["risk_score"]),
            "candidate_risk": float(candidate_metrics["risk_score"]),
            "fallback_risk": float(fallback_metrics["risk_score"]),
            "base_min_distance": float(base_metrics["min_distance"]),
            "candidate_min_distance": float(candidate_metrics["min_distance"]),
            "fallback_min_distance": float(fallback_metrics["min_distance"]),
            "base_ttc_cost": float(base_metrics["ttc_cost"]),
            "candidate_ttc_cost": float(candidate_metrics["ttc_cost"]),
            "fallback_ttc_cost": float(fallback_metrics["ttc_cost"]),
            "base_max_lateral_error": float(base_metrics["max_lateral_error"]),
            "candidate_max_lateral_error": float(candidate_metrics["max_lateral_error"]),
            "fallback_max_lateral_error": float(fallback_metrics["max_lateral_error"]),
            "base_progress": float(base_metrics["progress"]),
            "candidate_progress": float(candidate_metrics["progress"]),
            "fallback_progress": float(fallback_metrics["progress"]),
            "front_obstacle_distance": float(radar["front_min"]),
            "left_clearance": float(radar["left_min"]),
            "right_clearance": float(radar["right_min"]),
            "global_obstacle_distance": float(radar["global_min"]),
        }

    def _rollout_cost(
        self,
        action_sequence: np.ndarray,
        base_action: np.ndarray,
        planner_state: Dict[str, Any],
        obstacles: Optional[np.ndarray],
    ) -> Tuple[float, float, float, float, float]:
        metrics = self._rollout_metrics(action_sequence, base_action, planner_state, obstacles)
        return (
            metrics["total_cost"],
            metrics["min_distance"],
            metrics["dbas_cost"],
            metrics["ttc_cost"],
            metrics["out_of_bounds_cost"],
        )

    def _rollout_metrics(
        self,
        action_sequence: np.ndarray,
        base_action: np.ndarray,
        planner_state: Dict[str, Any],
        obstacles: Optional[np.ndarray],
    ) -> Dict[str, float]:
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
        max_lateral_error = 0.0
        max_heading_error = 0.0

        for action in action_sequence:
            effective_action = cfg.action_lag_alpha * prev_action + (1.0 - cfg.action_lag_alpha) * action
            control_input = np.array([float(effective_action[0]), 0.0, float(effective_action[1])], dtype=float)
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
                max_lateral_error = max(max_lateral_error, lateral_error)
                max_heading_error = max(max_heading_error, heading_error)
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
            prev_action = effective_action

        final_dist_to_goal = float(np.linalg.norm(target - position))
        progress = start_dist_to_goal - final_dist_to_goal
        collision_risk = 1.0 if min_distance < cfg.collision_distance else 0.0
        safety_violation = max(0.0, cfg.safe_distance - min_distance)
        out_of_bounds_risk = 1.0 if max_lateral_error > cfg.out_of_bounds_limit else 0.0
        risk_score = (
            10.0 * collision_risk
            + 4.0 * out_of_bounds_risk
            + 2.0 * safety_violation
            + total_ttc_cost
            + 0.2 * total_dbas_cost
        )

        return {
            "total_cost": float(total_cost),
            "min_distance": float(min_distance),
            "dbas_cost": float(total_dbas_cost),
            "ttc_cost": float(total_ttc_cost),
            "out_of_bounds_cost": float(total_oob_cost),
            "max_lateral_error": float(max_lateral_error),
            "max_heading_error": float(max_heading_error),
            "progress": float(progress),
            "final_distance_to_goal": float(final_dist_to_goal),
            "risk_score": float(risk_score),
            "collision_risk": float(collision_risk),
            "out_of_bounds_risk": float(out_of_bounds_risk),
        }

    def _base_action_needs_filter(self, base_metrics: Dict[str, float], radar: Dict[str, float]) -> bool:
        cfg = self.config
        if radar["global_min"] < cfg.fallback_trigger_distance:
            return True
        if radar["front_min"] < cfg.risk_activation_distance:
            return True
        if base_metrics["min_distance"] < cfg.safe_distance:
            return True
        if base_metrics["ttc_cost"] > 0.0:
            return True
        if base_metrics["out_of_bounds_risk"] > 0.0:
            return True
        return False

    def _fallback_should_run(self, base_metrics: Dict[str, float], radar: Dict[str, float]) -> bool:
        cfg = self.config
        if radar["global_min"] < cfg.fallback_trigger_distance:
            return True
        if base_metrics["collision_risk"] > 0.0:
            return True
        if base_metrics["min_distance"] < cfg.safe_distance:
            return True
        if base_metrics["ttc_cost"] > 0.0:
            return True
        return False

    def _accept_candidate(
        self,
        base_metrics: Dict[str, float],
        candidate_metrics: Dict[str, float],
        optimized_action: np.ndarray,
        base_action: np.ndarray,
    ) -> Tuple[bool, str]:
        cfg = self.config
        action_delta = np.abs(optimized_action - base_action)
        if np.any(action_delta > np.asarray(cfg.mppi_max_action_delta, dtype=float) + 1e-6):
            return False, "reject_trust_region"
        if candidate_metrics["collision_risk"] > base_metrics["collision_risk"]:
            return False, "reject_collision_risk"
        if candidate_metrics["min_distance"] < cfg.collision_distance:
            return False, "reject_collision_risk"
        if candidate_metrics["out_of_bounds_risk"] > 0.0:
            return False, "reject_out_of_bounds"
        if candidate_metrics["max_lateral_error"] > base_metrics["max_lateral_error"] + cfg.max_lateral_worsening:
            return False, "reject_out_of_bounds"
        if candidate_metrics["progress"] < base_metrics["progress"] - cfg.max_progress_loss:
            return False, "reject_progress_loss"

        distance_gain = candidate_metrics["min_distance"] - base_metrics["min_distance"]
        ttc_gain = base_metrics["ttc_cost"] - candidate_metrics["ttc_cost"]
        risk_gain = base_metrics["risk_score"] - candidate_metrics["risk_score"]
        if distance_gain >= cfg.safety_distance_margin or ttc_gain > 1e-6 or risk_gain > 0.05:
            return True, "accept_safety_gain"
        return False, "reject_no_safety_gain"

    def _accept_fallback(self, base_metrics: Dict[str, float], fallback_metrics: Dict[str, float]) -> Tuple[bool, str]:
        cfg = self.config
        if fallback_metrics["out_of_bounds_risk"] > 0.0:
            return False, "fallback_reject_out_of_bounds"
        if fallback_metrics["max_lateral_error"] > base_metrics["max_lateral_error"] + cfg.fallback_max_lateral_worsening:
            return False, "fallback_reject_out_of_bounds"
        risk_gain = base_metrics["risk_score"] - fallback_metrics["risk_score"]
        distance_gain = fallback_metrics["min_distance"] - base_metrics["min_distance"]
        ttc_gain = base_metrics["ttc_cost"] - fallback_metrics["ttc_cost"]
        if risk_gain > 0.0 or distance_gain >= 0.0 or ttc_gain > 0.0:
            return True, "fallback_accept_safety"
        return False, "fallback_reject_no_safety_gain"

    def _fallback_or_base(
        self,
        base_action: np.ndarray,
        base_metrics: Dict[str, float],
        planner_state: Dict[str, Any],
        obstacles: Optional[np.ndarray],
        radar: Dict[str, float],
        reason: str,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        if not self.config.enable_fallback or not self._fallback_should_run(base_metrics, radar):
            self.last_action = base_action.astype(np.float32)
            return base_action.astype(np.float32), self._make_passthrough_debug(
                base_action, radar["global_min"], base_metrics, radar, reason
            )
        fallback_action = self._fallback_action(base_action, planner_state, radar)
        fallback_sequence = np.tile(fallback_action.reshape(1, 2), (self.config.horizon, 1))
        fallback_metrics = self._rollout_metrics(fallback_sequence, base_action, planner_state, obstacles)
        accept, fallback_reason = self._accept_fallback(base_metrics, fallback_metrics)
        executed = fallback_action if accept else base_action
        self.last_action = executed.astype(np.float32)
        action_delta = executed - base_action
        debug = self._make_passthrough_debug(executed, radar["global_min"], base_metrics, radar, fallback_reason)
        debug.update({
            "raw_action_surge": float(base_action[0]),
            "raw_action_yaw": float(base_action[1]),
            "optimized_action_surge": float(executed[0]),
            "optimized_action_yaw": float(executed[1]),
            "action_delta_surge": float(action_delta[0]),
            "action_delta_yaw": float(action_delta[1]),
            "action_delta_norm": float(np.linalg.norm(action_delta)),
            "fallback_active": True,
            "fallback_accept": bool(accept),
            "action_source": "fallback" if accept else "sac",
            "terminal_source": "fallback" if accept else "sac",
            "fallback_risk": float(fallback_metrics["risk_score"]),
            "fallback_min_distance": float(fallback_metrics["min_distance"]),
            "fallback_ttc_cost": float(fallback_metrics["ttc_cost"]),
            "fallback_max_lateral_error": float(fallback_metrics["max_lateral_error"]),
            "fallback_progress": float(fallback_metrics["progress"]),
        })
        return executed.astype(np.float32), debug

    def _fallback_action(self, base_action: np.ndarray, planner_state: Dict[str, Any], radar: Dict[str, float]) -> np.ndarray:
        cfg = self.config
        action = base_action.astype(float).copy()
        lateral_error = 0.0
        frenet_transform = planner_state.get("frenet_transform")
        if frenet_transform is not None:
            position = np.asarray(planner_state["position"], dtype=float)
            _, frenet_d = frenet_transform.cartesian_to_frenet(position)
            lateral_error = abs(float(frenet_d))

        recover_speed = cfg.hard_brake_surge if (radar["global_min"] < cfg.safe_distance or lateral_error > 1.2) else cfg.fallback_surge
        action[0] = min(action[0], recover_speed)

        clearance_delta = radar["left_min"] - radar["right_min"]
        path_center_yaw = self._yaw_toward_path_center(planner_state, default_yaw=0.0)
        strong_center_recovery = lateral_error > cfg.hierarchical_lateral_trigger

        if strong_center_recovery and abs(path_center_yaw) > 1e-6:
            yaw = path_center_yaw
        elif abs(clearance_delta) >= cfg.fallback_min_clearance_delta:
            yaw = cfg.fallback_yaw if clearance_delta > 0.0 else -cfg.fallback_yaw
        else:
            yaw = path_center_yaw
            if abs(yaw) < 1e-6:
                yaw = cfg.fallback_yaw if radar["left_min"] >= radar["right_min"] else -cfg.fallback_yaw

        action[1] = yaw
        candidate = self._clip_action(action)
        if self._one_step_lateral_worsens(candidate, base_action, planner_state):
            candidate[0] = min(candidate[0], cfg.hard_brake_surge)
            candidate[1] = self._yaw_toward_path_center(planner_state, default_yaw=0.0)
            candidate = self._clip_action(candidate)
        return candidate

    def _yaw_toward_path_center(self, planner_state: Dict[str, Any], default_yaw: float) -> float:
        frenet_transform = planner_state.get("frenet_transform")
        if frenet_transform is None:
            return default_yaw
        position = np.asarray(planner_state["position"], dtype=float)
        _, frenet_d = frenet_transform.cartesian_to_frenet(position)
        if abs(float(frenet_d)) < 0.05:
            return default_yaw
        magnitude = min(1.0, self.config.fallback_yaw + 0.12 * abs(float(frenet_d)))
        return -math.copysign(magnitude, float(frenet_d))

    def _one_step_lateral_worsens(
        self,
        candidate_action: np.ndarray,
        base_action: np.ndarray,
        planner_state: Dict[str, Any],
    ) -> bool:
        frenet_transform = planner_state.get("frenet_transform")
        if frenet_transform is None:
            return False
        base_metrics = self._rollout_metrics(np.tile(base_action.reshape(1, 2), (2, 1)), base_action, planner_state, None)
        cand_metrics = self._rollout_metrics(np.tile(candidate_action.reshape(1, 2), (2, 1)), base_action, planner_state, None)
        return cand_metrics["max_lateral_error"] > base_metrics["max_lateral_error"] + self.config.fallback_max_lateral_worsening

    def _make_passthrough_debug(
        self,
        base_action: np.ndarray,
        current_min_dist: float,
        base_metrics: Optional[Dict[str, float]] = None,
        radar: Optional[Dict[str, float]] = None,
        reason: str = "base_safe",
    ) -> Dict[str, float]:
        max_range = current_min_dist
        radar = radar or {
            "global_min": current_min_dist,
            "front_min": max_range,
            "left_min": max_range,
            "right_min": max_range,
        }
        base_metrics = base_metrics or {
            "total_cost": 0.0,
            "risk_score": 0.0,
            "min_distance": current_min_dist,
            "dbas_cost": 0.0,
            "ttc_cost": 0.0,
            "out_of_bounds_cost": 0.0,
            "max_lateral_error": 0.0,
            "progress": 0.0,
            "collision_risk": 0.0,
            "out_of_bounds_risk": 0.0,
        }
        base_score = self._candidate_score(base_metrics)
        return {
            "raw_action_surge": float(base_action[0]),
            "raw_action_yaw": float(base_action[1]),
            "optimized_action_surge": float(base_action[0]),
            "optimized_action_yaw": float(base_action[1]),
            "candidate_action_surge": float(base_action[0]),
            "candidate_action_yaw": float(base_action[1]),
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
            "mppi_accept": False,
            "mppi_reject": False,
            "mppi_decision_reason": reason,
            "selected_reason": "select_sac",
            "reject_reason": reason,
            "mppi_prior_type": "none",
            "mppi_warm_start_used": False,
            "teacher_mppi_would_accept": False,
            "fallback_active": False,
            "fallback_accept": False,
            "action_source": "sac",
            "terminal_source": "sac",
            "candidate_sac_score": float(base_score),
            "candidate_fallback_score": float(base_score),
            "candidate_mppi_score": float(base_score),
            "sac_pred_collision": bool(base_metrics["collision_risk"] > 0.0),
            "fallback_pred_collision": bool(base_metrics["collision_risk"] > 0.0),
            "mppi_pred_collision": bool(base_metrics["collision_risk"] > 0.0),
            "sac_pred_out_of_bounds": bool(base_metrics["out_of_bounds_risk"] > 0.0),
            "fallback_pred_out_of_bounds": bool(base_metrics["out_of_bounds_risk"] > 0.0),
            "mppi_pred_out_of_bounds": bool(base_metrics["out_of_bounds_risk"] > 0.0),
            "sac_min_obstacle_distance": float(base_metrics["min_distance"]),
            "fallback_min_obstacle_distance": float(base_metrics["min_distance"]),
            "mppi_min_obstacle_distance": float(base_metrics["min_distance"]),
            "base_risk": float(base_metrics["risk_score"]),
            "candidate_risk": float(base_metrics["risk_score"]),
            "fallback_risk": float(base_metrics["risk_score"]),
            "base_min_distance": float(base_metrics["min_distance"]),
            "candidate_min_distance": float(base_metrics["min_distance"]),
            "fallback_min_distance": float(base_metrics["min_distance"]),
            "base_ttc_cost": float(base_metrics["ttc_cost"]),
            "candidate_ttc_cost": float(base_metrics["ttc_cost"]),
            "fallback_ttc_cost": float(base_metrics["ttc_cost"]),
            "base_max_lateral_error": float(base_metrics["max_lateral_error"]),
            "candidate_max_lateral_error": float(base_metrics["max_lateral_error"]),
            "fallback_max_lateral_error": float(base_metrics["max_lateral_error"]),
            "base_progress": float(base_metrics["progress"]),
            "candidate_progress": float(base_metrics["progress"]),
            "fallback_progress": float(base_metrics["progress"]),
            "front_obstacle_distance": float(radar["front_min"]),
            "left_clearance": float(radar["left_min"]),
            "right_clearance": float(radar["right_min"]),
            "global_obstacle_distance": float(radar["global_min"]),
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
        forward_mask = np.abs(np.array([self._wrap_angle(a) for a in scan_angles])) <= self.config.obstacle_sector_half_angle
        valid = (ranges > 0.02) & (ranges < self.config.risk_activation_distance) & forward_mask
        if not np.any(valid):
            return None

        world_angles = float(planner_state.get("yaw", 0.0)) + scan_angles[valid]
        position = np.asarray(planner_state["position"], dtype=float)
        return position + np.column_stack((np.cos(world_angles), np.sin(world_angles))) * ranges[valid, None]

    def _scan_risk_summary(self, planner_state: Dict[str, Any]) -> Dict[str, float]:
        scan = planner_state.get("scan")
        max_range = float(planner_state.get("max_laser_range", 10.0))
        if scan is None or not hasattr(scan, "ranges"):
            return {
                "global_min": max_range,
                "front_min": max_range,
                "left_min": max_range,
                "right_min": max_range,
            }
        ranges = np.asarray(scan.ranges, dtype=float)
        ranges = np.nan_to_num(ranges, nan=max_range, posinf=max_range, neginf=max_range)
        ranges = np.clip(ranges, 0.0, max_range)
        angle_min = float(getattr(scan, "angle_min", -math.pi))
        angle_increment = float(getattr(scan, "angle_increment", (2.0 * math.pi) / max(len(ranges), 1)))
        angles = np.array([self._wrap_angle(a) for a in angle_min + np.arange(len(ranges), dtype=float) * angle_increment])
        valid = ranges > 0.02

        def sector_min(mask: np.ndarray) -> float:
            sector = ranges[valid & mask]
            return float(np.min(sector)) if sector.size else max_range

        front_mask = np.abs(angles) <= self.config.front_sector_half_angle
        left_mask = (angles > 0.0) & (angles <= self.config.obstacle_sector_half_angle)
        right_mask = (angles < 0.0) & (angles >= -self.config.obstacle_sector_half_angle)
        return {
            "global_min": float(np.min(ranges[valid])) if np.any(valid) else max_range,
            "front_min": sector_min(front_mask),
            "left_min": sector_min(left_mask),
            "right_min": sector_min(right_mask),
        }

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
