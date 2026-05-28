import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from mppi_dbas import MPPIDBaSConfig, MPPIDBaSOptimizer


@dataclass
class RLDrivenMPPIConfig(MPPIDBaSConfig):
    """Configuration for the paper-style RL-driven MPPI controller."""

    num_rl_rollouts: int = 64
    num_mppi_rollouts: int = 64
    num_iterations: int = 3
    top_z: int = 64
    initial_sigma: Tuple[float, float] = (0.35, 0.35)
    sigma_min: Tuple[float, float] = (0.05, 0.05)
    use_rl_initialization: bool = True
    use_hss: bool = True
    update_sigma: bool = True
    use_terminal_q: bool = True
    terminal_q_weight: float = 0.001
    pure_mppi: bool = False
    strict_terminal_q: bool = False
    observation_stack: int = 1
    controller_name: str = "rl_driven_mppi"
    rlmppi_accept_gate: bool = False
    rlmppi_trigger_distance: float = 1.8
    rlmppi_lateral_trigger: float = 0.9
    rlmppi_min_reward_gain: float = 1.0
    rlmppi_min_action_delta: float = 0.08
    rlmppi_min_risk_gain: float = 0.05
    rlmppi_center_assist: bool = False
    rlmppi_center_assist_distance: float = 2.2
    rlmppi_center_assist_lateral: float = 0.75
    rlmppi_shield_fallback: bool = False
    rlmppi_shield_front_trigger_distance: float = 1.2
    rlmppi_shield_approach_surge: float = 0.15
    rlmppi_progress_assist: bool = False
    rlmppi_progress_assist_distance: float = 1.2
    rlmppi_progress_assist_surge: float = 0.85
    rlmppi_progress_heading_gain: float = 1.25
    rlmppi_progress_center_gain: float = 0.20
    rlmppi_progress_recovery_lateral: float = 1.0
    rlmppi_progress_recovery_surge: float = 0.55


class SB3SacPolicyAdapter:
    """Small adapter around a Stable-Baselines3 SAC policy.

    The optimizer still accepts the action produced by the evaluation model as
    a fallback mean, so this adapter degrades cleanly when no model is loaded.
    """

    def __init__(self, model: Any = None):
        self.model = model
        self.action_space = getattr(getattr(model, "env", None), "action_space", None)

    def predict_mean(self, observation: Any, fallback_action: np.ndarray) -> np.ndarray:
        if self.model is None or observation is None:
            return np.asarray(fallback_action, dtype=float)
        try:
            action, _ = self.model.predict(observation, deterministic=True)
            return np.asarray(action, dtype=float).reshape(-1)[:2]
        except Exception:
            return np.asarray(fallback_action, dtype=float)

    def sample_action(self, observation: Any, fallback_action: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.model is None or observation is None:
            noise = rng.normal(0.0, [0.25, 0.25], size=2)
            return np.asarray(fallback_action, dtype=float) + noise
        try:
            action, _ = self.model.predict(observation, deterministic=False)
            return np.asarray(action, dtype=float).reshape(-1)[:2]
        except Exception:
            noise = rng.normal(0.0, [0.25, 0.25], size=2)
            return np.asarray(fallback_action, dtype=float) + noise

    def action_std(self) -> np.ndarray:
        return np.array([0.35, 0.35], dtype=float)

    def estimate_terminal_cost(self, observation: Any, action: np.ndarray) -> Tuple[float, bool]:
        if self.model is None or observation is None:
            return 0.0, False
        try:
            import torch

            obs_tensor, _ = self.model.policy.obs_to_tensor(observation)
            action_arr = np.asarray(action, dtype=np.float32).reshape(1, -1)
            action_tensor = torch.as_tensor(action_arr, device=self.model.device)
            with torch.no_grad():
                q_values = self.model.critic(obs_tensor, action_tensor)
                q_value = torch.min(torch.cat(q_values, dim=1), dim=1).values
            return float(-q_value.detach().cpu().numpy().reshape(-1)[0]), True
        except Exception:
            return 0.0, False


class DSACPolicyAdapter:
    """Strict adapter for DSAC actor and distributional critic."""

    def __init__(self, policy: Any):
        self.policy = policy

    @classmethod
    def load(cls, path: str, device: str = "auto") -> "DSACPolicyAdapter":
        from dsac import DSACPolicy

        return cls(DSACPolicy.load(path, device=device))

    def predict_mean(self, observation: Any, fallback_action: np.ndarray) -> np.ndarray:
        action, _ = self.policy.predict(observation, deterministic=True)
        return np.asarray(action, dtype=float).reshape(-1)[:2]

    def sample_action(self, observation: Any, fallback_action: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        action, _ = self.policy.predict(observation, deterministic=False)
        return np.asarray(action, dtype=float).reshape(-1)[:2]

    def action_std(self, observation: Any = None) -> np.ndarray:
        if observation is None:
            raise ValueError("DSACPolicyAdapter requires an observation for action_std().")
        return np.asarray(self.policy.action_std(observation), dtype=float).reshape(-1)[:2]

    def estimate_terminal_cost(self, observation: Any, action: np.ndarray) -> Tuple[float, bool]:
        if observation is None:
            raise ValueError("DSAC terminal cost requires a stacked observation.")
        return float(self.policy.terminal_cost(observation, action)), True


class TransitionModel:
    """Interface reserved for learned dynamics models used by MPPI rollouts."""

    def predict_next(self, state: Dict[str, Any], action: np.ndarray) -> Dict[str, Any]:
        raise NotImplementedError


class USVApproxTransitionModel(TransitionModel):
    """Marker for the current analytic USV rollout implemented in the optimizer."""

    def predict_next(self, state: Dict[str, Any], action: np.ndarray) -> Dict[str, Any]:
        raise NotImplementedError("The analytic USV rollout is implemented by RLDrivenMPPIOptimizer._rollout_metrics().")


class RLDrivenMPPIOptimizer(MPPIDBaSOptimizer):
    """RL-driven MPPI following the paper's online Algorithm 1 structure."""

    def __init__(
        self,
        config: Optional[RLDrivenMPPIConfig] = None,
        policy_adapter: Optional[SB3SacPolicyAdapter] = None,
    ):
        super().__init__(config or RLDrivenMPPIConfig())
        self.config: RLDrivenMPPIConfig
        self.policy_adapter = policy_adapter or SB3SacPolicyAdapter()
        self.last_mean_sequence: Optional[np.ndarray] = None
        self.last_sigma_sequence: Optional[np.ndarray] = None

    def reset(self) -> None:
        super().reset()
        self.last_mean_sequence = None
        self.last_sigma_sequence = None

    def optimize(
        self,
        base_action: Any,
        planner_state: Dict[str, Any],
        observation: Any = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        start_time = time.perf_counter()
        cfg = self.config
        base_action = self._clip_action(base_action)
        radar = self._scan_risk_summary(planner_state)
        obstacles = self._scan_to_obstacle_points(planner_state)
        base_sequence = np.tile(base_action.reshape(1, 2), (cfg.horizon, 1))
        base_metrics = self._sequence_metrics(base_sequence, base_action, planner_state, obstacles)

        if cfg.rlmppi_accept_gate and not self._rlmppi_hazard_active(base_metrics, radar):
            executed, shield_metrics, shield_active, shield_accept, shield_reason = self._shield_fallback_action(
                base_action=base_action,
                base_metrics=base_metrics,
                planner_state=planner_state,
                obstacles=obstacles,
                radar=radar,
            )
            if not shield_accept:
                executed = self._progress_assist_action(base_action, planner_state, radar)
                executed = self._center_assist_action(executed, planner_state, radar)
            self.last_action = executed.astype(np.float32)
            selected_metrics = shield_metrics if shield_accept else base_metrics
            debug = self._make_debug(
                base_action=base_action,
                executed=executed,
                base_metrics=base_metrics,
                best_metrics=selected_metrics,
                best_sequence=base_sequence,
                costs=np.array([selected_metrics["total_cost"]], dtype=float),
                sigma_sequence=np.zeros_like(base_sequence, dtype=float),
                elapsed_ms=(time.perf_counter() - start_time) * 1000.0,
                init_source="shield_fallback" if shield_accept else "pre_gate",
                terminal_q_used=False,
                top_count=0,
                action_delta=executed - base_action,
            )
            reject_reason = shield_reason if shield_active else "base_safe_gate"
            self._apply_accept_gate_debug(debug, False, reject_reason, base_action, executed)
            debug["mppi_active"] = False
            debug["mppi_triggered"] = False
            debug["fallback_active"] = bool(shield_active)
            debug["fallback_accept"] = bool(shield_accept)
            debug["fallback_reason"] = shield_reason
            debug["fallback_pred_collision"] = bool(shield_metrics.get("collision_risk", 0.0) > 0.0)
            debug["fallback_pred_out_of_bounds"] = bool(shield_metrics.get("out_of_bounds_risk", 0.0) > 0.0)
            debug["fallback_min_obstacle_distance"] = float(shield_metrics.get("min_distance", base_metrics["min_distance"]))
            debug["fallback_risk"] = float(shield_metrics.get("risk_score", base_metrics["risk_score"]))
            debug["fallback_min_distance"] = float(shield_metrics.get("min_distance", base_metrics["min_distance"]))
            debug["fallback_ttc_cost"] = float(shield_metrics.get("ttc_cost", base_metrics["ttc_cost"]))
            debug["fallback_max_lateral_error"] = float(shield_metrics.get("max_lateral_error", base_metrics["max_lateral_error"]))
            debug["fallback_progress"] = float(shield_metrics.get("progress", base_metrics["progress"]))
            debug["candidate_fallback_score"] = float(-shield_metrics.get("total_cost", base_metrics["total_cost"]))
            if shield_accept:
                debug["selected_reason"] = "select_fallback"
                debug["action_source"] = "fallback"
                debug["terminal_source"] = "fallback"
                debug["mppi_decision_reason"] = shield_reason
            elif np.linalg.norm(executed - base_action) > 1e-6:
                debug["selected_reason"] = "select_progress_assist"
                debug["action_source"] = "sac"
                debug["terminal_source"] = "sac"
                debug["mppi_decision_reason"] = "progress_assist"
            debug["rlmppi_num_rl_rollouts"] = 0
            debug["rlmppi_num_mppi_rollouts"] = 0
            debug["rlmppi_num_iterations"] = 0
            debug["rlmppi_top_z"] = 0
            debug["rlmppi_online_time_ms"] = float((time.perf_counter() - start_time) * 1000.0)
            return executed.astype(np.float32), debug

        mean_sequence, sigma_sequence, init_source = self._initial_distribution(base_action, observation)
        guided_sequences = self._sample_guided_rollouts(mean_sequence, sigma_sequence, observation, base_action)

        all_iteration_costs = []
        best_metrics = base_metrics
        best_sequence = mean_sequence.copy()
        terminal_q_used = False
        selected_top_count = 0

        for _ in range(int(cfg.num_iterations)):
            mppi_sequences = self._sample_mppi_rollouts(mean_sequence, sigma_sequence)
            if cfg.use_hss and len(guided_sequences) > 0:
                candidates = np.concatenate([guided_sequences, mppi_sequences], axis=0)
            else:
                candidates = mppi_sequences

            costs, metrics, q_used_flags = self._score_sequences(candidates, base_action, planner_state, obstacles, observation)
            all_iteration_costs.append(costs)
            terminal_q_used = terminal_q_used or any(q_used_flags)
            top_count = max(1, min(int(cfg.top_z), len(candidates)))
            selected_top_count = top_count
            top_indices = np.argsort(costs)[:top_count]
            top_sequences = candidates[top_indices]
            top_costs = costs[top_indices]
            weights = self._mppi_weights(top_costs)
            mean_sequence = np.sum(top_sequences * weights.reshape(-1, 1, 1), axis=0)
            if cfg.update_sigma:
                centered = top_sequences - mean_sequence.reshape(1, cfg.horizon, 2)
                variance = np.sum((centered * centered) * weights.reshape(-1, 1, 1), axis=0)
                sigma_sequence = np.sqrt(np.maximum(variance, np.asarray(cfg.sigma_min, dtype=float) ** 2))
            sigma_sequence = np.maximum(sigma_sequence, np.asarray(cfg.sigma_min, dtype=float).reshape(1, 2))
            best_idx = int(top_indices[0])
            best_sequence = candidates[best_idx].copy()
            best_metrics = metrics[best_idx]

        candidate_action = self._clip_action(mean_sequence[0])
        accept_candidate, reject_reason = self._accept_rl_driven_candidate(
            base_metrics=base_metrics,
            candidate_metrics=best_metrics,
            candidate_action=candidate_action,
            base_action=base_action,
            radar=radar,
        )
        if accept_candidate:
            executed = candidate_action
        else:
            executed = base_action
        self.best_sequence = mean_sequence.copy()
        self.last_action = executed.astype(np.float32)
        self.last_mean_sequence = mean_sequence.copy()
        self.last_sigma_sequence = sigma_sequence.copy()

        action_delta = executed - base_action
        all_costs = np.concatenate(all_iteration_costs) if all_iteration_costs else np.array([base_metrics["total_cost"]])
        debug = self._make_debug(
            base_action=base_action,
            executed=executed,
            base_metrics=base_metrics,
            best_metrics=best_metrics,
            best_sequence=best_sequence,
            costs=all_costs,
            sigma_sequence=sigma_sequence,
            elapsed_ms=(time.perf_counter() - start_time) * 1000.0,
            init_source=init_source,
            terminal_q_used=terminal_q_used,
            top_count=selected_top_count,
            action_delta=action_delta,
        )
        if cfg.rlmppi_accept_gate:
            self._apply_accept_gate_debug(debug, accept_candidate, reject_reason, base_action, executed)
        return executed.astype(np.float32), debug

    def _sequence_metrics(
        self,
        sequence: np.ndarray,
        base_action: np.ndarray,
        planner_state: Dict[str, Any],
        obstacles: Optional[np.ndarray],
    ) -> Dict[str, float]:
        if self.config.use_reward_aligned_cost:
            return self._reward_aligned_rollout(sequence, base_action, planner_state, obstacles)
        return self._rollout_metrics(sequence, base_action, planner_state, obstacles)

    def _initial_distribution(
        self,
        base_action: np.ndarray,
        observation: Any,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        cfg = self.config
        if cfg.pure_mppi or not cfg.use_rl_initialization:
            if self.best_sequence is not None and self.best_sequence.shape == (cfg.horizon, 2):
                mean = np.vstack([self.best_sequence[1:], self.best_sequence[-1:]])
                source = "warm_start"
            else:
                mean = np.tile(base_action.reshape(1, 2), (cfg.horizon, 1))
                source = "base_action"
        else:
            policy_mean = self.policy_adapter.predict_mean(observation, base_action)
            policy_mean = self._clip_action(policy_mean)
            mean = np.tile(policy_mean.reshape(1, 2), (cfg.horizon, 1))
            source = "rl_policy"

        if cfg.use_rl_initialization and not cfg.pure_mppi:
            try:
                initial_sigma = self.policy_adapter.action_std(observation)
            except TypeError:
                initial_sigma = self.policy_adapter.action_std()
        else:
            initial_sigma = np.asarray(cfg.initial_sigma)
        sigma = np.tile(np.asarray(initial_sigma, dtype=float).reshape(1, 2), (cfg.horizon, 1))
        sigma = np.maximum(sigma, np.asarray(cfg.sigma_min, dtype=float).reshape(1, 2))
        return self._clip_sequence(mean), sigma, source

    def _sample_guided_rollouts(
        self,
        mean_sequence: np.ndarray,
        sigma_sequence: np.ndarray,
        observation: Any,
        base_action: np.ndarray,
    ) -> np.ndarray:
        cfg = self.config
        if cfg.pure_mppi or not cfg.use_hss or cfg.num_rl_rollouts <= 0:
            return np.zeros((0, cfg.horizon, 2), dtype=float)
        sequences = []
        for _ in range(int(cfg.num_rl_rollouts)):
            first = self.policy_adapter.sample_action(observation, base_action, self.rng)
            sequence = mean_sequence.copy()
            sequence[0] = self._clip_action(first)
            if cfg.horizon > 1:
                noise = self.rng.normal(0.0, sigma_sequence[1:], size=(cfg.horizon - 1, 2))
                sequence[1:] = mean_sequence[1:] + noise
            sequences.append(self._clip_sequence(sequence))
        return np.asarray(sequences, dtype=float)

    def _sample_mppi_rollouts(self, mean_sequence: np.ndarray, sigma_sequence: np.ndarray) -> np.ndarray:
        cfg = self.config
        count = max(1, int(cfg.num_mppi_rollouts))
        noise = self.rng.normal(0.0, sigma_sequence, size=(count, cfg.horizon, 2))
        sequences = mean_sequence.reshape(1, cfg.horizon, 2) + noise
        sequences[0] = mean_sequence
        return self._clip_sequence(sequences)

    def _score_sequences(
        self,
        sequences: np.ndarray,
        base_action: np.ndarray,
        planner_state: Dict[str, Any],
        obstacles: Optional[np.ndarray],
        observation: Any,
    ) -> Tuple[np.ndarray, list, list]:
        costs = []
        metrics = []
        q_used_flags = []
        for sequence in sequences:
            rollout_metrics = self._sequence_metrics(sequence, base_action, planner_state, obstacles)
            terminal_cost, q_used = self._terminal_cost(observation, sequence[-1])
            cost = float(rollout_metrics["total_cost"] + terminal_cost)
            enriched = dict(rollout_metrics)
            enriched["terminal_q_cost"] = float(terminal_cost)
            costs.append(cost)
            metrics.append(enriched)
            q_used_flags.append(q_used)
        return np.asarray(costs, dtype=float), metrics, q_used_flags

    def _terminal_cost(self, observation: Any, action: np.ndarray) -> Tuple[float, bool]:
        cfg = self.config
        if cfg.pure_mppi or not cfg.use_terminal_q:
            return 0.0, False
        try:
            terminal_cost, used = self.policy_adapter.estimate_terminal_cost(observation, action)
        except Exception:
            if cfg.strict_terminal_q:
                raise
            return 0.0, False
        if cfg.strict_terminal_q and not used:
            raise RuntimeError("Strict RL-driven MPPI requires DSAC terminal critic cost.")
        return float(cfg.terminal_q_weight * terminal_cost), bool(used)

    def _accept_rl_driven_candidate(
        self,
        base_metrics: Dict[str, float],
        candidate_metrics: Dict[str, float],
        candidate_action: np.ndarray,
        base_action: np.ndarray,
        radar: Dict[str, float],
    ) -> Tuple[bool, str]:
        cfg = self.config
        if not cfg.rlmppi_accept_gate:
            return True, "none"

        base_lateral = float(base_metrics.get("max_lateral_error", 0.0))
        candidate_lateral = float(candidate_metrics.get("max_lateral_error", 0.0))
        base_distance = float(base_metrics.get("min_distance", radar["global_min"]))
        candidate_distance = float(candidate_metrics.get("min_distance", base_distance))
        base_risk = float(base_metrics.get("risk_score", 0.0))
        candidate_risk = float(candidate_metrics.get("risk_score", 0.0))
        reward_gain = float(base_metrics.get("total_cost", 0.0) - candidate_metrics.get("total_cost", 0.0))
        action_delta = float(np.linalg.norm(candidate_action - base_action))

        if not self._rlmppi_hazard_active(base_metrics, radar):
            return False, "base_safe_gate"
        if action_delta < cfg.rlmppi_min_action_delta:
            return False, "reject_small_update"
        if reward_gain < cfg.rlmppi_min_reward_gain:
            return False, "reject_no_reward_gain"
        if candidate_metrics.get("collision_risk", 0.0) > base_metrics.get("collision_risk", 0.0):
            return False, "reject_collision_risk"
        if candidate_metrics.get("out_of_bounds_risk", 0.0) > base_metrics.get("out_of_bounds_risk", 0.0):
            return False, "reject_out_of_bounds"
        if candidate_lateral > base_lateral + cfg.max_lateral_worsening:
            return False, "reject_lateral_worsening"
        if candidate_distance < base_distance - cfg.safety_distance_margin:
            return False, "reject_clearance_loss"

        risk_gain = base_risk - candidate_risk
        clearance_gain = candidate_distance - base_distance
        lateral_gain = base_lateral - candidate_lateral
        safety_gain = (
            risk_gain >= cfg.rlmppi_min_risk_gain
            or clearance_gain >= cfg.safety_distance_margin
            or lateral_gain >= cfg.max_lateral_worsening
        )
        if not safety_gain:
            return False, "reject_no_safety_gain"
        return True, "none"

    def _rlmppi_hazard_active(self, base_metrics: Dict[str, float], radar: Dict[str, float]) -> bool:
        return False

    def _shield_fallback_action(
        self,
        base_action: np.ndarray,
        base_metrics: Dict[str, float],
        planner_state: Dict[str, Any],
        obstacles: Optional[np.ndarray],
        radar: Dict[str, float],
    ) -> Tuple[np.ndarray, Dict[str, float], bool, bool, str]:
        cfg = self.config
        if not cfg.rlmppi_shield_fallback or not cfg.enable_fallback:
            return base_action.astype(float).copy(), base_metrics, False, False, "shield_disabled"
        if not self._rlmppi_shield_should_run(base_action, base_metrics, radar):
            return base_action.astype(float).copy(), base_metrics, False, False, "shield_not_needed"

        fallback_action = self._fallback_action(base_action, planner_state, radar)
        fallback_sequence = np.tile(fallback_action.reshape(1, 2), (cfg.horizon, 1))
        fallback_metrics = self._sequence_metrics(fallback_sequence, base_action, planner_state, obstacles)
        accept, fallback_reason = self._accept_fallback(base_metrics, fallback_metrics)
        if self._emergency_brake_needed(base_metrics, fallback_metrics, radar):
            emergency_action = self._emergency_brake_action(base_action)
            emergency_sequence = np.tile(emergency_action.reshape(1, 2), (cfg.horizon, 1))
            emergency_metrics = self._sequence_metrics(emergency_sequence, base_action, planner_state, obstacles)
            emergency_accept, emergency_reason = self._accept_fallback(base_metrics, emergency_metrics)
            if emergency_accept or not accept:
                return emergency_action, emergency_metrics, True, bool(emergency_accept), emergency_reason
        executed = fallback_action if accept else base_action
        return executed.astype(float).copy(), fallback_metrics, True, bool(accept), fallback_reason

    def _rlmppi_shield_should_run(
        self,
        base_action: np.ndarray,
        base_metrics: Dict[str, float],
        radar: Dict[str, float],
    ) -> bool:
        cfg = self.config
        if radar["global_min"] < cfg.fallback_trigger_distance:
            return True
        if radar["front_min"] < cfg.rlmppi_shield_front_trigger_distance and float(base_action[0]) > cfg.rlmppi_shield_approach_surge:
            return True
        if base_metrics.get("collision_risk", 0.0) > 0.0:
            return True
        if base_metrics.get("min_distance", radar["global_min"]) < cfg.safe_distance:
            return True
        if base_metrics.get("ttc_cost", 0.0) > 0.0:
            return True
        return False

    def _center_assist_action(
        self,
        base_action: np.ndarray,
        planner_state: Dict[str, Any],
        radar: Dict[str, float],
    ) -> np.ndarray:
        cfg = self.config
        if not cfg.rlmppi_center_assist or radar["global_min"] < cfg.rlmppi_center_assist_distance:
            return base_action.astype(float).copy()
        frenet_transform = planner_state.get("frenet_transform")
        if frenet_transform is None:
            return base_action.astype(float).copy()
        position = np.asarray(planner_state.get("position", np.zeros(2)), dtype=float)
        _, frenet_d = frenet_transform.cartesian_to_frenet(position)
        if abs(float(frenet_d)) < cfg.rlmppi_center_assist_lateral:
            return base_action.astype(float).copy()
        assisted = base_action.astype(float).copy()
        assisted[1] = self._yaw_toward_path_center(planner_state, default_yaw=float(base_action[1]))
        return self._clip_action(assisted)

    def _progress_assist_action(
        self,
        base_action: np.ndarray,
        planner_state: Dict[str, Any],
        radar: Dict[str, float],
    ) -> np.ndarray:
        cfg = self.config
        if not cfg.rlmppi_progress_assist or radar["global_min"] < cfg.rlmppi_progress_assist_distance:
            return base_action.astype(float).copy()
        frenet_transform = planner_state.get("frenet_transform")
        if frenet_transform is None:
            return base_action.astype(float).copy()

        position = np.asarray(planner_state.get("position", np.zeros(2)), dtype=float)
        yaw = float(planner_state.get("yaw", 0.0))
        frenet_s, frenet_d = frenet_transform.cartesian_to_frenet(position)
        heading_error = float(frenet_transform.get_heading_error(yaw, frenet_s))
        assisted = base_action.astype(float).copy()
        target_surge = cfg.rlmppi_progress_assist_surge
        if abs(float(frenet_d)) > cfg.rlmppi_progress_recovery_lateral:
            target_surge = min(target_surge, cfg.rlmppi_progress_recovery_surge)
        if radar["front_min"] < cfg.rlmppi_shield_front_trigger_distance:
            target_surge = min(target_surge, cfg.fallback_surge)
        assisted[0] = max(float(assisted[0]), target_surge)
        yaw_cmd = cfg.rlmppi_progress_heading_gain * heading_error - cfg.rlmppi_progress_center_gain * float(frenet_d)
        assisted[1] = float(np.clip(yaw_cmd, cfg.action_low[1], cfg.action_high[1]))
        return self._clip_action(assisted)

    def _apply_accept_gate_debug(
        self,
        debug: Dict[str, Any],
        accepted: bool,
        reject_reason: str,
        base_action: np.ndarray,
        executed: np.ndarray,
    ) -> None:
        cfg = self.config
        debug["optimized_action"] = executed.astype(np.float32)
        debug["optimized_action_surge"] = float(executed[0])
        debug["optimized_action_yaw"] = float(executed[1])
        debug["action_delta_surge"] = float(executed[0] - base_action[0])
        debug["action_delta_yaw"] = float(executed[1] - base_action[1])
        debug["action_delta_norm"] = float(np.linalg.norm(executed - base_action))
        debug["mppi_delta_norm"] = debug["action_delta_norm"]
        debug["mppi_delta_penalty"] = float(np.dot(executed - base_action, executed - base_action))
        debug["mppi_accept"] = bool(accepted)
        debug["mppi_reject"] = not bool(accepted)
        debug["mppi_selected"] = bool(accepted)
        debug["candidate_accepted"] = bool(accepted)
        debug["teacher_mppi_would_accept"] = bool(accepted)
        debug["reject_reason"] = "none" if accepted else reject_reason
        debug["mppi_decision_reason"] = "rl_driven_mppi_update" if accepted else reject_reason
        if accepted:
            debug["selected_reason"] = "select_rl_driven_mppi" if not cfg.pure_mppi else "select_pure_mppi"
            debug["action_source"] = cfg.controller_name if not cfg.pure_mppi else "pure_mppi"
            debug["terminal_source"] = debug["action_source"]
        elif reject_reason == "emergency_brake":
            debug["selected_reason"] = "select_emergency_brake"
            debug["action_source"] = "fallback"
            debug["terminal_source"] = "fallback"
            debug["fallback_active"] = True
            debug["fallback_accept"] = True
        else:
            debug["selected_reason"] = "select_sac"
            debug["action_source"] = "sac"
            debug["terminal_source"] = "sac"

    def _make_debug(
        self,
        base_action: np.ndarray,
        executed: np.ndarray,
        base_metrics: Dict[str, float],
        best_metrics: Dict[str, float],
        best_sequence: np.ndarray,
        costs: np.ndarray,
        sigma_sequence: np.ndarray,
        elapsed_ms: float,
        init_source: str,
        terminal_q_used: bool,
        top_count: int,
        action_delta: np.ndarray,
    ) -> Dict[str, Any]:
        cfg = self.config
        return {
            "raw_action": base_action.astype(np.float32),
            "optimized_action": executed.astype(np.float32),
            "raw_action_surge": float(base_action[0]),
            "raw_action_yaw": float(base_action[1]),
            "optimized_action_surge": float(executed[0]),
            "optimized_action_yaw": float(executed[1]),
            "candidate_action_surge": float(best_sequence[0, 0]),
            "candidate_action_yaw": float(best_sequence[0, 1]),
            "action_delta_surge": float(action_delta[0]),
            "action_delta_yaw": float(action_delta[1]),
            "action_delta_norm": float(np.linalg.norm(action_delta)),
            "mppi_delta_norm": float(np.linalg.norm(action_delta)),
            "mppi_delta_penalty": float(np.dot(action_delta, action_delta)),
            "mppi_active": True,
            "mppi_accept": True,
            "mppi_reject": False,
            "mppi_selected": True,
            "mppi_dbas_enabled": True,
            "action_source": cfg.controller_name if not cfg.pure_mppi else "pure_mppi",
            "terminal_source": cfg.controller_name if not cfg.pure_mppi else "pure_mppi",
            "mppi_decision_reason": "rl_driven_mppi_update" if not cfg.pure_mppi else "pure_mppi_update",
            "selected_reason": "select_rl_driven_mppi" if not cfg.pure_mppi else "select_pure_mppi",
            "reject_reason": "none",
            "mppi_prior_type": init_source,
            "mppi_warm_start_used": init_source == "warm_start",
            "teacher_mppi_would_accept": False,
            "fallback_active": False,
            "fallback_accept": False,
            "mppi_cost": float(np.min(costs)),
            "mppi_mean_cost": float(np.mean(costs)),
            "candidate_mppi_score": float(-best_metrics["total_cost"]),
            "candidate_sac_score": float(-base_metrics["total_cost"]),
            "candidate_fallback_score": float(-base_metrics["total_cost"]),
            "dbas_cost": float(best_metrics.get("dbas_cost", 0.0)),
            "dbas_mean_cost": 0.0,
            "ttc_cost": float(best_metrics.get("ttc_cost", 0.0)),
            "out_of_bounds_cost": float(best_metrics.get("out_of_bounds_cost", 0.0)),
            "min_predicted_obstacle_distance": float(best_metrics.get("min_distance", 0.0)),
            "current_obstacle_distance": float(base_metrics.get("min_distance", 0.0)),
            "mppi_min_obstacle_distance": float(best_metrics.get("min_distance", 0.0)),
            "mppi_pred_collision": bool(best_metrics.get("collision_risk", 0.0) > 0.0),
            "mppi_pred_out_of_bounds": bool(best_metrics.get("out_of_bounds_risk", 0.0) > 0.0),
            "sac_pred_collision": bool(base_metrics.get("collision_risk", 0.0) > 0.0),
            "sac_pred_out_of_bounds": bool(base_metrics.get("out_of_bounds_risk", 0.0) > 0.0),
            "fallback_pred_collision": bool(base_metrics.get("collision_risk", 0.0) > 0.0),
            "fallback_pred_out_of_bounds": bool(base_metrics.get("out_of_bounds_risk", 0.0) > 0.0),
            "base_risk": float(base_metrics.get("risk_score", 0.0)),
            "candidate_risk": float(best_metrics.get("risk_score", 0.0)),
            "fallback_risk": float(base_metrics.get("risk_score", 0.0)),
            "base_min_distance": float(base_metrics.get("min_distance", 0.0)),
            "candidate_min_distance": float(best_metrics.get("min_distance", 0.0)),
            "fallback_min_distance": float(base_metrics.get("min_distance", 0.0)),
            "base_ttc_cost": float(base_metrics.get("ttc_cost", 0.0)),
            "candidate_ttc_cost": float(best_metrics.get("ttc_cost", 0.0)),
            "fallback_ttc_cost": float(base_metrics.get("ttc_cost", 0.0)),
            "base_max_lateral_error": float(base_metrics.get("max_lateral_error", 0.0)),
            "candidate_max_lateral_error": float(best_metrics.get("max_lateral_error", 0.0)),
            "fallback_max_lateral_error": float(base_metrics.get("max_lateral_error", 0.0)),
            "base_progress": float(base_metrics.get("progress", 0.0)),
            "candidate_progress": float(best_metrics.get("progress", 0.0)),
            "fallback_progress": float(base_metrics.get("progress", 0.0)),
            "predicted_reward_sac": float(-base_metrics.get("total_cost", 0.0)),
            "predicted_reward_mppi": float(-best_metrics.get("total_cost", 0.0)),
            "predicted_reward_delta": float(base_metrics.get("total_cost", 0.0) - best_metrics.get("total_cost", 0.0)),
            "exploration_noise_scale": float(np.mean(sigma_sequence)),
            "rlmppi_enabled": True,
            "rlmppi_init_source": init_source,
            "rlmppi_hss_enabled": bool(cfg.use_hss and not cfg.pure_mppi),
            "rlmppi_terminal_q_enabled": bool(cfg.use_terminal_q and not cfg.pure_mppi),
            "rlmppi_terminal_q_used": bool(terminal_q_used),
            "rlmppi_update_sigma": bool(cfg.update_sigma),
            "rlmppi_num_rl_rollouts": int(cfg.num_rl_rollouts if cfg.use_hss and not cfg.pure_mppi else 0),
            "rlmppi_num_mppi_rollouts": int(cfg.num_mppi_rollouts),
            "rlmppi_num_iterations": int(cfg.num_iterations),
            "rlmppi_top_z": int(top_count),
            "rlmppi_sigma_mean": float(np.mean(sigma_sequence)),
            "rlmppi_sigma_min": float(np.min(sigma_sequence)),
            "rlmppi_sigma_max": float(np.max(sigma_sequence)),
            "rlmppi_cost_best": float(np.min(costs)),
            "rlmppi_cost_mean": float(np.mean(costs)),
            "rlmppi_online_time_ms": float(elapsed_ms),
        }


def rl_driven_mppi_config(mode: str, seed: int) -> RLDrivenMPPIConfig:
    if mode == "dsac_rl_driven_mppi":
        return RLDrivenMPPIConfig(seed=seed, strict_terminal_q=True, observation_stack=4, controller_name="dsac_rl_driven_mppi", use_reward_aligned_cost=True, terminal_q_weight=0.05, rlmppi_accept_gate=True, rlmppi_lateral_trigger=99.0, rlmppi_min_reward_gain=5.0, rlmppi_min_action_delta=0.12, rlmppi_shield_fallback=True, fallback_trigger_distance=1.0, safe_distance=0.65, fallback_surge=0.35, hard_brake_surge=0.22, rlmppi_shield_front_trigger_distance=1.6, rlmppi_progress_assist=True, rlmppi_progress_assist_distance=1.0, rlmppi_progress_heading_gain=1.4, rlmppi_progress_center_gain=0.35)
    if mode == "dsac_rl_driven_mppi_no_hss":
        return RLDrivenMPPIConfig(seed=seed, use_hss=False, strict_terminal_q=True, observation_stack=4, controller_name="dsac_rl_driven_mppi", use_reward_aligned_cost=True, terminal_q_weight=0.05, rlmppi_accept_gate=True, rlmppi_lateral_trigger=99.0, rlmppi_min_reward_gain=5.0, rlmppi_min_action_delta=0.12)
    if mode == "dsac_rl_driven_mppi_fixed_sigma":
        return RLDrivenMPPIConfig(seed=seed, update_sigma=False, strict_terminal_q=True, observation_stack=4, controller_name="dsac_rl_driven_mppi", use_reward_aligned_cost=True, terminal_q_weight=0.05, rlmppi_accept_gate=True, rlmppi_lateral_trigger=99.0, rlmppi_min_reward_gain=5.0, rlmppi_min_action_delta=0.12)
    if mode == "dsac_rl_driven_mppi_no_q":
        return RLDrivenMPPIConfig(seed=seed, use_terminal_q=False, strict_terminal_q=False, observation_stack=4, controller_name="dsac_rl_driven_mppi", use_reward_aligned_cost=True, rlmppi_accept_gate=True, rlmppi_lateral_trigger=99.0, rlmppi_min_reward_gain=5.0, rlmppi_min_action_delta=0.12)
    if mode == "pure_mppi":
        return RLDrivenMPPIConfig(
            seed=seed,
            pure_mppi=True,
            use_rl_initialization=False,
            use_hss=False,
            use_terminal_q=False,
            use_reward_aligned_cost=True,
        )
    if mode == "rl_driven_mppi_no_hss":
        return RLDrivenMPPIConfig(seed=seed, use_hss=False)
    if mode == "rl_driven_mppi_fixed_sigma":
        return RLDrivenMPPIConfig(seed=seed, update_sigma=False)
    if mode == "rl_driven_mppi_no_q":
        return RLDrivenMPPIConfig(seed=seed, use_terminal_q=False)
    return RLDrivenMPPIConfig(seed=seed)


class RLDrivenMPPIActionWrapper(gym.Wrapper):
    """Apply RL-driven MPPI as an action post-processor."""

    def __init__(
        self,
        env: gym.Env,
        config: Optional[RLDrivenMPPIConfig] = None,
        policy_adapter: Optional[SB3SacPolicyAdapter] = None,
    ):
        super().__init__(env)
        self.optimizer = RLDrivenMPPIOptimizer(config, policy_adapter)
        self.last_observation = None
        self.observation_stack = int(self.optimizer.config.observation_stack)
        self._obs_history = []

    def reset(self, **kwargs):
        self.optimizer.reset()
        obs, info = self.env.reset(**kwargs)
        self.last_observation = obs
        self._obs_history = [np.asarray(obs, dtype=np.float32).copy() for _ in range(max(self.observation_stack, 1))]
        return obs, info

    def step(self, action):
        planner_state = self.env.get_planner_state()
        raw_action = np.asarray(action, dtype=np.float32).reshape(-1)[:2]
        adapter_observation = self._stacked_observation()
        optimized_action, debug = self.optimizer.optimize(raw_action, planner_state, adapter_observation)
        obs, reward, terminated, truncated, info = self.env.step(optimized_action)
        self.last_observation = obs
        self._push_observation(obs)
        info = dict(info)
        info.update(debug)
        info["raw_action"] = raw_action.astype(np.float32)
        info["optimized_action"] = optimized_action.astype(np.float32)
        info["mppi_dbas_enabled"] = True
        info["rl_driven_mppi_enabled"] = True
        return obs, reward, terminated, truncated, info

    def get_planner_state(self):
        return self.env.get_planner_state()

    def _push_observation(self, obs: Any) -> None:
        if self.observation_stack <= 1:
            self._obs_history = [np.asarray(obs, dtype=np.float32).copy()]
            return
        self._obs_history.append(np.asarray(obs, dtype=np.float32).copy())
        self._obs_history = self._obs_history[-self.observation_stack :]

    def _stacked_observation(self):
        if not self._obs_history:
            return self.last_observation
        if self.observation_stack <= 1:
            return self._obs_history[-1]
        return np.concatenate([np.asarray(obs, dtype=np.float32).reshape(-1) for obs in self._obs_history], axis=0)
