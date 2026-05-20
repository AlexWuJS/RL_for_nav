from typing import Optional

import gymnasium as gym
import numpy as np

from mppi_dbas import (
    MPPIDBaSConfig,
    MPPIDBaSOptimizer,
    V4_MODE_NAMES,
    decode_mode_intent_v4,
    intent_to_frenet_params_v3,
    intent_to_mode_params_v4,
    intent_to_params,
    intent_to_params_v2,
    intent_to_structured_params_v41,
)


def hierarchical_mppi_config(seed: Optional[int] = None) -> MPPIDBaSConfig:
    return MPPIDBaSConfig(
        seed=seed,
        num_samples=32,
        horizon=8,
        use_reward_aligned_cost=False,
        always_run_mppi=True,
        execute_mppi=True,
        final_safety_check=True,
        enable_mppi=True,
        enable_fallback=False,
        base_noise_std=(0.16, 0.18),
        residual_low=(-0.35, -0.45),
        residual_high=(0.35, 0.45),
        reward_aligned_residual_low=(-0.35, -0.45),
        reward_aligned_residual_high=(0.35, 0.45),
    )


def hierarchical_mppi_v2_config(seed: Optional[int] = None) -> MPPIDBaSConfig:
    return MPPIDBaSConfig(
        seed=seed,
        num_samples=32,
        horizon=8,
        use_reward_aligned_cost=False,
        always_run_mppi=False,
        execute_mppi=True,
        final_safety_check=True,
        enable_mppi=True,
        enable_fallback=True,
        base_noise_std=(0.10, 0.12),
        residual_low=(-0.22, -0.28),
        residual_high=(0.18, 0.28),
        reward_aligned_residual_low=(-0.22, -0.28),
        reward_aligned_residual_high=(0.18, 0.28),
        mppi_max_action_delta=(0.22, 0.28),
        hierarchical_front_trigger_distance=1.35,
        hierarchical_global_trigger_distance=1.0,
        hierarchical_lateral_trigger=0.55,
        hierarchical_heading_trigger=0.45,
    )


def hierarchical_mppi_v3_config(seed: Optional[int] = None) -> MPPIDBaSConfig:
    return MPPIDBaSConfig(
        seed=seed,
        num_samples=48,
        horizon=10,
        use_reward_aligned_cost=False,
        always_run_mppi=True,
        execute_mppi=True,
        final_safety_check=True,
        enable_mppi=True,
        enable_fallback=True,
        base_noise_std=(0.12, 0.16),
        residual_low=(-0.35, -0.42),
        residual_high=(0.35, 0.42),
        reward_aligned_residual_low=(-0.35, -0.42),
        reward_aligned_residual_high=(0.35, 0.42),
        mppi_max_action_delta=(0.35, 0.42),
        hierarchical_front_trigger_distance=1.35,
        hierarchical_global_trigger_distance=1.0,
        hierarchical_lateral_trigger=0.55,
        hierarchical_heading_trigger=0.45,
    )


def hierarchical_mppi_v4_config(seed: Optional[int] = None, reward_profile: str = "compat") -> MPPIDBaSConfig:
    reward_profile = str(reward_profile).lower()
    conservative = reward_profile == "enhanced"
    return MPPIDBaSConfig(
        seed=seed,
        num_samples=64,
        horizon=15,
        use_reward_aligned_cost=True,
        always_run_mppi=False,
        execute_mppi=True,
        final_safety_check=True,
        enable_mppi=True,
        enable_fallback=True,
        base_noise_std=(0.10, 0.12) if conservative else (0.11, 0.13),
        reward_aligned_residual_low=(-0.20, -0.24),
        reward_aligned_residual_high=(0.16, 0.24),
        mppi_max_action_delta=(0.22, 0.28) if conservative else (0.24, 0.30),
        hierarchical_front_trigger_distance=1.30,
        hierarchical_global_trigger_distance=0.95,
        hierarchical_lateral_trigger=0.52,
        hierarchical_heading_trigger=0.42,
        hierarchical_min_risk_gain=0.04 if conservative else 0.03,
        hierarchical_accept_score_margin=0.01,
        hierarchical_lateral_recovery_gain=0.06 if conservative else 0.05,
        hierarchical_v4_cruise_max_yaw_delta=0.16 if conservative else 0.18,
        hierarchical_v4_fallback_risk_margin=0.10 if conservative else 0.08,
    )


def hierarchical_mppi_v41_config(seed: Optional[int] = None, reward_profile: str = "guided") -> MPPIDBaSConfig:
    reward_profile = str(reward_profile).lower()
    guided = reward_profile == "guided"
    return MPPIDBaSConfig(
        seed=seed,
        num_samples=64,
        horizon=15,
        use_reward_aligned_cost=True,
        always_run_mppi=False,
        execute_mppi=True,
        final_safety_check=True,
        enable_mppi=True,
        enable_fallback=True,
        base_noise_std=(0.10, 0.12),
        reward_aligned_residual_low=(-0.20, -0.24),
        reward_aligned_residual_high=(0.16, 0.24),
        mppi_max_action_delta=(0.22, 0.28),
        hierarchical_front_trigger_distance=1.30,
        hierarchical_global_trigger_distance=0.95,
        hierarchical_lateral_trigger=0.55,
        hierarchical_heading_trigger=0.45,
        hierarchical_min_risk_gain=0.04 if guided else 0.03,
        hierarchical_accept_score_margin=0.01,
        hierarchical_lateral_recovery_gain=0.06,
        hierarchical_v41_gate_threshold=0.65,
        hierarchical_v41_gate_risk_distance=1.8,
    )


class HierarchicalMppiWrapper(gym.Wrapper):
    """Expose a 4D SAC intent action space and execute MPPI's low-level action."""

    def __init__(
        self,
        env: gym.Env,
        config: Optional[MPPIDBaSConfig] = None,
        optimizer: Optional[MPPIDBaSOptimizer] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(env)
        self.optimizer = optimizer or MPPIDBaSOptimizer(config or hierarchical_mppi_config(seed=seed))
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def reset(self, **kwargs):
        self.optimizer.reset()
        return self.env.reset(**kwargs)

    def step(self, intent):
        raw_intent = np.asarray(intent, dtype=np.float32).reshape(-1)[:4]
        if raw_intent.size < 4:
            raw_intent = np.pad(raw_intent, (0, 4 - raw_intent.size))
        raw_intent = np.clip(raw_intent, -1.0, 1.0).astype(np.float32)
        planner_state = self.env.get_planner_state()
        executed_action, debug = self.optimizer.optimize_from_intent(raw_intent, planner_state)
        obs, reward, terminated, truncated, info = self.env.step(executed_action)
        info = dict(info)
        debug.setdefault("action_source", "hierarchical_mppi")
        debug.setdefault("terminal_source", "hierarchical_mppi")
        debug["sac_intent_target_speed"] = float(raw_intent[0])
        debug["sac_intent_turn_bias"] = float(raw_intent[1])
        debug["sac_intent_path_weight"] = float(raw_intent[2])
        debug["sac_intent_safety_weight"] = float(raw_intent[3])
        debug.setdefault("mppi_executed_surge", float(executed_action[0]))
        debug.setdefault("mppi_executed_yaw", float(executed_action[1]))
        info.update(debug)
        info["raw_intent"] = raw_intent.copy()
        info["optimized_action"] = executed_action.astype(np.float32)
        info["hierarchical_mppi_enabled"] = True
        info["mppi_dbas_enabled"] = True
        return obs, reward, terminated, truncated, info

    def get_planner_state(self):
        return self.env.get_planner_state()


class HierarchicalMppiV2Wrapper(gym.Wrapper):
    """Trigger-based variant: SAC intent is the default, MPPI refines only risky steps."""

    def __init__(
        self,
        env: gym.Env,
        config: Optional[MPPIDBaSConfig] = None,
        optimizer: Optional[MPPIDBaSOptimizer] = None,
        seed: Optional[int] = None,
        intent_ema_alpha: float = 0.6,
        intent_hold_steps: int = 2,
        enable_delta_penalty: bool = False,
        delta_penalty_alpha: float = 0.08,
        delta_deadband: float = 0.10,
    ):
        super().__init__(env)
        self.optimizer = optimizer or MPPIDBaSOptimizer(config or hierarchical_mppi_v2_config(seed=seed))
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.intent_ema_alpha = float(np.clip(intent_ema_alpha, 0.0, 1.0))
        self.intent_hold_steps = max(1, int(intent_hold_steps))
        self.enable_delta_penalty = bool(enable_delta_penalty)
        self.delta_penalty_alpha = float(max(0.0, delta_penalty_alpha))
        self.delta_deadband = float(max(0.0, delta_deadband))
        self.smoothed_intent = np.zeros(4, dtype=np.float32)
        self.held_intent = np.zeros(4, dtype=np.float32)
        self.hold_counter = 0

    def reset(self, **kwargs):
        self.optimizer.reset()
        self.smoothed_intent = np.zeros(4, dtype=np.float32)
        self.held_intent = np.zeros(4, dtype=np.float32)
        self.hold_counter = 0
        return self.env.reset(**kwargs)

    def step(self, intent):
        raw_intent = np.asarray(intent, dtype=np.float32).reshape(-1)[:4]
        if raw_intent.size < 4:
            raw_intent = np.pad(raw_intent, (0, 4 - raw_intent.size))
        raw_intent = np.clip(raw_intent, -1.0, 1.0).astype(np.float32)

        self.smoothed_intent = (
            self.intent_ema_alpha * self.smoothed_intent
            + (1.0 - self.intent_ema_alpha) * raw_intent
        ).astype(np.float32)
        if self.hold_counter <= 0:
            self.held_intent = self.smoothed_intent.copy()
            self.hold_counter = self.intent_hold_steps
        self.hold_counter -= 1

        planner_state = self.env.get_planner_state()
        executed_action, debug = self.optimizer.optimize_from_intent_v2(self.held_intent, planner_state)
        obs, reward, terminated, truncated, info = self.env.step(executed_action)
        info = dict(info)
        params = intent_to_params_v2(self.held_intent)
        debug.setdefault("sac_intent_target_speed", float(raw_intent[0]))
        debug.setdefault("sac_intent_turn_bias", float(raw_intent[1]))
        debug.setdefault("sac_intent_path_weight", float(raw_intent[2]))
        debug.setdefault("sac_intent_safety_weight", float(raw_intent[3]))
        debug["raw_intent_target_speed"] = float(raw_intent[0])
        debug["raw_intent_turn_bias"] = float(raw_intent[1])
        debug["raw_intent_path_weight"] = float(raw_intent[2])
        debug["raw_intent_safety_weight"] = float(raw_intent[3])
        debug["smoothed_intent_target_speed"] = float(self.smoothed_intent[0])
        debug["smoothed_intent_turn_bias"] = float(self.smoothed_intent[1])
        debug["smoothed_intent_path_weight"] = float(self.smoothed_intent[2])
        debug["smoothed_intent_safety_weight"] = float(self.smoothed_intent[3])
        debug["held_intent_target_speed"] = float(self.held_intent[0])
        debug["held_intent_turn_bias"] = float(self.held_intent[1])
        debug["held_intent_path_weight"] = float(self.held_intent[2])
        debug["held_intent_safety_weight"] = float(self.held_intent[3])
        debug.setdefault("intent_prior_surge", float(params["target_speed"]))
        debug.setdefault("intent_prior_yaw", float(params["turn_bias"]))
        debug.setdefault("mppi_executed_surge", float(executed_action[0]))
        debug.setdefault("mppi_executed_yaw", float(executed_action[1]))
        info.update(debug)
        prior_action = np.array([info["intent_prior_surge"], info["intent_prior_yaw"]], dtype=np.float32)
        mppi_delta_norm = float(np.linalg.norm(executed_action.astype(np.float32) - prior_action))
        delta_penalty = self._mppi_delta_penalty(mppi_delta_norm, info)
        training_reward = float(reward) - delta_penalty
        info["env_reward"] = float(reward)
        info["training_reward"] = float(training_reward)
        info["mppi_delta_norm"] = float(mppi_delta_norm)
        info["mppi_delta_penalty"] = float(delta_penalty)
        info["delta_penalty_enabled"] = bool(self.enable_delta_penalty)
        info["raw_intent"] = raw_intent.copy()
        info["smoothed_intent"] = self.smoothed_intent.copy()
        info["held_intent"] = self.held_intent.copy()
        info["raw_action"] = prior_action
        info["optimized_action"] = executed_action.astype(np.float32)
        info["hierarchical_mppi_enabled"] = True
        info["hierarchical_mppi_v2_enabled"] = True
        info["mppi_dbas_enabled"] = True
        return obs, training_reward, terminated, truncated, info

    def get_planner_state(self):
        return self.env.get_planner_state()

    def _mppi_delta_penalty(self, mppi_delta_norm: float, info: dict) -> float:
        if not self.enable_delta_penalty:
            return 0.0
        if not self._delta_penalty_allowed(info):
            return 0.0
        excess = max(0.0, float(mppi_delta_norm) - self.delta_deadband)
        return float(self.delta_penalty_alpha * excess * excess)

    def _delta_penalty_allowed(self, info: dict) -> bool:
        source = str(info.get("action_source", "intent_prior"))
        trigger = str(info.get("mppi_trigger_reason", "none"))
        reject = str(info.get("reject_reason", "none"))
        if source == "fallback" or reject == "emergency_brake":
            return False
        emergency_triggers = {
            "trigger_collision_risk",
            "trigger_out_of_bounds",
            "trigger_ttc",
            "trigger_front_obstacle",
            "trigger_near_obstacle",
        }
        if trigger in emergency_triggers:
            return False
        current_distance = float(info.get("current_obstacle_distance", np.inf))
        safe_distance = float(getattr(self.optimizer.config, "safe_distance", 0.55))
        if current_distance < safe_distance:
            return False
        return True


class HierarchicalMppiV3Wrapper(gym.Wrapper):
    """Expose SAC's long-horizon Frenet intent and execute MPPI's low-level action."""

    def __init__(
        self,
        env: gym.Env,
        config: Optional[MPPIDBaSConfig] = None,
        optimizer: Optional[MPPIDBaSOptimizer] = None,
        seed: Optional[int] = None,
        intent_ema_alpha: float = 0.6,
        intent_hold_steps: int = 2,
    ):
        super().__init__(env)
        self.optimizer = optimizer or MPPIDBaSOptimizer(config or hierarchical_mppi_v3_config(seed=seed))
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.intent_ema_alpha = float(np.clip(intent_ema_alpha, 0.0, 1.0))
        self.intent_hold_steps = max(1, int(intent_hold_steps))
        self.smoothed_intent = np.zeros(4, dtype=np.float32)
        self.held_intent = np.zeros(4, dtype=np.float32)
        self.hold_counter = 0

    def reset(self, **kwargs):
        self.optimizer.reset()
        self.smoothed_intent = np.zeros(4, dtype=np.float32)
        self.held_intent = np.zeros(4, dtype=np.float32)
        self.hold_counter = 0
        return self.env.reset(**kwargs)

    def step(self, intent):
        raw_intent = np.asarray(intent, dtype=np.float32).reshape(-1)[:4]
        if raw_intent.size < 4:
            raw_intent = np.pad(raw_intent, (0, 4 - raw_intent.size))
        raw_intent = np.clip(raw_intent, -1.0, 1.0).astype(np.float32)

        self.smoothed_intent = (
            self.intent_ema_alpha * self.smoothed_intent
            + (1.0 - self.intent_ema_alpha) * raw_intent
        ).astype(np.float32)
        if self.hold_counter <= 0:
            self.held_intent = self.smoothed_intent.copy()
            self.hold_counter = self.intent_hold_steps
        self.hold_counter -= 1

        planner_state = self.env.get_planner_state()
        executed_action, debug = self.optimizer.optimize_from_frenet_intent_v3(self.held_intent, planner_state)
        obs, reward, terminated, truncated, info = self.env.step(executed_action)
        info = dict(info)
        params = intent_to_frenet_params_v3(self.held_intent, planner_state, self.optimizer.config)
        debug.setdefault("sac_intent_progress", float(raw_intent[0]))
        debug.setdefault("sac_intent_lateral_offset", float(raw_intent[1]))
        debug.setdefault("sac_intent_caution", float(raw_intent[2]))
        debug.setdefault("sac_intent_recovery", float(raw_intent[3]))
        debug["raw_intent_progress"] = float(raw_intent[0])
        debug["raw_intent_lateral_offset"] = float(raw_intent[1])
        debug["raw_intent_caution"] = float(raw_intent[2])
        debug["raw_intent_recovery"] = float(raw_intent[3])
        debug["held_intent_progress"] = float(self.held_intent[0])
        debug["held_intent_lateral_offset"] = float(self.held_intent[1])
        debug["held_intent_caution"] = float(self.held_intent[2])
        debug["held_intent_recovery"] = float(self.held_intent[3])
        debug.setdefault("target_progress_speed", float(params["target_progress_speed"]))
        debug.setdefault("target_lateral_offset", float(params["target_lateral_offset"]))
        debug.setdefault("caution_level", float(params["caution_level"]))
        debug.setdefault("recovery_level", float(params["recovery_level"]))
        debug.setdefault("mppi_executed_surge", float(executed_action[0]))
        debug.setdefault("mppi_executed_yaw", float(executed_action[1]))
        info.update(debug)
        info["env_reward"] = float(reward)
        info["training_reward"] = float(reward)
        info["raw_intent"] = raw_intent.copy()
        info["smoothed_intent"] = self.smoothed_intent.copy()
        info["held_intent"] = self.held_intent.copy()
        info["raw_action"] = np.array(
            [info.get("intent_prior_surge", 0.0), info.get("intent_prior_yaw", 0.0)],
            dtype=np.float32,
        )
        info["optimized_action"] = executed_action.astype(np.float32)
        info["hierarchical_mppi_enabled"] = True
        info["hierarchical_mppi_v3_enabled"] = True
        info["mppi_dbas_enabled"] = True
        return obs, reward, terminated, truncated, info

    def get_planner_state(self):
        return self.env.get_planner_state()


class HierarchicalMppiV4Wrapper(gym.Wrapper):
    """Semi-discrete v4 wrapper: mode scores + 2 continuous params with trigger-based MPPI correction."""

    def __init__(
        self,
        env: gym.Env,
        config: Optional[MPPIDBaSConfig] = None,
        optimizer: Optional[MPPIDBaSOptimizer] = None,
        seed: Optional[int] = None,
        reward_profile: str = "compat",
        mode_hold_steps: int = 3,
        direction_switch_cooldown: int = 2,
        switch_penalty: float = 0.05,
        direction_flip_penalty: float = 0.08,
        recover_center_bonus: float = 0.20,
        hazard_mode_bonus: float = 0.08,
        conservative_overuse_penalty: float = 0.04,
    ):
        super().__init__(env)
        self.reward_profile = str(reward_profile).lower()
        self.optimizer = optimizer or MPPIDBaSOptimizer(config or hierarchical_mppi_v4_config(seed=seed, reward_profile=self.reward_profile))
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        self.mode_hold_steps = max(1, int(mode_hold_steps))
        self.direction_switch_cooldown = max(0, int(direction_switch_cooldown))
        self.switch_penalty = float(max(0.0, switch_penalty))
        self.direction_flip_penalty = float(max(0.0, direction_flip_penalty))
        self.recover_center_bonus = float(max(0.0, recover_center_bonus))
        self.hazard_mode_bonus = float(max(0.0, hazard_mode_bonus))
        self.conservative_overuse_penalty = float(max(0.0, conservative_overuse_penalty))
        self.current_mode_name = "cruise"
        self.current_mode_index = 0
        self.hold_counter = 0
        self.direction_cooldown = 0
        self.prev_abs_lateral_error = 0.0
        self.last_mode_name = "cruise"
        self.conservative_mode_streak = 0

    def reset(self, **kwargs):
        self.optimizer.reset()
        self.current_mode_name = "cruise"
        self.current_mode_index = 0
        self.hold_counter = 0
        self.direction_cooldown = 0
        self.prev_abs_lateral_error = 0.0
        self.last_mode_name = "cruise"
        self.conservative_mode_streak = 0
        return self.env.reset(**kwargs)

    def step(self, intent):
        raw_intent = np.asarray(intent, dtype=np.float32).reshape(-1)[:8]
        if raw_intent.size < 8:
            raw_intent = np.pad(raw_intent, (0, 8 - raw_intent.size))
        raw_intent = np.clip(raw_intent, -1.0, 1.0).astype(np.float32)

        decoded = decode_mode_intent_v4(raw_intent)
        held_mode_name, held_mode_index, mode_switched = self._select_mode(decoded)
        held_intent = raw_intent.copy()
        held_intent[: len(V4_MODE_NAMES)] = -1.0
        held_intent[held_mode_index] = 1.0

        planner_state = self.env.get_planner_state()
        executed_action, debug = self.optimizer.optimize_from_mode_intent_v4(held_intent, planner_state)
        obs, reward, terminated, truncated, info = self.env.step(executed_action)
        info = dict(info)
        params = intent_to_mode_params_v4(held_intent, planner_state, self.optimizer.config)
        post_lateral_error = self._current_abs_lateral_error()

        debug["raw_mode"] = str(decoded["mode_name"])
        debug["raw_mode_index"] = int(decoded["mode_index"])
        debug["high_level_mode"] = str(held_mode_name)
        debug["high_level_mode_index"] = int(held_mode_index)
        debug["mode_switched"] = bool(mode_switched)
        debug["mode_score_margin"] = float(decoded["mode_margin"])
        debug["mode_target_speed"] = float(params["target_speed"])
        debug["mode_turn_bias"] = float(params["turn_bias"])
        debug["mode_safe_distance"] = float(params["safe_distance"])
        debug["mode_desired_lateral_offset"] = float(params["desired_lateral_offset"])
        debug.setdefault("mppi_executed_surge", float(executed_action[0]))
        debug.setdefault("mppi_executed_yaw", float(executed_action[1]))
        info.update(debug)

        training_reward, reward_terms = self._training_reward(
            env_reward=float(reward),
            info=info,
            prev_mode=self.last_mode_name,
            current_mode=held_mode_name,
            mode_switched=mode_switched,
            post_lateral_error=post_lateral_error,
        )
        info["env_reward"] = float(reward)
        info["training_reward"] = float(training_reward)
        info["reward_profile"] = self.reward_profile
        info["reward_mode_switch_penalty"] = float(reward_terms["switch_penalty"])
        info["reward_direction_flip_penalty"] = float(reward_terms["direction_flip_penalty"])
        info["reward_recover_center_bonus"] = float(reward_terms["recover_bonus"])
        info["reward_hazard_mode_bonus"] = float(reward_terms["hazard_bonus"])
        info["reward_conservative_overuse_penalty"] = float(reward_terms["conservative_penalty"])
        info["raw_intent"] = raw_intent.copy()
        info["held_intent"] = held_intent.copy()
        info["raw_action"] = np.array([info.get("intent_prior_surge", 0.0), info.get("intent_prior_yaw", 0.0)], dtype=np.float32)
        info["optimized_action"] = executed_action.astype(np.float32)
        info["hierarchical_mppi_enabled"] = True
        info["hierarchical_mppi_v4_enabled"] = True
        info["mppi_dbas_enabled"] = True

        self.prev_abs_lateral_error = post_lateral_error
        self.last_mode_name = held_mode_name
        return obs, training_reward, terminated, truncated, info

    def get_planner_state(self):
        return self.env.get_planner_state()

    def _select_mode(self, decoded: dict) -> tuple[str, int, bool]:
        raw_mode_name = str(decoded["mode_name"])
        raw_mode_index = int(decoded["mode_index"])
        emergency_mode = raw_mode_name == "brake"
        held_mode_name = self.current_mode_name
        held_mode_index = self.current_mode_index
        mode_switched = False

        if emergency_mode:
            held_mode_name = raw_mode_name
            held_mode_index = raw_mode_index
            mode_switched = held_mode_name != self.current_mode_name
            self.hold_counter = self.mode_hold_steps
        elif self.hold_counter > 0 and raw_mode_name != self.current_mode_name:
            self.hold_counter -= 1
        else:
            opposite_flip = (
                {self.current_mode_name, raw_mode_name} == {"avoid_left", "avoid_right"}
                and self.direction_cooldown > 0
            )
            if not opposite_flip:
                held_mode_name = raw_mode_name
                held_mode_index = raw_mode_index
                mode_switched = held_mode_name != self.current_mode_name
                if mode_switched:
                    self.hold_counter = self.mode_hold_steps
            elif self.hold_counter > 0:
                self.hold_counter -= 1

        if mode_switched and {self.current_mode_name, held_mode_name} == {"avoid_left", "avoid_right"}:
            self.direction_cooldown = self.direction_switch_cooldown
        else:
            self.direction_cooldown = max(0, self.direction_cooldown - 1)

        self.current_mode_name = held_mode_name
        self.current_mode_index = held_mode_index
        if held_mode_name in ("cautious_cruise", "brake"):
            self.conservative_mode_streak += 1
        else:
            self.conservative_mode_streak = 0
        return held_mode_name, held_mode_index, mode_switched

    def _current_abs_lateral_error(self) -> float:
        try:
            planner_state = self.env.get_planner_state()
            frenet_transform = planner_state.get("frenet_transform")
            if frenet_transform is None:
                return 0.0
            _, frenet_d = frenet_transform.cartesian_to_frenet(np.asarray(planner_state["position"], dtype=float))
            return abs(float(frenet_d))
        except Exception:
            return 0.0

    def _training_reward(
        self,
        env_reward: float,
        info: dict,
        prev_mode: str,
        current_mode: str,
        mode_switched: bool,
        post_lateral_error: float,
    ) -> tuple[float, dict]:
        if self.reward_profile != "enhanced":
            return float(env_reward), {
                "switch_penalty": 0.0,
                "direction_flip_penalty": 0.0,
                "recover_bonus": 0.0,
                "hazard_bonus": 0.0,
                "conservative_penalty": 0.0,
            }

        switch_penalty = self.switch_penalty if mode_switched and current_mode != "brake" else 0.0
        direction_flip_penalty = 0.0
        if {prev_mode, current_mode} == {"avoid_left", "avoid_right"}:
            direction_flip_penalty = self.direction_flip_penalty
        recover_bonus = 0.0
        if current_mode == "recover_center":
            lateral_improvement = max(0.0, self.prev_abs_lateral_error - post_lateral_error)
            recover_bonus = self.recover_center_bonus * lateral_improvement
        hazard_bonus = 0.0
        hazard = bool(info.get("mppi_triggered", False)) or float(info.get("current_obstacle_distance", np.inf)) < float(getattr(self.optimizer.config, "safe_distance", 0.55)) + 0.15
        if hazard and current_mode in ("cautious_cruise", "avoid_left", "avoid_right", "recover_center", "brake"):
            hazard_bonus = self.hazard_mode_bonus
        conservative_penalty = 0.0
        if current_mode in ("cautious_cruise", "brake"):
            obstacle_distance = float(info.get("current_obstacle_distance", np.inf))
            safe_distance = float(getattr(self.optimizer.config, "safe_distance", 0.55))
            if self.conservative_mode_streak > 6 and obstacle_distance > safe_distance + 0.25:
                conservative_penalty = self.conservative_overuse_penalty * (self.conservative_mode_streak - 6)
        training_reward = float(env_reward + recover_bonus + hazard_bonus - switch_penalty - direction_flip_penalty - conservative_penalty)
        return training_reward, {
            "switch_penalty": switch_penalty,
            "direction_flip_penalty": direction_flip_penalty,
            "recover_bonus": recover_bonus,
            "hazard_bonus": hazard_bonus,
            "conservative_penalty": conservative_penalty,
        }


class HierarchicalMppiV41Wrapper(gym.Wrapper):
    """Plan-B: SAC perceives MPPI delta; shared cost; decomposed reward."""

    def __init__(
        self,
        env: gym.Env,
        config: Optional[MPPIDBaSConfig] = None,
        optimizer: Optional[MPPIDBaSOptimizer] = None,
        seed: Optional[int] = None,
        reward_profile: str = "guided",
        correction_bonus_weight: float = 0.1,
        consistency_penalty_weight: float = 0.05,
    ):
        super().__init__(env)
        self.reward_profile = str(reward_profile).lower()
        self.optimizer = optimizer or MPPIDBaSOptimizer(config or hierarchical_mppi_v41_config(seed=seed, reward_profile=self.reward_profile))
        # SAC action space unchanged (4D intent)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        # Expand observation by 2: [mppi_delta_surge_norm, mppi_delta_yaw_norm]
        orig_obs_space = env.observation_space
        orig_dim = orig_obs_space.shape[0]
        self._base_obs_dim = orig_dim
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(orig_dim + 2,), dtype=np.float32,
        )
        self.correction_bonus_weight = float(max(0.0, correction_bonus_weight))
        self.consistency_penalty_weight = float(max(0.0, consistency_penalty_weight))
        self._prev_executed_action = np.zeros(2, dtype=np.float32)

    def reset(self, **kwargs):
        self.optimizer.reset()
        self._prev_executed_action = np.zeros(2, dtype=np.float32)
        obs, info = self.env.reset(**kwargs)
        # Pad initial obs with zero delta (no MPPI correction yet)
        obs = np.concatenate([obs, np.zeros(2, dtype=np.float32)]).astype(np.float32)
        return obs, info

    def step(self, intent):
        raw_intent = np.asarray(intent, dtype=np.float32).reshape(-1)[:4]
        if raw_intent.size < 4:
            raw_intent = np.pad(raw_intent, (0, 4 - raw_intent.size))
        raw_intent = np.clip(raw_intent, -1.0, 1.0).astype(np.float32)

        planner_state = self.env.get_planner_state()
        params = intent_to_structured_params_v41(raw_intent, planner_state, self.optimizer.config)
        executed_action, debug = self.optimizer.optimize_from_structured_intent_v41(raw_intent, planner_state)
        obs, reward, terminated, truncated, info = self.env.step(executed_action)
        info = dict(info)
        debug.setdefault("mppi_executed_surge", float(executed_action[0]))
        debug.setdefault("mppi_executed_yaw", float(executed_action[1]))
        info.update(debug)

        # --- Plan B: MPPI delta as observation ---
        prior_action = np.array([
            info.get("intent_prior_surge", 0.0),
            info.get("intent_prior_yaw", 0.0),
        ], dtype=np.float32)
        mppi_delta = executed_action.astype(np.float32) - prior_action

        act_low = self.env.action_space.low[:2].astype(np.float32)
        act_high = self.env.action_space.high[:2].astype(np.float32)
        act_range = act_high - act_low
        safe_range = np.maximum(act_range, 1e-6)
        delta_norm = np.clip(mppi_delta / (safe_range * 0.5), -1.0, 1.0).astype(np.float32)

        obs = np.concatenate([obs, delta_norm]).astype(np.float32)

        # --- Plan B: decomposed reward ---
        base_reward = float(reward)
        action_source = str(info.get("action_source", "intent_prior"))

        if self.reward_profile == "guided":
            # Correction bonus: reward MPPI when it finds a genuinely better action
            correction_bonus = 0.0
            if action_source == "hierarchical_mppi_v41" and not terminated:
                prior_pred = float(info.get("prior_predicted_reward", 0.0))
                mppi_pred = float(info.get("candidate_predicted_reward", 0.0))
                correction_bonus = self.correction_bonus_weight * max(0.0, mppi_pred - prior_pred)

            # Consistency penalty: discourage SAC from relying on MPPI corrections
            consistency_penalty = self.consistency_penalty_weight * float(np.sum(delta_norm ** 2))

            training_reward = base_reward + correction_bonus - consistency_penalty
        else:
            correction_bonus = 0.0
            consistency_penalty = 0.0
            training_reward = base_reward

        info["env_reward"] = base_reward
        info["training_reward"] = float(training_reward)
        info["base_reward"] = base_reward
        info["mppi_correction_bonus"] = float(correction_bonus)
        info["mppi_consistency_penalty"] = float(consistency_penalty)
        info["mppi_delta_surge"] = float(delta_norm[0])
        info["mppi_delta_yaw"] = float(delta_norm[1])
        info["mppi_delta_norm"] = float(np.linalg.norm(mppi_delta))
        info["reward_profile"] = self.reward_profile
        info["raw_intent"] = raw_intent.copy()
        info["raw_action"] = prior_action
        info["optimized_action"] = executed_action.astype(np.float32)
        info["hierarchical_mppi_enabled"] = True
        info["hierarchical_mppi_v41_enabled"] = True
        info["mppi_dbas_enabled"] = True

        self._prev_executed_action = executed_action.astype(np.float32)
        return obs, training_reward, terminated, truncated, info

    def get_planner_state(self):
        return self.env.get_planner_state()
