from typing import Optional

import gymnasium as gym
import numpy as np

from mppi_dbas import (
    MPPIDBaSConfig,
    MPPIDBaSOptimizer,
    intent_to_frenet_params_v3,
    intent_to_params,
    intent_to_params_v2,
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
