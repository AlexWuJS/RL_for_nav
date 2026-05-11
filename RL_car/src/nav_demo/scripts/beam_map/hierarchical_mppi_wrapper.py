from typing import Optional

import gymnasium as gym
import numpy as np

from mppi_dbas import MPPIDBaSConfig, MPPIDBaSOptimizer, intent_to_params


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
        debug.setdefault("sac_intent_target_speed", float(raw_intent[0]))
        debug.setdefault("sac_intent_turn_bias", float(raw_intent[1]))
        debug.setdefault("sac_intent_path_weight", float(raw_intent[2]))
        debug.setdefault("sac_intent_safety_weight", float(raw_intent[3]))
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
