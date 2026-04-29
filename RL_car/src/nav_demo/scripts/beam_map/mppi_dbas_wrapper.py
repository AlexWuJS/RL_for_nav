from typing import Optional

import gymnasium as gym
import numpy as np

from mppi_dbas import MPPIDBaSConfig, MPPIDBaSOptimizer


class MppiDbaSActionWrapper(gym.Wrapper):
    """Apply MPPI-DBaS as an evaluation-time action post-processor."""

    def __init__(self, env: gym.Env, config: Optional[MPPIDBaSConfig] = None):
        super().__init__(env)
        self.optimizer = MPPIDBaSOptimizer(config)

    def reset(self, **kwargs):
        self.optimizer.reset()
        return self.env.reset(**kwargs)

    def step(self, action):
        planner_state = self.env.get_planner_state()
        raw_action = np.asarray(action, dtype=np.float32).reshape(-1)[:2]
        optimized_action, debug = self.optimizer.optimize(raw_action, planner_state)
        obs, reward, terminated, truncated, info = self.env.step(optimized_action)
        info = dict(info)
        info.update(debug)
        info["raw_action"] = raw_action.astype(np.float32)
        info["optimized_action"] = optimized_action.astype(np.float32)
        info["mppi_dbas_enabled"] = True
        return obs, reward, terminated, truncated, info

    def get_planner_state(self):
        return self.env.get_planner_state()
