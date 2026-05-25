import math
import sys
import types
import unittest
from pathlib import Path

import numpy as np


BEAM_MAP_DIR = Path(__file__).resolve().parents[1] / "src" / "nav_demo" / "scripts" / "beam_map"
sys.path.insert(0, str(BEAM_MAP_DIR))


try:
    import gymnasium  # noqa: F401
except ModuleNotFoundError:
    gymnasium = types.ModuleType("gymnasium")

    class Wrapper:
        def __init__(self, env):
            self.env = env
            self.observation_space = getattr(env, "observation_space", None)
            self.action_space = getattr(env, "action_space", None)

        def reset(self, **kwargs):
            return self.env.reset(**kwargs)

        def step(self, action):
            return self.env.step(action)

    class Env:
        pass

    gymnasium.Env = Env
    gymnasium.Wrapper = Wrapper
    sys.modules["gymnasium"] = gymnasium


from rl_driven_mppi import RLDrivenMPPIConfig, RLDrivenMPPIOptimizer, SB3SacPolicyAdapter  # noqa: E402


class FakeScan:
    def __init__(self, ranges, angle_min=-math.pi / 2, angle_increment=math.pi / 4):
        self.ranges = ranges
        self.angle_min = angle_min
        self.angle_increment = angle_increment


class StraightFrenet:
    path_length = 10.0

    def cartesian_to_frenet(self, point):
        return float(point[0]), float(point[1])

    def get_heading_error(self, yaw, s):
        return -float(yaw)


def planner_state(scan_ranges=None, y=0.0):
    return {
        "position": np.array([0.0, y], dtype=float),
        "yaw": 0.0,
        "velocity": np.zeros(3, dtype=float),
        "target_position": np.array([10.0, 0.0], dtype=float),
        "frenet_transform": StraightFrenet(),
        "scan": FakeScan(scan_ranges or [10.0, 10.0, 10.0, 10.0, 10.0]),
        "dt": 0.1,
        "mass": 2.0,
        "damping": 0.5,
        "max_laser_range": 10.0,
        "last_action": np.array([0.5, 0.0], dtype=float),
    }


class FakePolicyAdapter(SB3SacPolicyAdapter):
    def __init__(self):
        super().__init__(None)
        self.sample_calls = 0
        self.q_calls = 0

    def predict_mean(self, observation, fallback_action):
        return np.array([0.8, 0.1], dtype=float)

    def sample_action(self, observation, fallback_action, rng):
        self.sample_calls += 1
        return np.array([0.75, -0.05], dtype=float)

    def action_std(self):
        return np.array([0.30, 0.20], dtype=float)

    def estimate_terminal_cost(self, observation, action):
        self.q_calls += 1
        return 5.0, True


class RLDrivenMPPITests(unittest.TestCase):
    def make_optimizer(self, **kwargs):
        config = RLDrivenMPPIConfig(
            seed=3,
            horizon=5,
            num_rl_rollouts=4,
            num_mppi_rollouts=5,
            num_iterations=2,
            top_z=3,
            sigma_min=(0.04, 0.04),
            **kwargs,
        )
        adapter = FakePolicyAdapter()
        return RLDrivenMPPIOptimizer(config, adapter), adapter

    def test_rl_initialization_uses_policy_mean_and_std(self):
        optimizer, _ = self.make_optimizer()

        mean, sigma, source = optimizer._initial_distribution(np.array([0.2, 0.0]), observation=np.zeros(4))

        self.assertEqual(source, "rl_policy")
        self.assertEqual(mean.shape, (5, 2))
        self.assertEqual(sigma.shape, (5, 2))
        np.testing.assert_allclose(mean[0], np.array([0.8, 0.1]))
        np.testing.assert_allclose(sigma[0], np.array([0.30, 0.20]))

    def test_guided_rollouts_are_sampled_once_per_online_control_step(self):
        optimizer, adapter = self.make_optimizer()

        action, debug = optimizer.optimize(np.array([0.2, 0.0]), planner_state(), observation=np.zeros(4))

        self.assertEqual(adapter.sample_calls, optimizer.config.num_rl_rollouts)
        self.assertEqual(debug["rlmppi_num_rl_rollouts"], optimizer.config.num_rl_rollouts)
        self.assertEqual(debug["rlmppi_num_iterations"], optimizer.config.num_iterations)
        self.assertEqual(action.shape, (2,))

    def test_top_z_update_keeps_sigma_above_minimum_and_actions_bounded(self):
        optimizer, _ = self.make_optimizer()

        action, debug = optimizer.optimize(np.array([2.0, 1.0]), planner_state(), observation=np.zeros(4))

        self.assertGreaterEqual(debug["rlmppi_sigma_min"], min(optimizer.config.sigma_min) - 1e-9)
        self.assertEqual(debug["rlmppi_top_z"], optimizer.config.top_z)
        self.assertTrue(np.all(action >= np.asarray(optimizer.config.action_low) - 1e-6))
        self.assertTrue(np.all(action <= np.asarray(optimizer.config.action_high) + 1e-6))

    def test_ablation_without_hss_disables_guided_rollouts(self):
        optimizer, adapter = self.make_optimizer(use_hss=False)

        _, debug = optimizer.optimize(np.array([0.2, 0.0]), planner_state(), observation=np.zeros(4))

        self.assertEqual(adapter.sample_calls, 0)
        self.assertFalse(debug["rlmppi_hss_enabled"])
        self.assertEqual(debug["rlmppi_num_rl_rollouts"], 0)

    def test_ablation_without_terminal_q_does_not_call_critic(self):
        optimizer, adapter = self.make_optimizer(use_terminal_q=False)

        _, debug = optimizer.optimize(np.array([0.2, 0.0]), planner_state(), observation=np.zeros(4))

        self.assertEqual(adapter.q_calls, 0)
        self.assertFalse(debug["rlmppi_terminal_q_used"])

    def test_fixed_sigma_ablation_reports_sigma_update_disabled(self):
        optimizer, _ = self.make_optimizer(update_sigma=False)

        _, debug = optimizer.optimize(np.array([0.2, 0.0]), planner_state(), observation=np.zeros(4))

        self.assertFalse(debug["rlmppi_update_sigma"])


if __name__ == "__main__":
    unittest.main()
