import sys
import types
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - depends on local ML runtime
    torch = None


BEAM_MAP_DIR = Path(__file__).resolve().parents[1] / "src" / "nav_demo" / "scripts" / "beam_map"
sys.path.insert(0, str(BEAM_MAP_DIR))


try:
    import gymnasium  # noqa: F401
except ModuleNotFoundError:
    gymnasium = types.ModuleType("gymnasium")
    spaces = types.ModuleType("gymnasium.spaces")

    class Box:
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.low = np.full(shape, low, dtype=dtype) if shape is not None and np.isscalar(low) else np.asarray(low, dtype=dtype)
            self.high = np.full(shape, high, dtype=dtype) if shape is not None and np.isscalar(high) else np.asarray(high, dtype=dtype)
            self.shape = tuple(shape) if shape is not None else self.low.shape
            self.dtype = dtype

    class Wrapper:
        def __init__(self, env):
            self.env = env
            self.observation_space = getattr(env, "observation_space", None)
            self.action_space = getattr(env, "action_space", None)

    class Env:
        pass

    gymnasium.Env = Env
    gymnasium.Wrapper = Wrapper
    spaces.Box = Box
    gymnasium.spaces = spaces
    sys.modules["gymnasium"] = gymnasium
    sys.modules["gymnasium.spaces"] = spaces


if torch is not None:
    from dsac import DSACConfig, DSACPolicy, DSACReplayBuffer, DistributionalCritic  # noqa: E402
    from rl_driven_mppi import DSACPolicyAdapter, RLDrivenMPPIConfig, RLDrivenMPPIOptimizer  # noqa: E402
    from test_rl_driven_mppi import planner_state  # noqa: E402


@unittest.skipIf(torch is None, "torch is not installed in this Python environment")
class DSACModuleTests(unittest.TestCase):
    def make_policy(self):
        config = DSACConfig(
            observation_dim=8,
            action_dim=2,
            hidden_dim=32,
            num_quantiles=8,
            learning_starts=2,
            batch_size=2,
            buffer_size=16,
            seed=7,
        )
        return DSACPolicy(config, device="cpu")

    def test_actor_outputs_bounded_actions_and_supports_sampling(self):
        policy = self.make_policy()
        obs = np.zeros((1, 8), dtype=np.float32)

        deterministic_action, _ = policy.predict(obs, deterministic=True)
        sampled_action, _ = policy.predict(obs, deterministic=False)

        self.assertEqual(deterministic_action.shape, (1, 2))
        self.assertEqual(sampled_action.shape, (1, 2))
        self.assertTrue(np.all(deterministic_action >= np.asarray(policy.config.action_low) - 1e-6))
        self.assertTrue(np.all(deterministic_action <= np.asarray(policy.config.action_high) + 1e-6))

    def test_distributional_critic_outputs_quantile_distribution(self):
        critic = DistributionalCritic(obs_dim=8, action_dim=2, hidden_dim=32, num_quantiles=8)
        obs = torch.zeros((3, 8), dtype=torch.float32)
        action = torch.zeros((3, 2), dtype=torch.float32)

        q1, q2 = critic(obs, action)
        mean_q = critic.mean_q(obs, action)

        self.assertEqual(q1.shape, (3, 8))
        self.assertEqual(q2.shape, (3, 8))
        self.assertEqual(mean_q.shape, (3, 1))

    def test_replay_buffer_samples_frame_stacked_observations(self):
        buffer = DSACReplayBuffer(obs_dim=8, action_dim=2, size=8, seed=0)
        for idx in range(4):
            obs = np.full(8, idx, dtype=np.float32)
            buffer.add(obs, np.array([0.1, -0.1]), 1.0, obs + 1.0, False)

        batch = buffer.sample(3, torch.device("cpu"))

        self.assertEqual(batch["obs"].shape, (3, 8))
        self.assertEqual(batch["actions"].shape, (3, 2))
        self.assertEqual(batch["rewards"].shape, (3, 1))
        self.assertEqual(batch["dones"].shape, (3, 1))

    def test_dsac_policy_adapter_provides_rlmppi_contract(self):
        policy = self.make_policy()
        adapter = DSACPolicyAdapter(policy)
        obs = np.zeros(8, dtype=np.float32)

        mean = adapter.predict_mean(obs, np.zeros(2))
        sample = adapter.sample_action(obs, np.zeros(2), np.random.default_rng(0))
        std = adapter.action_std(obs)
        terminal_cost, used = adapter.estimate_terminal_cost(obs, np.array([0.0, 0.0]))

        self.assertEqual(mean.shape, (2,))
        self.assertEqual(sample.shape, (2,))
        self.assertEqual(std.shape, (2,))
        self.assertTrue(np.isfinite(terminal_cost))
        self.assertTrue(used)

    def test_strict_terminal_q_fails_when_critic_is_unavailable(self):
        class NoCriticAdapter:
            def predict_mean(self, observation, fallback_action):
                return np.array([0.5, 0.0], dtype=float)

            def sample_action(self, observation, fallback_action, rng):
                return np.array([0.5, 0.0], dtype=float)

            def action_std(self, observation=None):
                return np.array([0.1, 0.1], dtype=float)

            def estimate_terminal_cost(self, observation, action):
                return 0.0, False

        optimizer = RLDrivenMPPIOptimizer(
            RLDrivenMPPIConfig(
                seed=1,
                horizon=3,
                num_rl_rollouts=2,
                num_mppi_rollouts=2,
                num_iterations=1,
                top_z=2,
                strict_terminal_q=True,
            ),
            NoCriticAdapter(),
        )

        with self.assertRaises(RuntimeError):
            optimizer.optimize(np.array([0.5, 0.0]), planner_state(), observation=np.zeros(8))


if __name__ == "__main__":
    unittest.main()
