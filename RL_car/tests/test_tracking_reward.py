import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_rl_driven_mppi import planner_state  # noqa: E402
from dsac_mppi.envs.frenet_utils import compute_tracking_reward, obstacle_tracking_scale, piecewise_lateral_penalty  # noqa: E402
from dsac_mppi.controllers.rl_driven_mppi import RLDrivenMPPIConfig, RLDrivenMPPIOptimizer  # noqa: E402


class TrackingRewardTests(unittest.TestCase):
    def test_lateral_penalty_is_piecewise_and_barrier_rises_fast(self):
        penalties = [
            piecewise_lateral_penalty(0.2),
            piecewise_lateral_penalty(0.8),
            piecewise_lateral_penalty(1.8),
            piecewise_lateral_penalty(3.0),
        ]

        self.assertEqual(penalties[0], 0.0)
        self.assertGreater(penalties[1], penalties[0])
        self.assertGreater(penalties[2], penalties[1])
        self.assertGreater(penalties[3] - penalties[2], penalties[2] - penalties[1])

    def test_obstacle_pressure_temporarily_reduces_tracking_weight(self):
        near = obstacle_tracking_scale(0.4)
        mid = obstacle_tracking_scale(1.0)
        far = obstacle_tracking_scale(2.0)

        self.assertLess(near, mid)
        self.assertLess(mid, far)
        self.assertAlmostEqual(far, 1.0)

    def test_progress_reward_is_weakened_when_far_from_path(self):
        centered = compute_tracking_reward(0.2, 0.0, 0.0)
        far = compute_tracking_reward(0.2, 2.5, 0.0)

        self.assertGreater(centered["components"]["s_progress"], far["components"]["s_progress"])
        self.assertLess(far["total"], centered["total"])

    def test_recovering_toward_center_is_rewarded(self):
        recovering = compute_tracking_reward(0.0, 1.0, 0.0, previous_abs_frenet_d=1.5)
        worsening = compute_tracking_reward(0.0, 1.5, 0.0, previous_abs_frenet_d=1.0)

        self.assertGreater(
            recovering["components"]["recover_center_bonus"],
            worsening["components"]["recover_center_bonus"],
        )
        self.assertGreater(recovering["total"], worsening["total"])

    def test_reward_aligned_rollout_cost_tracks_negative_shared_reward(self):
        config = RLDrivenMPPIConfig(
            seed=0,
            horizon=1,
            use_reward_aligned_cost=True,
            use_terminal_q=False,
            action_lag_alpha=0.0,
            trust_region_weight=0.0,
            dbas_weight=0.0,
            ttc_weight=0.0,
            out_of_bounds_weight=0.0,
        )
        optimizer = RLDrivenMPPIOptimizer(config)
        state = planner_state(scan_ranges=[10.0, 10.0, 10.0, 10.0, 10.0], y=0.0)
        action = np.array([0.5, 0.0], dtype=float)
        sequence = action.reshape(1, 2)
        metrics = optimizer._sequence_metrics(sequence, action, state, optimizer._scan_to_obstacle_points(state))

        self.assertAlmostEqual(metrics["total_cost"], -metrics["predicted_reward"], places=6)


if __name__ == "__main__":
    unittest.main()
