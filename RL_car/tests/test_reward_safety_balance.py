import sys
import unittest
from pathlib import Path


BEAM_MAP_DIR = Path(__file__).resolve().parents[1] / "src" / "nav_demo" / "scripts" / "beam_map"
sys.path.insert(0, str(BEAM_MAP_DIR))

from frenet_utils import compute_tracking_reward, obstacle_avoidance_penalty  # noqa: E402
from mppi_dbas import MPPIDBaSConfig  # noqa: E402


class RewardSafetyBalanceTests(unittest.TestCase):
    def test_obstacle_penalty_grows_near_collision_and_clears_at_safe_distance(self):
        close_penalty = obstacle_avoidance_penalty(0.25, safe_distance=0.7)
        mid_penalty = obstacle_avoidance_penalty(0.55, safe_distance=0.7)
        clear_penalty = obstacle_avoidance_penalty(0.7, safe_distance=0.7)

        self.assertGreaterEqual(close_penalty, 49.0)
        self.assertGreater(close_penalty, mid_penalty)
        self.assertEqual(clear_penalty, 0.0)

    def test_recovering_toward_path_scores_better_than_drifting_away(self):
        recovering = compute_tracking_reward(
            delta_s=0.0,
            frenet_d=1.0,
            heading_error=0.0,
            min_obstacle_dist=10.0,
            previous_abs_frenet_d=1.4,
        )
        drifting = compute_tracking_reward(
            delta_s=0.0,
            frenet_d=1.4,
            heading_error=0.0,
            min_obstacle_dist=10.0,
            previous_abs_frenet_d=1.0,
        )

        self.assertGreater(recovering["total"], drifting["total"])
        self.assertGreater(recovering["components"]["recover_center_bonus"], 0.0)
        self.assertLess(drifting["components"]["recover_center_bonus"], 0.0)

    def test_mppi_reward_constants_match_safety_balance(self):
        cfg = MPPIDBaSConfig()

        self.assertEqual(cfg.env_terminal_reward, 1000.0)
        self.assertEqual(cfg.env_success_reward, 1000.0)
        self.assertEqual(cfg.env_safe_distance, 0.7)
        self.assertEqual(cfg.env_out_of_bounds_limit, 3.0)


if __name__ == "__main__":
    unittest.main()
