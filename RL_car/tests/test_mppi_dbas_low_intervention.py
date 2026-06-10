import math
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from dsac_mppi.controllers.mppi_dbas import MPPIDBaSConfig, MPPIDBaSOptimizer  # noqa: E402


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
        "velocity": np.zeros(2, dtype=float),
        "target_position": np.array([10.0, 0.0], dtype=float),
        "frenet_transform": StraightFrenet(),
        "scan": FakeScan(scan_ranges or [10.0, 10.0, 10.0, 10.0, 10.0]),
        "dt": 0.1,
        "surge_time_constant": 0.6,
        "yaw_time_constant": 0.4,
        "max_du": 0.15,
        "max_dr": 0.12,
        "max_laser_range": 10.0,
        "last_action": np.array([0.5, 0.0], dtype=float),
        "last_command": np.array([0.5, 0.0], dtype=float),
    }


class LowInterventionMppiTests(unittest.TestCase):
    def test_reward_aligned_filter_leaves_safe_sac_action_unchanged(self):
        optimizer = MPPIDBaSOptimizer(
            MPPIDBaSConfig(seed=0, use_reward_aligned_cost=True, always_run_mppi=False)
        )

        action, debug = optimizer.optimize(np.array([0.5, 0.0]), planner_state())

        np.testing.assert_allclose(action, np.array([0.5, 0.0]), atol=1e-6)
        self.assertIs(debug["mppi_active"], False)
        self.assertEqual(debug["action_source"], "sac")
        self.assertEqual(debug["mppi_decision_reason"], "base_safe")

    def test_teacher_only_records_mppi_candidate_without_executing_it(self):
        optimizer = MPPIDBaSOptimizer(
            MPPIDBaSConfig(
                seed=1,
                use_reward_aligned_cost=True,
                always_run_mppi=True,
                teacher_only=True,
                execute_mppi=False,
                enable_fallback=False,
                reward_improvement_threshold=-999.0,
            )
        )

        action, debug = optimizer.optimize(np.array([0.5, 0.0]), planner_state([1.0, 1.0, 1.0, 1.0, 1.0]))

        np.testing.assert_allclose(action, np.array([0.5, 0.0]), atol=1e-6)
        self.assertIs(debug["mppi_active"], True)
        self.assertIs(debug["mppi_accept"], False)
        self.assertIs(debug["teacher_mppi_would_accept"], True)
        self.assertEqual(debug["action_source"], "sac")
        self.assertEqual(debug["reject_reason"], "teacher_record_only")

    def test_mppi_candidate_must_beat_accepted_fallback_before_execution(self):
        optimizer = MPPIDBaSOptimizer(MPPIDBaSConfig(seed=0))
        base_metrics = {
            "collision_risk": 1.0,
            "out_of_bounds_risk": 0.0,
            "min_distance": 0.20,
            "max_lateral_error": 0.2,
            "progress": 1.0,
            "ttc_cost": 1.0,
            "risk_score": 10.0,
            "total_cost": 10.0,
        }
        fallback_metrics = {
            **base_metrics,
            "collision_risk": 0.0,
            "min_distance": 0.70,
            "ttc_cost": 0.0,
            "risk_score": 0.1,
            "total_cost": 0.1,
        }
        candidate_metrics = {
            **base_metrics,
            "collision_risk": 0.0,
            "min_distance": 0.80,
            "ttc_cost": 0.0,
            "risk_score": 0.2,
            "total_cost": 0.2,
        }

        accepted, reason = optimizer._accept_candidate_against_baselines(
            base_metrics,
            fallback_metrics,
            candidate_metrics,
            np.array([0.55, 0.05]),
            np.array([0.5, 0.0]),
            fallback_accept=True,
        )

        self.assertIs(accepted, False)
        self.assertEqual(reason, "reject_not_better_than_fallback")

    def test_piecewise_lateral_penalty_has_center_buffer_and_hard_zone(self):
        optimizer = MPPIDBaSOptimizer(MPPIDBaSConfig())

        self.assertEqual(optimizer._piecewise_lateral_penalty(0.25), 0.0)
        self.assertLess(
            optimizer._piecewise_lateral_penalty(1.0),
            optimizer._piecewise_lateral_penalty(2.0),
        )
        self.assertGreater(optimizer._piecewise_lateral_penalty(1.0), 0.0)
        self.assertGreater(optimizer._piecewise_lateral_penalty(3.2), 100.0)

    def test_dbas_barrier_is_zero_outside_safe_distance_and_progressive_inside(self):
        optimizer = MPPIDBaSOptimizer(MPPIDBaSConfig(safe_distance=0.55, collision_distance=0.25))

        self.assertEqual(optimizer._dbas_cost(0.70), 0.0)
        self.assertLess(optimizer._dbas_cost(0.45), optimizer._dbas_cost(0.30))
        self.assertGreater(optimizer._dbas_cost(0.45), 0.0)


if __name__ == "__main__":
    unittest.main()
