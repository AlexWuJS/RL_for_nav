import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from dsac_mppi.controllers.reference_tracker import ReferenceLineTracker, ReferenceTrackerConfig  # noqa: E402
from dsac_mppi.envs.frenet_utils import FrenetTransform  # noqa: E402
from dsac_mppi.envs.usv_dynamics import USVDynamicsConfig, frenet_outputs, step_pose  # noqa: E402


class ReferenceTrackerTests(unittest.TestCase):
    def test_positive_lateral_error_commands_right_turn_on_straight_path(self):
        frenet = FrenetTransform(np.array([0.0, 0.0]), np.array([10.0, 0.0]), curve_offset=0.0)
        tracker = ReferenceLineTracker()

        action, debug = tracker.compute_action(
            position=np.array([0.0, 1.0], dtype=float),
            yaw=0.0,
            target_position=np.array([10.0, 0.0], dtype=float),
            frenet_transform=frenet,
        )

        self.assertGreater(debug["frenet_d"], 0.0)
        self.assertLess(action[1], 0.0)
        self.assertGreater(action[0], 0.0)

    def test_offline_straight_line_tracking_reaches_goal_region(self):
        dynamics = USVDynamicsConfig(dt=0.1, max_du=0.15, max_dr=0.12)
        tracker = ReferenceLineTracker(
            ReferenceTrackerConfig(lookahead_distance=1.5, target_speed=0.85, heading_gain=1.4, lateral_gain=0.30),
            dynamics,
        )
        start = np.array([0.0, 0.0], dtype=float)
        goal = np.array([10.0, 0.0], dtype=float)
        frenet = FrenetTransform(start, goal, curve_offset=0.0)
        position = np.array([0.0, 0.8], dtype=float)
        yaw = 0.3
        velocity = np.array([0.0, 0.0], dtype=float)
        previous_command = np.array([0.0, 0.0], dtype=float)
        first_metrics = frenet_outputs(position, yaw, goal, frenet)
        max_abs_d = abs(first_metrics["frenet_d"])

        for _ in range(400):
            action, _ = tracker.compute_action(position, yaw, goal, frenet)
            position, yaw, velocity, previous_command = step_pose(
                position,
                yaw,
                velocity,
                action,
                previous_command,
                dynamics,
                "first_order",
            )
            metrics = frenet_outputs(position, yaw, goal, frenet)
            max_abs_d = max(max_abs_d, abs(metrics["frenet_d"]))
            if metrics["remaining_path"] < 0.4 and abs(metrics["frenet_d"]) < 0.5:
                break

        final_metrics = frenet_outputs(position, yaw, goal, frenet)
        self.assertGreater(final_metrics["frenet_s"], 0.9 * frenet.path_length)
        self.assertLess(final_metrics["remaining_path"], first_metrics["remaining_path"])
        self.assertLess(abs(final_metrics["frenet_d"]), 0.5)
        self.assertLess(max_abs_d, 2.0)


if __name__ == "__main__":
    unittest.main()
