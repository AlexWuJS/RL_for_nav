import math
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from dsac_mppi.envs.usv_dynamics import USVDynamicsConfig, step_pose, step_velocity  # noqa: E402


class USVDynamicsTests(unittest.TestCase):
    def test_first_order_velocity_applies_command_and_acceleration_limits(self):
        config = USVDynamicsConfig(dt=0.1, surge_time_constant=0.6, yaw_time_constant=0.4, max_du=0.15, max_dr=0.12)

        velocity, command = step_velocity(
            velocity=np.array([0.0, 0.0], dtype=float),
            command=np.array([1.0, 0.4], dtype=float),
            previous_command=np.array([0.0, 0.0], dtype=float),
            config=config,
            dynamics_model="first_order",
        )

        np.testing.assert_allclose(command, np.array([0.15, 0.12], dtype=float), atol=1e-8)
        np.testing.assert_allclose(velocity, np.array([0.025, 0.03], dtype=float), atol=1e-8)

    def test_pose_step_uses_x_y_psi_u_r_state(self):
        config = USVDynamicsConfig(dt=0.1, surge_time_constant=0.5, yaw_time_constant=0.5, max_du=1.5, max_dr=0.6)

        position, yaw, velocity, command = step_pose(
            position=np.array([1.0, 2.0], dtype=float),
            yaw=math.pi / 2.0,
            velocity=np.array([0.5, 0.0], dtype=float),
            command=np.array([1.0, 0.5], dtype=float),
            previous_command=np.array([0.5, 0.0], dtype=float),
            config=config,
            dynamics_model="first_order",
        )

        np.testing.assert_allclose(command, np.array([1.0, 0.5], dtype=float), atol=1e-8)
        np.testing.assert_allclose(velocity, np.array([0.6, 0.1], dtype=float), atol=1e-8)
        np.testing.assert_allclose(position, np.array([1.0, 2.06], dtype=float), atol=1e-8)
        self.assertAlmostEqual(yaw, math.pi / 2.0 + 0.01, places=8)


if __name__ == "__main__":
    unittest.main()
