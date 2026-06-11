import math
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from dsac_mppi.envs.frenet_utils import FrenetTransform  # noqa: E402


class FrenetTransformTests(unittest.TestCase):
    def test_straight_path_projection_s_d_and_sign(self):
        frenet = FrenetTransform(np.array([0.0, 0.0]), np.array([10.0, 0.0]), curve_offset=0.0)

        s, d = frenet.cartesian_to_frenet(np.array([4.0, 1.5]))
        self.assertAlmostEqual(s, 4.0, places=5)
        self.assertAlmostEqual(d, 1.5, places=5)

        s, d = frenet.cartesian_to_frenet(np.array([4.0, -1.5]))
        self.assertAlmostEqual(s, 4.0, places=5)
        self.assertAlmostEqual(d, -1.5, places=5)

    def test_s_is_clamped_to_path_endpoints_for_outside_points(self):
        frenet = FrenetTransform(np.array([0.0, 0.0]), np.array([10.0, 0.0]), curve_offset=0.0)

        before_s, before_d = frenet.cartesian_to_frenet(np.array([-2.0, 0.5]))
        after_s, after_d = frenet.cartesian_to_frenet(np.array([12.0, -0.5]))

        self.assertAlmostEqual(before_s, 0.0, places=5)
        self.assertAlmostEqual(before_d, 0.5, places=5)
        self.assertAlmostEqual(after_s, frenet.path_length, places=5)
        self.assertAlmostEqual(after_d, -0.5, places=5)

    def test_curved_path_round_trip_uses_segment_projection(self):
        frenet = FrenetTransform(np.array([0.0, 0.0]), np.array([10.0, 0.0]), curve_offset=2.0)

        for s_expected in np.linspace(0.5, frenet.path_length - 0.5, 5):
            for d_expected in (-0.4, 0.0, 0.6):
                point = frenet.frenet_to_cartesian(float(s_expected), float(d_expected))
                s, d = frenet.cartesian_to_frenet(point)
                reconstructed = frenet.frenet_to_cartesian(s, d)

                self.assertAlmostEqual(s, s_expected, delta=0.03)
                self.assertAlmostEqual(d, d_expected, delta=0.03)
                self.assertLess(np.linalg.norm(reconstructed - point), 0.02)

    def test_heading_error_is_normalized(self):
        frenet = FrenetTransform(np.array([0.0, 0.0]), np.array([10.0, 0.0]), curve_offset=0.0)

        error = frenet.get_heading_error(robot_yaw=3.5 * math.pi, s=5.0)

        self.assertGreaterEqual(error, -math.pi)
        self.assertLessEqual(error, math.pi)
        self.assertAlmostEqual(error, math.pi / 2.0, places=6)

    def test_lookahead_point_on_straight_path(self):
        frenet = FrenetTransform(np.array([0.0, 0.0]), np.array([10.0, 0.0]), curve_offset=0.0)

        lookahead_s, point = frenet.get_lookahead_point(4.0, 3.0)

        self.assertAlmostEqual(lookahead_s, 7.0, places=5)
        np.testing.assert_allclose(point, np.array([7.0, 0.0]), atol=1e-5)

    def test_lookahead_point_clamps_near_goal_and_non_positive_distance(self):
        frenet = FrenetTransform(np.array([0.0, 0.0]), np.array([10.0, 0.0]), curve_offset=0.0)

        lookahead_s, point = frenet.get_lookahead_point(9.0, 3.0)
        self.assertAlmostEqual(lookahead_s, frenet.path_length, places=5)
        np.testing.assert_allclose(point, np.array([10.0, 0.0]), atol=1e-5)

        same_s, same_point = frenet.get_lookahead_point(4.0, -1.0)
        self.assertAlmostEqual(same_s, 4.0, places=5)
        np.testing.assert_allclose(same_point, frenet.frenet_to_cartesian(4.0, 0.0), atol=1e-5)

    def test_lookahead_point_on_curved_path_matches_centerline_conversion(self):
        frenet = FrenetTransform(np.array([0.0, 0.0]), np.array([10.0, 0.0]), curve_offset=2.0)

        lookahead_s, point = frenet.get_lookahead_point(2.0, 3.0)

        self.assertAlmostEqual(lookahead_s, 5.0, places=5)
        np.testing.assert_allclose(point, frenet.frenet_to_cartesian(5.0, 0.0), atol=1e-8)
        _, d = frenet.cartesian_to_frenet(point)
        self.assertAlmostEqual(d, 0.0, delta=0.03)


if __name__ == "__main__":
    unittest.main()
