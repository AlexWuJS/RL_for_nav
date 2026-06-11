import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from dsac_mppi.envs.dynamic_obstacles import (  # noqa: E402
    CVObstaclePredictor,
    CVPredictionConfig,
    DynamicObstacle,
    DynamicObstacleScenarioFactory,
    compute_tcpa_dcpa_risk,
    step_obstacles,
)


class DynamicObstacleTests(unittest.TestCase):
    def test_crossing_obstacle_moves_continuously_with_stable_velocity(self):
        dt = 0.1
        obstacles = DynamicObstacleScenarioFactory.crossing(x=5.0, y=3.0, vy=-0.5)
        positions = [obstacles[0].position.copy()]
        for _ in range(10):
            step_obstacles(obstacles, dt)
            positions.append(obstacles[0].position.copy())
        positions = np.asarray(positions)
        deltas = np.diff(positions, axis=0)

        np.testing.assert_allclose(deltas[:, 0], 0.0, atol=1e-9)
        np.testing.assert_allclose(deltas[:, 1], -0.05, atol=1e-9)
        self.assertTrue(np.all(deltas[:, 1] < 0.0))
        self.assertLess(float(np.max(np.linalg.norm(np.diff(deltas, axis=0), axis=1))), 1e-9)

    def test_cv_predictor_estimates_velocity_from_history(self):
        dt = 0.1
        predictor = CVObstaclePredictor(CVPredictionConfig(history_len=4, prediction_horizon=12, dt=dt))
        obstacle = DynamicObstacle(0, 5.0, 3.0, 0.0, -0.5, 0.4, "crossing")

        for _ in range(4):
            predictor.update([obstacle])
            obstacle.step(dt)

        velocity = predictor.estimate_velocity(obstacle)

        np.testing.assert_allclose(velocity, np.array([0.0, -0.5]), atol=1e-9)

    def test_future_prediction_extends_along_motion_direction(self):
        dt = 0.1
        predictor = CVObstaclePredictor(CVPredictionConfig(history_len=4, prediction_horizon=12, dt=dt))
        obstacle = DynamicObstacle(0, 5.0, 3.0, 0.0, -0.5, 0.4, "crossing")
        for _ in range(4):
            predictor.update([obstacle])
            obstacle.step(dt)
        current = obstacle.copy()
        predictions = predictor.predict([current], usv_position=np.array([0.0, 0.0]), usv_velocity=np.array([0.8, 0.0]))
        positions = np.asarray(predictions[0]["positions"], dtype=float)
        expected_final = current.position + np.array([0.0, -0.5]) * dt * 12.0

        self.assertEqual(positions.shape, (12, 2))
        np.testing.assert_allclose(np.diff(positions, axis=0)[:, 0], 0.0, atol=1e-9)
        self.assertTrue(np.all(np.diff(positions, axis=0)[:, 1] < 0.0))
        np.testing.assert_allclose(positions[-1], expected_final, atol=1e-9)

    def test_tcpa_dcpa_risk_rises_then_falls_for_crossing_and_clears_when_receding(self):
        usv_position = np.array([0.0, 0.0])
        usv_velocity = np.array([1.0, 0.0])
        obstacle_velocity = np.array([0.0, -0.5])
        samples = [
            np.array([2.0, 3.0]),
            np.array([2.0, 2.0]),
            np.array([2.0, 1.0]),
            np.array([2.0, 0.0]),
        ]
        metrics = [
            compute_tcpa_dcpa_risk(pos, obstacle_velocity, usv_position, usv_velocity, radius=0.4, t_max=3.0)
            for pos in samples
        ]
        tcpas = [row[0] for row in metrics]
        dcpas = [row[1] for row in metrics]
        risks = [row[2] for row in metrics]

        self.assertTrue(all(tcpa > 0.0 for tcpa in tcpas))
        self.assertLess(dcpas[1], dcpas[0])
        self.assertLess(dcpas[2], dcpas[1])
        self.assertGreater(dcpas[3], dcpas[2])
        self.assertGreater(risks[1], risks[0])
        self.assertGreater(risks[2], risks[1])
        self.assertLess(risks[3], risks[2])

        receding = compute_tcpa_dcpa_risk(
            obstacle_position=np.array([-1.0, 0.0]),
            obstacle_velocity=np.array([-0.5, 0.0]),
            usv_position=usv_position,
            usv_velocity=usv_velocity,
            radius=0.4,
            t_max=3.0,
        )
        self.assertLess(receding[0], 0.0)
        self.assertAlmostEqual(receding[2], 0.0, places=9)

    def test_prediction_output_is_mppi_readable(self):
        predictor = CVObstaclePredictor(CVPredictionConfig(history_len=4, prediction_horizon=12, dt=0.1))
        obstacle = DynamicObstacleScenarioFactory.crossing()[0]
        predictor.update([obstacle])

        predictions = predictor.predict([obstacle], usv_position=np.array([0.0, 0.0]), usv_velocity=np.array([0.8, 0.0]))
        row = predictions[0]

        self.assertEqual(set(["id", "positions", "velocity", "radius", "tcpa", "dcpa", "risk", "type"]), set(row.keys()))
        self.assertEqual(row["id"], 0)
        self.assertEqual(row["type"], "crossing")
        self.assertEqual(len(row["positions"]), 12)
        self.assertEqual(len(row["positions"][0]), 2)
        self.assertEqual(len(row["velocity"]), 2)
        self.assertIsInstance(row["tcpa"], float)
        self.assertIsInstance(row["dcpa"], float)
        self.assertGreaterEqual(row["risk"], 0.0)
        self.assertLessEqual(row["risk"], 1.0)

    def test_scenario_factory_provides_all_first_stage_scenarios(self):
        for scenario in (
            "crossing",
            "oncoming",
            "overtaking_slow_ahead",
            "overtaking_fast_behind",
            "mixed",
        ):
            obstacles = DynamicObstacleScenarioFactory.create(scenario)
            self.assertGreaterEqual(len(obstacles), 1)
            self.assertLessEqual(len(obstacles), 5)
            self.assertTrue(all(isinstance(obstacle, DynamicObstacle) for obstacle in obstacles))


if __name__ == "__main__":
    unittest.main()
