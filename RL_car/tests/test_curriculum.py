import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from dsac_mppi.envs.curriculum import CurriculumConfig, CurriculumManager  # noqa: E402


class CurriculumManagerTests(unittest.TestCase):
    def test_auto_curriculum_advances_when_window_meets_thresholds(self):
        manager = CurriculumManager(
            "auto",
            CurriculumConfig(
                window_size=3,
                success_threshold=0.8,
                collision_threshold=0.1,
                mean_abs_frenet_d_threshold=1.5,
            ),
        )

        for _ in range(3):
            manager.record_episode(
                {
                    "is_success": True,
                    "is_collision": False,
                    "frenet_d": 0.4,
                    "min_laser_dist": 3.0,
                }
            )

        self.assertEqual(manager.stage, 1)
        self.assertEqual(len(manager.history), 0)

    def test_curriculum_does_not_degrade_when_stage_is_hard(self):
        manager = CurriculumManager("auto", CurriculumConfig(window_size=2))
        manager.stage = 2

        for _ in range(2):
            manager.record_episode({"is_success": False, "is_collision": True, "frenet_d": 3.0})

        self.assertEqual(manager.stage, 2)


if __name__ == "__main__":
    unittest.main()
