import sys
import types
import unittest
from pathlib import Path


BEAM_MAP_DIR = Path(__file__).resolve().parents[1] / "src" / "nav_demo" / "scripts" / "beam_map"
sys.path.insert(0, str(BEAM_MAP_DIR))

sys.modules.setdefault("matplotlib", types.ModuleType("matplotlib"))
sys.modules.setdefault("matplotlib.pyplot", types.ModuleType("matplotlib.pyplot"))

import plot_comparison_curves as plots  # noqa: E402


class PlotComparisonCurvesTest(unittest.TestCase):
    def test_low_intervention_modes_have_matplotlib_colors(self):
        modes = [
            "baseline",
            "shield_only",
            "shield_first",
            "shield_mppi_teacher",
            "shield_mppi_execute",
        ]

        for mode in modes:
            self.assertIsInstance(plots.color_for(mode), str)

    def test_unknown_modes_use_a_stable_fallback_color(self):
        self.assertIsInstance(plots.color_for("future_ablation_mode"), str)
        self.assertEqual(plots.color_for("future_ablation_mode"), plots.color_for("future_ablation_mode"))


if __name__ == "__main__":
    unittest.main()
