import sys
import types
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

sys.modules.setdefault("matplotlib", types.ModuleType("matplotlib"))
sys.modules.setdefault("matplotlib.pyplot", types.ModuleType("matplotlib.pyplot"))

from scripts.analysis import plot_comparison_curves as plots  # noqa: E402


class PlotComparisonCurvesTest(unittest.TestCase):
    def test_low_intervention_modes_have_matplotlib_colors(self):
        modes = [
            "dsac",
            "pure_mppi",
            "dsac_rl_driven_mppi",
            "dsac_rl_driven_mppi_no_hss",
            "dsac_rl_driven_mppi_fixed_sigma",
            "dsac_rl_driven_mppi_no_q",
        ]

        for mode in modes:
            self.assertIsInstance(plots.color_for(mode), str)

    def test_unknown_modes_use_a_stable_fallback_color(self):
        self.assertIsInstance(plots.color_for("future_ablation_mode"), str)
        self.assertEqual(plots.color_for("future_ablation_mode"), plots.color_for("future_ablation_mode"))


if __name__ == "__main__":
    unittest.main()
