import math
import sys
import types
import unittest
from pathlib import Path

import numpy as np


BEAM_MAP_DIR = Path(__file__).resolve().parents[1] / "src" / "nav_demo" / "scripts" / "beam_map"
sys.path.insert(0, str(BEAM_MAP_DIR))


try:
    import gymnasium  # noqa: F401
except ModuleNotFoundError:
    gymnasium = types.ModuleType("gymnasium")
    spaces = types.ModuleType("gymnasium.spaces")

    class Box:
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.low = np.full(shape, low, dtype=dtype) if shape is not None and np.isscalar(low) else np.asarray(low, dtype=dtype)
            self.high = np.full(shape, high, dtype=dtype) if shape is not None and np.isscalar(high) else np.asarray(high, dtype=dtype)
            self.shape = tuple(shape) if shape is not None else self.low.shape
            self.dtype = dtype

    class Env:
        pass

    class Wrapper:
        def __init__(self, env):
            self.env = env
            self.observation_space = getattr(env, "observation_space", None)
            self.action_space = getattr(env, "action_space", None)

        @property
        def unwrapped(self):
            return getattr(self.env, "unwrapped", self.env)

        def reset(self, **kwargs):
            return self.env.reset(**kwargs)

        def step(self, action):
            return self.env.step(action)

    spaces.Box = Box
    gymnasium.Env = Env
    gymnasium.Wrapper = Wrapper
    gymnasium.spaces = spaces
    sys.modules["gymnasium"] = gymnasium
    sys.modules["gymnasium.spaces"] = spaces


from hierarchical_mppi_wrapper import (  # noqa: E402
    HierarchicalMppiV2Wrapper,
    HierarchicalMppiV3Wrapper,
    HierarchicalMppiV41Wrapper,
    HierarchicalMppiV4Wrapper,
    HierarchicalMppiWrapper,
    decode_mode_intent_v4,
    intent_to_frenet_params_v3,
    intent_to_mode_params_v4,
    intent_to_params,
    intent_to_params_v2,
    intent_to_structured_params_v41,
)
from mppi_dbas import MPPIDBaSConfig, MPPIDBaSOptimizer  # noqa: E402


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
        "velocity": np.zeros(3, dtype=float),
        "target_position": np.array([10.0, 0.0], dtype=float),
        "frenet_transform": StraightFrenet(),
        "scan": FakeScan(scan_ranges or [10.0, 10.0, 10.0, 10.0, 10.0]),
        "dt": 0.1,
        "mass": 2.0,
        "damping": 0.5,
        "max_laser_range": 10.0,
        "last_action": np.array([0.5, 0.0], dtype=float),
    }


class FakeEnv:
    def __init__(self):
        self.action_space = gymnasium.spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([2.0, 1.0]),
            dtype=np.float32,
        )
        self.observation_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.last_action = None

    def get_planner_state(self):
        return planner_state()

    def reset(self, **kwargs):
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self.last_action = np.asarray(action, dtype=np.float32)
        return np.ones(4, dtype=np.float32), 1.0, False, False, {"terminal_reason": "running"}


class FakeOptimizer:
    def reset(self):
        self.reset_called = True

    def optimize_from_intent(self, intent, planner_state):
        return np.array([0.7, -0.2], dtype=np.float32), {
            "mppi_cost": 3.0,
            "min_predicted_obstacle_distance": 1.2,
            "mppi_pred_collision": False,
            "mppi_pred_out_of_bounds": False,
        }


class FakeV2Optimizer:
    def reset(self):
        self.reset_called = True

    def optimize_from_intent_v2(self, intent, planner_state):
        return np.array([0.6, 0.1], dtype=np.float32), {
            "action_source": "intent_prior",
            "mppi_active": False,
            "mppi_triggered": False,
            "mppi_trigger_reason": "base_safe",
            "intent_prior_surge": 0.6,
            "intent_prior_yaw": 0.1,
            "mppi_executed_surge": 0.6,
            "mppi_executed_yaw": 0.1,
        }


class FakeV2DeltaOptimizer(FakeV2Optimizer):
    def __init__(self, source="hierarchical_mppi", trigger="trigger_lateral_error"):
        self.config = MPPIDBaSConfig()
        self.source = source
        self.trigger = trigger

    def optimize_from_intent_v2(self, intent, planner_state):
        return np.array([1.0, 0.5], dtype=np.float32), {
            "action_source": self.source,
            "mppi_active": True,
            "mppi_triggered": True,
            "mppi_trigger_reason": self.trigger,
            "reject_reason": "none",
            "intent_prior_surge": 0.6,
            "intent_prior_yaw": 0.1,
            "mppi_executed_surge": 1.0,
            "mppi_executed_yaw": 0.5,
            "current_obstacle_distance": 10.0,
        }


class FakeV3Optimizer:
    def __init__(self):
        self.config = MPPIDBaSConfig()

    def reset(self):
        self.reset_called = True

    def optimize_from_frenet_intent_v3(self, intent, planner_state):
        return np.array([0.8, -0.15], dtype=np.float32), {
            "action_source": "hierarchical_mppi_v3",
            "mppi_active": True,
            "mppi_fallback_active": False,
            "target_progress_speed": 0.75,
            "target_lateral_offset": -0.5,
            "caution_level": 0.5,
            "recovery_level": 0.5,
            "intent_feasible": True,
            "intent_feasibility_cost": 0.0,
            "mppi_best_cost": 1.25,
            "mppi_predicted_progress": 0.4,
            "mppi_predicted_lateral_error": 0.1,
            "mppi_predicted_min_obstacle_distance": 2.0,
            "mppi_predicted_oob_risk": 0.0,
        }


class FakeV4Optimizer:
    def __init__(self):
        self.config = MPPIDBaSConfig()

    def reset(self):
        self.reset_called = True

    def optimize_from_mode_intent_v4(self, intent, planner_state):
        return np.array([0.45, 0.2], dtype=np.float32), {
            "action_source": "hierarchical_mppi_v4",
            "mppi_active": True,
            "mppi_triggered": True,
            "mppi_trigger_reason": "mode_avoid_left",
            "mppi_accept": True,
            "candidate_accepted": True,
            "high_level_mode": "avoid_left",
            "high_level_mode_index": 2,
            "mode_target_speed": 0.45,
            "mode_turn_bias": 0.2,
            "intent_prior_surge": 0.4,
            "intent_prior_yaw": 0.15,
            "mppi_executed_surge": 0.45,
            "mppi_executed_yaw": 0.2,
            "current_obstacle_distance": 0.8,
        }


class FakeV41Optimizer:
    def __init__(self, source="intent_prior", risk_score=0.0, triggered=False):
        self.config = MPPIDBaSConfig()
        self.source = source
        self.risk_score = risk_score
        self.triggered = triggered

    def reset(self):
        self.reset_called = True

    def optimize_from_structured_intent_v41(self, intent, planner_state):
        return np.array([0.55, 0.05], dtype=np.float32), {
            "action_source": self.source,
            "mppi_active": self.triggered,
            "mppi_triggered": self.triggered,
            "mppi_trigger_reason": "trigger_mppi_gate" if self.triggered else "base_safe",
            "mppi_accept": self.source == "hierarchical_mppi_v41",
            "candidate_accepted": self.source == "hierarchical_mppi_v41",
            "intent_prior_surge": 0.52,
            "intent_prior_yaw": 0.04,
            "mppi_executed_surge": 0.55,
            "mppi_executed_yaw": 0.05,
            "current_obstacle_distance": 1.2,
            "structured_target_progress_speed": 0.65,
            "structured_target_lateral_offset": 0.1,
            "structured_safety_margin": 0.2,
            "structured_mppi_gate": 0.8,
            "structured_dynamic_offset_limit": 1.0,
            "structured_target_feasible": True,
            "structured_prior_risk_score": self.risk_score,
            "structured_candidate_risk_score": max(0.0, self.risk_score - 0.2),
            "reject_reason": "none",
        }


class HierarchicalSacMppiTests(unittest.TestCase):
    def test_wrapper_exposes_four_dimensional_intent_action_space_and_executes_mppi_action(self):
        env = FakeEnv()
        wrapper = HierarchicalMppiWrapper(env, optimizer=FakeOptimizer())

        self.assertEqual(wrapper.action_space.shape, (4,))
        self.assertTrue(np.allclose(wrapper.action_space.low, -1.0))
        self.assertTrue(np.allclose(wrapper.action_space.high, 1.0))

        _, _, _, _, info = wrapper.step(np.array([0.0, -0.5, 0.25, 1.0], dtype=np.float32))

        np.testing.assert_allclose(env.last_action, np.array([0.7, -0.2], dtype=np.float32))
        self.assertEqual(info["action_source"], "hierarchical_mppi")
        self.assertAlmostEqual(info["sac_intent_turn_bias"], -0.5)
        self.assertAlmostEqual(info["mppi_executed_surge"], 0.7)
        self.assertAlmostEqual(info["mppi_executed_yaw"], -0.2)

    def test_intent_mapping_uses_expected_ranges(self):
        params = intent_to_params(np.array([-1.0, 1.0, -1.0, 1.0], dtype=np.float32))

        self.assertAlmostEqual(params["target_speed"], 0.0)
        self.assertAlmostEqual(params["turn_bias"], 0.8)
        self.assertAlmostEqual(params["path_weight"], 0.5)
        self.assertAlmostEqual(params["safety_weight"], 4.0)

    def test_v2_intent_mapping_uses_conservative_ranges(self):
        params = intent_to_params_v2(np.array([-1.0, 1.0, -1.0, 1.0], dtype=np.float32))

        self.assertAlmostEqual(params["target_speed"], 0.0)
        self.assertAlmostEqual(params["turn_bias"], 0.55)
        self.assertAlmostEqual(params["path_weight"], 0.8)
        self.assertAlmostEqual(params["safety_weight"], 3.2)

    def test_v3_frenet_intent_mapping_uses_long_horizon_semantics(self):
        params = intent_to_frenet_params_v3(
            np.array([-1.0, 1.0, -1.0, 1.0], dtype=np.float32),
            planner_state(),
            MPPIDBaSConfig(),
        )

        self.assertAlmostEqual(params["target_progress_speed"], 0.25)
        self.assertAlmostEqual(params["target_lateral_offset"], 1.0)
        self.assertAlmostEqual(params["caution_level"], 0.0)
        self.assertAlmostEqual(params["recovery_level"], 1.0)
        self.assertAlmostEqual(params["path_relaxation"], 0.0)
        self.assertGreater(params["oob_weight"], params["base_oob_weight"])
        self.assertGreater(params["lateral_target_weight"], params["base_lateral_weight"])

    def test_v3_lateral_offset_maps_right_center_left(self):
        config = MPPIDBaSConfig()
        right = intent_to_frenet_params_v3(np.array([0.0, -1.0, 0.0, 0.0]), planner_state(), config)
        center = intent_to_frenet_params_v3(np.array([0.0, 0.0, 0.0, 0.0]), planner_state(), config)
        left = intent_to_frenet_params_v3(np.array([0.0, 1.0, 0.0, 0.0]), planner_state(), config)

        self.assertLess(right["target_lateral_offset"], 0.0)
        self.assertAlmostEqual(center["target_lateral_offset"], 0.0)
        self.assertGreater(left["target_lateral_offset"], 0.0)

    def test_v3_dynamic_offset_limit_shrinks_near_boundary(self):
        config = MPPIDBaSConfig()
        centered = intent_to_frenet_params_v3(np.array([0.0, 1.0, 0.0, 0.0]), planner_state(y=0.0), config)
        near_boundary = intent_to_frenet_params_v3(np.array([0.0, 1.0, 0.0, 0.0]), planner_state(y=2.85), config)

        self.assertLess(near_boundary["dynamic_offset_limit"], centered["dynamic_offset_limit"])
        self.assertLess(abs(near_boundary["target_lateral_offset"]), abs(centered["target_lateral_offset"]))

    def test_v3_caution_and_recovery_raise_safety_and_centering_weights(self):
        config = MPPIDBaSConfig()
        low = intent_to_frenet_params_v3(np.array([0.0, 0.0, -1.0, -1.0]), planner_state(), config)
        high = intent_to_frenet_params_v3(np.array([0.0, 0.0, 1.0, 1.0]), planner_state(), config)

        self.assertGreater(high["safe_distance"], low["safe_distance"])
        self.assertGreater(high["ttc_weight"], low["ttc_weight"])
        self.assertGreater(high["obstacle_weight"], low["obstacle_weight"])
        self.assertGreater(high["oob_weight"], low["oob_weight"])
        self.assertGreater(high["lateral_target_weight"], low["lateral_target_weight"])

    def test_v3_wrapper_exposes_four_dimensional_intent_and_debug_fields(self):
        env = FakeEnv()
        wrapper = HierarchicalMppiV3Wrapper(env, optimizer=FakeV3Optimizer())

        self.assertEqual(wrapper.action_space.shape, (4,))
        _, reward, _, _, info = wrapper.step(np.array([0.0, -1.0, 0.0, 0.0], dtype=np.float32))

        self.assertAlmostEqual(reward, 1.0)
        np.testing.assert_allclose(env.last_action, np.array([0.8, -0.15], dtype=np.float32))
        self.assertTrue(info["hierarchical_mppi_v3_enabled"])
        self.assertEqual(info["action_source"], "hierarchical_mppi_v3")
        self.assertIn("target_progress_speed", info)
        self.assertIn("target_lateral_offset", info)
        self.assertIn("intent_feasible", info)
        self.assertIn("mppi_predicted_oob_risk", info)

    def test_v4_decode_chooses_mode_and_scales_params(self):
        decoded = decode_mode_intent_v4(np.array([-1.0, -0.5, 0.8, 0.4, 0.2, -0.2, 1.0, -1.0], dtype=np.float32))

        self.assertEqual(decoded["mode_name"], "avoid_left")
        self.assertEqual(decoded["mode_index"], 2)
        self.assertAlmostEqual(decoded["speed_scale"], 1.0)
        self.assertAlmostEqual(decoded["avoid_strength"], 0.0)

    def test_v4_mapping_uses_mode_semantics(self):
        params = intent_to_mode_params_v4(np.array([-1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 0.0, 1.0], dtype=np.float32), planner_state(), MPPIDBaSConfig())

        self.assertEqual(params["mode_name"], "avoid_left")
        self.assertGreater(params["turn_bias"], 0.0)
        self.assertGreater(params["safety_weight"], 2.0)
        self.assertGreater(params["safe_distance"], MPPIDBaSConfig().safe_distance)

    def test_v4_wrapper_exposes_mode_debug_and_enhanced_reward(self):
        env = FakeEnv()
        wrapper = HierarchicalMppiV4Wrapper(
            env,
            optimizer=FakeV4Optimizer(),
            reward_profile="enhanced",
            mode_hold_steps=1,
        )

        self.assertEqual(wrapper.action_space.shape, (8,))
        _, reward, _, _, info = wrapper.step(np.array([-1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 0.0, 1.0], dtype=np.float32))

        np.testing.assert_allclose(env.last_action, np.array([0.45, 0.2], dtype=np.float32))
        self.assertTrue(info["hierarchical_mppi_v4_enabled"])
        self.assertEqual(info["high_level_mode"], "avoid_left")
        self.assertEqual(info["action_source"], "hierarchical_mppi_v4")
        self.assertIn("reward_hazard_mode_bonus", info)
        self.assertGreaterEqual(reward, 1.0)

    def test_v41_mapping_uses_continuous_structured_targets(self):
        params = intent_to_structured_params_v41(
            np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            planner_state(y=2.85),
            MPPIDBaSConfig(),
        )

        self.assertGreater(params["target_progress_speed"], 1.0)
        self.assertLess(params["dynamic_offset_limit"], 1.0)
        self.assertLessEqual(abs(params["target_lateral_offset"]), params["dynamic_offset_limit"])
        self.assertGreater(params["safe_distance"], MPPIDBaSConfig().safe_distance)
        self.assertAlmostEqual(params["mppi_gate"], 1.0)

    def test_v41_wrapper_compat_reward_matches_env_reward(self):
        wrapper = HierarchicalMppiV41Wrapper(
            FakeEnv(),
            optimizer=FakeV41Optimizer(source="intent_prior", risk_score=5.0, triggered=True),
            reward_profile="compat",
        )

        _, reward, _, _, info = wrapper.step(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))

        self.assertEqual(wrapper.action_space.shape, (4,))
        self.assertAlmostEqual(reward, 1.0)
        self.assertAlmostEqual(info["training_reward"], 1.0)
        self.assertTrue(info["hierarchical_mppi_v41_enabled"])

    def test_v41_wrapper_guided_reward_penalizes_risk_and_intervention(self):
        wrapper = HierarchicalMppiV41Wrapper(
            FakeEnv(),
            optimizer=FakeV41Optimizer(source="hierarchical_mppi_v41", risk_score=5.0, triggered=True),
            reward_profile="guided",
            risk_penalty_weight=0.1,
            intervention_penalty_weight=0.1,
            infeasible_penalty_weight=0.0,
            jitter_penalty_weight=0.0,
        )

        _, reward, _, _, info = wrapper.step(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))

        self.assertLess(reward, 1.0)
        self.assertGreater(info["structured_risk_exposure_penalty"], 0.0)
        self.assertGreater(info["structured_intervention_penalty"], 0.0)

    def test_v41_wrapper_guided_reward_penalizes_target_jitter(self):
        wrapper = HierarchicalMppiV41Wrapper(
            FakeEnv(),
            optimizer=FakeV41Optimizer(source="intent_prior", risk_score=0.0, triggered=False),
            reward_profile="guided",
            risk_penalty_weight=0.0,
            intervention_penalty_weight=0.0,
            infeasible_penalty_weight=0.0,
            jitter_penalty_weight=1.0,
        )

        wrapper.step(np.array([-1.0, -1.0, 0.0, -1.0], dtype=np.float32))
        _, reward, _, _, info = wrapper.step(np.array([1.0, 1.0, 0.0, -1.0], dtype=np.float32))

        self.assertLess(reward, 1.0)
        self.assertGreater(info["structured_target_jitter_penalty"], 0.0)

    def test_v2_wrapper_smooths_and_holds_intent_prior_action(self):
        env = FakeEnv()
        wrapper = HierarchicalMppiV2Wrapper(
            env,
            optimizer=FakeV2Optimizer(),
            intent_ema_alpha=0.5,
            intent_hold_steps=2,
        )

        _, _, _, _, first_info = wrapper.step(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        _, _, _, _, second_info = wrapper.step(np.array([-1.0, 1.0, 1.0, 1.0], dtype=np.float32))

        self.assertEqual(first_info["action_source"], "intent_prior")
        self.assertFalse(first_info["mppi_triggered"])
        self.assertAlmostEqual(first_info["held_intent_target_speed"], 0.5)
        self.assertAlmostEqual(second_info["held_intent_target_speed"], 0.5)
        np.testing.assert_allclose(env.last_action, np.array([0.6, 0.1], dtype=np.float32))

    def test_v2_delta_penalty_disabled_returns_env_reward(self):
        wrapper = HierarchicalMppiV2Wrapper(FakeEnv(), optimizer=FakeV2DeltaOptimizer())

        _, reward, _, _, info = wrapper.step(np.zeros(4, dtype=np.float32))

        self.assertAlmostEqual(reward, 1.0)
        self.assertAlmostEqual(info["env_reward"], 1.0)
        self.assertAlmostEqual(info["training_reward"], 1.0)
        self.assertAlmostEqual(info["mppi_delta_penalty"], 0.0)

    def test_v2_delta_penalty_applies_to_non_emergency_large_delta(self):
        wrapper = HierarchicalMppiV2Wrapper(
            FakeEnv(),
            optimizer=FakeV2DeltaOptimizer(),
            enable_delta_penalty=True,
            delta_penalty_alpha=1.0,
            delta_deadband=0.1,
        )

        _, reward, _, _, info = wrapper.step(np.zeros(4, dtype=np.float32))

        expected_delta = float(np.linalg.norm(np.array([1.0, 0.5]) - np.array([0.6, 0.1])))
        expected_penalty = (expected_delta - 0.1) ** 2
        self.assertAlmostEqual(info["mppi_delta_norm"], expected_delta, places=6)
        self.assertAlmostEqual(info["mppi_delta_penalty"], expected_penalty, places=6)
        self.assertAlmostEqual(reward, 1.0 - expected_penalty, places=6)

    def test_v2_delta_penalty_ignores_emergency_fallback(self):
        wrapper = HierarchicalMppiV2Wrapper(
            FakeEnv(),
            optimizer=FakeV2DeltaOptimizer(source="fallback", trigger="trigger_collision_risk"),
            enable_delta_penalty=True,
            delta_penalty_alpha=1.0,
            delta_deadband=0.1,
        )

        _, reward, _, _, info = wrapper.step(np.zeros(4, dtype=np.float32))

        self.assertAlmostEqual(reward, 1.0)
        self.assertAlmostEqual(info["mppi_delta_penalty"], 0.0)

    def test_intent_conditioned_mppi_outputs_bounded_action_and_debug(self):
        optimizer = MPPIDBaSOptimizer(
            MPPIDBaSConfig(seed=0, num_samples=8, horizon=4)
        )

        action, debug = optimizer.optimize_from_intent(np.array([1.0, 1.0, 0.0, 0.0]), planner_state())

        self.assertEqual(action.shape, (2,))
        self.assertGreaterEqual(action[0], -1.0)
        self.assertLessEqual(action[0], 2.0)
        self.assertGreaterEqual(action[1], -1.0)
        self.assertLessEqual(action[1], 1.0)
        self.assertEqual(debug["action_source"], "hierarchical_mppi")
        self.assertIn("sac_intent_target_speed", debug)
        self.assertIn("mppi_best_cost", debug)

    def test_v2_optimizer_skips_mppi_when_prior_is_safe(self):
        optimizer = MPPIDBaSOptimizer(MPPIDBaSConfig(seed=0, num_samples=8, horizon=4))

        action, debug = optimizer.optimize_from_intent_v2(np.array([0.0, 0.0, 0.0, 0.0]), planner_state())

        self.assertEqual(debug["action_source"], "intent_prior")
        self.assertFalse(debug["mppi_triggered"])
        self.assertFalse(debug["mppi_active"])
        np.testing.assert_allclose(action, np.array([0.75, 0.0], dtype=np.float32), atol=1e-6)

    def test_v2_optimizer_triggers_mppi_when_prior_is_risky(self):
        optimizer = MPPIDBaSOptimizer(MPPIDBaSConfig(seed=0, num_samples=8, horizon=4))

        _, debug = optimizer.optimize_from_intent_v2(
            np.array([0.0, 0.0, 0.0, 0.0]),
            planner_state(scan_ranges=[0.5, 0.5, 0.5, 0.5, 0.5]),
        )

        self.assertTrue(debug["mppi_triggered"])
        self.assertIn(debug["action_source"], ("intent_prior", "hierarchical_mppi", "fallback"))

    def test_v2_optimizer_triggers_on_lateral_deviation(self):
        optimizer = MPPIDBaSOptimizer(MPPIDBaSConfig(seed=0, num_samples=8, horizon=4))

        _, debug = optimizer.optimize_from_intent_v2(
            np.array([0.0, 0.0, 0.0, 0.0]),
            planner_state(y=1.0),
        )

        self.assertTrue(debug["mppi_triggered"])
        self.assertEqual(debug["mppi_trigger_reason"], "trigger_lateral_error")

    def test_v3_optimizer_outputs_bounded_action_and_frenet_debug(self):
        optimizer = MPPIDBaSOptimizer(MPPIDBaSConfig(seed=0, num_samples=8, horizon=4))

        action, debug = optimizer.optimize_from_frenet_intent_v3(
            np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32),
            planner_state(),
        )

        self.assertEqual(action.shape, (2,))
        self.assertGreaterEqual(action[0], -1.0)
        self.assertLessEqual(action[0], 2.0)
        self.assertGreaterEqual(action[1], -1.0)
        self.assertLessEqual(action[1], 1.0)
        self.assertTrue(debug["mppi_active"])
        self.assertIn(debug["action_source"], ("hierarchical_mppi_v3", "fallback"))
        self.assertIn("target_lateral_offset", debug)
        self.assertIn("mppi_predicted_progress", debug)

    def test_v41_optimizer_keeps_safe_prior_without_mppi(self):
        optimizer = MPPIDBaSOptimizer(MPPIDBaSConfig(seed=0, num_samples=8, horizon=4))

        action, debug = optimizer.optimize_from_structured_intent_v41(
            np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32),
            planner_state(),
        )

        self.assertEqual(action.shape, (2,))
        self.assertEqual(debug["action_source"], "intent_prior")
        self.assertFalse(debug["mppi_triggered"])
        self.assertTrue(debug["hierarchical_mppi_v41_enabled"])
        self.assertIn("structured_target_progress_speed", debug)
        self.assertIn("structured_mppi_gate", debug)

    def test_v41_gate_alone_does_not_trigger_without_local_risk(self):
        optimizer = MPPIDBaSOptimizer(MPPIDBaSConfig(seed=0, num_samples=8, horizon=4))

        _, debug = optimizer.optimize_from_structured_intent_v41(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            planner_state(),
        )

        self.assertEqual(debug["action_source"], "intent_prior")
        self.assertFalse(debug["mppi_triggered"])
        self.assertEqual(debug["mppi_trigger_reason"], "base_safe")

    def test_v41_optimizer_triggers_on_obstacle_risk(self):
        optimizer = MPPIDBaSOptimizer(MPPIDBaSConfig(seed=0, num_samples=8, horizon=4))

        _, debug = optimizer.optimize_from_structured_intent_v41(
            np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32),
            planner_state(scan_ranges=[0.5, 0.5, 0.5, 0.5, 0.5]),
        )

        self.assertTrue(debug["mppi_triggered"])
        self.assertIn(debug["action_source"], ("intent_prior", "hierarchical_mppi_v41", "fallback"))

    def test_v41_accept_rejects_candidate_with_higher_oob_risk(self):
        optimizer = MPPIDBaSOptimizer(MPPIDBaSConfig(seed=0))
        params = intent_to_structured_params_v41(np.zeros(4, dtype=np.float32), planner_state(), optimizer.config)
        prior_metrics = {
            "risk_score": 0.0,
            "collision_risk": 0.0,
            "out_of_bounds_risk": 0.0,
            "progress": 0.4,
            "max_lateral_error": 0.1,
            "ttc_cost": 0.0,
            "final_lateral_offset": 0.0,
        }
        candidate_metrics = {
            **prior_metrics,
            "risk_score": 4.0,
            "out_of_bounds_risk": 1.0,
            "max_lateral_error": 4.0,
        }

        accepted, reason = optimizer._accept_structured_candidate_v41(
            params=params,
            prior_metrics=prior_metrics,
            candidate_metrics=candidate_metrics,
            candidate_action=np.array([0.5, 0.0], dtype=float),
            prior_action=np.array([0.5, 0.0], dtype=float),
            prior_score=10.0,
            candidate_score=1.0,
        )

        self.assertFalse(accepted)
        self.assertEqual(reason, "reject_out_of_bounds")

    def test_v41_prefers_fallback_when_candidate_is_not_clearly_safer(self):
        optimizer = MPPIDBaSOptimizer(MPPIDBaSConfig(seed=0))
        candidate_metrics = {
            "risk_score": 1.0,
            "out_of_bounds_risk": 0.0,
            "collision_risk": 0.0,
        }
        fallback_metrics = {
            "risk_score": 0.95,
            "out_of_bounds_risk": 0.0,
            "collision_risk": 0.0,
        }

        prefer_mppi = optimizer._structured_prefer_mppi_over_fallback_v41(
            candidate_metrics,
            fallback_metrics,
            accepted=True,
        )

        self.assertFalse(prefer_mppi)

    def test_v3_fallback_activates_when_best_mppi_is_still_high_risk(self):
        optimizer = MPPIDBaSOptimizer(
            MPPIDBaSConfig(seed=0, num_samples=8, horizon=4, collision_distance=9.0, safe_distance=9.5)
        )

        _, debug = optimizer.optimize_from_frenet_intent_v3(
            np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32),
            planner_state(scan_ranges=[0.2, 0.2, 0.2, 0.2, 0.2]),
        )

        self.assertEqual(debug["action_source"], "fallback")
        self.assertTrue(debug["mppi_fallback_active"])

    def test_v2_accept_rejects_high_progress_candidate_without_safety_or_path_gain(self):
        optimizer = MPPIDBaSOptimizer(MPPIDBaSConfig(seed=0))
        prior_metrics = {
            "risk_score": 0.0,
            "collision_risk": 0.0,
            "out_of_bounds_risk": 0.0,
            "progress": 0.1,
            "max_lateral_error": 0.2,
            "ttc_cost": 0.0,
        }
        candidate_metrics = {
            **prior_metrics,
            "progress": 2.0,
            "max_lateral_error": 0.2,
        }

        accepted, reason = optimizer._accept_intent_v2_candidate(
            prior_metrics,
            candidate_metrics,
            candidate_action=np.array([0.5, 0.1], dtype=float),
            prior_action=np.array([0.5, 0.0], dtype=float),
            prior_score=10.0,
            candidate_score=1.0,
        )

        self.assertFalse(accepted)
        self.assertEqual(reason, "reject_no_safety_or_path_gain")

    def test_v2_accept_rejects_large_prior_deviation_without_safety_gain(self):
        optimizer = MPPIDBaSOptimizer(MPPIDBaSConfig(seed=0, mppi_max_action_delta=(0.1, 0.1)))
        metrics = {
            "risk_score": 0.0,
            "collision_risk": 0.0,
            "out_of_bounds_risk": 0.0,
            "progress": 0.1,
            "max_lateral_error": 0.2,
            "ttc_cost": 0.0,
        }

        accepted, reason = optimizer._accept_intent_v2_candidate(
            metrics,
            metrics,
            candidate_action=np.array([1.0, 0.6], dtype=float),
            prior_action=np.array([0.5, 0.0], dtype=float),
            prior_score=10.0,
            candidate_score=1.0,
        )

        self.assertFalse(accepted)
        self.assertEqual(reason, "reject_trust_region")

    def test_safety_and_path_intents_change_conditioned_cost(self):
        optimizer = MPPIDBaSOptimizer(MPPIDBaSConfig(seed=0))
        metrics = {
            "total_cost": 1.0,
            "dbas_cost": 2.0,
            "ttc_cost": 1.0,
            "out_of_bounds_cost": 0.5,
            "max_lateral_error": 1.5,
            "max_heading_error": 0.4,
            "progress": 0.2,
            "min_distance": 0.4,
            "collision_risk": 0.0,
            "out_of_bounds_risk": 0.0,
        }
        sequence = np.tile(np.array([0.4, 0.1], dtype=float), (4, 1))
        prior = np.array([0.5, 0.0], dtype=float)

        low_safety = {"target_speed": 0.5, "turn_bias": 0.0, "path_weight": 1.0, "safety_weight": 0.5}
        high_safety = {**low_safety, "safety_weight": 4.0}
        low_path = {"target_speed": 0.5, "turn_bias": 0.0, "path_weight": 0.5, "safety_weight": 1.0}
        high_path = {**low_path, "path_weight": 3.0}

        self.assertGreater(
            optimizer._intent_conditioned_cost(metrics, sequence, prior, high_safety),
            optimizer._intent_conditioned_cost(metrics, sequence, prior, low_safety),
        )
        self.assertGreater(
            optimizer._intent_conditioned_cost(metrics, sequence, prior, high_path),
            optimizer._intent_conditioned_cost(metrics, sequence, prior, low_path),
        )

    def test_fallback_action_yaws_toward_path_center_when_laterally_offset(self):
        optimizer = MPPIDBaSOptimizer(MPPIDBaSConfig(seed=0))

        action = optimizer._fallback_action(
            np.array([0.7, 0.0], dtype=float),
            planner_state(y=1.0),
            {"global_min": 10.0, "front_min": 10.0, "left_min": 10.0, "right_min": 10.0},
        )

        self.assertLess(action[1], 0.0)


if __name__ == "__main__":
    unittest.main()
