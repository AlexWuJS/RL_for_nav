import argparse
import csv
import json
import os
import subprocess
import sys
from typing import Any, Dict, List

import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from hierarchical_mppi_wrapper import HierarchicalMppiWrapper, hierarchical_mppi_config
from mppi_dbas import MPPIDBaSConfig
from mppi_dbas_wrapper import MppiDbaSActionWrapper
from ros_env import MyCarEnv


SHIELD_RESIDUAL_LOW = (-0.15, -0.18)
SHIELD_RESIDUAL_HIGH = (0.06, 0.18)
DICT_OBS_KEYS = {"radar_image", "kinematics"}


def unwrap_env(env):
    current = env
    while hasattr(current, "env"):
        current = current.env
    return current


def config_for_mode(mode: str, seed: int) -> MPPIDBaSConfig:
    if mode in ("shield_first", "shield_mppi_execute", "mppi_dbas"):
        return MPPIDBaSConfig(
            seed=seed,
            use_reward_aligned_cost=True,
            always_run_mppi=False,
            execute_mppi=True,
            final_safety_check=True,
            enable_fallback=True,
            teacher_only=False,
            reward_aligned_residual_low=SHIELD_RESIDUAL_LOW,
            reward_aligned_residual_high=SHIELD_RESIDUAL_HIGH,
        )
    if mode == "shield_mppi_teacher":
        return MPPIDBaSConfig(
            seed=seed,
            use_reward_aligned_cost=True,
            always_run_mppi=False,
            execute_mppi=False,
            final_safety_check=True,
            enable_fallback=True,
            teacher_only=True,
            reward_aligned_residual_low=SHIELD_RESIDUAL_LOW,
            reward_aligned_residual_high=SHIELD_RESIDUAL_HIGH,
        )
    if mode == "sac_mppi":
        return MPPIDBaSConfig(
            seed=seed,
            use_reward_aligned_cost=True,
            always_run_mppi=True,
            execute_mppi=True,
            final_safety_check=False,
            enable_fallback=False,
            reward_aligned_residual_low=(-0.20, -0.25),
            reward_aligned_residual_high=(0.10, 0.25),
        )
    if mode == "sac_mppi_safe":
        return MPPIDBaSConfig(
            seed=seed,
            use_reward_aligned_cost=True,
            always_run_mppi=True,
            execute_mppi=True,
            final_safety_check=True,
            enable_fallback=False,
            reward_aligned_residual_low=(-0.20, -0.25),
            reward_aligned_residual_high=(0.10, 0.25),
        )
    if mode == "shield_only":
        return MPPIDBaSConfig(seed=seed, enable_mppi=False, enable_fallback=True)
    if mode == "hybrid_mppi":
        return MPPIDBaSConfig(seed=seed, enable_mppi=True, enable_fallback=True)
    if mode == "mppi_teacher":
        return MPPIDBaSConfig(
            seed=seed,
            use_reward_aligned_cost=True,
            always_run_mppi=True,
            execute_mppi=False,
            final_safety_check=True,
            enable_fallback=False,
            teacher_only=True,
            reward_aligned_residual_low=(-0.20, -0.25),
            reward_aligned_residual_high=(0.10, 0.25),
        )
    if mode == "trust_mppi":
        return MPPIDBaSConfig(seed=seed, dbas_weight=0.0, risk_activation_distance=10.0, enable_fallback=False)
    if mode == "trust_mppi_dbas":
        return MPPIDBaSConfig(seed=seed, risk_activation_distance=10.0, enable_fallback=False)
    return MPPIDBaSConfig(seed=seed)


class RadarDictObservationWrapper(gym.ObservationWrapper):
    """Adapt the flat MyCarEnv observation to dict observations used by image models."""

    def __init__(self, env: gym.Env, image_size: int = 256):
        super().__init__(env)
        self.image_size = int(image_size)
        self.observation_space = gym.spaces.Dict(
            {
                "radar_image": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(1, self.image_size, self.image_size),
                    dtype=np.uint8,
                ),
                "kinematics": gym.spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(4,),
                    dtype=np.float32,
                ),
            }
        )

    def observation(self, observation):
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        if obs.size < 5:
            laser = np.zeros(self.image_size, dtype=np.float32)
            kinematics = np.zeros(4, dtype=np.float32)
        else:
            laser = obs[:-4]
            kinematics = np.clip(obs[-4:], -1.0, 1.0).astype(np.float32)
        radar_image = self._laser_to_image(laser)
        return {"radar_image": radar_image, "kinematics": kinematics}

    def _laser_to_image(self, laser: np.ndarray) -> np.ndarray:
        laser = np.asarray(laser, dtype=np.float32).reshape(-1)
        if laser.size == 0:
            resized = np.ones(self.image_size, dtype=np.float32)
        else:
            laser = np.nan_to_num(laser, nan=1.0, posinf=1.0, neginf=0.0)
            laser = np.clip(laser, 0.0, 1.0)
            src_x = np.linspace(0.0, 1.0, laser.size)
            dst_x = np.linspace(0.0, 1.0, self.image_size)
            resized = np.interp(dst_x, src_x, laser).astype(np.float32)
        image = np.tile(resized.reshape(1, self.image_size), (self.image_size, 1))
        return np.rint(image.reshape(1, self.image_size, self.image_size) * 255.0).astype(np.uint8)

    def get_planner_state(self):
        return self.env.get_planner_state()


def observation_space_kind(space) -> str:
    if isinstance(space, gym.spaces.Dict) and DICT_OBS_KEYS.issubset(set(space.spaces.keys())):
        return "dict"
    return "flat"


def resolve_obs_mode(requested_mode: str, model_space) -> str:
    if requested_mode != "auto":
        return requested_mode
    return observation_space_kind(model_space)


def make_env(mode: str, seed: int, log_dir: str, obs_mode: str):
    def _init():
        env = MyCarEnv()
        if mode in ("hierarchical_mppi", "hierarchical_mppi_shield"):
            env = HierarchicalMppiWrapper(env, config=hierarchical_mppi_config(seed=seed))
        elif mode != "baseline":
            env = MppiDbaSActionWrapper(env, config_for_mode(mode, seed))
        if obs_mode == "dict":
            env = RadarDictObservationWrapper(env)
        env = Monitor(env, filename=os.path.join(log_dir, mode))
        return env

    return _init


def build_eval_env(mode: str, seed: int, log_dir: str, frame_stack: int, obs_mode: str):
    env = DummyVecEnv([make_env(mode, seed, log_dir, obs_mode)])
    if obs_mode == "flat" and frame_stack > 1:
        env = VecFrameStack(env, n_stack=frame_stack)
    return env


def get_base_env(vec_env):
    current = vec_env
    while hasattr(current, "venv"):
        current = current.venv
    env = current.envs[0]
    return unwrap_env(env)


def get_metric_state(vec_env) -> Dict[str, float]:
    env = get_base_env(vec_env)
    state = env.get_planner_state()
    pos = np.asarray(state["position"], dtype=float)
    target = np.asarray(state["target_position"], dtype=float)
    dist_to_goal = float(np.linalg.norm(target - pos))
    frenet_d = 0.0
    frenet = state.get("frenet_transform")
    if frenet is not None:
        _, frenet_d = frenet.cartesian_to_frenet(pos)
    scan = state.get("scan")
    min_scan = float(state.get("max_laser_range", 10.0))
    if scan is not None and hasattr(scan, "ranges"):
        ranges = np.asarray(scan.ranges, dtype=float)
        ranges = np.nan_to_num(ranges, nan=min_scan, posinf=min_scan, neginf=0.0)
        min_scan = float(np.min(np.clip(ranges, 0.0, min_scan)))
    return {
        "distance_to_goal": dist_to_goal,
        "frenet_abs_d": abs(float(frenet_d)),
        "min_scan_distance": min_scan,
        "step_count": float(getattr(env, "step_count", 0)),
        "dt": float(state.get("dt", 0.1)),
    }


def run_episode(model: SAC, vec_env, deterministic: bool, mode: str, episode_idx: int, trace_dir: str) -> Dict[str, Any]:
    obs = vec_env.reset()
    episode_reward = 0.0
    path_length = 0.0
    min_scan_distance = float("inf")
    frenet_errors: List[float] = []
    action_changes: List[float] = []
    raw_action_changes: List[float] = []
    mppi_active_flags: List[float] = []
    mppi_debug: List[Dict[str, float]] = []
    trace_rows: List[Dict[str, Any]] = []
    prev_pos = None
    prev_action = None
    prev_raw_action = None
    last_info: Dict[str, Any] = {}
    step_idx = 0

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = vec_env.step(action)
        info = dict(infos[0])
        last_info = info
        episode_reward += float(rewards[0])

        metrics = get_metric_state(vec_env)
        if "min_laser_dist" in info:
            metrics["min_scan_distance"] = float(info["min_laser_dist"])
        if "frenet_d" in info:
            metrics["frenet_abs_d"] = abs(float(info["frenet_d"]))
        if "distance_to_goal" in info:
            metrics["distance_to_goal"] = float(info["distance_to_goal"])
        min_scan_distance = min(min_scan_distance, metrics["min_scan_distance"])
        frenet_errors.append(metrics["frenet_abs_d"])

        env = get_base_env(vec_env)
        pos = np.asarray(info.get("current_position", env.current_pos), dtype=float)
        if prev_pos is not None:
            path_length += float(np.linalg.norm(pos - prev_pos))
        prev_pos = pos.copy()

        executed_action = np.asarray(info.get("optimized_action", action[0]), dtype=float).reshape(-1)[:2]
        raw_action = np.asarray(info.get("raw_action", action[0]), dtype=float).reshape(-1)[:2]
        if prev_action is not None:
            action_changes.append(float(np.linalg.norm(executed_action - prev_action)))
        if prev_raw_action is not None:
            raw_action_changes.append(float(np.linalg.norm(raw_action - prev_raw_action)))
        prev_action = executed_action
        prev_raw_action = raw_action

        mppi_active = float(bool(info.get("mppi_active", False)))
        mppi_active_flags.append(mppi_active)
        if info.get("mppi_dbas_enabled"):
            reason = str(info.get("mppi_decision_reason", "none"))
            mppi_debug.append({
                "dbas_cost": float(info.get("dbas_cost", 0.0)),
                "ttc_cost": float(info.get("ttc_cost", 0.0)),
                "out_of_bounds_cost": float(info.get("out_of_bounds_cost", 0.0)),
                "action_delta_norm": float(info.get("action_delta_norm", 0.0)),
                "current_obstacle_distance": float(info.get("current_obstacle_distance", 0.0)),
                "min_predicted_obstacle_distance": float(info.get("min_predicted_obstacle_distance", 0.0)),
                "exploration_noise_scale": float(info.get("exploration_noise_scale", 0.0)),
                "mppi_accept": float(bool(info.get("mppi_accept", False))),
                "mppi_reject": float(bool(info.get("mppi_reject", False))),
                "fallback_active": float(bool(info.get("fallback_active", False))),
                "fallback_accept": float(bool(info.get("fallback_accept", False))),
                "reject_collision_risk": float(reason == "reject_collision_risk"),
                "reject_out_of_bounds": float(reason == "reject_out_of_bounds"),
                "reject_progress_loss": float(reason == "reject_progress_loss"),
                "reject_no_safety_gain": float(reason == "reject_no_safety_gain"),
                "teacher_mppi_would_accept": float(bool(info.get("teacher_mppi_would_accept", False))),
                "mppi_warm_start_used": float(bool(info.get("mppi_warm_start_used", False))),
                "sac_pred_collision": float(bool(info.get("sac_pred_collision", False))),
                "fallback_pred_collision": float(bool(info.get("fallback_pred_collision", False))),
                "mppi_pred_collision": float(bool(info.get("mppi_pred_collision", False))),
                "sac_pred_out_of_bounds": float(bool(info.get("sac_pred_out_of_bounds", False))),
                "fallback_pred_out_of_bounds": float(bool(info.get("fallback_pred_out_of_bounds", False))),
                "mppi_pred_out_of_bounds": float(bool(info.get("mppi_pred_out_of_bounds", False))),
                "candidate_sac_score": float(info.get("candidate_sac_score", 0.0)),
                "candidate_fallback_score": float(info.get("candidate_fallback_score", 0.0)),
                "candidate_mppi_score": float(info.get("candidate_mppi_score", 0.0)),
                "predicted_reward_sac": float(info.get("predicted_reward_sac", 0.0)),
                "predicted_reward_mppi": float(info.get("predicted_reward_mppi", 0.0)),
                "predicted_reward_delta": float(info.get("predicted_reward_delta", 0.0)),
                "reward_prediction_error": float(info.get("predicted_reward_delta", 0.0) - float(rewards[0])),
                "mppi_selected": float(bool(info.get("mppi_selected", False))),
                "source_sac": float(info.get("action_source", "sac") == "sac"),
                "source_mppi": float(info.get("action_source", "sac") == "mppi"),
                "source_fallback": float(info.get("action_source", "sac") == "fallback"),
                "source_hierarchical_mppi": float(info.get("action_source", "sac") == "hierarchical_mppi"),
                "sac_intent_target_speed": float(info.get("sac_intent_target_speed", 0.0)),
                "sac_intent_turn_bias": float(info.get("sac_intent_turn_bias", 0.0)),
                "sac_intent_path_weight": float(info.get("sac_intent_path_weight", 0.0)),
                "sac_intent_safety_weight": float(info.get("sac_intent_safety_weight", 0.0)),
            })

        trace_rows.append({
            "step": step_idx,
            "reward": float(rewards[0]),
            "x": float(pos[0]),
            "y": float(pos[1]),
            "distance_to_goal": float(metrics["distance_to_goal"]),
            "frenet_abs_d": float(metrics["frenet_abs_d"]),
            "min_scan_distance": float(metrics["min_scan_distance"]),
            "raw_surge": float(raw_action[0]),
            "raw_yaw": float(raw_action[1]),
            "optimized_surge": float(executed_action[0]),
            "optimized_yaw": float(executed_action[1]),
            "action_delta_norm": float(info.get("action_delta_norm", 0.0)),
            "current_obstacle_distance": float(info.get("current_obstacle_distance", metrics["min_scan_distance"])),
            "mppi_active": mppi_active,
            "mppi_accept": int(bool(info.get("mppi_accept", False))),
            "mppi_reject": int(bool(info.get("mppi_reject", False))),
            "mppi_decision_reason": info.get("mppi_decision_reason", "none"),
            "selected_reason": info.get("selected_reason", "none"),
            "reject_reason": info.get("reject_reason", "none"),
            "mppi_rejected_reason": info.get("mppi_rejected_reason", info.get("reject_reason", "none")),
            "mppi_prior_type": info.get("mppi_prior_type", "none"),
            "mppi_warm_start_used": int(bool(info.get("mppi_warm_start_used", False))),
            "teacher_mppi_would_accept": int(bool(info.get("teacher_mppi_would_accept", False))),
            "mppi_selected": int(bool(info.get("mppi_selected", False))),
            "fallback_active": int(bool(info.get("fallback_active", False))),
            "fallback_accept": int(bool(info.get("fallback_accept", False))),
            "action_source": info.get("action_source", "sac"),
            "predicted_reward_sac": float(info.get("predicted_reward_sac", 0.0)),
            "predicted_reward_mppi": float(info.get("predicted_reward_mppi", 0.0)),
            "predicted_reward_delta": float(info.get("predicted_reward_delta", 0.0)),
            "actual_step_reward": float(rewards[0]),
            "reward_prediction_error": float(info.get("predicted_reward_delta", 0.0) - float(rewards[0])),
            "sac_rollout_terminal": info.get("sac_rollout_terminal", "none"),
            "mppi_rollout_terminal": info.get("mppi_rollout_terminal", "none"),
            "candidate_sac_score": float(info.get("candidate_sac_score", 0.0)),
            "candidate_fallback_score": float(info.get("candidate_fallback_score", 0.0)),
            "candidate_mppi_score": float(info.get("candidate_mppi_score", 0.0)),
            "sac_pred_collision": int(bool(info.get("sac_pred_collision", False))),
            "fallback_pred_collision": int(bool(info.get("fallback_pred_collision", False))),
            "mppi_pred_collision": int(bool(info.get("mppi_pred_collision", False))),
            "sac_pred_out_of_bounds": int(bool(info.get("sac_pred_out_of_bounds", False))),
            "fallback_pred_out_of_bounds": int(bool(info.get("fallback_pred_out_of_bounds", False))),
            "mppi_pred_out_of_bounds": int(bool(info.get("mppi_pred_out_of_bounds", False))),
            "sac_min_obstacle_distance": float(info.get("sac_min_obstacle_distance", metrics["min_scan_distance"])),
            "fallback_min_obstacle_distance": float(info.get("fallback_min_obstacle_distance", metrics["min_scan_distance"])),
            "mppi_min_obstacle_distance": float(info.get("mppi_min_obstacle_distance", metrics["min_scan_distance"])),
            "base_risk": float(info.get("base_risk", 0.0)),
            "candidate_risk": float(info.get("candidate_risk", 0.0)),
            "fallback_risk": float(info.get("fallback_risk", 0.0)),
            "base_min_distance": float(info.get("base_min_distance", metrics["min_scan_distance"])),
            "candidate_min_distance": float(info.get("candidate_min_distance", metrics["min_scan_distance"])),
            "fallback_min_distance": float(info.get("fallback_min_distance", metrics["min_scan_distance"])),
            "base_ttc_cost": float(info.get("base_ttc_cost", 0.0)),
            "candidate_ttc_cost": float(info.get("candidate_ttc_cost", 0.0)),
            "fallback_ttc_cost": float(info.get("fallback_ttc_cost", 0.0)),
            "base_max_lateral_error": float(info.get("base_max_lateral_error", 0.0)),
            "candidate_max_lateral_error": float(info.get("candidate_max_lateral_error", 0.0)),
            "fallback_max_lateral_error": float(info.get("fallback_max_lateral_error", 0.0)),
            "base_progress": float(info.get("base_progress", 0.0)),
            "candidate_progress": float(info.get("candidate_progress", 0.0)),
            "fallback_progress": float(info.get("fallback_progress", 0.0)),
            "front_obstacle_distance": float(info.get("front_obstacle_distance", metrics["min_scan_distance"])),
            "left_clearance": float(info.get("left_clearance", metrics["min_scan_distance"])),
            "right_clearance": float(info.get("right_clearance", metrics["min_scan_distance"])),
            "sac_intent_target_speed": float(info.get("sac_intent_target_speed", 0.0)),
            "sac_intent_turn_bias": float(info.get("sac_intent_turn_bias", 0.0)),
            "sac_intent_path_weight": float(info.get("sac_intent_path_weight", 0.0)),
            "sac_intent_safety_weight": float(info.get("sac_intent_safety_weight", 0.0)),
            "intent_target_speed": float(info.get("intent_target_speed", 0.0)),
            "intent_turn_bias": float(info.get("intent_turn_bias", 0.0)),
            "intent_path_weight": float(info.get("intent_path_weight", 0.0)),
            "intent_safety_weight": float(info.get("intent_safety_weight", 0.0)),
            "mppi_best_cost": float(info.get("mppi_best_cost", info.get("mppi_cost", 0.0))),
            "terminal_reason": info.get("terminal_reason", "running"),
        })
        step_idx += 1

        if bool(dones[0]):
            final_step_count = int(metrics.get("step_count", getattr(env, "step_count", 0)))
            if "episode" in last_info and isinstance(last_info["episode"], dict):
                final_step_count = int(last_info["episode"].get("l", final_step_count))
            terminal_reason = str(last_info.get("terminal_reason", "unknown"))
            terminated_by_success = terminal_reason == "success" or bool(
                last_info.get("is_success", metrics["distance_to_goal"] < getattr(env, "goal_reach_threshold", 0.4))
            )
            collision = terminal_reason == "collision" or bool(last_info.get("is_collision", min_scan_distance < 0.25))
            out_of_bounds = terminal_reason == "out_of_bounds" or bool(last_info.get("is_out_of_bounds", False))
            timeout = terminal_reason == "timeout" or bool(last_info.get("TimeLimit.truncated", False)) or (
                not terminated_by_success
                and not collision
                and not out_of_bounds
                and bool(last_info.get("is_timeout", final_step_count >= getattr(env, "max_steps", 0)))
            )
            write_csv(os.path.join(trace_dir, f"{mode}_episode_{episode_idx:03d}.csv"), trace_rows)
            return {
                "reward": episode_reward,
                "success": int(terminated_by_success),
                "collision": int(collision),
                "out_of_bounds": int(out_of_bounds),
                "timeout": int(timeout),
                "terminal_reason": terminal_reason,
                "steps": final_step_count,
                "duration": float(final_step_count * metrics["dt"]),
                "path_length": path_length,
                "min_obstacle_distance": min_scan_distance,
                "mean_frenet_abs_d": float(np.mean(frenet_errors)) if frenet_errors else 0.0,
                "mean_action_change": float(np.mean(action_changes)) if action_changes else 0.0,
                "mean_raw_action_change": float(np.mean(raw_action_changes)) if raw_action_changes else 0.0,
                "mean_mppi_active": float(np.mean(mppi_active_flags)) if mppi_active_flags else 0.0,
                "mean_mppi_accept": mean_debug(mppi_debug, "mppi_accept"),
                "mean_mppi_reject": mean_debug(mppi_debug, "mppi_reject"),
                "mean_fallback_active": mean_debug(mppi_debug, "fallback_active"),
                "mean_fallback_accept": mean_debug(mppi_debug, "fallback_accept"),
                "reject_collision_risk": mean_debug(mppi_debug, "reject_collision_risk"),
                "reject_out_of_bounds": mean_debug(mppi_debug, "reject_out_of_bounds"),
                "reject_progress_loss": mean_debug(mppi_debug, "reject_progress_loss"),
                "reject_no_safety_gain": mean_debug(mppi_debug, "reject_no_safety_gain"),
                "mean_teacher_mppi_would_accept": mean_debug(mppi_debug, "teacher_mppi_would_accept"),
                "mean_mppi_warm_start_used": mean_debug(mppi_debug, "mppi_warm_start_used"),
                "mean_sac_pred_collision": mean_debug(mppi_debug, "sac_pred_collision"),
                "mean_fallback_pred_collision": mean_debug(mppi_debug, "fallback_pred_collision"),
                "mean_mppi_pred_collision": mean_debug(mppi_debug, "mppi_pred_collision"),
                "mean_sac_pred_out_of_bounds": mean_debug(mppi_debug, "sac_pred_out_of_bounds"),
                "mean_fallback_pred_out_of_bounds": mean_debug(mppi_debug, "fallback_pred_out_of_bounds"),
                "mean_mppi_pred_out_of_bounds": mean_debug(mppi_debug, "mppi_pred_out_of_bounds"),
                "mean_candidate_sac_score": mean_debug(mppi_debug, "candidate_sac_score"),
                "mean_candidate_fallback_score": mean_debug(mppi_debug, "candidate_fallback_score"),
                "mean_candidate_mppi_score": mean_debug(mppi_debug, "candidate_mppi_score"),
                "mean_predicted_reward_sac": mean_debug(mppi_debug, "predicted_reward_sac"),
                "mean_predicted_reward_mppi": mean_debug(mppi_debug, "predicted_reward_mppi"),
                "mean_predicted_reward_delta": mean_debug(mppi_debug, "predicted_reward_delta"),
                "mean_reward_prediction_error": mean_debug(mppi_debug, "reward_prediction_error"),
                "mean_mppi_selected": mean_debug(mppi_debug, "mppi_selected"),
                "mean_source_sac": mean_debug(mppi_debug, "source_sac"),
                "mean_source_mppi": mean_debug(mppi_debug, "source_mppi"),
                "mean_source_fallback": mean_debug(mppi_debug, "source_fallback"),
                "mean_source_hierarchical_mppi": mean_debug(mppi_debug, "source_hierarchical_mppi"),
                "mean_sac_intent_target_speed": mean_debug(mppi_debug, "sac_intent_target_speed"),
                "mean_sac_intent_turn_bias": mean_debug(mppi_debug, "sac_intent_turn_bias"),
                "mean_sac_intent_path_weight": mean_debug(mppi_debug, "sac_intent_path_weight"),
                "mean_sac_intent_safety_weight": mean_debug(mppi_debug, "sac_intent_safety_weight"),
                "mean_action_delta_norm": mean_debug(mppi_debug, "action_delta_norm"),
                "mean_current_obstacle_distance": mean_debug(mppi_debug, "current_obstacle_distance"),
                "mean_front_obstacle_distance": mean_trace(trace_rows, "front_obstacle_distance"),
                "mean_left_clearance": mean_trace(trace_rows, "left_clearance"),
                "mean_right_clearance": mean_trace(trace_rows, "right_clearance"),
                "terminal_source": terminal_source_from_trace(trace_rows),
                "mean_dbas_cost": mean_debug(mppi_debug, "dbas_cost"),
                "mean_ttc_cost": mean_debug(mppi_debug, "ttc_cost"),
                "mean_out_of_bounds_cost": mean_debug(mppi_debug, "out_of_bounds_cost"),
                "mean_min_predicted_obstacle_distance": mean_debug(mppi_debug, "min_predicted_obstacle_distance"),
                "mean_exploration_noise_scale": mean_debug(mppi_debug, "exploration_noise_scale"),
            }


def mean_debug(rows: List[Dict[str, float]], key: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([row[key] for row in rows]))


def mean_trace(rows: List[Dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row]
    return float(np.mean(values)) if values else 0.0


def terminal_source_from_trace(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "unknown"
    return str(rows[-1].get("action_source", "unknown"))


def run_mode(args, mode: str, model_path: str, obs_mode: str) -> List[Dict[str, Any]]:
    log_dir = os.path.join(args.output_dir, "monitor_logs")
    trace_dir = os.path.join(args.output_dir, "traces")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(trace_dir, exist_ok=True)
    vec_env = build_eval_env(mode, args.seed, log_dir, args.frame_stack, obs_mode)
    model = SAC.load(model_path, env=vec_env, device=args.device)

    rows = []
    for episode_idx in range(args.episodes):
        np.random.seed(args.seed + episode_idx)
        row = run_episode(model, vec_env, deterministic=not args.stochastic, mode=mode, episode_idx=episode_idx, trace_dir=trace_dir)
        row["episode"] = episode_idx
        row["seed"] = args.seed + episode_idx
        row["mode"] = mode
        rows.append(row)
        print(
            f"[{mode}] episode={episode_idx + 1}/{args.episodes} "
            f"success={row['success']} collision={row['collision']} "
            f"oob={row['out_of_bounds']} reward={row['reward']:.2f} "
            f"delta={row['mean_action_delta_norm']:.3f} active={row['mean_mppi_active']:.2f}"
        )

    vec_env.close()
    return rows


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {}
    numeric_keys = [key for key, value in rows[0].items() if isinstance(value, (int, float))]
    summary = {"episodes": len(rows)}
    for key in numeric_keys:
        if key in ("episode", "seed"):
            continue
        summary[f"mean_{key}"] = float(np.mean([float(row[key]) for row in rows]))
    return summary


def paired_summary(all_rows: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    baseline = {row["episode"]: row for row in all_rows.get("baseline", [])}
    result: Dict[str, Any] = {}
    for mode, rows in all_rows.items():
        if mode == "baseline":
            continue
        pairs = []
        for row in rows:
            base = baseline.get(row["episode"])
            if base is None:
                continue
            pairs.append({
                "episode": row["episode"],
                "seed": row["seed"],
                "reward_delta": row["reward"] - base["reward"],
                "success_delta": row["success"] - base["success"],
                "collision_delta": row["collision"] - base["collision"],
                "out_of_bounds_delta": row["out_of_bounds"] - base["out_of_bounds"],
                "min_obstacle_distance_delta": row["min_obstacle_distance"] - base["min_obstacle_distance"],
            })
        result[mode] = {
            "pairs": len(pairs),
            "mean_reward_delta": float(np.mean([p["reward_delta"] for p in pairs])) if pairs else 0.0,
            "mean_success_delta": float(np.mean([p["success_delta"] for p in pairs])) if pairs else 0.0,
            "mean_collision_delta": float(np.mean([p["collision_delta"] for p in pairs])) if pairs else 0.0,
            "mean_out_of_bounds_delta": float(np.mean([p["out_of_bounds_delta"] for p in pairs])) if pairs else 0.0,
            "mean_min_obstacle_distance_delta": float(np.mean([p["min_obstacle_distance_delta"] for p in pairs])) if pairs else 0.0,
        }
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Compare SAC baseline and SAC+MPPI-DBaS for USV navigation.")
    parser.add_argument("--model", default=None, help="Path to a trained SAC model zip.")
    parser.add_argument("--baseline-model", default=None, help="Path to the old pure SAC baseline model.")
    parser.add_argument("--hierarchical-model", default=None, help="Path to the hierarchical SAC-MPPI model.")
    parser.add_argument("--episodes", "--episode", type=int, default=30)
    parser.add_argument(
        "--mode",
        choices=[
            "baseline",
            "shield_first",
            "sac_mppi",
            "sac_mppi_safe",
            "shield_only",
            "shield_mppi_teacher",
            "shield_mppi_execute",
            "hybrid_mppi",
            "mppi_teacher",
            "mppi_dbas",
            "trust_mppi",
            "trust_mppi_dbas",
            "hierarchical_mppi",
            "hierarchical_mppi_shield",
            "hierarchical_ablation",
            "both",
            "ablation",
        ],
        default="both",
    )
    parser.add_argument("--output-dir", default="./comparison_results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-stack", type=int, default=4, help="Use 4 for models trained by beam_map/train.py; use 1 for plain models.")
    parser.add_argument(
        "--obs-mode",
        choices=["auto", "flat", "dict"],
        default="auto",
        help="auto detects the saved model observation space; dict adapts flat lidar obs to radar_image/kinematics.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic SAC actions during evaluation.")
    parser.add_argument("--plot", action="store_true", help="Generate comparison plots after evaluation.")
    parser.add_argument("--plot-dir", default=None, help="Directory for plots. Defaults to <output-dir>/plots.")
    parser.add_argument("--plot-max-steps", type=int, default=180)
    return parser.parse_args()


def model_path_for_mode(args, mode: str) -> str:
    if mode in ("hierarchical_mppi", "hierarchical_mppi_shield"):
        model_path = args.hierarchical_model or args.model
    else:
        model_path = args.baseline_model or args.model
    if not model_path:
        raise ValueError(f"No model path provided for mode '{mode}'. Use --model or the mode-specific model argument.")
    return model_path


def obs_mode_for_model(args, model_path: str) -> str:
    probe_model = SAC.load(model_path, env=None, device=args.device)
    return resolve_obs_mode(args.obs_mode, probe_model.observation_space)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.mode == "both":
        modes = ["baseline", "shield_first"]
    elif args.mode == "ablation":
        modes = ["baseline", "shield_only", "shield_mppi_teacher", "shield_mppi_execute"]
    elif args.mode == "hierarchical_ablation":
        modes = ["baseline", "shield_only", "hierarchical_mppi"]
    else:
        modes = [args.mode]

    all_rows: Dict[str, List[Dict[str, Any]]] = {}
    summary = {}
    for mode in modes:
        model_path = model_path_for_mode(args, mode)
        obs_mode = obs_mode_for_model(args, model_path)
        if obs_mode == "dict" and args.frame_stack != 1:
            print(f"[compare] Dict observation model detected for {mode}; ignoring frame_stack and using radar_image/kinematics observations.")
        rows = run_mode(args, mode, model_path, obs_mode)
        all_rows[mode] = rows
        write_csv(os.path.join(args.output_dir, f"{mode}_metrics.csv"), rows)
        summary[mode] = summarize(rows)

    summary["paired"] = paired_summary(all_rows)
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.plot:
        plot_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_comparison_curves.py")
        plot_dir = args.plot_dir or os.path.join(args.output_dir, "plots")
        subprocess.run(
            [
                sys.executable,
                plot_script,
                "--result-dir",
                args.output_dir,
                "--output-dir",
                plot_dir,
                "--max-steps",
                str(args.plot_max_steps),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
