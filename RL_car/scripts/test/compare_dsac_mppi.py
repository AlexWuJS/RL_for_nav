import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import gymnasium as gym
import numpy as np

from dsac_mppi.algorithms.dsac import DSACPolicy
from dsac_mppi.controllers.rl_driven_mppi import (
    DSACPolicyAdapter,
    RLDrivenMPPIActionWrapper,
    rl_driven_mppi_config,
)
from dsac_mppi.envs.ros_env import MyCarEnv
from scripts.test.observation_stack import PolicyObservationStacker


DSAC_RLMPPI_MODES = {
    "dsac_rl_driven_mppi",
    "dsac_rl_driven_mppi_no_hss",
    "dsac_rl_driven_mppi_fixed_sigma",
    "dsac_rl_driven_mppi_no_q",
}
SINGLE_MODES = {"dsac", "pure_mppi"} | DSAC_RLMPPI_MODES
RUN_MODES = tuple(sorted(SINGLE_MODES | {"ablation_dsac_rlmppi"}))


def default_model_path(model_name: str) -> str:
    return os.path.join("data", model_name, "models", "best_model")


def default_output_dir(model_name: str, run_name: str) -> str:
    return os.path.join("data", model_name, "eval", run_name)


def resolve_model_path(path: str) -> str:
    candidates = [path]
    if not path.endswith(".pt"):
        candidates.append(path + ".pt")
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"DSAC model not found: {path}")


def config_for_mode(mode: str, seed: int):
    return rl_driven_mppi_config(mode, seed)


def modes_for_request(mode: str) -> List[str]:
    if mode == "ablation_dsac_rlmppi":
        return [
            "pure_mppi",
            "dsac",
            "dsac_rl_driven_mppi",
            "dsac_rl_driven_mppi_no_hss",
            "dsac_rl_driven_mppi_fixed_sigma",
            "dsac_rl_driven_mppi_no_q",
        ]
    return [mode]


def make_env(mode: str, model_path: str, seed: int, device: str, control_mode: str, dynamics_model: str) -> gym.Env:
    env = MyCarEnv(control_mode=control_mode, dynamics_model=dynamics_model)
    if hasattr(env, "seed"):
        env.seed(seed)
    if mode == "dsac":
        return env
    if mode == "pure_mppi":
        return RLDrivenMPPIActionWrapper(env, config_for_mode(mode, seed))
    if mode in DSAC_RLMPPI_MODES:
        return RLDrivenMPPIActionWrapper(
            env,
            config_for_mode(mode, seed),
            policy_adapter=DSACPolicyAdapter.load(model_path, device=device),
        )
    raise ValueError(f"Unsupported mode: {mode}")


def jsonable(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return value


def numeric(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(jsonable(row.get(key)), ensure_ascii=False) if isinstance(row.get(key), (dict, list, tuple, np.ndarray)) else row.get(key) for key in fieldnames})


def summarize(rows: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    summary = {
        "episodes": float(len(rows)),
        "mean_reward": float(np.mean([numeric(row.get("reward", row.get("episode_reward"))) for row in rows])),
        "mean_steps": float(np.mean([numeric(row.get("steps")) for row in rows])),
        "mean_success": float(np.mean([numeric(row.get("success")) for row in rows])),
        "mean_collision": float(np.mean([numeric(row.get("collision")) for row in rows])),
        "mean_out_of_bounds": float(np.mean([numeric(row.get("out_of_bounds")) for row in rows])),
        "mean_timeout": float(np.mean([numeric(row.get("timeout")) for row in rows])),
        "mean_min_obstacle_distance": float(np.mean([numeric(row.get("min_obstacle_distance"), np.nan) for row in rows])),
        "mean_mean_frenet_abs_d": float(np.mean([numeric(row.get("mean_frenet_abs_d"), np.nan) for row in rows])),
        "mean_mean_action_delta_norm": float(np.mean([numeric(row.get("mean_action_delta_norm")) for row in rows])),
        "mean_mean_mppi_active": float(np.mean([numeric(row.get("mean_mppi_active")) for row in rows])),
        "mean_mean_mppi_accept": float(np.mean([numeric(row.get("mean_mppi_accept")) for row in rows])),
        "mean_mean_fallback_active": float(np.mean([numeric(row.get("mean_fallback_active")) for row in rows])),
        "mean_rlmppi_online_time_ms": float(np.mean([numeric(row.get("rlmppi_online_time_ms")) for row in rows])),
        "mean_rlmppi_terminal_q_used": float(np.mean([numeric(row.get("rlmppi_terminal_q_used")) for row in rows])),
    }
    summary["mean_min_distance"] = summary["mean_min_obstacle_distance"]
    return summary


def run_episode(policy: DSACPolicy, env: gym.Env, mode: str, seed: int, episode_idx: int, max_steps: int, trace_dir: str) -> Dict[str, Any]:
    obs, _ = env.reset(seed=seed + episode_idx)
    obs_stacker = PolicyObservationStacker(policy.config.observation_dim)
    policy_obs = obs_stacker.reset(obs)
    episode_reward = 0.0
    trace_rows: List[Dict[str, Any]] = []
    last_info: Dict[str, Any] = {}

    for step_idx in range(max_steps):
        action, _ = policy.predict(policy_obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(np.asarray(action).reshape(-1)[: policy.config.action_dim])
        policy_obs = obs_stacker.update(obs)
        last_info = dict(info)
        episode_reward += float(reward)
        current_position = info.get("current_position", [np.nan, np.nan])
        raw_action = np.asarray(info.get("raw_action", info.get("executed_low_level_action", [np.nan, np.nan])), dtype=float).reshape(-1)
        optimized_action = np.asarray(info.get("optimized_action", info.get("executed_low_level_action", raw_action)), dtype=float).reshape(-1)
        action_delta = optimized_action[:2] - raw_action[:2] if raw_action.size >= 2 and optimized_action.size >= 2 else np.array([np.nan, np.nan])
        min_scan_distance = numeric(
            info.get(
                "min_laser_dist",
                info.get("current_obstacle_distance", info.get("mppi_min_obstacle_distance", np.nan)),
            ),
            np.nan,
        )
        action_source = str(info.get("action_source", "dsac" if mode == "dsac" else mode))
        trace_rows.append(
            {
                "episode": episode_idx,
                "step": step_idx,
                "reward": float(reward),
                "x": numeric(current_position[0], np.nan) if len(current_position) >= 1 else np.nan,
                "y": numeric(current_position[1], np.nan) if len(current_position) >= 2 else np.nan,
                "distance_to_goal": numeric(info.get("distance_to_goal"), np.nan),
                "frenet_abs_d": abs(numeric(info.get("frenet_d"), np.nan)),
                "min_scan_distance": min_scan_distance,
                "raw_surge": numeric(raw_action[0], np.nan) if raw_action.size >= 1 else np.nan,
                "raw_yaw": numeric(raw_action[1], np.nan) if raw_action.size >= 2 else np.nan,
                "optimized_surge": numeric(optimized_action[0], np.nan) if optimized_action.size >= 1 else np.nan,
                "optimized_yaw": numeric(optimized_action[1], np.nan) if optimized_action.size >= 2 else np.nan,
                "action_delta_norm": float(np.linalg.norm(action_delta)) if np.all(np.isfinite(action_delta)) else np.nan,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "action_source": action_source,
                "mppi_active": info.get("mppi_active", False),
                "mppi_accept": info.get("mppi_accept", False),
                "fallback_active": info.get("fallback_active", False),
                "fallback_accept": info.get("fallback_accept", False),
                "mppi_selected": info.get("mppi_selected", info.get("mppi_accept", False)),
                "mppi_decision_reason": info.get("mppi_decision_reason", info.get("reject_reason", "none")),
                "current_obstacle_distance": info.get("current_obstacle_distance", min_scan_distance),
                "mppi_min_obstacle_distance": info.get("mppi_min_obstacle_distance", info.get("min_predicted_obstacle_distance", np.nan)),
                "rlmppi_online_time_ms": info.get("rlmppi_online_time_ms", info.get("mppi_elapsed_ms", 0.0)),
                "rlmppi_terminal_q_used": info.get("terminal_q_used", False),
                "terminal_reason": info.get("terminal_reason", "running"),
            }
        )
        if terminated or truncated:
            break

    write_csv(os.path.join(trace_dir, f"{mode}_episode_{episode_idx:03d}.csv"), trace_rows)
    min_distances = [numeric(row.get("min_scan_distance"), np.nan) for row in trace_rows]
    frenet_abs = [numeric(row.get("frenet_abs_d"), np.nan) for row in trace_rows]
    action_deltas = [numeric(row.get("action_delta_norm")) for row in trace_rows]
    min_obstacle_distance = float(np.nanmin(min_distances)) if min_distances and np.any(np.isfinite(min_distances)) else np.nan
    mean_frenet_abs_d = float(np.nanmean(frenet_abs)) if frenet_abs and np.any(np.isfinite(frenet_abs)) else np.nan
    terminal_reason = str(last_info.get("terminal_reason", "timeout" if len(trace_rows) >= max_steps else "done"))
    return {
        "mode": mode,
        "episode": episode_idx,
        "reward": episode_reward,
        "episode_reward": episode_reward,
        "steps": len(trace_rows),
        "success": bool(last_info.get("is_success", last_info.get("success", False))),
        "collision": bool(last_info.get("is_collision", last_info.get("collision", False))),
        "out_of_bounds": bool(last_info.get("is_out_of_bounds", last_info.get("out_of_bounds", False))),
        "timeout": bool(last_info.get("is_timeout", len(trace_rows) >= max_steps)),
        "terminal_reason": terminal_reason,
        "min_obstacle_distance": min_obstacle_distance,
        "min_distance": min_obstacle_distance,
        "mean_frenet_abs_d": mean_frenet_abs_d,
        "mean_action_delta_norm": float(np.mean(action_deltas)) if action_deltas else 0.0,
        "mean_mppi_active": float(np.mean([numeric(row.get("mppi_active")) for row in trace_rows])) if trace_rows else 0.0,
        "mean_mppi_accept": float(np.mean([numeric(row.get("mppi_accept")) for row in trace_rows])) if trace_rows else 0.0,
        "mean_fallback_active": float(np.mean([numeric(row.get("fallback_active")) for row in trace_rows])) if trace_rows else 0.0,
        "rlmppi_online_time_ms": last_info.get("rlmppi_online_time_ms", last_info.get("mppi_elapsed_ms", 0.0)),
        "rlmppi_terminal_q_used": bool(last_info.get("terminal_q_used", False)),
    }


def run_mode(args, mode: str, model_path: str) -> List[Dict[str, Any]]:
    policy = DSACPolicy.load(model_path, device=args.device)
    control_mode = args.control_mode
    if control_mode == "auto":
        control_mode = "high_level_frenet" if policy.config.action_dim == 3 else "low_level_velocity"
    env = make_env(mode, model_path, args.seed, args.device, control_mode, args.dynamics_model)
    trace_dir = os.path.join(args.output_dir, "traces")
    rows: List[Dict[str, Any]] = []
    try:
        for episode_idx in range(args.episodes):
            rows.append(run_episode(policy, env, mode, args.seed, episode_idx, args.max_steps, trace_dir))
    finally:
        env.close()
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DSAC, pure MPPI, and DSAC+RL-driven MPPI.")
    parser.add_argument("--mode", choices=RUN_MODES, default="dsac_rl_driven_mppi")
    parser.add_argument("--model", "--dsac-model", dest="model", default=None)
    parser.add_argument("--model-name", default="dsac_high_level")
    parser.add_argument("--control-mode", choices=("auto", "low_level_velocity", "high_level_frenet"), default="auto")
    parser.add_argument("--dynamics-model", choices=("ideal", "first_order", "inertia"), default="first_order")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--episodes", "--episode", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-dir", default=None)
    parser.add_argument("--plot-max-steps", type=int, default=180)
    return parser.parse_args()


def main():
    args = parse_args()
    args.run_name = args.run_name or args.mode
    args.output_dir = args.output_dir or default_output_dir(args.model_name, args.run_name)
    model_path = resolve_model_path(args.model or default_model_path(args.model_name))
    os.makedirs(args.output_dir, exist_ok=True)

    all_rows: Dict[str, List[Dict[str, Any]]] = {}
    summary: Dict[str, Any] = {}
    for mode in modes_for_request(args.mode):
        rows = run_mode(args, mode, model_path)
        all_rows[mode] = rows
        write_csv(os.path.join(args.output_dir, f"{mode}_metrics.csv"), rows)
        summary[mode] = summarize(rows)

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(jsonable(summary), f, indent=2, ensure_ascii=False)
    print(json.dumps(jsonable(summary), indent=2, ensure_ascii=False))

    if args.plot:
        plot_dir = args.plot_dir or os.path.join(args.output_dir, "plots")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.analysis.plot_comparison_curves",
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
