import argparse
import csv
import json
import os
from typing import Any, Dict, List

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from mppi_dbas import MPPIDBaSConfig
from mppi_dbas_wrapper import MppiDbaSActionWrapper
from ros_env import MyCarEnv


def unwrap_env(env):
    current = env
    while hasattr(current, "env"):
        current = current.env
    return current


def config_for_mode(mode: str, seed: int) -> MPPIDBaSConfig:
    if mode == "trust_mppi":
        return MPPIDBaSConfig(seed=seed, dbas_weight=0.0, risk_activation_distance=10.0)
    if mode == "trust_mppi_dbas":
        return MPPIDBaSConfig(seed=seed, risk_activation_distance=10.0)
    return MPPIDBaSConfig(seed=seed)


def make_env(mode: str, seed: int, log_dir: str):
    def _init():
        env = MyCarEnv()
        if mode != "baseline":
            env = MppiDbaSActionWrapper(env, config_for_mode(mode, seed))
        env = Monitor(env, filename=os.path.join(log_dir, mode))
        return env

    return _init


def build_eval_env(mode: str, seed: int, log_dir: str, frame_stack: int):
    env = DummyVecEnv([make_env(mode, seed, log_dir)])
    if frame_stack > 1:
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
                "reject_collision_risk": float(reason == "reject_collision_risk"),
                "reject_out_of_bounds": float(reason == "reject_out_of_bounds"),
                "reject_progress_loss": float(reason == "reject_progress_loss"),
                "reject_no_safety_gain": float(reason == "reject_no_safety_gain"),
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
            "base_risk": float(info.get("base_risk", 0.0)),
            "candidate_risk": float(info.get("candidate_risk", 0.0)),
            "base_min_distance": float(info.get("base_min_distance", metrics["min_scan_distance"])),
            "candidate_min_distance": float(info.get("candidate_min_distance", metrics["min_scan_distance"])),
            "base_ttc_cost": float(info.get("base_ttc_cost", 0.0)),
            "candidate_ttc_cost": float(info.get("candidate_ttc_cost", 0.0)),
            "base_max_lateral_error": float(info.get("base_max_lateral_error", 0.0)),
            "candidate_max_lateral_error": float(info.get("candidate_max_lateral_error", 0.0)),
            "base_progress": float(info.get("base_progress", 0.0)),
            "candidate_progress": float(info.get("candidate_progress", 0.0)),
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
                "reject_collision_risk": mean_debug(mppi_debug, "reject_collision_risk"),
                "reject_out_of_bounds": mean_debug(mppi_debug, "reject_out_of_bounds"),
                "reject_progress_loss": mean_debug(mppi_debug, "reject_progress_loss"),
                "reject_no_safety_gain": mean_debug(mppi_debug, "reject_no_safety_gain"),
                "mean_action_delta_norm": mean_debug(mppi_debug, "action_delta_norm"),
                "mean_current_obstacle_distance": mean_debug(mppi_debug, "current_obstacle_distance"),
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


def run_mode(args, mode: str) -> List[Dict[str, Any]]:
    log_dir = os.path.join(args.output_dir, "monitor_logs")
    trace_dir = os.path.join(args.output_dir, "traces")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(trace_dir, exist_ok=True)
    vec_env = build_eval_env(mode, args.seed, log_dir, args.frame_stack)
    model = SAC.load(args.model, env=vec_env, device=args.device)

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
    parser.add_argument("--model", required=True, help="Path to a trained SAC model zip.")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument(
        "--mode",
        choices=["baseline", "mppi_dbas", "trust_mppi", "trust_mppi_dbas", "both", "ablation"],
        default="both",
    )
    parser.add_argument("--output-dir", default="./comparison_results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-stack", type=int, default=4, help="Use 4 for models trained by beam_map/train.py; use 1 for plain models.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic SAC actions during evaluation.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.mode == "both":
        modes = ["baseline", "mppi_dbas"]
    elif args.mode == "ablation":
        modes = ["baseline", "trust_mppi", "trust_mppi_dbas", "mppi_dbas"]
    else:
        modes = [args.mode]

    all_rows: Dict[str, List[Dict[str, Any]]] = {}
    summary = {}
    for mode in modes:
        rows = run_mode(args, mode)
        all_rows[mode] = rows
        write_csv(os.path.join(args.output_dir, f"{mode}_metrics.csv"), rows)
        summary[mode] = summarize(rows)

    summary["paired"] = paired_summary(all_rows)
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
