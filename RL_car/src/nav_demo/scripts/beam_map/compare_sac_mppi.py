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


def make_env(use_mppi: bool, seed: int, log_dir: str):
    def _init():
        env = MyCarEnv()
        if use_mppi:
            env = MppiDbaSActionWrapper(env, MPPIDBaSConfig(seed=seed))
        env = Monitor(env, filename=os.path.join(log_dir, "mppi_dbas" if use_mppi else "baseline"))
        return env

    return _init


def build_eval_env(use_mppi: bool, seed: int, log_dir: str, frame_stack: int):
    env = DummyVecEnv([make_env(use_mppi, seed, log_dir)])
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


def run_episode(model: SAC, vec_env, deterministic: bool) -> Dict[str, Any]:
    obs = vec_env.reset()
    episode_reward = 0.0
    path_length = 0.0
    min_scan_distance = float("inf")
    frenet_errors: List[float] = []
    action_changes: List[float] = []
    raw_action_changes: List[float] = []
    mppi_debug: List[Dict[str, float]] = []
    prev_pos = None
    prev_action = None
    prev_raw_action = None
    last_info: Dict[str, Any] = {}

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

        if info.get("mppi_dbas_enabled"):
            mppi_debug.append({
                "dbas_cost": float(info.get("dbas_cost", 0.0)),
                "min_predicted_obstacle_distance": float(info.get("min_predicted_obstacle_distance", 0.0)),
                "exploration_noise_scale": float(info.get("exploration_noise_scale", 0.0)),
            })

        if bool(dones[0]):
            final_step_count = int(metrics.get("step_count", getattr(env, "step_count", 0)))
            if "episode" in last_info and isinstance(last_info["episode"], dict):
                final_step_count = int(last_info["episode"].get("l", final_step_count))
            terminated_by_success = bool(last_info.get("is_success", metrics["distance_to_goal"] < getattr(env, "goal_reach_threshold", 0.4)))
            collision = bool(last_info.get("is_collision", min_scan_distance < 0.25))
            timeout = bool(last_info.get("TimeLimit.truncated", False)) or (
                not terminated_by_success and not collision and bool(last_info.get("is_timeout", final_step_count >= getattr(env, "max_steps", 0)))
            )
            return {
                "reward": episode_reward,
                "success": int(terminated_by_success),
                "collision": int(collision),
                "timeout": int(timeout),
                "steps": final_step_count,
                "duration": float(final_step_count * metrics["dt"]),
                "path_length": path_length,
                "min_obstacle_distance": min_scan_distance,
                "mean_frenet_abs_d": float(np.mean(frenet_errors)) if frenet_errors else 0.0,
                "mean_action_change": float(np.mean(action_changes)) if action_changes else 0.0,
                "mean_raw_action_change": float(np.mean(raw_action_changes)) if raw_action_changes else 0.0,
                "mean_dbas_cost": mean_debug(mppi_debug, "dbas_cost"),
                "mean_min_predicted_obstacle_distance": mean_debug(mppi_debug, "min_predicted_obstacle_distance"),
                "mean_exploration_noise_scale": mean_debug(mppi_debug, "exploration_noise_scale"),
            }


def mean_debug(rows: List[Dict[str, float]], key: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([row[key] for row in rows]))


def run_mode(args, mode: str) -> List[Dict[str, Any]]:
    use_mppi = mode == "mppi_dbas"
    log_dir = os.path.join(args.output_dir, "monitor_logs")
    os.makedirs(log_dir, exist_ok=True)
    vec_env = build_eval_env(use_mppi, args.seed, log_dir, args.frame_stack)
    model = SAC.load(args.model, env=vec_env, device=args.device)

    rows = []
    for episode_idx in range(args.episodes):
        np.random.seed(args.seed + episode_idx)
        row = run_episode(model, vec_env, deterministic=not args.stochastic)
        row["episode"] = episode_idx
        row["mode"] = mode
        rows.append(row)
        print(
            f"[{mode}] episode={episode_idx + 1}/{args.episodes} "
            f"success={row['success']} collision={row['collision']} "
            f"reward={row['reward']:.2f} min_obs={row['min_obstacle_distance']:.2f}"
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
        if key == "episode":
            continue
        summary[f"mean_{key}"] = float(np.mean([float(row[key]) for row in rows]))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Compare SAC baseline and SAC+MPPI-DBaS for USV navigation.")
    parser.add_argument("--model", required=True, help="Path to a trained SAC model zip.")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--mode", choices=["baseline", "mppi_dbas", "both"], default="both")
    parser.add_argument("--output-dir", default="./comparison_results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-stack", type=int, default=4, help="Use 4 for models trained by beam_map/train.py; use 1 for plain models.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic SAC actions during evaluation.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    modes = ["baseline", "mppi_dbas"] if args.mode == "both" else [args.mode]

    summary = {}
    for mode in modes:
        rows = run_mode(args, mode)
        write_csv(os.path.join(args.output_dir, f"{mode}_metrics.csv"), rows)
        summary[mode] = summarize(rows)

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
