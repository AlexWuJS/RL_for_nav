import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import numpy as np
import rospy
from geometry_msgs.msg import Twist

from dsac_mppi.algorithms.dsac import DSACPolicy
from dsac_mppi.controllers.rl_driven_mppi import (
    DSACPolicyAdapter,
    RLDrivenMPPIActionWrapper,
    rl_driven_mppi_config,
)
from dsac_mppi.envs.ros_env import MyCarEnv
from scripts.test.observation_stack import PolicyObservationStacker


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


DSAC_RLMPPI_MODES = {
    "dsac_rl_driven_mppi",
    "dsac_rl_driven_mppi_no_hss",
    "dsac_rl_driven_mppi_fixed_sigma",
    "dsac_rl_driven_mppi_no_q",
}
DSAC_MODES = {"dsac"} | DSAC_RLMPPI_MODES
RUN_MODES = ("dsac", "pure_mppi", *sorted(DSAC_RLMPPI_MODES))


def config_for_mode(mode: str, seed: int):
    return rl_driven_mppi_config(mode, seed)


def candidate_paths(path: str) -> list[Path]:
    raw_path = Path(path).expanduser()
    if raw_path.is_absolute():
        return [raw_path]
    return [raw_path, PROJECT_ROOT / raw_path]


def find_existing_model_path(path: str) -> str | None:
    candidates = [path]
    if not path.endswith(".pt"):
        candidates.append(path + ".pt")
    for candidate in candidates:
        for candidate_path in candidate_paths(candidate):
            if candidate_path.exists():
                return str(candidate_path)
    return None


def resolve_model_path(path: str) -> str:
    existing_path = find_existing_model_path(path)
    if existing_path:
        return existing_path
    raise FileNotFoundError(f"DSAC model not found: {path}")


def choose_model_path(model_arg: str | None, mode: str, model_name: str) -> str | None:
    mode_candidates = {
        "dsac": [
            f"./data/{model_name}/models/best_model",
            f"./data/{model_name}/models/final_model_dsac",
            "./data/dsac_usv/models/best_model",
            "./data/dsac_usv/models/final_model_dsac",
        ],
        "dsac_rl_driven_mppi": [
            f"./data/{model_name}/models/best_model",
            f"./data/{model_name}/models/final_model_dsac",
            "./data/dsac_usv/models/best_model",
            "./data/dsac_usv/models/final_model_dsac",
        ],
        "dsac_rl_driven_mppi_no_hss": [
            f"./data/{model_name}/models/best_model",
            f"./data/{model_name}/models/final_model_dsac",
            "./data/dsac_usv/models/best_model",
            "./data/dsac_usv/models/final_model_dsac",
        ],
        "dsac_rl_driven_mppi_fixed_sigma": [
            f"./data/{model_name}/models/best_model",
            f"./data/{model_name}/models/final_model_dsac",
            "./data/dsac_usv/models/best_model",
            "./data/dsac_usv/models/final_model_dsac",
        ],
        "dsac_rl_driven_mppi_no_q": [
            f"./data/{model_name}/models/best_model",
            f"./data/{model_name}/models/final_model_dsac",
            "./data/dsac_usv/models/best_model",
            "./data/dsac_usv/models/final_model_dsac",
        ],
        "pure_mppi": [
            f"./data/{model_name}/models/best_model",
            f"./data/{model_name}/models/final_model_dsac",
            "./data/dsac_usv/models/best_model",
            "./data/dsac_usv/models/final_model_dsac",
        ],
    }
    candidates = [model_arg] + mode_candidates.get(mode, [])
    for path in candidates:
        if not path:
            continue
        existing_path = find_existing_model_path(path)
        if existing_path:
            return existing_path
    return None


def make_env(
    mode: str,
    seed: int,
    model_path: str | None = None,
    device: str = "auto",
    control_mode: str = "low_level_velocity",
    dynamics_model: str = "first_order",
) -> gym.Env:
    env = MyCarEnv(control_mode=control_mode, dynamics_model=dynamics_model)
    if hasattr(env, "seed"):
        env.seed(seed)
    if mode == "pure_mppi":
        return RLDrivenMPPIActionWrapper(env, config=config_for_mode(mode, seed))
    if mode in DSAC_RLMPPI_MODES:
        if not model_path:
            raise ValueError(f"{mode} requires a DSAC model path.")
        return RLDrivenMPPIActionWrapper(
            env,
            config=config_for_mode(mode, seed),
            policy_adapter=DSACPolicyAdapter.load(model_path, device=device),
        )
    if mode == "dsac":
        return env
    raise ValueError(f"Unsupported mode: {mode}")


def unwrap_env(env: gym.Env) -> gym.Env:
    current = env
    while hasattr(current, "env"):
        current = current.env
    return current


def parse_args():
    parser = argparse.ArgumentParser(description="Run DSAC policy in Gazebo, optionally with RL-driven MPPI.")
    parser.add_argument(
        "--mode",
        choices=RUN_MODES,
        default="dsac_rl_driven_mppi",
        help="dsac keeps raw policy actions; dsac_rl_driven_mppi lets MPPI refine unsafe/uncertain actions.",
    )
    parser.add_argument("--model", default=None, help="Path to DSAC model. Defaults to data/<model-name>/models/best_model.")
    parser.add_argument("--model-name", default="dsac_high_level")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--control-mode", choices=("auto", "low_level_velocity", "high_level_frenet"), default="auto")
    parser.add_argument("--dynamics-model", choices=("ideal", "first_order", "inertia"), default="first_order")
    parser.add_argument("--log-every", type=int, default=10, help="Print MPPI debug every N steps. Use 0 to disable periodic logs.")
    parser.add_argument("--max-steps", type=int, default=0, help="Maximum steps per episode. Use 0 to run until env done.")
    parser.add_argument("--deterministic", action="store_true", default=True)
    return parser.parse_args()


def print_episode_header(raw_env: gym.Env, label: str) -> None:
    base_env = unwrap_env(raw_env)
    current_pos = getattr(base_env, "current_pos", None)
    target_pos = getattr(base_env, "target_pos", None)
    if current_pos is None or target_pos is None:
        print(f"\n[{label}] 新回合开始")
        return
    print(
        f"\n[{label}] 起点: ({current_pos[0]:.2f}, {current_pos[1]:.2f}) "
        f"-> 终点: ({target_pos[0]:.2f}, {target_pos[1]:.2f})"
    )


def print_mppi_debug(step: int, info: dict, last_source: str | None, log_every: int) -> str | None:
    source = str(info.get("action_source", "dsac"))
    reason = str(info.get("mppi_decision_reason", info.get("terminal_reason", "none")))
    should_print = source != last_source or (log_every > 0 and step % log_every == 0)
    if should_print:
        raw_action = info.get("raw_action", info.get("raw_intent"))
        optimized_action = info.get("optimized_action")
        delta = info.get("action_delta_norm", 0.0)
        current_dist = info.get("current_obstacle_distance", info.get("min_laser_dist", 0.0))
        distance_to_goal = info.get("distance_to_goal", info.get("distance_remaining", 0.0))
        print(
            f"[step {step:04d}] source={source} active={int(bool(info.get('mppi_active', False)))} "
            f"accept={int(bool(info.get('mppi_accept', False)))} reason={reason} "
            f"delta={float(delta):.3f} obs_dist={float(current_dist):.2f} "
            f"goal_dist={float(distance_to_goal):.2f} raw={raw_action} opt={optimized_action}"
        )
    return source


def stop_robot() -> None:
    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    rospy.sleep(0.1)
    pub.publish(Twist())


def main():
    args = parse_args()
    print(f"DEBUG: starting Gazebo evaluation, mode={args.mode}")

    model_path_arg = choose_model_path(args.model, args.mode, args.model_name)
    if model_path_arg is None:
        print("ERROR: model file not found. Please pass --model with a DSAC model path.")
        return
    model_path = resolve_model_path(model_path_arg)

    print(f"DEBUG: 加载模型: {Path(model_path)}")
    model = DSACPolicy.load(model_path, device=args.device)
    control_mode = args.control_mode
    if control_mode == "auto":
        control_mode = "high_level_frenet" if model.config.action_dim == 3 else "low_level_velocity"

    rospy.init_node("dsac_mppi_test01", anonymous=True, disable_signals=True)
    raw_env = make_env(
        args.mode,
        args.seed,
        model_path=model_path,
        device=args.device,
        control_mode=control_mode,
        dynamics_model=args.dynamics_model,
    )
    print(f"DEBUG: policy_obs_dim={model.config.observation_dim}")
    print(f"DEBUG: action_dim={model.config.action_dim}, control_mode={control_mode}, dynamics_model={args.dynamics_model}")
    print("DEBUG: model loaded; navigation test started. Press Ctrl+C to stop.")

    obs, _ = raw_env.reset(seed=args.seed)
    obs_stacker = PolicyObservationStacker(model.config.observation_dim)
    policy_obs = obs_stacker.reset(obs)
    print_episode_header(raw_env, "第一回合")

    episode_reward = 0.0
    steps = 0
    episode_idx = 1
    last_source = None

    try:
        while not rospy.is_shutdown():
            action, _states = model.predict(policy_obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = raw_env.step(np.asarray(action).reshape(-1)[: model.config.action_dim])
            policy_obs = obs_stacker.update(obs)

            episode_reward += float(reward)
            steps += 1
            last_source = print_mppi_debug(steps, info, last_source, args.log_every)

            max_steps_reached = args.max_steps > 0 and steps >= args.max_steps
            if terminated or truncated or max_steps_reached:
                if episode_reward > 100:
                    print(f"任务完成：用时 {steps} 步，总奖励 {episode_reward:.1f}")
                else:
                    reason = info.get("terminal_reason", "max_steps" if max_steps_reached else "done")
                    print(f"任务结束：{reason}，用时 {steps} 步，总奖励 {episode_reward:.1f}")

                print("-" * 40)
                episode_idx += 1
                obs, _ = raw_env.reset(seed=args.seed + episode_idx - 1)
                obs_stacker = PolicyObservationStacker(model.config.observation_dim)
                policy_obs = obs_stacker.reset(obs)
                print_episode_header(raw_env, "新回合")
                episode_reward = 0.0
                steps = 0
                last_source = None

    except KeyboardInterrupt:
        print("\n测试停止，发送零速度。")
        stop_robot()
    finally:
        raw_env.close()


if __name__ == "__main__":
    main()
