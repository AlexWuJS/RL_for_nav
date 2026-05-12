import argparse
import os

import gymnasium as gym
import rospy
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from hierarchical_mppi_wrapper import (
    HierarchicalMppiV2Wrapper,
    HierarchicalMppiWrapper,
    hierarchical_mppi_config,
    hierarchical_mppi_v2_config,
)
from mppi_dbas import MPPIDBaSConfig
from mppi_dbas_wrapper import MppiDbaSActionWrapper
from ros_env import MyCarEnv


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def config_for_mode(mode: str, seed: int) -> MPPIDBaSConfig | None:
    if mode in ("baseline", "hierarchical_mppi", "hierarchical_mppi_v2"):
        return None
    if mode == "shield_only":
        return MPPIDBaSConfig(seed=seed, enable_mppi=False, enable_fallback=True)
    if mode == "shield_first":
        return MPPIDBaSConfig(
            seed=seed,
            enable_mppi=True,
            enable_fallback=True,
            always_run_mppi=False,
            execute_mppi=True,
            final_safety_check=True,
            use_reward_aligned_cost=True,
        )
    if mode == "shield_mppi_teacher":
        return MPPIDBaSConfig(
            seed=seed,
            enable_mppi=True,
            enable_fallback=True,
            always_run_mppi=False,
            execute_mppi=False,
            teacher_only=True,
            final_safety_check=True,
            use_reward_aligned_cost=True,
        )
    if mode == "shield_mppi_execute":
        return MPPIDBaSConfig(
            seed=seed,
            enable_mppi=True,
            enable_fallback=True,
            always_run_mppi=False,
            execute_mppi=True,
            final_safety_check=True,
            use_reward_aligned_cost=True,
        )
    raise ValueError(f"Unsupported mode: {mode}")


def make_env(mode: str, seed: int) -> gym.Env:
    env = MyCarEnv()
    if mode == "hierarchical_mppi":
        return HierarchicalMppiWrapper(env, config=hierarchical_mppi_config(seed=seed))
    if mode == "hierarchical_mppi_v2":
        return HierarchicalMppiV2Wrapper(env, config=hierarchical_mppi_v2_config(seed=seed))
    config = config_for_mode(mode, seed)
    if config is None:
        return env
    return MppiDbaSActionWrapper(env, config)


def unwrap_env(env: gym.Env) -> gym.Env:
    return getattr(env, "unwrapped", env)


def choose_model_path(model_arg: str | None) -> str | None:
    candidates = [
        model_arg,
        "./training_hierarchical_mppi_v2_results/best_model.zip",
        "./training_hierarchical_mppi_v2_results/best_model",
        "./training_hierarchical_mppi_v2_results/final_model_hierarchical_v2.zip",
        "./training_hierarchical_mppi_results/best_model.zip",
        "./training_hierarchical_mppi_results/best_model",
        "./training_hierarchical_mppi_results/final_model_hierarchical.zip",
        "./training_usv_v2_results/best_model.zip",
        "./training_usv_v2_results/best_model",
        "./training_usv_v2_results/final_model_stacked.zip",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Run SAC policy in Gazebo, optionally with shield/MPPI-DBaS action filtering.")
    parser.add_argument(
        "--mode",
        choices=["baseline", "shield_only", "shield_first", "shield_mppi_teacher", "shield_mppi_execute", "hierarchical_mppi", "hierarchical_mppi_v2"],
        default="baseline",
        help="baseline keeps raw SAC actions; shield_only enables the low-intervention safety fallback.",
    )
    parser.add_argument("--model", default=None, help="Path to SAC model. Defaults to training_usv_v2_results/best_model.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=10, help="Print shield debug every N steps. Use 0 to disable periodic logs.")
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


def print_shield_debug(step: int, info: dict, last_source: str | None, log_every: int) -> str | None:
    if not info.get("mppi_dbas_enabled"):
        return last_source

    source = str(info.get("action_source", "sac"))
    reason = str(info.get("mppi_decision_reason", "none"))
    should_print = source != last_source or (log_every > 0 and step % log_every == 0)
    if should_print:
        raw_action = info.get("raw_action", info.get("raw_intent"))
        optimized_action = info.get("optimized_action")
        delta = info.get("action_delta_norm", 0.0)
        current_dist = info.get("current_obstacle_distance", 0.0)
        print(
            f"[step {step:04d}] source={source} active={int(bool(info.get('mppi_active', False)))} "
            f"accept={int(bool(info.get('mppi_accept', False)))} reason={reason} "
            f"delta={float(delta):.3f} obs_dist={float(current_dist):.2f} "
            f"raw={raw_action} opt={optimized_action}"
        )
    return source


def main():
    args = parse_args()
    print(f"DEBUG: 启动仿真测试，mode={args.mode}")

    raw_env_holder: dict[str, gym.Env] = {}

    def env_factory():
        env = make_env(args.mode, args.seed)
        raw_env_holder["env"] = env
        return env

    env = DummyVecEnv([env_factory])
    if args.frame_stack > 1:
        env = VecFrameStack(env, n_stack=args.frame_stack)

    raw_env = raw_env_holder["env"]

    model_path = choose_model_path(args.model)
    if model_path is None:
        print("错误：找不到模型文件，请使用 --model 指定 SAC 模型路径。")
        return

    print(f"DEBUG: 加载模型: {model_path}")
    model = SAC.load(model_path, env=env, device=args.device)
    print("DEBUG: 模型加载成功，开始导航测试。按 Ctrl+C 停止。")

    obs = env.reset()
    print_episode_header(raw_env, "第一回合")

    episode_reward = 0.0
    steps = 0
    last_source = None

    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            reward = float(rewards[0])
            done = bool(dones[0])
            info = infos[0] if infos else {}

            episode_reward += reward
            steps += 1
            last_source = print_shield_debug(steps, info, last_source, args.log_every)

            if done:
                if episode_reward > 100:
                    print(f"任务完成：用时 {steps} 步，总奖励 {episode_reward:.1f}")
                else:
                    print(f"任务结束：可能碰撞/越界/超时，用时 {steps} 步，总奖励 {episode_reward:.1f}")

                print("-" * 40)
                print_episode_header(raw_env, "新回合")
                episode_reward = 0.0
                steps = 0
                last_source = None

    except KeyboardInterrupt:
        print("\n测试停止，发送零速度。")
        from geometry_msgs.msg import Twist

        pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        pub.publish(Twist())


if __name__ == "__main__":
    main()
