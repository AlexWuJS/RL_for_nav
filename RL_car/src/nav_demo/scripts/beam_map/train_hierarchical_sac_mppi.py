import argparse
import os

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from hierarchical_mppi_wrapper import HierarchicalMppiWrapper, hierarchical_mppi_config
from lidar_compress_net import LidarProcessor
from ros_env import MyCarEnv


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def make_env(rank: int, log_dir: str, seed: int):
    def _init():
        env = MyCarEnv()
        env = HierarchicalMppiWrapper(env, config=hierarchical_mppi_config(seed=seed + rank))
        env = Monitor(env, filename=os.path.join(log_dir, f"hierarchical_monitor_{rank}"))
        return env

    return _init


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC high-level intent policy with MPPI low-level control.")
    parser.add_argument("--total-timesteps", type=int, default=300000)
    parser.add_argument("--save-dir", default="./training_hierarchical_mppi_results/")
    parser.add_argument("--log-dir", default="./logs_hierarchical_mppi/")
    parser.add_argument("--tensorboard-log", default="./sac_hierarchical_mppi_log/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-freq", type=int, default=50000)
    parser.add_argument("--learning-starts", type=int, default=10000)
    parser.add_argument("--buffer-size", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    train_env = DummyVecEnv([make_env(0, args.log_dir, args.seed)])
    eval_env = DummyVecEnv([make_env(1, args.log_dir, args.seed + 10000)])
    if args.frame_stack > 1:
        train_env = VecFrameStack(train_env, n_stack=args.frame_stack)
        eval_env = VecFrameStack(eval_env, n_stack=args.frame_stack)

    policy_kwargs = dict(
        features_extractor_class=LidarProcessor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256], qf=[256, 256]),
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.save_dir,
        log_path=args.log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=args.save_dir,
        name_prefix="hierarchical_sac_checkpoint",
    )

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=args.tensorboard_log,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        ent_coef="auto",
        gamma=0.99,
        tau=0.005,
        learning_starts=args.learning_starts,
        train_freq=(10, "step"),
        gradient_steps=10,
        seed=args.seed,
        device=device,
    )

    print(f"Training hierarchical SAC-MPPI with action space: {train_env.action_space}")
    print(f"Saving results to: {args.save_dir}")
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[eval_callback, checkpoint_callback],
        )
    except KeyboardInterrupt:
        print("Training interrupted; saving current model.")

    model.save(os.path.join(args.save_dir, "final_model_hierarchical"))
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
