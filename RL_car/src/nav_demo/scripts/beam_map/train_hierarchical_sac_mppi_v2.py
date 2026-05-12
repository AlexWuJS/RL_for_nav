import argparse
import os

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import constant_fn
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from hierarchical_mppi_wrapper import HierarchicalMppiV2Wrapper, hierarchical_mppi_v2_config
from lidar_compress_net import LidarProcessor
from ros_env import MyCarEnv


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class EntropyControlCallback(BaseCallback):
    def __init__(self, switch_step: int = 120000, target_ent_coef: float = 0.02, target_lr: float = 3e-5, verbose: int = 0):
        super().__init__(verbose)
        self.switch_step = int(switch_step)
        self.target_ent_coef = float(target_ent_coef)
        self.target_lr = float(target_lr)
        self.switched = False

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.switch_step and not self.switched:
            print(f"[hierarchical-v2] switching to fine-tune phase at {self.num_timesteps} steps")
            self.model.ent_coef_optimizer = None
            self.model.ent_coef_tensor = torch.tensor(self.target_ent_coef, device=self.model.device)
            self.model.ent_coef = self.target_ent_coef
            self.model.lr_schedule = constant_fn(self.target_lr)
            for optimizer in [self.model.actor.optimizer, self.model.critic.optimizer]:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = self.target_lr
            self.switched = True
        return True


def make_env(rank: int, log_dir: str, seed: int, intent_ema_alpha: float, intent_hold_steps: int):
    def _init():
        env = MyCarEnv()
        env = HierarchicalMppiV2Wrapper(
            env,
            config=hierarchical_mppi_v2_config(seed=seed + rank),
            intent_ema_alpha=intent_ema_alpha,
            intent_hold_steps=intent_hold_steps,
        )
        env = Monitor(env, filename=os.path.join(log_dir, f"hierarchical_v2_monitor_{rank}"))
        return env

    return _init


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC high-level intent policy with trigger-based MPPI control.")
    parser.add_argument("--total-timesteps", type=int, default=300000)
    parser.add_argument("--save-dir", default="./training_hierarchical_mppi_v2_results/")
    parser.add_argument("--log-dir", default="./logs_hierarchical_mppi_v2/")
    parser.add_argument("--tensorboard-log", default="./sac_hierarchical_mppi_v2_log/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-freq", type=int, default=50000)
    parser.add_argument("--learning-starts", type=int, default=10000)
    parser.add_argument("--buffer-size", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--intent-ema-alpha", type=float, default=0.6)
    parser.add_argument("--intent-hold-steps", type=int, default=2)
    parser.add_argument("--entropy-switch-step", type=int, default=120000)
    parser.add_argument("--target-ent-coef", type=float, default=0.02)
    parser.add_argument("--target-lr", type=float, default=3e-5)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    train_env = DummyVecEnv([make_env(0, args.log_dir, args.seed, args.intent_ema_alpha, args.intent_hold_steps)])
    eval_env = DummyVecEnv([make_env(1, args.log_dir, args.seed + 10000, args.intent_ema_alpha, args.intent_hold_steps)])
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
        name_prefix="hierarchical_v2_sac_checkpoint",
    )
    entropy_callback = EntropyControlCallback(
        switch_step=args.entropy_switch_step,
        target_ent_coef=args.target_ent_coef,
        target_lr=args.target_lr,
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

    print(f"Training hierarchical SAC-MPPI v2 with action space: {train_env.action_space}")
    print(f"Saving results to: {args.save_dir}")
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[eval_callback, checkpoint_callback, entropy_callback],
        )
    except KeyboardInterrupt:
        print("Training interrupted; saving current model.")

    model.save(os.path.join(args.save_dir, "final_model_hierarchical_v2"))
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
