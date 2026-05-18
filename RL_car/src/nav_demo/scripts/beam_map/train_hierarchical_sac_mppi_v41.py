import argparse
import os

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import constant_fn
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from hierarchical_mppi_wrapper import HierarchicalMppiV41Wrapper, hierarchical_mppi_v41_config
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
            print(f"[hierarchical-v41] switching to fine-tune phase at {self.num_timesteps} steps")
            self.model.ent_coef_optimizer = None
            self.model.ent_coef_tensor = torch.tensor(self.target_ent_coef, device=self.model.device)
            self.model.ent_coef = self.target_ent_coef
            self.model.lr_schedule = constant_fn(self.target_lr)
            for optimizer in [self.model.actor.optimizer, self.model.critic.optimizer]:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = self.target_lr
            self.switched = True
        return True


def make_env(
    rank: int,
    log_dir: str,
    seed: int,
    reward_profile: str,
):
    def _init():
        env = MyCarEnv()
        env = HierarchicalMppiV41Wrapper(
            env,
            config=hierarchical_mppi_v41_config(seed=seed + rank, reward_profile=reward_profile),
            reward_profile=reward_profile,
        )
        env = Monitor(env, filename=os.path.join(log_dir, f"hierarchical_v41_{reward_profile}_monitor_{rank}"))
        return env

    return _init


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC structured-intent v4.1 policy with trigger-based MPPI control.")
    parser.add_argument("--total-timesteps", type=int, default=300000)
    parser.add_argument("--load-model", default=None, help="Resume training from an existing SAC model zip/path.")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--tensorboard-log", default=None)
    parser.add_argument("--reward-profile", choices=["compat", "guided"], default="guided")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-freq", type=int, default=50000)
    parser.add_argument("--learning-starts", type=int, default=10000)
    parser.add_argument("--buffer-size", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--entropy-switch-step", type=int, default=120000)
    parser.add_argument("--target-ent-coef", type=float, default=0.02)
    parser.add_argument("--target-lr", type=float, default=3e-5)
    return parser.parse_args()


def resolve_model_path(path: str) -> str:
    if os.path.exists(path):
        return path
    if not path.endswith(".zip") and os.path.exists(f"{path}.zip"):
        return f"{path}.zip"
    raise FileNotFoundError(f"Could not find model file: {path}")


def main():
    args = parse_args()
    profile = args.reward_profile
    save_dir = args.save_dir or f"./training_hierarchical_mppi_v41_{profile}_results/"
    log_dir = args.log_dir or f"./logs_hierarchical_mppi_v41_{profile}/"
    tensorboard_log = args.tensorboard_log or f"./sac_hierarchical_mppi_v41_{profile}_log/"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_env = DummyVecEnv([make_env(0, log_dir, args.seed, profile)])
    eval_env = DummyVecEnv([make_env(1, log_dir, args.seed + 10000, profile)])
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
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=save_dir,
        name_prefix=f"hierarchical_v41_{profile}_sac_checkpoint",
    )
    entropy_callback = EntropyControlCallback(
        switch_step=args.entropy_switch_step,
        target_ent_coef=args.target_ent_coef,
        target_lr=args.target_lr,
    )

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.load_model:
        model_path = resolve_model_path(args.load_model)
        print(f"Resuming hierarchical SAC-MPPI v4.1 ({profile}) from: {model_path}")
        model = SAC.load(model_path, env=train_env, device=device)
        model.tensorboard_log = tensorboard_log
        model.verbose = 1
        completed_timesteps = int(getattr(model, "num_timesteps", 0))
        remaining_timesteps = args.total_timesteps - completed_timesteps
        if remaining_timesteps <= 0:
            print(
                f"Requested total_timesteps={args.total_timesteps}, "
                f"but loaded model already has num_timesteps={completed_timesteps}."
            )
            train_env.close()
            eval_env.close()
            return
    else:
        model = SAC(
            "MlpPolicy",
            train_env,
            verbose=1,
            tensorboard_log=tensorboard_log,
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
        completed_timesteps = 0
        remaining_timesteps = args.total_timesteps

    print(f"Training hierarchical SAC-MPPI v4.1 ({profile}) with action space: {train_env.action_space}")
    print(f"Saving results to: {save_dir}")
    print(
        f"Completed timesteps: {completed_timesteps} | "
        f"Target total timesteps: {args.total_timesteps} | "
        f"Timesteps to run now: {remaining_timesteps}"
    )
    try:
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=[eval_callback, checkpoint_callback, entropy_callback],
            reset_num_timesteps=False,
        )
    except KeyboardInterrupt:
        print("Training interrupted; saving current model.")

    model.save(os.path.join(save_dir, f"final_model_hierarchical_v41_{profile}"))
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
