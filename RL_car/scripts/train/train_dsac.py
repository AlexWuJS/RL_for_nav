import os

import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from dsac_mppi.algorithms.dsac import DSACConfig, DSACPolicy, DSACTrainer, parse_dsac_args
from dsac_mppi.envs.curriculum import CurriculumManager
from dsac_mppi.envs.ros_env import MyCarEnv


def make_env(rank: int, log_dir: str, args, curriculum_manager: CurriculumManager | None = None):
    def _init():
        env = MyCarEnv(
            control_mode=args.control_mode,
            dynamics_model=args.dynamics_model,
            curriculum_stage=curriculum_manager.stage if curriculum_manager is not None else None,
        )
        env.curriculum_manager = curriculum_manager
        return Monitor(env, filename=os.path.join(log_dir, f"dsac_monitor_{rank}"))

    return _init


def main():
    args = parse_dsac_args()
    model_root = os.path.join("data", args.model_name)
    args.save_dir = args.save_dir or os.path.join(model_root, "models")
    args.log_dir = args.log_dir or os.path.join(model_root, "logs")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    curriculum_manager = CurriculumManager(args.curriculum)
    env = DummyVecEnv([make_env(0, args.log_dir, args, curriculum_manager)])
    if args.frame_stack > 1:
        env = VecFrameStack(env, n_stack=args.frame_stack)

    obs_dim = int(np.prod(env.observation_space.shape))
    action_low = tuple(float(x) for x in env.action_space.low.reshape(-1))
    action_high = tuple(float(x) for x in env.action_space.high.reshape(-1))
    config = DSACConfig(
        observation_dim=obs_dim,
        action_dim=int(np.prod(env.action_space.shape)),
        action_low=action_low,
        action_high=action_high,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        seed=args.seed,
    )
    policy = DSACPolicy(config, device=args.device)
    tensorboard_log = args.tensorboard_log or os.path.join(args.log_dir, "tensorboard")
    trainer = DSACTrainer(policy, env, config, tensorboard_log=tensorboard_log)

    print(f"DSAC observation_dim={obs_dim}, action_low={action_low}, action_high={action_high}")
    print(
        "DSAC environment: "
        f"control_mode={args.control_mode}, dynamics_model={args.dynamics_model}, curriculum={args.curriculum}"
    )
    print(f"Saving DSAC models to: {args.save_dir}")
    try:
        trainer.learn(
            args.total_timesteps,
            args.save_dir,
            log_interval=args.log_interval,
            episode_log_interval=args.episode_log_interval,
        )
    except KeyboardInterrupt:
        print("DSAC training interrupted; saving current model.")
    finally:
        trainer.close()

    policy.save(os.path.join(args.save_dir, "final_model_dsac"))
    if not os.path.exists(os.path.join(args.save_dir, "best_model", "model.pt")):
        policy.save(os.path.join(args.save_dir, "best_model"))
    env.close()
    print("DSAC training finished.")


if __name__ == "__main__":
    main()
