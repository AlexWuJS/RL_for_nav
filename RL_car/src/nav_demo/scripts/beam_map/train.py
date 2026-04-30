#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import yaml
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import constant_fn
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# 尝试导入Gazebo环境，失败不阻塞
try:
    from ros_env import MyCarEnv
    from lidar_compress_net import LidarProcessor
    GAZEB0_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Gazebo environment not available: {e}")
    GAZEB0_AVAILABLE = False


# ==========================================
# 0. 自定义回调：两阶段熵控制 (探索 -> 利用)
# ==========================================
class EntropyControlCallback(BaseCallback):
    """
    在指定步数时，强制关闭 SAC 的自动熵调节，锁死为极小值，并降低学习率。
    实现从"野蛮探索"到"老司机微调"的完美过渡。
    """
    def __init__(self, switch_step=100000, target_ent_coef=0.02, target_lr=3e-5, verbose=0):
        super(EntropyControlCallback, self).__init__(verbose)
        self.switch_step = switch_step
        self.target_ent_coef = target_ent_coef
        self.target_lr = target_lr
        self.switched = False

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.switch_step and not self.switched:
            print(f"\n[🚀 {self.num_timesteps} 步触发] 正在执行两阶段训练切换！")

            # 1. 锁死固定熵
            self.model.ent_coef_optimizer = None
            self.model.ent_coef_tensor = torch.tensor(float(self.target_ent_coef), device=self.model.device)
            self.model.ent_coef = self.target_ent_coef
            print(f"   -> 熵系数 (ent_coef) 已强制锁死为: {self.target_ent_coef}")

            # 2. 降低学习率
            self.model.lr_schedule = constant_fn(self.target_lr)
            for optimizer in [self.model.actor.optimizer, self.model.critic.optimizer]:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.target_lr
            print(f"   -> 学习率 (learning_rate) 已降至: {self.target_lr}")

            print("✅ 切换完成！模型进入稳健微调模式！\n")
            self.switched = True

        return True


# ==========================================
# 1. Grid环境工厂函数
# ==========================================
def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def make_grid_env(rank: int, log_dir: str, config: dict, seed: int = None) -> callable:
    """创建Grid环境的工厂函数"""
    def _init():
        from grid_env import GridDynamicObstacleEnv

        # 合并config中的seed和传入的seed
        env_seed = seed if seed is not None else config.get('seed', None)

        env = GridDynamicObstacleEnv(
            map_path=config.get('map_path', None),
            trajectory_path=config.get('trajectory_path', None),
            robot_radius=config.get('robot_radius', 0.15),
            v_max=config.get('v_max', 1.0),
            w_max=config.get('w_max', 1.0),
            dynamic_obstacle_radius=config.get('dynamic_obstacle_radius', 0.25),
            use_local_patch=config.get('use_local_patch', True),
            patch_size=config.get('patch_size', 21),
            include_dynamic_in_patch=config.get('include_dynamic_in_patch', True),
            include_nearest_dynamic=config.get('include_nearest_dynamic', True),
            nearest_dynamic_k=config.get('nearest_dynamic_k', 3),
            dt=config.get('dt', 0.1),
            max_episode_steps=config.get('max_episode_steps', 500),
            goal_reward=config.get('goal_reward', 100.0),
            collision_penalty=config.get('collision_penalty', -100.0),
            step_penalty=config.get('step_penalty', -0.1),
            progress_weight=config.get('progress_weight', 2.0),
            safe_distance=config.get('safe_distance', 0.5),
            safe_distance_penalty=config.get('safe_distance_penalty', -5.0),
            render_mode='human' if rank == 0 else None,
            seed=env_seed
        )

        if log_dir:
            env = Monitor(env, filename=os.path.join(log_dir, f"monitor_{rank}"))
        return env
    return _init


def make_gazebo_env(rank: int, log_dir: str) -> callable:
    """创建Gazebo环境的工厂函数"""
    def _init():
        if not GAZEB0_AVAILABLE:
            raise RuntimeError("Gazebo environment not available")
        env = MyCarEnv()
        if log_dir:
            env = Monitor(env, filename=os.path.join(log_dir, f"monitor_{rank}"))
        return env
    return _init


# ==========================================
# 2. 主程序
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SAC agent')
    parser.add_argument('--env_type', type=str, default='grid',
                       choices=['gazebo', 'grid'],
                       help='Environment type: gazebo or grid')
    parser.add_argument('--config', type=str, default='configs/train_grid.yaml',
                       help='Path to config file (for grid env)')
    parser.add_argument('--total_timesteps', type=int, default=100000,
                       help='Total training timesteps')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()

    # 根据环境类型设置路径
    if args.env_type == 'grid':
        config_path = args.config

        # 加载配置
        if os.path.exists(config_path):
            full_config = load_config(config_path)
            env_config = full_config.get('env', {})
            sac_config = full_config.get('sac', {})
            training_config = full_config.get('training', {})
            save_dir = training_config.get('save_dir', './training_grid_results/')
            log_dir = training_config.get('log_dir', './logs_grid/')
            tensorboard_log = training_config.get('tensorboard_log', './3_sac_grid_log/')
            total_timesteps = args.total_timesteps  # CLI args override config
            eval_freq = training_config.get('eval_freq', 5000)
            save_freq = training_config.get('save_freq', 10000)
            n_eval_episodes = training_config.get('n_eval_episodes', 5)
        else:
            print(f"Warning: Config file {config_path} not found, using defaults")
            env_config = {}
            sac_config = {}
            save_dir = './training_grid_results/'
            log_dir = './logs_grid/'
            tensorboard_log = './3_sac_grid_log/'
            total_timesteps = args.total_timesteps
            eval_freq = 5000
            save_freq = 10000
            n_eval_episodes = 5

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # 创建环境（传入CLI的seed）
        print("正在初始化Grid训练环境...")
        train_env = DummyVecEnv([make_grid_env(0, log_dir, env_config, seed=args.seed)])
        print("正在初始化Grid评估环境...")
        eval_env = DummyVecEnv([make_grid_env(1, log_dir, env_config, seed=args.seed)])

        # Grid环境不需要LidarProcessor，直接使用MLP
        policy_kwargs = sac_config.get('policy_kwargs', dict(
            net_arch=dict(pi=[256, 256], qf=[256, 256])
        ))

        learning_rate = sac_config.get('learning_rate', 3e-4)
        batch_size = sac_config.get('batch_size', 256)
        buffer_size = sac_config.get('buffer_size', 100000)
        gamma = sac_config.get('gamma', 0.99)
        tau = sac_config.get('tau', 0.005)
        train_freq = sac_config.get('train_freq', 1)
        gradient_steps = sac_config.get('gradient_steps', 1)

        print(f"Grid Environment observation space: {train_env.observation_space}")

    elif args.env_type == 'gazebo':
        if not GAZEB0_AVAILABLE:
            raise RuntimeError("Gazebo environment not available. Please ensure ROS/Gazebo is properly sourced.")

        save_dir = "./training_usv_v2_results/"
        log_dir = "./logs/"
        tensorboard_log = "./3_sac_nav_car_log/"
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        print("正在初始化Gazebo训练环境...")
        train_env = DummyVecEnv([make_gazebo_env(0, log_dir)])
        print("正在初始化Gazebo评估环境...")
        eval_env = DummyVecEnv([make_gazebo_env(1, log_dir)])

        # Gazebo环境使用LidarProcessor
        policy_kwargs = dict(
            features_extractor_class=LidarProcessor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[256, 256], qf=[256, 256])
        )

        learning_rate = 3e-4
        batch_size = 128
        buffer_size = 200000
        gamma = 0.99
        tau = 0.005
        train_freq = (10, "step")
        gradient_steps = 10
        total_timesteps = args.total_timesteps
        eval_freq = 5000
        save_freq = 50000
        n_eval_episodes = 5

    else:
        raise ValueError(f"Unknown env_type: {args.env_type}")

    # ==========================================
    # 3. 定义回调函数列表
    # ==========================================
    # A. 自动保存最佳模型
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1
    )

    # B. 定期保存检查点
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=save_dir,
        name_prefix="sac_checkpoint"
    )

    # C. 两阶段训练控制
    entropy_callback = EntropyControlCallback(
        switch_step=120000,
        target_ent_coef=0.02,
        target_lr=3e-5
    )

    # ==========================================
    # 4. 初始化 SAC 模型
    # ==========================================
    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        ent_coef='auto',
        gamma=gamma,
        tau=tau,
        learning_starts=10000,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Environment: {args.env_type}")
    print(f"Observation Space: {train_env.observation_space.shape}")
    print(f"Action Space: {train_env.action_space}")
    print(f"Total timesteps: {total_timesteps}")
    print("🚀 开始SAC训练...")

    # ==========================================
    # 5. 开始训练
    # ==========================================
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback, entropy_callback]
        )
    except KeyboardInterrupt:
        print("\n⚠️ 训练被手动中断！正在保存当前模型...")

    # 保存最终模型
    model.save(os.path.join(save_dir, "final_model"))
    print("✅ 训练结束，模型已保存。")

    # ==========================================
    # 6. 简单测试循环
    # ==========================================
    print("开始最终评估演示...")
    obs = eval_env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_env.step(action)
