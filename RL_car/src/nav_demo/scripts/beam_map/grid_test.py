"""
Grid Environment Model Testing Script
用于测试grid环境下训练好的SAC模型

用法:
    python grid_test.py --model ./training_grid_results/best_model.zip
    python grid_test.py --model ./training_grid_results/final_model.zip --episodes 5
    python grid_test.py --episodes 10 --render  # 全程渲染
    python grid_test.py --random  # 随机Agent基线测试
"""

import os
import sys
import argparse
import numpy as np

import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

# 尝试导入grid环境
try:
    from grid_env import GridDynamicObstacleEnv
    from grid_render import SimpleRenderer, TrajectoryRecorder
    GRID_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Grid environment not available: {e}")
    GRID_AVAILABLE = False


def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def make_grid_env(config: dict, render_mode=None, seed: int = 42):
    """创建grid环境的工厂函数"""
    def _init():
        env = GridDynamicObstacleEnv(
            map_path=config.get('map_path', 'example_data/sample_map.npy'),
            trajectory_path=config.get('trajectory_path', 'example_data/sample_trajectories.json'),
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
            render_mode=render_mode,
            seed=seed
        )
        return env
    return _init


def _make_batched_action(action, num_envs=1):
    """确保action是VecEnv期望的批处理格式 (num_envs, action_dim)"""
    if not isinstance(action, np.ndarray):
        action = np.array(action, dtype=np.float32)
    # 如果action是1D（单个动作），添加batch维度
    if action.ndim == 1:
        action = action.reshape(1, -1)  # shape: (1, action_dim)
    # 如果action已经有正确的batch维度但num_envs>1，需要确保匹配
    if action.shape[0] != num_envs:
        if action.shape[0] == 1 and num_envs > 1:
            action = np.repeat(action, num_envs, axis=0)
        else:
            action = action.reshape(1, -1)
    return action


def test_model(model_path, episodes=3, render=True, record_gif=False, gif_path='test_traj.gif', config: dict = None):
    """测试训练好的模型"""
    if not GRID_AVAILABLE:
        raise RuntimeError("Grid environment not available")

    # 创建环境（使用DummyVecEnv来匹配stable-baselines3接口）
    render_mode = 'human' if render else None
    if config is None:
        config = {}
    env = DummyVecEnv([make_grid_env(config, render_mode=render_mode, seed=42)])

    # 加载模型
    print(f"Loading model from: {model_path}")
    model = SAC.load(model_path, env=env)
    print("Model loaded successfully!")

    # 渲染器（可选）
    renderer = None
    recorder = None
    if render and GRID_AVAILABLE:
        try:
            from grid_render import SimpleRenderer
            # DummyVecEnv.envs[0] 是实际的 Gym 环境
            raw_env = env.envs[0]
            renderer = SimpleRenderer(raw_env)
        except ImportError:
            print("Warning: Renderer not available")

    if record_gif and GRID_AVAILABLE:
        try:
            from grid_render import TrajectoryRecorder
            raw_env = env.envs[0]
            recorder = TrajectoryRecorder(raw_env, output_path=gif_path)
        except ImportError:
            print("Warning: GIF recorder not available")

    # 测试循环
    episode_rewards = []
    episode_steps = []
    collision_count = 0
    goal_reached_count = 0

    for ep in range(episodes):
        # DummyVecEnv.reset() 返回 obs（已经batch的）
        obs = env.reset()
        ep_reward = 0.0
        ep_steps = 0

        # 获取初始状态（通过get_attr获取原始环境的状态）
        goal_pos = env.get_attr('goal_pos')[0]
        robot_pos = env.get_attr('robot_pos')[0]
        print(f"\n{'='*50}")
        print(f"Episode {ep+1}/{episodes}")
        print(f"  Start: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f})")
        print(f"  Goal:  ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})")
        print(f"{'='*50}")

        while True:
            # 渲染
            if renderer:
                renderer.render()

            # 预测动作（obs已经是numpy数组格式 (1, obs_dim)）
            action, _ = model.predict(obs, deterministic=True)
            # 确保action是批处理格式
            action = _make_batched_action(action, env.num_envs)

            # 执行
            obs, reward, dones, infos = env.step(action)
            ep_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            ep_steps += 1

            # 记录GIF
            if recorder:
                recorder.capture_frame()

            # SB3的step返回 dones（不是terminated/truncated分开）
            done = dones[0] if isinstance(dones, np.ndarray) else dones

            if done:
                # 获取终止原因
                collision = infos.get('collision', [False])[0] if isinstance(infos, dict) else False
                goal_reached = infos.get('goal_reached', [False])[0] if isinstance(infos, dict) else False

                if collision:
                    result = "💥 COLLISION"
                    collision_count += 1
                elif goal_reached:
                    result = "🎉 GOAL REACHED"
                    goal_reached_count += 1
                else:
                    result = "⏰ TIMEOUT"

                print(f"\n  Episode {ep+1} finished:")
                print(f"    Result: {result}")
                print(f"    Steps: {ep_steps}")
                print(f"    Reward: {ep_reward:.1f}")

                episode_rewards.append(ep_reward)
                episode_steps.append(ep_steps)
                break

    # 统计
    print(f"\n{'='*60}")
    print("📊 Test Summary")
    print(f"{'='*60}")
    print(f"  Total Episodes: {episodes}")
    print(f"  Goals Reached: {goal_reached_count} ({100*goal_reached_count/episodes:.1f}%)")
    print(f"  Collisions: {collision_count} ({100*collision_count/episodes:.1f}%)")
    print(f"  Timeouts: {episodes - collision_count - goal_reached_count}")
    if episode_steps:
        print(f"  Avg Steps: {np.mean(episode_steps):.1f} ± {np.std(episode_steps):.1f}")
    if episode_rewards:
        print(f"  Avg Reward: {np.mean(episode_rewards):.1f} ± {np.std(episode_rewards):.1f}")
    print(f"{'='*60}")

    # 保存GIF
    if recorder:
        recorder.save()

    # 清理
    if renderer:
        input("Press Enter to close...")
        renderer.close()

    env.close()

    return {
        'episode_rewards': episode_rewards,
        'episode_steps': episode_steps,
        'collision_count': collision_count,
        'goal_reached_count': goal_reached_count
    }


def random_agent_test(episodes=1, render=True, config: dict = None):
    """随机动作测试（验证环境正常工作）"""
    if not GRID_AVAILABLE:
        raise RuntimeError("Grid environment not available")

    render_mode = 'human' if render else None
    if config is None:
        config = {}
    env = DummyVecEnv([make_grid_env(config, render_mode=render_mode, seed=42)])

    renderer = None
    if render and GRID_AVAILABLE:
        try:
            raw_env = env.envs[0]
            renderer = SimpleRenderer(raw_env)
        except ImportError:
            pass

    print("\n🎲 Testing with Random Agent (baseline)...")

    for ep in range(episodes):
        obs = env.reset()
        ep_reward = 0.0
        ep_steps = 0

        while True:
            if renderer:
                renderer.render()

            # 获取原始action_space来sample
            raw_env_instance = env.envs[0]
            action = raw_env_instance.action_space.sample()
            # 确保是numpy数组格式，并批处理为(num_envs, action_dim)
            action = _make_batched_action(action, env.num_envs)

            obs, reward, dones, infos = env.step(action)
            ep_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            ep_steps += 1

            done = dones[0] if isinstance(dones, np.ndarray) else dones
            if done:
                collision = infos.get('collision', [False])[0] if isinstance(infos, dict) else False
                goal_reached = infos.get('goal_reached', [False])[0] if isinstance(infos, dict) else False
                print(f"  Episode {ep+1}: steps={ep_steps}, reward={ep_reward:.1f}, "
                      f"collision={collision}, goal={goal_reached}")
                break

    if renderer:
        input("Press Enter to close...")
        renderer.close()

    env.close()


def main():
    parser = argparse.ArgumentParser(description='Test grid-trained SAC model')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model zip file (default: auto-detect best/final)')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of test episodes')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable rendering')
    parser.add_argument('--record-gif', action='store_true',
                       help='Record trajectory as GIF')
    parser.add_argument('--gif-path', type=str, default='test_traj.gif',
                       help='Output GIF path')
    parser.add_argument('--random', action='store_true',
                       help='Run random agent test instead of loading model')
    parser.add_argument('--config', type=str, default='configs/train_grid.yaml',
                       help='Path to config file')

    args = parser.parse_args()

    # 加载配置文件
    full_config = load_config(args.config)
    env_config = full_config.get('env', {})

    # 自动检测模型路径（仅当需要加载模型时）
    if not args.random and args.model is None:
        possible_paths = [
            "./training_grid_results/best_model.zip",
            "./training_grid_results/final_model.zip",
            "./training_grid_results/best_model",
            "./training_grid_results/final_model",
        ]
        for p in possible_paths:
            if os.path.exists(p):
                args.model = p
                print(f"Auto-detected model: {p}")
                break
        if args.model is None:
            print("❌ No model found! Please train a model first:")
            print("   python train.py --env_type grid --total_timesteps 50000")
            print("\nOr specify model path:")
            print("   python grid_test.py --model ./training_grid_results/best_model.zip")
            return

    if args.random:
        random_agent_test(episodes=args.episodes, render=not args.no_render, config=env_config)
    else:
        test_model(
            model_path=args.model,
            episodes=args.episodes,
            render=not args.no_render,
            record_gif=args.record_gif,
            gif_path=args.gif_path,
            config=env_config
        )


if __name__ == "__main__":
    main()