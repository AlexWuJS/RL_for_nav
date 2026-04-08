"""
Grid Environment Model Testing Script
用于测试grid环境下训练好的SAC模型

用法:
    python grid_test.py --model ./training_grid_results/best_model.zip
    python grid_test.py --model ./training_grid_results/final_model.zip --episodes 5
    python grid_test.py --episodes 10 --render  # 全程渲染
"""

import os
import sys
import argparse
import numpy as np

import gymnasium as gym
from stable_baselines3 import SAC

# 尝试导入grid环境
try:
    from grid_env import GridDynamicObstacleEnv
    from grid_render import SimpleRenderer, TrajectoryRecorder
    GRID_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Grid environment not available: {e}")
    GRID_AVAILABLE = False


def make_grid_env(render_mode='human'):
    """创建grid环境的工厂函数"""
    def _init():
        env = GridDynamicObstacleEnv(
            map_path="example_data/sample_map.npy",
            trajectory_path="example_data/sample_trajectories.json",
            robot_radius=0.15,
            v_max=1.0,
            w_max=1.0,
            dynamic_obstacle_radius=0.25,
            use_local_patch=True,
            patch_size=21,
            include_dynamic_in_patch=True,
            include_nearest_dynamic=True,
            nearest_dynamic_k=3,
            dt=0.1,
            max_episode_steps=500,
            goal_reward=100.0,
            collision_penalty=-100.0,
            step_penalty=-0.1,
            progress_weight=2.0,
            safe_distance=0.5,
            safe_distance_penalty=-5.0,
            render_mode=render_mode,
            seed=42
        )
        return env
    return _init


def test_model(model_path, episodes=3, render=True, record_gif=False, gif_path='test_traj.gif'):
    """测试训练好的模型"""
    if not GRID_AVAILABLE:
        raise RuntimeError("Grid environment not available")

    # 创建环境
    render_mode = 'human' if render else None
    env = gym.make('custom', make_grid_env=make_grid_env(render_mode))

    # 加载模型
    print(f"Loading model from: {model_path}")
    model = SAC.load(model_path, env=env)
    print("Model loaded successfully!")

    # 渲染器（可选）
    renderer = None
    recorder = None
    if render:
        try:
            from grid_render import SimpleRenderer
            raw_env = env.envs[0]
            renderer = SimpleRenderer(raw_env)
        except ImportError:
            print("Warning: Renderer not available")

    if record_gif:
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
        obs, info = env.reset()
        ep_reward = 0.0
        ep_steps = 0

        # 打印起点终点
        goal_pos = info.get('goal_pos', env.envs[0].goal_pos)
        robot_pos = info.get('robot_pos', env.envs[0].robot_pos)
        print(f"\n{'='*50}")
        print(f"Episode {ep+1}/{episodes}")
        print(f"  Start: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f})")
        print(f"  Goal:  ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})")
        print(f"{'='*50}")

        while True:
            # 渲染
            if renderer:
                renderer.render()

            # 预测动作
            action, _ = model.predict(obs, deterministic=True)

            # 执行
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps += 1

            # 记录GIF
            if recorder:
                recorder.capture_frame()

            if terminated or truncated:
                # 获取终止原因
                collision = info.get('collision', False)
                goal_reached = info.get('goal_reached', False)

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
    print(f"  Avg Steps: {np.mean(episode_steps):.1f} ± {np.std(episode_steps):.1f}")
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


def random_agent_test(episodes=1, render=True):
    """随机动作测试（验证环境正常工作）"""
    if not GRID_AVAILABLE:
        raise RuntimeError("Grid environment not available")

    render_mode = 'human' if render else None
    env = GridDynamicObstacleEnv(
        map_path="example_data/sample_map.npy",
        trajectory_path="example_data/sample_trajectories.json",
        use_local_patch=True,
        patch_size=21,
        render_mode=render_mode,
        seed=42
    )

    renderer = None
    if render:
        try:
            renderer = SimpleRenderer(env)
        except ImportError:
            pass

    print("\n🎲 Testing with Random Agent (baseline)...")

    for ep in range(episodes):
        obs, info = env.reset()
        ep_reward = 0.0
        ep_steps = 0

        while True:
            if renderer:
                renderer.render()

            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps += 1

            if terminated or truncated:
                print(f"  Episode {ep+1}: steps={ep_steps}, reward={ep_reward:.1f}, "
                      f"collision={info.get('collision', False)}, "
                      f"goal={info.get('goal_reached', False)}")
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

    # 自动检测模型路径
    if args.model is None:
        # 尝试多个可能路径
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
        random_agent_test(episodes=args.episodes, render=not args.no_render)
    else:
        test_model(
            model_path=args.model,
            episodes=args.episodes,
            render=not args.no_render,
            record_gif=args.record_gif,
            gif_path=args.gif_path
        )


if __name__ == "__main__":
    main()