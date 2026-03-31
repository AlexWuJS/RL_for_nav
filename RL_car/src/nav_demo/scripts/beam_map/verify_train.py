"""
USV 强化学习训练验证脚本 (微型过拟合测试)
验证 RL 算法能否在 USV 惯性环境下正常训练并开始收敛
"""
import os
import sys
import numpy as np
import torch

# 确保能导入同级模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from ros_env import MyCarEnv


# ==========================================
# 自定义回调：打印训练进度
# ==========================================
class ProgressCallback(BaseCallback):
    def __init__(self, print_freq=1000, verbose=1):
        super(ProgressCallback, self).__init__(verbose)
        self.print_freq = print_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_ep_reward = 0
        self.current_ep_len = 0
        self.total_timesteps_logged = set()

    def _on_step(self) -> bool:
        # 从 info 中获取 reward 和 done 状态
        # SB3 的 step 返回已经是展开的
        self.current_ep_reward += self.locals["rewards"][0]
        self.current_ep_len += 1

        # 检测回合结束 (SB3 VecEnv 的 dones 是一个数组)
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_ep_reward)
            self.episode_lengths.append(self.current_ep_len)

            # 每 print_freq 步打印一次进度
            if len(self.episode_rewards) % self.print_freq == 0:
                recent_rews = self.episode_rewards[-self.print_freq:]
                recent_lens = self.episode_lengths[-self.print_freq:]
                ep_rew_mean = np.mean(recent_rews)
                ep_len_mean = np.mean(recent_lens)
                print(f"[Step {self.num_timesteps:>8}] "
                      f"ep_rew_mean: {ep_rew_mean:>8.2f} | "
                      f"ep_len_mean: {ep_len_mean:>6.1f} | "
                      f"episodes: {len(self.episode_rewards)}")

            self.current_ep_reward = 0
            self.current_ep_len = 0

        return True


# ==========================================
# 辅助函数：创建并包装环境
# ==========================================
def make_env(log_dir=None):
    def _init():
        env = MyCarEnv()
        if log_dir:
            env = Monitor(env, filename=os.path.join(log_dir, "monitor_verify"))
        return env
    return _init


# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    log_dir = "./verify_logs/"
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 60)
    print("USV 惯性环境 - 训练验证 (微型过拟合测试)")
    print("=" * 60)

    # 1. 创建环境 (DummyVecEnv for SB3)
    print("\n[1/5] 初始化环境...")
    env = DummyVecEnv([make_env(log_dir)])
    print(f"    Action space:     {env.action_space}")
    print(f"    Observation space: {env.observation_space}")

    # 2. 初始化 SAC 模型
    print("\n[2/5] 初始化 SAC 模型...")
    model = SAC(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        buffer_size=10000,
        batch_size=64,
        ent_coef='auto',
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        # 目标熵自动调节
    )
    print(f"    Device: {model.device}")
    print(f"    Buffer size: 10000")

    # 3. 创建进度回调
    print("\n[3/5] 配置训练回调...")
    progress_callback = ProgressCallback(print_freq=5)  # 每5个回合打印一次

    # 4. 开始训练
    total_timesteps = 20000
    print(f"\n[4/5] 开始训练 (total_timesteps={total_timesteps})...")
    print("-" * 60)

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=progress_callback,
            progress_bar=False,
        )
    except Exception as e:
        print(f"\n[ERROR] 训练过程出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("-" * 60)
    print(f"\n[5/5] 训练完成!")

    # 5. 打印统计信息
    if progress_callback.episode_rewards:
        rewards = progress_callback.episode_rewards
        lengths = progress_callback.episode_lengths
        print(f"\n=== 训练统计 ===")
        print(f"    总回合数:  {len(rewards)}")
        print(f"    平均奖励:  {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"    平均长度:  {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
        print(f"    最高奖励:  {np.max(rewards):.2f}")
        print(f"    最近5回合奖励: {[f'{r:.1f}' for r in rewards[-5:]]}")
    else:
        print("\n    (未能收集到完整回合数据)")

    # 保存模型
    save_path = os.path.join(log_dir, "verify_usv_model")
    model.save(save_path)
    print(f"\n    模型已保存至: {save_path}")
    print("\n✅ 验证脚本执行完毕!")
