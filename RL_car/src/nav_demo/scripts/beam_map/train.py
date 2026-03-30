import os
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from ros_env import MyCarEnv
from lidar_compress_net import LidarProcessor

# ==========================================
# 0. 自定义回调：两阶段熵控制 (探索 -> 利用)
# ==========================================
class EntropyControlCallback(BaseCallback):
    """
    在指定步数时，强制关闭 SAC 的自动熵调节，锁死为极小值，并降低学习率。
    实现从“野蛮探索”到“老司机微调”的完美过渡。
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
            self.model.learning_rate = self.target_lr
            for optimizer in [self.model.actor.optimizer, self.model.critic.optimizer]:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.target_lr
            print(f"   -> 学习率 (learning_rate) 已降至: {self.target_lr}")
            
            print("✅ 切换完成！模型进入稳健微调模式！\n")
            self.switched = True
            
        return True

# ==========================================
# 1. 辅助函数：创建并包装环境
# ==========================================
def make_env(rank, log_dir):
    """
    用于创建环境的工厂函数。
    Stable-Baselines3 的向量环境需要传入一个生成环境的无参函数。
    """
    def _init():
        env = MyCarEnv()
        # Monitor 用于记录每回合的 Reward 和 Length，生成 monitor.csv
        env = Monitor(env, filename=os.path.join(log_dir, f"monitor_{rank}"))
        return env
    return _init

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # 1. 路径设置
    save_dir = "./training_frenet_results/"
    log_dir = "./logs/"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # ==========================================
    # 2. 构建训练环境 (核心修改：Frame Stacking)
    # ==========================================
    print("正在初始化训练环境...")
    
    # A. 创建基础向量化环境
    train_env = DummyVecEnv([make_env(0, log_dir)])
    
    # B. 堆叠最近 4 帧 (n_stack=4)
    # 这一步非常关键！它把 [t, t-1, t-2, t-3] 时刻的数据拼在一起。
    train_env = VecFrameStack(train_env, n_stack=4)

    # ==========================================
    # 3. 构建评估环境 (结构必须完全一致)
    # ==========================================
    print("正在初始化评估环境...")
    eval_env = DummyVecEnv([make_env(1, log_dir)])
    eval_env = VecFrameStack(eval_env, n_stack=4) 

    # ==========================================
    # 4. 定义网络参数
    # ==========================================
    policy_kwargs = dict(
        features_extractor_class=LidarProcessor,
        # features_dim 是经过特征提取网络压缩后的特征向量长度
        features_extractor_kwargs=dict(features_dim=256), 
        net_arch=dict(pi=[256, 256], qf=[256, 256])
    )

    # ==========================================
    # 5. 定义回调函数列表
    # ==========================================
    # A. 自动保存最佳模型
    eval_callback = EvalCallback(
        eval_env,                   
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=5000,             # 每 5000 步测试一次
        n_eval_episodes=5,          # 每次测试跑 5 个回合取平均
        deterministic=True,         # 测试时使用确定性策略
        render=False,
        verbose=1
    )
    
    # B. 定期保存检查点 (防止断电白跑)
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path=save_dir, 
        name_prefix="sac_checkpoint"
    )

    # C. 两阶段训练控制 (第 12 万步切断探索)
    # 根据你之前的崩溃日志，12万步左右效果最好，所以设为 120000
    entropy_callback = EntropyControlCallback(
        switch_step=120000, 
        target_ent_coef=0.02, 
        target_lr=3e-5
    )

    # ==========================================
    # 6. 初始化 SAC 模型
    # ==========================================
    model = SAC(
        "MlpPolicy",
        train_env,                  
        verbose=1,
        tensorboard_log="./3_sac_nav_car_log/",
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,         # 前期较高学习率
        buffer_size=200000,         # 加大经验池容量
        batch_size=512,             # 加大 Batch Size
        ent_coef='auto',            # 前期让它自动调节探索
        gamma=0.99,
        tau=0.005,
        train_freq=1,               
        gradient_steps=1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Observation Space (Stacked): {train_env.observation_space.shape}")
    print("🚀 开始多阶段 SAC 训练... (FrameStack Enabled)")

    # ==========================================
    # 7. 开始训练
    # ==========================================
    try:
        model.learn(
            total_timesteps=300000,     
            # 同时挂载三个回调函数
            callback=[eval_callback, checkpoint_callback, entropy_callback] 
        )
    except KeyboardInterrupt:
        print("\n⚠️ 训练被手动中断！正在保存当前模型...")

    # 保存最终模型
    model.save(os.path.join(save_dir, "final_model_stacked"))
    print("✅ 训练结束，模型已保存。")

    # ==========================================
    # 8. 简单测试循环
    # ==========================================
    print("开始最终评估演示...")
    obs = eval_env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)