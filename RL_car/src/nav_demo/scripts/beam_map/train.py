import os
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback # 引入回调函数
from stable_baselines3.common.monitor import Monitor        # 引入监控器
from ros_env import MyCarEnv
from lidar_compress_net import LidarProcessor

# 1. 创建保存目录
save_dir = "./training_results/"
os.makedirs(save_dir, exist_ok=True)

# 2. 创建环境
# 注意：为了让回调函数能读取到奖励数据，建议用 Monitor 包裹环境
env = MyCarEnv()
env = Monitor(env, filename=os.path.join(save_dir, "monitor_log"))

# 定义策略网络参数
policy_kwargs = dict(
    features_extractor_class=LidarProcessor,
    features_extractor_kwargs=dict(features_dim=128)
)

# ==========================================
# 3. 定义自动保存最佳模型的回调函数 (核心部分)
# ==========================================
eval_callback = EvalCallback(
    env,                            # 用于测试的环境 (在ROS中通常就复用训练环境)
    best_model_save_path=save_dir,  # 最佳模型保存的文件夹
    log_path=save_dir,              # 评估日志保存路径
    eval_freq=10000,                # 多少步评估一次 (例如每1万步测一次)
    n_eval_episodes=5,              # 每次评估跑多少个回合取平均值
    deterministic=True,             # 测试时使用确定性策略 (不探索，只用最强实力)
    render=False,
    verbose=1
)

# 4. 初始化模型
model = SAC(
    "MlpPolicy", 
    env, 
    verbose=1, 
    tensorboard_log="./sac_nav_car_log/", 
    policy_kwargs=policy_kwargs,
    learning_rate=3e-4,
    buffer_size=50000,      # 建议改回 50000，太小了虽然跑得快但忘得快
    batch_size=256,
    ent_coef='auto',
    train_freq=(10, "step"), # 之前优化 FPS 建议的设置
    gradient_steps=10
)

print("开始训练... 最好的模型将保存在 best_model.zip")

# 5. 开始训练，并挂载回调函数
model.learn(
    total_timesteps=200000, 
    callback=eval_callback  # <--- 把回调函数传进去
)

# 保存最终模型 (作为备份，但不一定是最好的)
model.save(os.path.join(save_dir, "final_model"))

print("训练结束。")

# ==========================================
# 测试部分
# ==========================================
# 如果你想直接加载跑出来的最好的模型进行测试：
# model = SAC.load(os.path.join(save_dir, "best_model.zip"))

obs, info = env.reset() 
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()