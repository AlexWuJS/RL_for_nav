import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack 
from ros_env import MyCarEnv
from lidar_compress_net import LidarProcessor 
import os
import rospy
import numpy as np

# 防止库冲突
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def main():
    print("DEBUG: 程序开始运行...")

    # ==========================================
    # 1. 初始化环境
    # ==========================================
    env = DummyVecEnv([lambda: MyCarEnv()]) 
    env = VecFrameStack(env, n_stack=4)
    
    # === 新增：获取底层环境引用 ===
    # env 是 VecFrameStack -> env.venv 是 DummyVecEnv -> env.venv.envs[0] 是原始 MyCarEnv
    # 这样我们要读取 self.target_pos 或 self.current_pos 就方便了
    raw_env = env.venv.envs[0] 

    # ==========================================
    # 2. 指定模型路径
    # ==========================================
    best_model_path = "./training_usv_v2_results/best_model.zip"
    final_model_path = "./training_usv_v2_results/final_model_stacked.zip"

    if os.path.exists(best_model_path):
        model_path = best_model_path
        print(f"✅ 发现最佳模型，正在加载: {model_path}")
    elif os.path.exists(final_model_path):
        model_path = final_model_path
        print(f"⚠️ 未找到最佳模型，尝试加载最终模型: {model_path}")
    else:
        print(f"❌ 未找到新版本模型，请先运行 train.py 进行训练")
        print(f"   期望路径: {best_model_path}")
        return

    # ==========================================
    # 3. 加载模型
    # ==========================================
    model = SAC.load(model_path, env=env)
    print("DEBUG: 模型加载成功！开始导航测试...")

    obs = env.reset()
    
    # === 新增：打印初始起点和终点 ===
    print(f"\n📍 [第一轮] 起点: ({raw_env.current_pos[0]:.2f}, {raw_env.current_pos[1]:.2f}) "
          f"-> 🎯 终点: ({raw_env.target_pos[0]:.2f}, {raw_env.target_pos[1]:.2f})")
    
    episode_reward = 0.0
    steps = 0
    
    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            
            obs, rewards, dones, infos = env.step(action)
            
            reward = rewards[0]
            done = dones[0]
            
            episode_reward += reward
            steps += 1
            
            if done:
                if episode_reward > 100: 
                    print(f"🎉 任务完成！ 用时: {steps}步, 总得分: {episode_reward:.1f}")
                else:
                    print(f"💥 任务结束 (碰撞/超时)！ 用时: {steps}步, 总得分: {episode_reward:.1f}")
                
                print("-" * 30)
                
                # === 关键修改：VecEnv 自动 Reset 后，打印新一轮的起点终点 ===
                # 因为 VecEnv 在 done=True 时已经自动调用了 reset()，
                # 所以此时 raw_env 里的 target_pos 已经是新生成的了。
                print(f"📍 [新一轮] 起点: ({raw_env.current_pos[0]:.2f}, {raw_env.current_pos[1]:.2f}) "
                      f"-> 🎯 终点: ({raw_env.target_pos[0]:.2f}, {raw_env.target_pos[1]:.2f})")

                episode_reward = 0
                steps = 0
                
    except KeyboardInterrupt:
        print("\n测试停止")
        from geometry_msgs.msg import Twist
        pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        pub.publish(Twist())

if __name__ == '__main__':
    main()