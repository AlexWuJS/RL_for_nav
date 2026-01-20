import gymnasium as gym
from stable_baselines3 import SAC
from ros_env import MyCarEnv
from lidar_compress_net import LidarProcessor # 必须导入这个，否则加载模型会报错
import os
import rospy

# 防止库冲突
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def main():
    print("DEBUG: 程序开始运行...")

    # 1. 初始化环境
    # 注意：这里的环境逻辑必须和训练时的一模一样（Observation空间维度必须一致）
    env = MyCarEnv()
    
    # 2. 指定模型路径
    # 优先使用自动保存的最高分模型
    best_model_path = "./training_results/best_model.zip"
    last_model_path = "nav_car_sac.zip"
    
    if os.path.exists(best_model_path):
        model_path = best_model_path
        print(f"✅ 发现最佳模型，正在加载: {model_path}")
    else:
        model_path = last_model_path
        print(f"⚠️ 未找到最佳模型，尝试加载最终模型: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ 错误：找不到文件 {model_path}")
        return

    # 3. 加载模型
    # 这里的 custom_objects 主要是为了防止有些版本不兼容，通常直接 load 即可
    model = SAC.load(model_path, env=env)
    print("DEBUG: 模型加载成功！开始导航测试...")

    obs, info = env.reset()
    
    episode_reward = 0
    steps = 0
    
    try:
        while True:
            # === 关键点：deterministic=True ===
            # 训练时我们需要随机性来探索(False)，测试时我们需要最强的执行力(True)
            action, _states = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            # 可以在这里打印当前距离目标的距离 (如果你的 info 里有的话，或者是 env 里的变量)
            # print(f"Action: {action}, Reward: {reward:.2f}")

            if terminated or truncated:
                if reward > 50: # 粗略判断，如果是正向大奖励，说明到了
                    print(f"🎉 成功到达终点！ 用时: {steps}步, 总得分: {episode_reward:.1f}")
                else:
                    print(f"💥 碰撞或超时！ 用时: {steps}步, 总得分: {episode_reward:.1f}")
                
                obs, info = env.reset()
                episode_reward = 0
                steps = 0
                print("-" * 30)
                
    except KeyboardInterrupt:
        print("\n测试停止")

if __name__ == '__main__':
    main()