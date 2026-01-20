import gymnasium as gym
from stable_baselines3 import PPO,SAC
from ros_env import MyCarEnv
import os

# 防止库冲突
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def main():
    print("DEBUG: 程序开始运行...") # <--- 调试点 1

    print("DEBUG: 正在初始化环境 (如果卡在这里，请检查 Gazebo 是否开启)...")
    env = MyCarEnv()
    print("DEBUG: 环境初始化完成！")

    # 请根据你实际有的文件修改这里！
    # 如果你还没有 logs/best_model.zip，就用 my_car_ppo.zip
    # model_path = "my_car_ppo.zip" 
    model_path = "my_car_sac.zip"
    
    print(f"DEBUG: 正在寻找模型文件: {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ 错误：找不到文件 {model_path}，请检查路径！")
        return

    print("DEBUG: 正在加载模型...")
    # 强制使用 CPU 运行
    # model = PPO.load(model_path, env=env, device='cpu')
    model = SAC.load(model_path, env=env)
    print("DEBUG: 模型加载成功！开始控制小车...")

    obs, info = env.reset()
    
    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print(">>> 回合结束，重置环境")
                obs, info = env.reset()
                
    except KeyboardInterrupt:
        print("\n测试停止")

# === 这一段绝对不能少！===
if __name__ == '__main__':
    main()