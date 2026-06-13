from ros_env import MyCarEnv  # 请替换为你的实际环境类名
import time

env = MyCarEnv()
obs = env.reset()

print("--- 惯性加速测试 ---")
action = [1.0, 0.0]  # 给定恒定推力 [Surge, Yaw]
for i in range(10):
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i+1} | 实际 Surge 速度: {env.velocity[0]:.4f}")
    time.sleep(0.1)

print("--- 阻尼刹车测试 ---")
action = [0.0, 0.0]  # 推力归零
for i in range(5):
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Brake Step {i+1} | 实际 Surge 速度: {env.velocity[0]:.4f}")
    time.sleep(0.1)