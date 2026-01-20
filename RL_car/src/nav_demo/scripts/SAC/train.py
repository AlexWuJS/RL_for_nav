import rospy
import os
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from robot_env import MyRobotEnv  # 导入第一步写的文件

def main():
    # 1. 初始化 ROS 节点 (必须在实例化 Env 之前)
    rospy.init_node('sac_train_node', disable_signals=True) # disable_signals允许Ctrl+C退出

    # 2. 创建环境
    env = MyRobotEnv()
    
    # 3. 创建 SAC 模型
    # MlpPolicy: 适用于雷达这种数值型输入
    # ent_coef='auto': SAC 的精髓，自动调整探索程度
    model = SAC(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log="./sac_tensorboard/",
        learning_rate=3e-4,
        buffer_size=100000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        ent_coef='auto'
    )

    # 4. 设置自动保存 (每 5000 步保存一次模型)
    checkpoint_callback = CheckpointCallback(
        save_freq=5000, 
        save_path='./models/', 
        name_prefix='sac_nav'
    )

    print("-------------- Start Training --------------")
    try:
        # 开始训练，total_timesteps 根据你的时间决定，通常几十万步起
        model.learn(total_timesteps=100000, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("Training interrupted manually.")
    
    # 5. 保存最终模型
    model.save("sac_nav_final")
    print("Model saved. Training finished.")

if __name__ == '__main__':
    main()