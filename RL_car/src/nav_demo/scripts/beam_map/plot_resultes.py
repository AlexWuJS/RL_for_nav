import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

def plot_training_results(log_dir):
    # 找到最新的日志文件夹
    subdirs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
    # 按时间排序，取最新的一个
    latest_subdir = max(subdirs, key=os.path.getmtime)
    print(f"正在读取日志: {latest_subdir}")

    # 加载日志数据
    event_acc = EventAccumulator(latest_subdir)
    event_acc.Reload()

    # 获取标量数据 (Tags)
    tags = event_acc.Tags()['scalars']
    
    # 我们主要关心的指标
    # metrics = ['rollout/ep_rew_mean', 'rollout/ep_len_mean', 'train/loss']
    metrics = ['rollout/ep_rew_mean', 'rollout/ep_len_mean', 'train/actor_loss','train/critic_loss']
    
    plt.figure(figsize=(15, 5))

    for i, metric in enumerate(metrics):
        if metric in tags:
            # 提取数据
            events = event_acc.Scalars(metric)
            steps = [x.step for x in events]
            values = [x.value for x in events]
            
            # 画图
            plt.subplot(1, 4, i+1)
            plt.plot(steps, values, label=metric, color='b')
            plt.xlabel('Timesteps')
            plt.title(metric)
            plt.grid(True)
        else:
            print(f"警告: 在日志中未找到 {metric}")

    plt.tight_layout()
    plt.savefig('training_sac_result.png') # 保存为图片
    print("图表已保存为 training_sac_result.png")
    plt.show() # 显示图片

if __name__ == "__main__":
    # 你的日志目录
    # log_directory = "./ppo_car_log/"
    log_directory = "./sac_car_log/"
    
    try:
        plot_training_results(log_directory)
    except ImportError:
        print("错误：缺少绘图库，请运行: pip install matplotlib pandas")
    except Exception as e:
        print(f"发生错误: {e}")