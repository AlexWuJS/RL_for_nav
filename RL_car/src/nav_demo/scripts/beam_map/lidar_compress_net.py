# 你的自定义网络文件 (例如 custom_net.py)

import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch

# 建议把类名拼写改对，看起来舒服点
class LidarProcessor(BaseFeaturesExtractor): 
    # 这里的参数名必须和 train.py 中字典的 key 完全一致
    def __init__(self, observation_space, features_dim=64): # <--- 必须是 features_dim
        super().__init__(observation_space, features_dim)
        
        n_input_features = observation_space.shape[0]
        
        self.compress_net = nn.Sequential(
            nn.Linear(n_input_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim), # 使用这个变量
            nn.ReLU()
        )

    def forward(self, observations):
        return self.compress_net(observations)