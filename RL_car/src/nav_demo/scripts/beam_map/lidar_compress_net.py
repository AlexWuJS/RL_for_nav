import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class LidarProcessor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        # 1. 获取输入总维度 (例如 4016)
        n_input_features = observation_space.shape[0]
        
        # 2. 定义基础单帧维度
        self.base_lidar_dim = 1000
        self.base_frenet_dim = 4
        
        # 3. 自动计算堆叠了多少帧 (例如 4016 / 1004 = 4)
        self.n_stack = n_input_features // (self.base_lidar_dim + self.base_frenet_dim)
        
        # 4. 计算堆叠后的总特征维度
        self.total_lidar_dim = self.base_lidar_dim * self.n_stack   # 1000 * 4 = 4000
        self.total_frenet_dim = self.base_frenet_dim * self.n_stack # 4 * 4 = 16
        
        # === 分支 A：雷达处理网络 ===
        # 注意：这里的输入变成了堆叠后的总维度 4000
        self.lidar_net = nn.Sequential(
            nn.Linear(self.total_lidar_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 240),
            nn.ReLU()
        )

        # === 分支 B：导航处理网络 ===
        # 注意：这里的输入变成了堆叠后的总维度 16
        self.frenet_net = nn.Sequential(
            nn.Linear(self.total_frenet_dim, 16),
            nn.ReLU()
        )

    def forward(self, observations):
        # observations 形状为 (Batch, 4016)
        batch_size = observations.shape[0]

        # 1. 重新塑形 (Reshape) 为立体结构，方便安全切分
        # 将 (Batch, 4016) 转换成 (Batch, 4帧, 1004)
        obs_reshaped = observations.view(batch_size, self.n_stack, self.base_lidar_dim + self.base_frenet_dim)

        # 2. 在最后一个维度上精准切割
        # lidar_data 形状: (Batch, 4帧, 1000)
        lidar_data = obs_reshaped[:, :, :self.base_lidar_dim]    
        # frenet_data 形状: (Batch, 4帧, 4)
        frenet_data = obs_reshaped[:, :, self.base_lidar_dim:]   

        # 3. 把切好的数据重新展平，喂给各自的线性层
        # 形状变成 (Batch, 4000)
        lidar_data = lidar_data.reshape(batch_size, -1)     
        # 形状变成 (Batch, 16)
        frenet_data = frenet_data.reshape(batch_size, -1)   

        # 4. 分别通过各自的网络
        lidar_features = self.lidar_net(lidar_data)
        frenet_features = self.frenet_net(frenet_data)

        # 5. 拼接输出 (240 + 16 = 256)
        return torch.cat((lidar_features, frenet_features), dim=1)