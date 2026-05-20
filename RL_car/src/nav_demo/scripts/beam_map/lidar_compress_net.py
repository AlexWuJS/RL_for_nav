import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LidarProcessor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        n_input_features = observation_space.shape[0]

        self.base_lidar_dim = 400
        self.base_frenet_dim = 4
        self.base_delta_dim = 2  # Plan B: MPPI delta [surge, yaw]

        per_frame = self.base_lidar_dim + self.base_frenet_dim + self.base_delta_dim
        self.n_stack = n_input_features // per_frame

        self.total_lidar_dim = self.base_lidar_dim * self.n_stack
        self.total_frenet_dim = (self.base_frenet_dim + self.base_delta_dim) * self.n_stack

        self.lidar_net = nn.Sequential(
            nn.Linear(self.total_lidar_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 240),
            nn.ReLU(),
        )

        self.frenet_net = nn.Sequential(
            nn.Linear(self.total_frenet_dim, 16),
            nn.ReLU(),
        )

    def forward(self, observations):
        batch_size = observations.shape[0]
        per_frame = self.base_lidar_dim + self.base_frenet_dim + self.base_delta_dim
        obs_reshaped = observations.view(batch_size, self.n_stack, per_frame)

        lidar_data = obs_reshaped[:, :, :self.base_lidar_dim]
        # frenet + MPPI delta (delta goes through the same pathway)
        frenet_data = obs_reshaped[:, :, self.base_lidar_dim:]

        lidar_data = lidar_data.reshape(batch_size, -1)
        frenet_data = frenet_data.reshape(batch_size, -1)

        lidar_features = self.lidar_net(lidar_data)
        frenet_features = self.frenet_net(frenet_data)

        return torch.cat((lidar_features, frenet_features), dim=1)
