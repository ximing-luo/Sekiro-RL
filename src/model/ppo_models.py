import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from .resnet import BasicBlock, BottleNeck

class SekiroStableExtractor(BaseFeaturesExtractor):
    """
    更稳定的卷积神经网络特征提取器。
    
    改进点：
    1. 采用 ResNet 架构：使用残差连接提高深层网络训练稳定性。
    2. 输入归一化：将 0-255 的像素值缩放到 0-1。
    3. BatchNorm2d：在卷积层后添加批归一化，加速收敛并提高稳定性。
    4. LayerNorm：在全连接层前添加层归一化，防止梯度爆炸。
    5. 紧凑的结构：参考 Isaac Lab 的设计风格。
    """
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0] # 通常为 3 (RGB)
        
        self.in_channels = 16
        
        # 参照 resnet.py 中的 DQN 结构构造极简残差网络 (超轻量化)
        # 1. 输入对齐模块：使用步幅 2 进行初步下采样，显著降低显存占用
        # 图像：240x135 -> 60x34
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=n_input_channels, out_channels=n_input_channels, kernel_size=3, stride=4, groups=n_input_channels, padding=1, bias=False),
            nn.Conv2d(in_channels=n_input_channels, out_channels=16, kernel_size=1, padding=0, bias=False), 
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # 2. 残差层阶段 (增加深度：每个阶段使用 2 个残差块)
        self.conv2_x = self._make_layer(BottleNeck, 64, 2, 2)
        self.conv3_x = self._make_layer(BottleNeck, 128, 4, 2)
        self.conv4_x = self._make_layer(BottleNeck, 256, 4   , 2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.cnn = nn.Sequential(
            self.conv1,
            self.conv2_x,
            self.conv3_x,
            self.conv4_x,
            self.avg_pool,
            nn.Flatten(),
        )

        # 动态计算卷积后的输出维度
        with torch.no_grad():
            # 假设输入是 [1, C, H, W]
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            # 注意：这里模拟前向传播时不需要归一化，只为了拿维度
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.LayerNorm(features_dim), # 在全连接层使用 LayerNorm 增加稳定性
            nn.ReLU(inplace=True)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        注意：SB3 的 CnnPolicy 默认会将 uint8 图像缩放到 [0, 1]。
        如果输入已经是 float 且在 [0, 255]，我们需要手动缩放。
        """
        # 确保输入在 [0, 1] 范围内（如果是 uint8，SB3 通常已经处理过）
        if observations.dtype == torch.uint8:
            observations = observations.float() / 255.0
            
        return self.linear(self.cnn(observations))

    def _make_layer(self, block, out_channels, num_block, stride):
        """构造一个残差层阶段。"""
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
