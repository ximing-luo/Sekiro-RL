import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from src.framework.ashina.models.resnet import BasicBlock, BottleNeck

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (通道注意力机制)
    通过全局平均池化捕捉通道间的全局统计信息，学习通道重要性权重。
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # 1. Squeeze: (B, C, H, W) -> (B, C)
        y = self.avg_pool(x).view(b, c)
        # 2. Excitation: (B, C) -> (B, C, 1, 1)
        y = self.fc(y).view(b, c, 1, 1)
        # 3. Reweight
        return x * y.expand_as(x)

class SekiroStableExtractor(BaseFeaturesExtractor):
    """
    更稳定的卷积神经网络特征提取器。
    
    改进点：
    1. 采用 ResNet 架构：使用残差连接提高深层网络训练稳定性。
    2. 输入归一化：将 0-255 的像素值缩放到 0-1。
    3. GroupNorm：使用组归一化替代 BatchNorm，增强在 RL 采样阶段（Batch Size=1）的稳定性。
    4. LayerNorm：在全连接层前添加层归一化，防止梯度爆炸。
    5. 采用 BasicBlock 替代 BottleNeck，减少参数并增加训练稳定性。
    """
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0] # 通常为 3 (RGB)
        
        self.in_channels = 32
        
        # 1. 输入模块：降低下采样攻击性，保留更多空间细节
        # 引入谱归一化和 SEBlock 增强特征提取的辨识度
        # 图像：240x135 -> 120x68
        self.conv1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)),
            SEBlock(32),
            nn.SiLU(inplace=True)
        )
        
        # 2. 残差层阶段 (使用 BasicBlock)
        # 120x68 -> 60x34
        self.layer1 = self._make_layer(BasicBlock, 64, 2, 2)
        # 60x34 -> 30x17
        self.layer2 = self._make_layer(BasicBlock, 128, 2, 2)
        # 30x17 -> 15x9
        self.layer3 = self._make_layer(BasicBlock, 256, 2, 2)
        
        # 3. 破局改进：保留空间语义的聚合方式
        # 使用 1x1 卷积进行跨通道语义整合，将 256 通道精炼为 64 通道
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256 * BasicBlock.expansion, 64, kernel_size=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True)
        )
        
        # 使用较大的池化目标尺寸 (4x7)，保留空间拓扑结构
        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 7))
        
        self.cnn = nn.Sequential(
            self.conv1,
            self.layer1,
            self.layer2,
            self.layer3,
            self.bottleneck,
            self.spatial_pool,
            nn.Flatten(),
        )

        # 动态计算卷积后的输出维度 (64 * 4 * 7 = 1792)
        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample_input).shape[1]

        # 4. 强化线性层语义分析能力
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(inplace=True),
            nn.Linear(1024, features_dim),
            nn.LayerNorm(features_dim)
        )
        
        # 5. 初始化权重 (正交初始化 Orthogonal)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # 针对 SiLU 的初始化增益修复：SiLU 近似于 LeakyReLU
                # PyTorch 不原生支持 calculate_gain('SiLU')
                gain = nn.init.calculate_gain('leaky_relu', 0.01)
                
                # 兼容谱归一化：如果使用了 spectral_norm，权重存储在 weight_orig 中
                target_weight = m.weight_orig if hasattr(m, 'weight_orig') else m.weight
                nn.init.orthogonal_(target_weight, gain=gain)
                
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dtype == torch.uint8:
            observations = observations.float() / 255.0
            
        return self.linear(self.cnn(observations))

    def _make_layer(self, block, out_channels, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

