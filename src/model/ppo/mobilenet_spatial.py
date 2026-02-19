import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

class InvertedResidual(nn.Module):
    """
    MobileNetV2 Inverted Residual Block
    Structure: 1x1 (Expand) -> 3x3 DW (Spatial) -> 1x1 (Project)
    """
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # 1. 1x1 Conv: Expansion (ReLU6)
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.GroupNorm(min(8, hidden_dim), hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        
        # 2. 3x3 Depthwise Conv (ReLU6)
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim),
            nn.ReLU6(inplace=True),
            
            # 3. 1x1 Conv: Projection (Linear)
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetSpatial(BaseFeaturesExtractor):
    """
    MobileNetV2 style architecture optimized for Sekiro-RL (240x136 input).
    Focus: Parameter Efficiency, Inverted Residuals.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        # Stage 0: Stem (Rapid Downsampling)
        # 240x136 -> 60x34 (Stride 4)
        # Use 4x4 conv with stride 4
        self.stem = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=4, stride=4, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU6(inplace=True)
        )
        
        # Inverted Residual Blocks configuration
        # t: expand_ratio, c: out_channels, n: num_blocks, s: stride
        interverted_residual_setting = [
            # Stage 1: 60x34 -> 60x34
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1], # Usually stride 2 here, but we want to keep 60x34 a bit? Or drop to 30x17?
                           # ResNetSpatial dropped to 30x17 at Layer 2.
                           # Let's match ResNetSpatial stages.
            # Stage 2: 60x34 -> 30x17
            [6, 32, 3, 2], 
            # Stage 3: 30x17 -> 15x8
            [6, 64, 4, 2],
            # Stage 4: 15x8 -> 7x4
            [6, 96, 3, 2],
            [6, 160, 3, 1], # Keep 7x4, expand channels
            [6, 320, 1, 1], # Project to 320
        ]

        # Building Inverted Residual blocks
        input_channel = 32
        features = []
        for t, c, n, s in interverted_residual_setting:
            output_channel = c
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        
        self.features = nn.Sequential(*features)

        # Building last several layers
        # 1x1 Conv to expand to 1280
        self.conv_last = nn.Sequential(
            nn.Conv2d(input_channel, 1280, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, 1280), 1280),
            nn.ReLU6(inplace=True)
        )

        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(1280, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU6(inplace=True) # or SiLU
        )

        # Weight Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.stem(observations)
        x = self.features(x)
        x = self.conv_last(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
