import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

class StandardBottleneck(nn.Module):
    """
    Standard ResNet Bottleneck Block
    Structure: 1x1 (Compress) -> 3x3 (Spatial) -> 1x1 (Expand)
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()
        width = int(out_channels * (base_width / 64.))
        
        # 1. 1x1 Conv: Compression
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(min(8, width), width) # Use GroupNorm for RL
        
        # 2. 3x3 Conv: Spatial
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(min(8, width), width)
        
        # 3. 1x1 Conv: Expansion
        self.conv3 = nn.Conv2d(width, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(min(8, out_channels * self.expansion), out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(min(8, out_channels * self.expansion), out_channels * self.expansion)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.shortcut(identity)
        out = self.relu(out)

        return out

class ResNetSpatial(BaseFeaturesExtractor):
    """
    ResNet-50 style architecture optimized for Sekiro-RL (240x136 input).
    Focus: Pyramid structure, Fast Downsampling.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        # Stage 0: Stem (Rapid Downsampling)
        # 240x136 -> 60x34 (Stride 4)
        self.stem = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.in_channels = 64

        # Stage 1: 60x34 -> 60x34 (No downsampling)
        # Bottleneck: 64 -> 64 (internal) -> 256 (out)
        self.layer1 = self._make_layer(StandardBottleneck, 64, 3, stride=1)

        # Stage 2: 60x34 -> 30x17
        # Bottleneck: 128 -> 128 (internal) -> 512 (out)
        self.layer2 = self._make_layer(StandardBottleneck, 128, 4, stride=2)

        # Stage 3: 30x17 -> 15x9
        # Bottleneck: 256 -> 256 (internal) -> 1024 (out)
        self.layer3 = self._make_layer(StandardBottleneck, 256, 6, stride=2)
        
        # Stage 4: 15x9 -> 8x5
        # Bottleneck: 512 -> 512 (internal) -> 2048 (out)
        self.layer4 = self._make_layer(StandardBottleneck, 512, 3, stride=2)

        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # MLP Head
        self.linear = nn.Sequential(
            nn.Linear(2048 * StandardBottleneck.expansion // 4, features_dim), # 2048
            nn.LayerNorm(features_dim),
            nn.ReLU(inplace=True)
        )
        
        # Weight Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.stem(observations)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x
