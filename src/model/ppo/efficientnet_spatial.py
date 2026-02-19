import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25):
        super().__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        
        hidden_dim = int(in_channels * expand_ratio)
        reduced_dim = max(1, int(in_channels * se_ratio))
        
        layers = []
        
        # 1. Expansion
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, bias=False))
            layers.append(nn.GroupNorm(min(8, hidden_dim), hidden_dim))
            layers.append(nn.SiLU())
            
        # 2. Depthwise Convolution
        # Padding calculation: (kernel_size - 1) // 2
        padding = (kernel_size - 1) // 2
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                                padding=padding, groups=hidden_dim, bias=False))
        layers.append(nn.GroupNorm(min(8, hidden_dim), hidden_dim))
        layers.append(nn.SiLU())
        
        # 3. Squeeze and Excitation
        layers.append(SqueezeExcitation(hidden_dim, reduced_dim))
        
        # 4. Projection
        layers.append(nn.Conv2d(hidden_dim, out_channels, 1, bias=False))
        layers.append(nn.GroupNorm(min(8, out_channels), out_channels))
        
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

class EfficientNetSpatial(BaseFeaturesExtractor):
    """
    EfficientNet-B0 style architecture adapted for Sekiro-RL.
    Uses MBConv blocks with Squeeze-and-Excitation.
    Adapted strides for 240x136 input.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
        # Stem: Rapid downsampling (Stride 4) to 60x34
        self.stem = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            # Additional downsampling to match MobileNetSpatial's aggressive stem
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        
        # EfficientNet-B0 Config (Adapted Strides)
        # operator | kernel | in | out | se | stride | expand
        # MBConv1  | 3x3    | 32 | 16  | 0.25 | 1    | 1
        # MBConv6  | 3x3    | 16 | 24  | 0.25 | 2    | 6
        # MBConv6  | 5x5    | 24 | 40  | 0.25 | 2    | 6
        # MBConv6  | 3x3    | 40 | 80  | 0.25 | 1    | 6  (Stride 2 in orig)
        # MBConv6  | 5x5    | 80 | 112 | 0.25 | 1    | 6
        # MBConv6  | 5x5    | 112| 192 | 0.25 | 2    | 6
        # MBConv6  | 3x3    | 192| 320 | 0.25 | 1    | 6
        
        # settings: [expand, out_c, repeats, stride, kernel]
        settings = [
            [1, 16, 1, 1, 3],
            [6, 24, 2, 2, 3], # -> 30x17
            [6, 40, 2, 2, 5], # -> 15x9
            [6, 80, 3, 1, 3], # -> 15x9 (Kept resolution)
            [6, 112, 3, 1, 5],# -> 15x9
            [6, 192, 4, 2, 5],# -> 8x5
            [6, 320, 1, 1, 3] # -> 8x5
        ]
        
        layers = []
        in_channels = 32
        for expand, out_c, repeats, stride, kernel in settings:
            for i in range(repeats):
                s = stride if i == 0 else 1
                layers.append(MBConvBlock(in_channels, out_c, kernel, s, expand))
                in_channels = out_c
                
        self.features = nn.Sequential(*layers)
        
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 1280, 1, bias=False),
            nn.GroupNorm(min(8, 1280), 1280),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, features_dim),
            nn.LayerNorm(features_dim),
            nn.SiLU()
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.stem(observations)
        x = self.features(x)
        x = self.head(x)
        return x
