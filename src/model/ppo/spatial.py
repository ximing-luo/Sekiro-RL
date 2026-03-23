import torch
import torch.nn as nn
from typing import Dict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from src.model.components import BasicBlock, BottleNeck, SEBlock
from gymnasium import spaces
import torch.nn.functional as F

class Focus(nn.Module):
    """
    Focus layer (Space-to-Depth): 将空间信息切片并堆叠到通道维度。
    作用：无损下采样，大幅降低计算量。
    输入：(B, C, H, W) -> 输出：(B, C*4, H/2, W/2)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2]
        ], dim=1)

class SekiroStableExtractor(BaseFeaturesExtractor):
    """
    更稳定的卷积神经网络特征提取器。
    
    改进点：
    1. 采用 ResNet 架构：使用残差连接提高深层网络训练稳定性。
    2. 输入归一化：将 0-255 的像素值缩放到 0-1。
    3. GroupNorm：使用组归一化替代 BatchNorm，增强在 RL 采样阶段（Batch Size=1）的稳定性。
    4. LayerNorm：在全连接层前添加层归一化，防止梯度爆炸。
    5. 采用 BasicBlock 替代 BottleNeck，减少参数并增加训练稳定性。
    6. [New] 极限压榨架构：Focus + Heavy Middle + Capped Width。
    """
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0] # 通常为 3 (RGB)
        
        self.in_channels = 32  # Stem 输出通道数
        
        # 1. 输入模块：Focus + 1x1 卷积
        # 预处理：Resize (240x135) -> Focus -> (120x67)
        # 通道：3 -> 12 -> 64
        self.stem = nn.Sequential(
            Focus(),
            nn.Conv2d(n_input_channels * 4, 32, kernel_size=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True)
        )
        
        # 2. 残差层阶段 (Heavy Middle 策略)
        # Stage 1: 120x67 -> 120x67 (保持分辨率)
        self.layer1 = self._make_layer(BasicBlock, 32, 1, 1)
        # Stage 2: 120x67 -> 60x34
        self.layer2 = self._make_layer(BasicBlock, 64, 2, 2)
        # Stage 3: 60x34 -> 30x17 (思考核心)
        self.layer3 = self._make_layer(BasicBlock, 128, 4, 2)
        # Stage 4: 30x17 -> 15x9 (宽度克制，保持 256)
        self.layer4 = self._make_layer(BasicBlock, 128, 2, 2)
        
        # 3. 空间感知池化
        # 放弃 1x1 Global Pool，保留 (3, 5) 网格
        # 3行：上/中/下段
        # 5列：左/中左/中/中右/右
        self.spatial_pool = nn.AdaptiveAvgPool2d((3, 5))
        
        self.cnn = nn.Sequential(
            self.stem,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.spatial_pool,
            nn.Flatten(),
        )

        # 动态计算卷积后的输出维度 (256 * 3 * 5 = 3840)
        # 注意：这里我们手动模拟一次 Resize 后的 forward
        with torch.no_grad():
            # 假设输入是任意大小，经过 resize 后变为 240x136
            # 但为了计算 shape，我们需要模拟 resize 后的 tensor
            dummy_resized = torch.zeros(1, n_input_channels, 136, 240)
            n_flatten = self.cnn(dummy_resized).shape[1]

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
                gain = nn.init.calculate_gain('leaky_relu', 0.01)
                target_weight = m.weight_orig if hasattr(m, 'weight_orig') else m.weight
                nn.init.orthogonal_(target_weight, gain=gain)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 0. 归一化 (兼容 uint8 和 float 输入)
        if observations.dtype == torch.uint8:
            observations = observations.float() / 255.0
        elif observations.max() > 1.0:
            observations = observations.float() / 255.0
            
        # 1. 强制 Resize 到 240x136 (1080p 的 1/8 附近，微调为偶数)
        # 这是为了确保 Focus 层能获得固定大小的输入，并大幅降低计算量
        # 高度 135 -> 136 (偶数)，避免 Focus 切片时奇偶行数量不一致 (68 vs 67)
        # align_corners=False 对于下采样通常更好
        # x = F.interpolate(observations, size=(136, 240), mode='bilinear', align_corners=False)
            
        return self.linear(self.cnn(observations))

    def _make_layer(self, block, out_channels, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

class SekiroMultiInputExtractor(BaseFeaturesExtractor):
    """
    多输入特征提取器 (Isaac Lab 风格)：
    1. 视觉分支：使用 SekiroStableExtractor 处理图像。
    2. 遥测分支：直接处理数值特征。
    3. 特征融合：拼接视觉与遥测特征并映射到统一维度。
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        # A. 视觉分支 (使用原有的稳定提取器)
        self.image_extractor = SekiroStableExtractor(observation_space["policy"], features_dim=features_dim)
        
        # B. 遥测分支
        telemetry_dim = observation_space["telemetry"].shape[0]
        
        # C. 特征融合层
        # 融合视觉 (features_dim) 和 遥测 (telemetry_dim)
        self.fusion = nn.Sequential(
            nn.Linear(features_dim + telemetry_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.SiLU(inplace=True)
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 1. 提取视觉特征
        img_feats = self.image_extractor(observations["policy"])
        
        # 2. 获取遥测特征
        tele_feats = observations["telemetry"]
        
        # 3. 拼接并融合
        combined = torch.cat([img_feats, tele_feats], dim=1)
        return self.fusion(combined)

