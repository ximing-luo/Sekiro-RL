import torch
import torch.nn as nn
import torch.nn.functional as F
from .rms import RMSNorm, RMSNorm2d

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
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class BottleNeck(nn.Module):
    """
    通用瓶颈块 (BottleNeck)
    统一了 ResNet (压缩型) 和 MobileNetV2 (扩张型/倒残差) 的逻辑。
    
    设计要点:
    1. 计算中心化: 关注 mid_channels，即中间 3x3 深度卷积的计算维度。
    2. 架构统一: in -> mid (1x1) -> mid (3x3 DW) -> out (1x1)。
    3. LLM 风格: Pre-Norm (RMSNorm) + Linear Bottleneck (末尾无激活)。
    4. ResNet-D 优化: Shortcut 路径使用 AvgPool 进行平滑降维。
    """
    expansion = 4 # 默认 ResNet 风格
    
    def __init__(self, in_channels, out_channels, mid_channels=None, stride=1):
        super().__init__()
        
        if mid_channels is None:
            mid_channels = out_channels // 4 # 默认压缩 4 倍
            
        self.residual_function = nn.Sequential(
            # 1. 投影层: in -> mid (SiLU)
            nn.utils.spectral_norm(nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)),
            nn.SiLU(inplace=True),
            # 中继归一化：压制由于通道扩张导致的内部数值膨胀
            RMSNorm2d(mid_channels),
            
            # 2. 空间层: mid -> mid (3x3 Depthwise, SiLU)
            nn.utils.spectral_norm(nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, 
                                             groups=mid_channels, padding=1, bias=False)),
            nn.SiLU(inplace=True),
            
            # 3. 输出层: mid -> out (无激活, 线性瓶颈)
            nn.utils.spectral_norm(nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)),
        )
        
        # 残差分支输出归一化 (Post-Norm)
        # 确保残差路径的输出量纲也被物理锁定在 1.0 附近
        self.res_norm = RMSNorm2d(out_channels)
        
        # Shortcut 对齐 (ResNet-D 优化版)
        layers = []
        if stride > 1:
            # 空间下采样: 使用 AvgPool 保持平滑
            layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True))

        if in_channels != out_channels or stride > 1:
            # 通道投影: 仅在通道不匹配或空间变化时添加 1x1 卷积
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            # 既然有了 Norm 保护，可以使用标准正交初始化提高初期特征丰富度
            nn.init.orthogonal_(conv.weight, gain=1.0)
            layers.append(conv)
            # 物理断路器：在捷径末端强制量纲回归
            layers.append(RMSNorm2d(out_channels))

        self.shortcut = nn.Sequential(*layers)

    def forward(self, x):
        identity = self.shortcut(x)
        # 核心改进：双路径输出归一化
        # 1. 计算残差分支并立即归一化
        out = self.res_norm(self.residual_function(x))
        # 2. 与同样归一化过的捷径分支相加 (1.0 + 1.0 ≈ 2.0)
        return out + identity

class InitialFrameConv(nn.Module):
    """
    多帧输入对齐模块 (来自 SimpleDQN)
    """
    def __init__(self, in_channels):
        super().__init__()
        assert in_channels % 3 == 0, "in_channels must be a multiple of 3 (RGB per frame)"
        num_frames = in_channels // 3
        out_channels = num_frames * 32
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=11, stride=4, padding=5, groups=num_frames, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.num_frames = num_frames

    def forward(self, x):
        return self.block(x)
