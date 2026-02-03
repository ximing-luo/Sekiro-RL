import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from .backbone.gate import GatedMLP
from .backbone.rms import RMSNorm

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
    def __init__(self, in_channels, out_channels, mid_channels, stride=1):
        super().__init__()
        
        self.residual_function = nn.Sequential(
            # 1. 投影层: in -> mid (SiLU)
            nn.utils.spectral_norm(nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)),
            nn.SiLU(inplace=True),
            
            # 2. 空间层: mid -> mid (3x3 Depthwise, SiLU)
            nn.utils.spectral_norm(nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, 
                                             groups=mid_channels, padding=1, bias=False)),
            nn.SiLU(inplace=True),
            
            # 3. 输出层: mid -> out (无激活, 线性瓶颈)
            nn.utils.spectral_norm(nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)),
        )
        
        # Pre-Norm
        self.norm = RMSNorm(in_channels)
        
        # Shortcut 对齐 (ResNet-D 优化版)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            if stride > 1:
                # 使用平均池化进行平滑降维，避免 1x1 卷积步长直接丢弃信息
                # 添加 ceil_mode=True 确保与 Conv2d 的步长输出尺寸对齐
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True),
                    nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
                )
            else:
                # 仅通道对齐
                self.shortcut = nn.Sequential(
                    nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
                )

    def forward(self, x):
        identity = self.shortcut(x)
        # 仅对分支信号归一化，主干保持纯净
        out = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = self.residual_function(out)
        return out + identity

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

class StaticBranch(nn.Module):
    """
    静态分支 (Spatial Stream)
    封装 Stem (7x7->5x5), Backbone (ResNet Stages) 和 Head。
    针对 480P 输入设计。
    """
    def __init__(self, in_channels, base_channels):
        super().__init__()
        c = base_channels
        
        # --- 1. Stem (前哨站) ---
        # 第一层: 7x7 步长 2, 无激活
        self.conv1 = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, 2, kernel_size=7, stride=2, padding=3, bias=True)
        )
        # 第二层: 5x5 步长 2, SiLU
        self.conv2 = nn.utils.spectral_norm(
            nn.Conv2d(2, c, kernel_size=5, stride=2, padding=2, bias=True)
        )
        self.act = nn.SiLU(inplace=True)
        self.norm = RMSNorm(c)
        self.se = SEBlock(c)
        
        # --- 2. Backbone (骨架) ---
        self.in_channels = c
        self.backbone = nn.Sequential(
            # Stage 1: 120x68 (基础型: mid=out)
            self._make_layer(c, c, 1, 1),
            # Stage 2: 60x34 (基础型)
            self._make_layer(c * 4, c * 4, 2, 2),
            # Stage 3: 30x17 (压缩型: mid=out/4)
            self._make_layer(c * 16, c * 4, 1, 2),
            # Stage 4: 15x9 (压缩型)
            self._make_layer(c * 32, c * 8, 1, 2),
            # Stage 5: 7x4 (压缩型)
            self._make_layer(c * 64, c * 16, 1, 2),
            # Stage 6: 4x2 (压缩型)
            self._make_layer(c * 32, c * 16, 1, 2),
        )
        
        # --- 3. Head (压缩与平坦化) ---
        self.head = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(c * 32, c * 8, kernel_size=1, bias=True)),
            nn.SiLU(inplace=True),
            nn.Flatten()
        )

    def _make_layer(self, out_channels, mid_channels, num_block, stride):
        """
        构建 BottleNeck 层
        """
        layers = []
        layers.append(BottleNeck(self.in_channels, out_channels, mid_channels, stride))
        self.in_channels = out_channels
        for _ in range(num_block - 1):
            layers.append(BottleNeck(self.in_channels, out_channels, mid_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.act(self.conv2(x))
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.se(x)
        # Backbone & Head
        x = self.backbone(x)
        return self.head(x)

class DynamicBranch(nn.Module):
    """
    动态分支 (Temporal Stream)
    封装 Stem (3x3->5x5), Backbone (BasicBlock 深度堆叠) 和 Head。
    针对帧差信号设计。
    """
    def __init__(self, in_channels, base_channels):
        super().__init__()
        c = base_channels
        
        # --- 1. Stem (前哨站) ---
        # 第一层: 3x3 步长 2, SiLU
        self.conv1 = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, 2, kernel_size=3, stride=2, padding=1, bias=True)
        )
        self.act1 = nn.SiLU(inplace=True)
        # 第二层: 5x5 步长 2, SiLU
        self.conv2 = nn.utils.spectral_norm(
            nn.Conv2d(2, c, kernel_size=5, stride=2, padding=2, bias=True)
        )
        self.act2 = nn.SiLU(inplace=True)
        self.norm = RMSNorm(c)
        self.se = SEBlock(c)
        
        # --- 2. Backbone (骨架) ---
        self.in_channels = c
        self.backbone = nn.Sequential(
            # Stage 2: 60x34 (基础型)
            self._make_layer(c * 4, c * 4, 2, 2),
            # Stage 3: 30x17 (基础型)
            self._make_layer(c * 8, c * 8, 2, 2),
            # Stage 4: 15x9 (扩张型/倒残差 - 深度加固 4 层)
            self._make_layer(c * 16, c * 64, 4, 2),
            # Stage 5: 8x5 (扩张型/倒残差 - 额外深化下采样)
            self._make_layer(c * 8, c * 64, 2, 2),
        )
        
        # --- 3. Head (压缩与平坦化) ---
        self.head = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(c * 8, c * 4, kernel_size=1, bias=True)),
            nn.SiLU(inplace=True),
            nn.Flatten()
        )

    def _make_layer(self, out_channels, mid_channels, num_block, stride):
        """
        构建 BottleNeck 层
        """
        layers = []
        layers.append(BottleNeck(self.in_channels, out_channels, mid_channels, stride))
        self.in_channels = out_channels
        for _ in range(num_block - 1):
            layers.append(BottleNeck(self.in_channels, out_channels, mid_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Stem
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.se(x)
        # Backbone & Head
        x = self.backbone(x)
        return self.head(x)

class SekiroMADSExtractor(BaseFeaturesExtractor):
    """
    Deep Asymmetric MADS (Motion-Aware Dual Stream) 特征提取器。
    
    设计理念：
    1. 静态流 (Spatial)：5阶段深层架构，480P 输入。
    2. 动态流 (Temporal)：轻量级长时感知架构，帧差输入。
    3. 门控融合：LLM 风格 SwiGLU 蒸馏与 Refiner 思考块。
    """
    def __init__(self, observation_space, features_dim=512, base_channels=4):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0] # 12 通道
        c = base_channels
        
        # 使用全封装的分支类重构
        self.spatial_cnn = StaticBranch(3, c)
        self.temporal_cnn = DynamicBranch(3, c)

        # 动态计算展平后的维度
        with torch.no_grad():
            # 使用 observation_space 的实际形状进行预计算
            sample_input = torch.zeros(1, *observation_space.shape)
            s_out = self.spatial_cnn(sample_input[:, -3:]).shape[1]
            d1 = self._rgb_to_gray(sample_input[:, 9:12] - sample_input[:, 6:9])
            d2 = self._rgb_to_gray(sample_input[:, 6:9] - sample_input[:, 3:6])
            d3 = self._rgb_to_gray(sample_input[:, 3:6] - sample_input[:, 0:3])
            diff_input = torch.cat([d1, d2, d3], dim=1)
            t_out = self.temporal_cnn(diff_input).shape[1]

        # --- 3. 门控融合流水线 (LLM 风格 SwiGLU) ---
        # 3.1 独立蒸馏层：使用紧凑型 1536 中间维 (1.5x output_dim)
        self.spatial_distill = GatedMLP(s_out, 512, intermediate_size=768)
        self.temporal_distill = GatedMLP(t_out, 512, intermediate_size=768)
        
        # 3.2 融合层：2048 -> 768 -> 512
        self.fusion = GatedMLP(1024, features_dim, intermediate_size=768)
        
        # 3.3 Refiner (深度思考块)：512 -> 768 -> 512
        self.refiner = GatedMLP(features_dim, features_dim, intermediate_size=768)
        
        # 3.4 最终归一化 (防止输出特征范数爆炸)
        self.final_norm = RMSNorm(features_dim)
        
        self._initialize_weights()

    def _rgb_to_gray(self, rgb):
        # 使用心理学权重进行灰度转换
        return 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # 统一获取权重：优先处理 spectral_norm 包装的 weight_orig
                weight = getattr(m, 'weight_orig', m.weight)
                # 设置增益：卷积层使用 ReLU 增益，线性层默认使用 1.0
                gain = nn.init.calculate_gain('relu') if isinstance(m, nn.Conv2d) else 1.0
                nn.init.orthogonal_(weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dtype == torch.uint8:
            observations = observations.float() / 255.0
            
        # 核心鲁棒性增强：自动对齐通道数
        # 如果输入通道不足 12 (例如 3 通道)，重复 4 次以匹配 MADS 逻辑
        if observations.shape[1] < 12:
            n_repeat = (12 + observations.shape[1] - 1) // observations.shape[1]
            observations = observations.repeat(1, n_repeat, 1, 1)[:, :12]

        # 1. 颜色抖动增强
        if self.training:
            observations = self._random_shift(observations)
            
        # 2. 静态流
        spatial_feat = self.spatial_cnn(observations[:, -3:])
        
        # 3. 动态流：灰度帧差
        d1 = self._rgb_to_gray(observations[:, 9:12] - observations[:, 6:9])
        d2 = self._rgb_to_gray(observations[:, 6:9] - observations[:, 3:6])
        d3 = self._rgb_to_gray(observations[:, 3:6] - observations[:, 0:3])
        diff_input = torch.cat([d1, d2, d3], dim=1)
        temporal_feat = self.temporal_cnn(diff_input)
        
        # 4. 融合流水线
        # 4.1 独立蒸馏
        s_distilled = self.spatial_distill(spatial_feat)
        t_distilled = self.temporal_distill(temporal_feat)
        
        # 4.2 拼接融合
        combined = torch.cat([s_distilled, t_distilled], dim=1)
        base_features = self.fusion(combined)
        
        # 4.3 深度思考 (Refiner)
        features = self.refiner(base_features)
        
        # 5. 最终归一化输出
        # return features
        return self.final_norm(features)

    def _random_shift(self, imgs, pad=4):
        n, c, h, w = imgs.shape
        # 颜色抖动
        brightness = 0.8 + torch.rand(n, 1, 1, 1, device=imgs.device) * 0.4
        contrast = 0.8 + torch.rand(n, 1, 1, 1, device=imgs.device) * 0.4
        imgs = imgs * brightness
        imgs = (imgs - imgs.mean(dim=[2, 3], keepdim=True)) * contrast + imgs.mean(dim=[2, 3], keepdim=True)
        imgs = torch.clamp(imgs, 0, 1)

        # 随机位移
        padding = (pad, pad, pad, pad)
        imgs = F.pad(imgs, padding, mode='replicate')
        shift_h = torch.randint(0, 2 * pad + 1, (n,), device=imgs.device)
        shift_w = torch.randint(0, 2 * pad + 1, (n,), device=imgs.device)
        new_imgs = torch.zeros(n, c, h, w, device=imgs.device)
        for i in range(n):
            new_imgs[i] = imgs[i, :, shift_h[i]:shift_h[i]+h, shift_w[i]:shift_w[i]+w]
        return new_imgs

