import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from ..components import RMSNorm, GatedMLP, BottleNeck, SEBlock

class StaticBranch(nn.Module):
    """
    静态分支 (Spatial Stream)
    """
    def __init__(self, in_channels, base_channels):
        super().__init__()
        c = base_channels
        
        self.conv1 = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, 2, kernel_size=7, stride=2, padding=3, bias=True)
        )
        self.conv2 = nn.utils.spectral_norm(
            nn.Conv2d(2, c, kernel_size=5, stride=2, padding=2, bias=True)
        )
        self.act = nn.SiLU(inplace=True)
        self.norm = RMSNorm(c)
        self.se = SEBlock(c)
        
        self.in_channels = c
        self.backbone = nn.Sequential(
            self._make_layer(c, c, 1, 1),
            self._make_layer(c * 4, c * 4, 2, 2),
            self._make_layer(c * 16, c * 4, 1, 2),
            self._make_layer(c * 32, c * 8, 1, 2),
            self._make_layer(c * 64, c * 16, 1, 2),
            self._make_layer(c * 32, c * 16, 1, 2),
        )
        
        self.head = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(c * 32, c * 8, kernel_size=1, bias=True)),
            nn.SiLU(inplace=True),
            nn.Flatten()
        )

    def _make_layer(self, out_channels, mid_channels, num_block, stride):
        layers = []
        layers.append(BottleNeck(self.in_channels, out_channels, mid_channels, stride))
        self.in_channels = out_channels
        for _ in range(num_block - 1):
            layers.append(BottleNeck(self.in_channels, out_channels, mid_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(self.conv2(x))
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.se(x)
        x = self.backbone(x)
        return self.head(x)

class DynamicBranch(nn.Module):
    """
    动态分支 (Temporal Stream)
    """
    def __init__(self, in_channels, base_channels):
        super().__init__()
        c = base_channels
        
        self.conv1 = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, 2, kernel_size=3, stride=2, padding=1, bias=True)
        )
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = nn.utils.spectral_norm(
            nn.Conv2d(2, c, kernel_size=5, stride=2, padding=2, bias=True)
        )
        self.act2 = nn.SiLU(inplace=True)
        self.norm = RMSNorm(c)
        self.se = SEBlock(c)
        
        self.in_channels = c
        self.backbone = nn.Sequential(
            self._make_layer(c * 4, c * 4, 2, 2),
            self._make_layer(c * 8, c * 8, 2, 2),
            self._make_layer(c * 16, c * 64, 4, 2),
            self._make_layer(c * 8, c * 64, 2, 2),
        )
        
        self.head = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(c * 8, c * 4, kernel_size=1, bias=True)),
            nn.SiLU(inplace=True),
            nn.Flatten()
        )

    def _make_layer(self, out_channels, mid_channels, num_block, stride):
        layers = []
        layers.append(BottleNeck(self.in_channels, out_channels, mid_channels, stride))
        self.in_channels = out_channels
        for _ in range(num_block - 1):
            layers.append(BottleNeck(self.in_channels, out_channels, mid_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.se(x)
        x = self.backbone(x)
        return self.head(x)

class SekiroMADSExtractor(BaseFeaturesExtractor):
    """
    双流 PPO 特征提取器 (MADS)
    """
    def __init__(self, observation_space, features_dim=512, base_channels=4):
        super().__init__(observation_space, features_dim)
        c = base_channels
        
        self.spatial_cnn = StaticBranch(3, c)
        self.temporal_cnn = DynamicBranch(3, c)

        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space.shape)
            s_out = self.spatial_cnn(sample_input[:, -3:]).shape[1]
            d1 = self._rgb_to_gray(sample_input[:, 9:12] - sample_input[:, 6:9])
            d2 = self._rgb_to_gray(sample_input[:, 6:9] - sample_input[:, 3:6])
            d3 = self._rgb_to_gray(sample_input[:, 3:6] - sample_input[:, 0:3])
            diff_input = torch.cat([d1, d2, d3], dim=1)
            t_out = self.temporal_cnn(diff_input).shape[1]

        self.spatial_distill = GatedMLP(s_out, 512, intermediate_size=768)
        self.temporal_distill = GatedMLP(t_out, 512, intermediate_size=768)
        
        self.fusion = GatedMLP(1024, features_dim, intermediate_size=768)
        self.refiner = GatedMLP(features_dim, features_dim, intermediate_size=768)
        self.final_norm = RMSNorm(features_dim)
        
        self._initialize_weights()

    def _rgb_to_gray(self, rgb):
        return 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                weight = getattr(m, 'weight_orig', m.weight)
                gain = nn.init.calculate_gain('relu') if isinstance(m, nn.Conv2d) else 1.0
                nn.init.orthogonal_(weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dtype == torch.uint8:
            observations = observations.float() / 255.0
            
        if observations.shape[1] < 12:
            n_repeat = (12 + observations.shape[1] - 1) // observations.shape[1]
            observations = observations.repeat(1, n_repeat, 1, 1)[:, :12]

        if self.training:
            observations = self._random_shift(observations)
            
        spatial_feat = self.spatial_cnn(observations[:, -3:])
        
        d1 = self._rgb_to_gray(observations[:, 9:12] - observations[:, 6:9])
        d2 = self._rgb_to_gray(observations[:, 6:9] - observations[:, 3:6])
        d3 = self._rgb_to_gray(observations[:, 3:6] - observations[:, 0:3])
        diff_input = torch.cat([d1, d2, d3], dim=1)
        temporal_feat = self.temporal_cnn(diff_input)
        
        s_distilled = self.spatial_distill(spatial_feat)
        t_distilled = self.temporal_distill(temporal_feat)
        
        combined = torch.cat([s_distilled, t_distilled], dim=1)
        base_features = self.fusion(combined)
        features = self.refiner(base_features)
        
        return self.final_norm(features)

    def _random_shift(self, imgs, pad=4):
        n, c, h, w = imgs.shape
        brightness = 0.8 + torch.rand(n, 1, 1, 1, device=imgs.device) * 0.4
        contrast = 0.8 + torch.rand(n, 1, 1, 1, device=imgs.device) * 0.4
        imgs = imgs * brightness
        imgs = (imgs - imgs.mean(dim=[2, 3], keepdim=True)) * contrast + imgs.mean(dim=[2, 3], keepdim=True)
        imgs = torch.clamp(imgs, 0, 1)

        padding = (pad, pad, pad, pad)
        imgs = F.pad(imgs, padding, mode='replicate')
        shift_h = torch.randint(0, 2 * pad + 1, (n,), device=imgs.device)
        shift_w = torch.randint(0, 2 * pad + 1, (n,), device=imgs.device)
        new_imgs = torch.zeros(n, c, h, w, device=imgs.device)
        for i in range(n):
            new_imgs[i] = imgs[i, :, shift_h[i]:shift_h[i]+h, shift_w[i]:shift_w[i]+w]
        return new_imgs
