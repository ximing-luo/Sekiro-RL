import torch
import torch.nn as nn


class InitialFrameConv(nn.Module):
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


class StaticDynamicConcat(nn.Module):
    def __init__(self, num_frames):
        super().__init__()
        in_c = num_frames * 32

        self.static = nn.Sequential(
            nn.Conv2d(in_c, num_frames * 64, kernel_size=7, stride=2, padding=3, groups=num_frames, bias=True),
            nn.BatchNorm2d(num_frames * 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(num_frames * 64, 64, kernel_size=5, stride=2, padding=2, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
        )

        self.dynamic = nn.Sequential(
            nn.Conv2d(in_c, 512, kernel_size=7, stride=2, padding=3, groups=32, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=32, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, groups=32, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
        )

    def forward(self, x):
        s = self.static(x)
        d = self.dynamic(x)
        return torch.cat([s, d], dim=1)


class SimpleDQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super().__init__()
        assert in_channels % 3 == 0, "in_channels must be a multiple of 3 (RGB per frame)"
        num_frames = in_channels // 3

        self.features = nn.Sequential(
            InitialFrameConv(in_channels),
            StaticDynamicConcat(num_frames),
        )

        self.pre_fc_norm = nn.LayerNorm(1024)

        self.initial_fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )
        self.fc_adv = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_actions)
        )
        self.fc_val = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pre_fc_norm(x)
        x = self.initial_fc(x)
        adv = self.fc_adv(x)
        val = self.fc_val(x).expand(x.size(0), adv.size(1))
        output = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), adv.size(1))
        return output


def dqn_simple(in_channels, num_actions):
    return SimpleDQN(in_channels=in_channels, num_actions=num_actions)