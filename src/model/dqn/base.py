import torch
import torch.nn as nn
from ..components import RMSNorm, BottleNeck

class DQN(nn.Module):
    """
    正常 DQN (ResNet 风格)
    """
    def __init__(self, in_channels, block, num_blocks, num_actions):
        super(DQN, self).__init__()

        self.in_channels = 64

        # 输入对齐模块
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, groups=in_channels, padding=1, bias=False),
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=1, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True)
        )

        self.conv2_x = self._make_layer(block, 64, num_blocks[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_blocks[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_blocks[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_blocks[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        
        self.fc = nn.Linear(512 * block.expansion, num_actions)

    def _make_layer(self, block, out_channels, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride=stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

class Dueling_DQN(nn.Module):
    """
    Dueling DQN (ResNet 风格)
    """
    def __init__(self, in_channels, block, num_blocks, num_actions):
        super(Dueling_DQN, self).__init__()
        self.in_channels = 64
        self.num_actions = num_actions

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, groups=in_channels, padding=1, bias=False),
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=1, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True)
        )

        self.conv2_x = self._make_layer(block, 64, num_blocks[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_blocks[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_blocks[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_blocks[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        
        self.fc_adv = nn.Linear(512 * block.expansion, num_actions)
        self.fc_val = nn.Linear(512 * block.expansion, 1)

    def _make_layer(self, block, out_channels, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride=stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def to_grayscale(self, x, order: str = 'auto'):
        n, c, h, w = x.shape
        assert c % 3 == 0
        k = c // 3
        xr = x.view(n, k, 3, h, w)
        if isinstance(order, str) and order.upper() == 'RGB':
            wts = torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype, device=x.device)
        else:
            wts = torch.tensor([0.114, 0.587, 0.299], dtype=x.dtype, device=x.device)
        y = (xr * wts.view(1, 1, 3, 1, 1)).sum(dim=2)
        return y

    def forward(self, x):
        if x.shape[1] != self.conv1[0].in_channels:
            x = self.to_grayscale(x)
        
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        
        adv = self.fc_adv(output)
        val = self.fc_val(output).expand(output.size(0), self.num_actions)

        output = val + adv - adv.mean(1).unsqueeze(1).expand(output.size(0), self.num_actions)
        return output

def dqn_res18(in_channels, num_actions):
    return DQN(in_channels=in_channels, block=BottleNeck, num_blocks=[2, 2, 2, 2], num_actions=num_actions)

def ddqn_res18(in_channels, num_actions):
    return Dueling_DQN(in_channels=in_channels, block=BottleNeck, num_blocks=[2, 2, 2, 2], num_actions=num_actions)
