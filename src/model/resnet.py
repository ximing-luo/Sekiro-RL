'''
Description: 
Version: 2.0
Autor: Zhang
Date: 2021-11-16 10:37:59
LastEditors: Zhang
LastEditTime: 2021-12-08 22:50:57
'''

import torch
import torch.nn as nn
import torchvision.models as models

a = models.AlexNet()

# nn.Conv2d 参数说明（本文件大量使用）：
# - in_channels：输入特征图的通道数（C_in）
# - out_channels：输出特征图的通道数（C_out）
# - kernel_size：卷积核大小（例如 3 表示 3x3，1 表示 1x1）
# - stride：步幅（>1 时做下采样，空间尺寸缩小）
# - padding：填充像素数量（通常与 kernel_size 协同保证尺寸不变）
# - groups：分组卷积；当 groups == in_channels 时为“深度卷积”（每个通道独立卷积）
# - bias：是否使用偏置项；本文件多数卷积不使用偏置（bias=False），由 BatchNorm 负责偏移
# - dilation：膨胀卷积的扩张系数（默认 1）
# 结构备注：下文的残差块采用“深度卷积 + 1x1 点卷积”的可分离卷积，以降低参数量与计算量。

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # 残差主分支（Residual）：
        # - 第1层：3x3 深度卷积（groups=in_channels），按通道独立提取局部特征；stride 控制是否下采样
        # - 第2层：1x1 点卷积，将通道从 in_channels 映射到 out_channels，实现通道混合
        # - 第3层：GroupNorm，做归一化与可学习的缩放/平移（替代 BatchNorm 以增强 RL 稳定性）
        # - 第4层：SiLU 激活
        # - 第5层：3x3 深度卷积（groups=out_channels），再次提取局部特征
        # - 第6层：1x1 点卷积，将通道映射到 out_channels * expansion（BasicBlock.expansion=1）
        # - 第7层：GroupNorm
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, groups=in_channels, padding=1, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, groups=out_channels, padding=1, bias=False),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(min(8, out_channels * BasicBlock.expansion), out_channels * BasicBlock.expansion)
        )

        # 残差捷径分支（Shortcut）：默认恒等映射（不改变尺寸与通道）
        self.shortcut = nn.Sequential()

        # 关键步骤（对应 37-41 行）：
        # 当 stride != 1（需要下采样）或 in_channels != out_channels * expansion（通道不匹配）时，
        # 使用 1x1 卷积（可带 stride）与 BatchNorm 将捷径分支的空间尺寸与通道数“对齐”到主分支输出，
        # 这样两条分支才能逐元素相加。
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(min(8, out_channels * BasicBlock.expansion), out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        # 前向：主分支与捷径分支输出相加后再 SiLU，形成残差学习
        return nn.SiLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):

    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # BottleNeck：沙漏/瓶颈结构 (ResNet-50/101/152 使用)
        # 特点：先压缩通道（省计算量），中间做 3x3，最后大幅扩张通道（增强表达能力）。
        self.residual_function = nn.Sequential(
            # -- 第一步：压缩 (Reduce) --
            # 第1层：1x1 卷积 - 把高通道压缩到低通道 (out_channels)，就像把宽路缩成窄瓶颈
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),

            # -- 第二步：卷积 (Convolution) --
            # 第2层：3x3 深度卷积 + 1x1 点卷积 - 在低维空间做核心特征提取，非常省显存
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, groups=out_channels, padding=1, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),

            # -- 第三步：扩张 (Expand) --
            # 第3层：1x1 卷积 - 把通道数暴力弹射起步，变成原来的 4 倍 (BottleNeck.expansion=4)
            nn.Conv2d(out_channels, out_channels* BottleNeck.expansion, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, out_channels * BottleNeck.expansion), out_channels * BottleNeck.expansion)
        )

        # 捷径分支默认恒等映射
        self.shortcut = nn.Sequential()

        # 关键步骤（对应 68-72 行）：
        # 当需要下采样或通道不匹配时，使用 1x1 卷积 + GroupNorm 将捷径分支对齐到
        # out_channels * expansion 的通道与空间尺寸，确保可相加。
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.GroupNorm(min(8, out_channels * BottleNeck.expansion), out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        # 前向：主分支 + 捷径分支，再 SiLU
        return nn.SiLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class DQN(nn.Module):
    def __init__(self, in_channels, block, num_blocks, num_actions):
        super(DQN, self).__init__()

        self.in_channels = 64

        # 输入对齐模块：
        # - 3x3 深度卷积（groups=in_channels）按通道独立提取局部特征（可视为每帧/每色通道的空间处理）
        # - 1x1 点卷积将通道映射到 64，统一后续残差层的输入通道数
        # - GroupNorm + SiLU 稳定训练
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3,groups=in_channels, padding=1, bias=False),
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=1, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True)
        )

        self.conv2_x = self._make_layer(block, 64, num_blocks[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_blocks[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_blocks[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_blocks[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        
        self.fc = nn.Linear(512 * block.expansion, num_actions)

    def _make_layer(self, block, out_channels, num_block, stride):
        # 逻辑说明（对应 100-109 行）：
        # - 构造一个“阶段”（layer），由 num_block 个残差块串联而成；
        # - 第一个块使用传入的 stride（常为 2）做下采样，其余块 stride=1 保持尺寸；
        # - 每追加一个块后，将下一块的输入通道更新为 out_channels * block.expansion，
        #   以匹配残差块的输出通道数（BasicBlock.expansion=1，BottleNeck.expansion=4）。
        strides = [stride] + [1] * (num_block - 1)  # 构造步幅列表：长度为 num_block；首块用传入的 stride（通常为 2）以实现下采样，其余块使用 1 保持特征图尺寸不变
        layers = []  # 暂存当前阶段中的所有残差块（模块列表），稍后包装为 nn.Sequential 以顺序执行

        for stride in strides:  # 逐个生成残差块；每次循环对应一个块，其步幅由上面的 strides 列表给出
            layers.append(block(self.in_channels, out_channels, stride))  # 创建并追加一个残差块：输入通道为当前 self.in_channels，输出通道为 out_channels，步幅为 stride
            self.in_channels = out_channels * block.expansion  # 更新下一块的输入通道数：等于本块的输出通道乘以 expansion（BasicBlock=1，BottleNeck=4），保证后续块的输入与前一块输出对齐

        return nn.Sequential(*layers)  # 将收集到的残差块按顺序打包为一个顺序容器，形成当前阶段的子网络


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
    def __init__(self, in_channels, block, num_blocks, num_actions):
        super(Dueling_DQN,self).__init__()
        self.in_channels = 64
        self.num_actions = num_actions

        # 与 DQN 相同的特征提取主干（ResNet 风格），仅在尾部使用 Dueling 结构拆分为 Advantage 与 Value 两个头
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3,groups=in_channels, padding=1, bias=False),
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=1, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True)
        )

        self.conv2_x = self._make_layer(block, 64, num_blocks[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_blocks[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_blocks[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_blocks[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        
        self.fc_adv = nn.Linear(512 * block.expansion, num_actions)
        self.fc_val = nn.Linear(512 * block.expansion, 1)

    def _make_layer(self, block, out_channels, num_block, stride):
        # 同上：首块下采样，其余保持尺寸，逐块更新 in_channels
        strides = [stride] + [1] * (num_block - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
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
        # 如果输入通道数不等于第一层卷积的输入通道数，尝试转换为灰度图
        # 例如：输入为 RGB 堆叠 (12通道)，第一层期望灰度堆叠 (4通道)
        if x.shape[1] != self.conv1[0].in_channels:
            x = self.to_grayscale(x)
        
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        # Dueling 头：分别计算优势函数 A(s,a) 与状态价值 V(s)
        # 形状：adv -> (N, num_actions)，val -> (N, 1)
        adv = self.fc_adv(output)
        # 将 V(s) 扩展到动作维度，便于与 A(s,a) 逐元素组合
        # 形状：(N, num_actions)，不拷贝数据，仅视图扩展
        val = self.fc_val(output).expand(output.size(0), self.num_actions)

        # Dueling 合成：Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
        # 通过减去均值实现优势的中心化，避免 V 与 A 的不唯一分解
        output = val + adv - adv.mean(1).unsqueeze(1).expand(output.size(0), self.num_actions)

        return output



def dqn_res18(in_channels, num_actions):
    return DQN(in_channels=in_channels, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_actions=num_actions)

def ddqn_res18(in_channels, num_actions):
    return Dueling_DQN(in_channels=in_channels, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_actions=num_actions)