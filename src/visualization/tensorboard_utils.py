import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class TensorboardHookManager:
    """
    TensorBoard Hook 管理器：负责自动注册模型 Hook 并记录特征图。
    已适配 Stable-Baselines3 模型结构。
    """
    def __init__(self, model, writer: SummaryWriter, log_interval: int = 1000, max_layers: int = 5):
        self.model = model
        self.writer = writer
        self.log_interval = log_interval
        self.max_layers = max_layers
        self.hooks = []

    def _get_activation_hook(self, name):
        """生成用于记录特征图的 Hook 函数。"""
        def hook(module, input, output):
            # 关键优化：如果是训练模式（反向传播），直接跳过，不记录特征图
            if module.training:
                return

            # 获取当前的全局步数
            step = getattr(self.model, 'num_timesteps', 0)
            if step > 0 and step % self.log_interval == 0:
                try:
                    # output shape: (Batch, Channel, H, W)
                    img = output[0].detach().cpu()
                    
                    # 归一化
                    mn, mx = img.min(), img.max()
                    img = (img - mn) / (mx - mn + 1e-9)

                    # 限制通道数
                    if img.shape[0] > 64:
                        img = img[:64]
                    
                    # 制作网格
                    grid = torchvision.utils.make_grid(img.unsqueeze(1), nrow=8, padding=2)
                    self.writer.add_image(f'Features/{name}', grid, step)
                except Exception as e:
                    pass 
        return hook

    def register_hooks(self):
        """动态发现卷积层并注册 Hook。"""
        self.remove_hooks()
        
        # 针对 SB3 PPO 的结构定位 CNN
        if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'features_extractor'):
            target_module = self.model.policy.features_extractor
        else:
            target_module = self.model

        conv_layers = []
        for name, module in target_module.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append((name, module))
        
        if not conv_layers:
            return

        indices = np.linspace(0, len(conv_layers) - 1, min(self.max_layers, len(conv_layers)), dtype=int)
        for i, idx in enumerate(indices):
            name, layer = conv_layers[idx]
            display_name = f"{i}_{name.replace('.', '_')}"
            hook_handle = layer.register_forward_hook(self._get_activation_hook(display_name))
            self.hooks.append(hook_handle)

    def remove_hooks(self):
        """移除所有已注册的 Hook。"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

def register_tensorboard_hooks(agent, writer: SummaryWriter, log_interval: int = 1000):
    """便捷函数：初始化并注册 TensorBoard Hooks。"""
    manager = TensorboardHookManager(agent, writer, log_interval=log_interval)
    manager.register_hooks()
    return manager