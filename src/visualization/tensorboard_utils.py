import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class TensorboardHookManager:
    """
    TensorBoard Hook 管理器：负责自动注册模型 Hook 并记录特征图。
    遵循 Isaac Lab 的解耦设计理念，将可视化逻辑从训练主流程中分离。
    """
    def __init__(self, agent, writer: SummaryWriter, log_interval: int = 1000, max_layers: int = 5):
        self.agent = agent
        self.writer = writer
        self.log_interval = log_interval
        self.max_layers = max_layers
        self.hooks = []

    def _get_activation_hook(self, name):
        """生成用于记录特征图的 Hook 函数。"""
        def hook(model, input, output):
            # 仅在特定步数记录
            step = getattr(self.agent, 'current_step', 0)
            if step > 0 and step % self.log_interval == 0:
                try:
                    # output shape: (Batch, Channel, H, W)
                    # 取第一个样本: (Channel, H, W)
                    img = output[0].detach().cpu()
                    
                    # 归一化到 [0, 1]
                    mn, mx = img.min(), img.max()
                    if mx - mn > 1e-9:
                        img = (img - mn) / (mx - mn)
                    else:
                        img = torch.zeros_like(img)

                    # 如果通道数过多，只取前 64 个通道进行展示
                    if img.shape[0] > 64:
                        img = img[:64]
                    
                    # 将通道维度作为 batch 维度，制作网格: (C, 1, H, W) -> Grid
                    grid = torchvision.utils.make_grid(img.unsqueeze(1), nrow=8, padding=2, normalize=False)
                    self.writer.add_image(f'Features/{name}', grid, step)
                except Exception as e:
                    print(f"Hook error for {name}: {e}")
        return hook

    def register_hooks(self):
        """动态发现卷积层并注册 Hook。"""
        # 清除现有 Hook
        self.remove_hooks()
        
        model = self.agent.algorithm.eval_net
        conv_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append((name, module))
        
        if not conv_layers:
            print("未发现可注册 Hook 的卷积层。")
            return

        # 均匀采样卷积层进行可视化
        indices = np.linspace(0, len(conv_layers) - 1, min(self.max_layers, len(conv_layers)), dtype=int)
        
        registered_count = 0
        for i, idx in enumerate(indices):
            name, layer = conv_layers[idx]
            display_name = name.split('.')[-1] if '.' in name else name
            full_name = f"{i}_{display_name}"
            
            hook_handle = layer.register_forward_hook(self._get_activation_hook(full_name))
            self.hooks.append(hook_handle)
            registered_count += 1
            
        print(f"已自动注册 {registered_count} 个 TensorBoard Hooks。")

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
