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
    def __init__(self, model, writer: SummaryWriter, log_interval: int = 128, max_layers: int = 5):
        self.model = model
        self.writer = writer
        self.log_interval = log_interval
        # 特殊处理：训练初期的记录间隔可以更小
        self.initial_log_interval = 16 
        self.max_layers = max_layers
        self.hooks = []

    def _get_activation_hook(self, name):
        """生成用于记录特征图的 Hook 函数。"""
        def hook(module, input, output):
            # 获取当前的全局步数
            step = getattr(self.model, 'num_timesteps', 0)
            
            # 记录频率控制：初期高频
            current_interval = self.initial_log_interval if step < 2000 else self.log_interval
            
            if step % current_interval == 0:
                try:
                    # 如果在训练模式，改变 Tag 名防止覆盖采集阶段的图
                    tag_name = f'Features/{name}'
                    if module.training:
                        tag_name += '_train'

                    # 如果是 2D 特征向量 (Batch, Dim)，升维成图片显示
                    if output.ndim == 2:
                        b, d = output.shape
                        side = int(np.sqrt(d))
                        if side * side == d:
                            img = output[0].view(1, side, side).detach().cpu()
                        else:
                            img = output[0].view(1, 1, -1).detach().cpu()
                    elif output.ndim == 4:
                        img = output[0].detach().cpu()
                    else:
                        return

                    # 归一化
                    mn, mx = img.min(), img.max()
                    img = (img - mn) / (mx - mn + 1e-9)

                    # 制作网格
                    if img.ndim == 3: # (C, H, W)
                        if img.shape[0] > 64: img = img[:64]
                        grid = torchvision.utils.make_grid(img.unsqueeze(1), nrow=8, padding=2)
                    else: # (H, W)
                        grid = img

                    self.writer.add_image(tag_name, grid, step)
                    self.writer.flush() # 强制写入磁盘
                except Exception as e:
                    pass 
        return hook

    def register_hooks(self):
        """动态发现模型层并注册 Hook。"""
        self.remove_hooks()
        
        # 针对 SB3 PPO 的结构定位目标模块
        target_modules = []
        if hasattr(self.model, 'policy'):
            if hasattr(self.model.policy, 'features_extractor'):
                target_modules.append(("Extractor", self.model.policy.features_extractor))
            if hasattr(self.model.policy, 'mlp_extractor'):
                target_modules.append(("Heads", self.model.policy.mlp_extractor))
        
        if not target_modules:
            target_modules = [("Model", self.model)]

        for module_prefix, target_module in target_modules:
            monitored_layers = []
            for name, module in target_module.named_modules():
                # 记录卷积层和线性层（或 NormalizedMLP）
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    monitored_layers.append((name, module))
            
            if not monitored_layers:
                continue

            # 均匀采样指定数量的层进行记录
            indices = np.linspace(0, len(monitored_layers) - 1, min(self.max_layers, len(monitored_layers)), dtype=int)
            for i, idx in enumerate(indices):
                name, layer = monitored_layers[idx]
                display_name = f"{module_prefix}_{i}_{name.replace('.', '_')}"
                hook_handle = layer.register_forward_hook(self._get_activation_hook(display_name))
                self.hooks.append(hook_handle)

    def remove_hooks(self):
        """移除所有已注册的 Hook。"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def log_gradients(self, step: int):
        """记录模型所有层的梯度范数，并检测 NaN。"""
        if self.writer is None:
            return

        # 动态调整记录间隔：前 2000 步使用更小的间隔
        current_interval = self.initial_log_interval if step < 2000 else self.log_interval

        # 适配 SB3：如果是算法对象，则访问其 policy 属性
        target = self.model.policy if hasattr(self.model, "policy") else self.model

        for name, param in target.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_max = param.grad.abs().max().item()
                
                # NaN 检测
                is_nan = np.isnan(grad_norm) or np.isinf(grad_norm)
                if is_nan:
                    print(f"\033[91m[CRITICAL] 检测到梯度爆炸/NaN! 层名: {name} | Norm: {grad_norm:.2e} | Max: {grad_max:.2e}\033[0m")
                
                # 记录到 TensorBoard
                # 优化：在前 10 步强制记录，确保立刻看到反馈
                if step < 10 or step % current_interval == 0 or is_nan:
                    self.writer.add_scalar(f"Gradients_Norm/{name}", grad_norm, step)
                    self.writer.add_scalar(f"Gradients_Max/{name}", grad_max, step)
                    self.writer.flush()

    def log_weights(self, step: int):
        """记录模型所有层的权重范数，并检测 NaN。"""
        if self.writer is None:
            return

        # 动态调整记录间隔
        current_interval = self.initial_log_interval if step < 2000 else self.log_interval

        # 适配 SB3
        target = self.model.policy if hasattr(self.model, "policy") else self.model

        for name, param in target.named_parameters():
            weight_norm = param.data.norm().item()
            
            # NaN 检测
            is_nan = np.isnan(weight_norm) or np.isinf(weight_norm)
            if is_nan:
                print(f"\033[91m[CRITICAL] 检测到权重损坏/NaN! 层名: {name} | Norm: {weight_norm:.2e}\033[0m")
            
            # 记录到 TensorBoard
            if step < 10 or step % current_interval == 0 or is_nan:
                self.writer.add_scalar(f"Weights_Norm/{name}", weight_norm, step)
                if step < 10 or step % (current_interval * 5) == 0: # 初期强制直方图
                    self.writer.add_histogram(f"Weights_Hist/{name}", param.data, step)
                self.writer.flush()

def register_tensorboard_hooks(agent, writer: SummaryWriter, log_interval: int = 1000):
    """便捷函数：初始化并注册 TensorBoard Hooks。"""
    manager = TensorboardHookManager(agent, writer, log_interval=log_interval)
    manager.register_hooks()
    return manager