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
    def __init__(self, model, writer: SummaryWriter, log_interval: int = 1024, max_layers: int = 5):
        self.model = model
        self.writer = writer
        self.log_interval = log_interval
        self.max_layers = max_layers
        self.hooks = []

    def _get_activation_hook(self, clean_name, module_prefix):
        """生成用于特征图可视化的前向钩子。"""
        def hook(module, input, output):
            # 关键修复：如果在训练模式（反向传播期间），绝对不要记录特征图或 Histogram
            # 否则会干扰计算图，导致 loss.backward() 崩溃
            if module.training:
                return

            # 获取当前的全局步数
            step = getattr(self.model, 'num_timesteps', 0)
            
            if step > 0 and step % self.log_interval == 0:
                try:
                    # 1. 记录数值分布 (Histogram)
                    hist_tag = f"{module_prefix}/Activations/{clean_name}"
                    self.writer.add_histogram(hist_tag, output, step)

                    # 2. 如果是 4D 图片张量，记录图片可视化
                    tag_name = f"{module_prefix}/Features/{clean_name}"
                    if module.training:
                        tag_name += '_train'
                    
                    if output.ndim == 4:
                        img = output[0].detach().cpu()
                        # 归一化
                        mn, mx = img.min(), img.max()
                        img = (img - mn) / (mx - mn + 1e-9)
                        # 制作网格
                        if img.shape[0] > 64: img = img[:64]
                        grid = torchvision.utils.make_grid(img.unsqueeze(1), nrow=8, padding=2)
                        self.writer.add_image(tag_name, grid, step)
                    
                    # 3. 如果是 2D 特征向量 (如全连接层输出)
                    elif output.ndim == 2:
                        # 获取单个样本的特征向量 [D]
                        feat = output[0].detach().cpu()
                        d = feat.shape[0]
                        
                        # 自动寻找最接近正方形的矩形尺寸 (w, h)
                        w = int(np.sqrt(d))
                        if w == 0: w = 1
                        h = int(np.ceil(d / w))
                        
                        # 如果不能整除，补零填充到 w * h
                        if d < w * h:
                            padding = torch.zeros(w * h - d)
                            feat = torch.cat([feat, padding])
                        
                        img = feat.view(1, h, w)
                        
                        # 归一化并保存
                        mn, mx = img.min(), img.max()
                        img = (img - mn) / (mx - mn + 1e-9)
                        self.writer.add_image(tag_name, img, step)

                    self.writer.flush()
                except Exception:
                    pass 
        return hook

    def register_hooks(self):
        """为所有层注册 Hook（包括卷积层、全连接层、甚至激活层）。"""
        self.remove_hooks()
        
        # 针对 SB3 PPO 的结构定位 CNN 和整个 Policy
        if hasattr(self.model, 'policy'):
            target_module = self.model.policy
        else:
            target_module = self.model

        # 遍历所有子模块
        for name, module in target_module.named_modules():
            # 过滤掉容器类模块（如 Sequential），只给原子层加 Hook
            if len(list(module.children())) == 0:
                clean_name = name.replace('.', '_')
                prefix = "Policy"
                hook_handle = module.register_forward_hook(self._get_activation_hook(clean_name, prefix))
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