import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from src.envs.mdp.actions import ACTION_LABELS, MULTI_DISCRETE_LABELS, MULTI_DISCRETE_HEAD_NAMES

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
            # 获取当前的全局步数
            step = getattr(self.model, 'num_timesteps', 0)
            
            # 1. 只有在达到日志间隔时才处理，避免每一帧都进行昂贵的检查
            if step <= 0 or step % self.log_interval != 0:
                return

            # 2. 关键修复：如果在训练模式，直接跳过。
            if module.training:
                return

            # 3. 显存优化核心：立即将 output 彻底脱离计算图并移至 CPU
            try:
                # 处理可能出现的元组输出 (如 RNN 或某些自定义层)
                real_output = output[0] if isinstance(output, (tuple, list)) else output
                
                # 统一使用 cpu 端的 tensor 进行后续所有操作
                output_cpu = real_output.detach().cpu()
                
                # 1. 记录数值分布 (Histogram) - 扁平化目录：Category/LayerName
                hist_tag = f"Activations/{clean_name}"
                self.writer.add_histogram(hist_tag, output_cpu, step)

                # 2. 图像可视化 - 扁平化目录：Category/LayerName
                tag_name = f"Features/{clean_name}"
                
                if output_cpu.ndim == 4:
                    img = output_cpu[0]
                    # 归一化
                    mn, mx = img.min(), img.max()
                    img = (img - mn) / (mx - mn + 1e-9)
                    # 制作网格
                    if img.shape[0] > 64: img = img[:64]
                    grid = torchvision.utils.make_grid(img.unsqueeze(1), nrow=8, padding=2)
                    self.writer.add_image(tag_name, grid, step)
                
                # 3. 如果是 2D 特征向量 (如全连接层输出)
                elif output_cpu.ndim == 2:
                    # 获取单个样本的特征向量 [D]
                    feat = output_cpu[0]
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
        """为所有层注册 Hook，并按顺序进行 01, 02 编号命名。"""
        self.remove_hooks()
        
        # 针对 SB3 PPO 的结构定位 CNN 和整个 Policy
        if hasattr(self.model, 'policy'):
            target_module = self.model.policy
        else:
            target_module = self.model

        # 遍历所有子模块
        layer_idx = 1
        for name, module in target_module.named_modules():
            # 过滤掉容器类模块（如 Sequential），只给原子层加 Hook
            if len(list(module.children())) == 0:
                # 命名格式：01_features_cnn_0
                clean_name = f"{layer_idx:02d}_{name.replace('.', '_')}"
                prefix = "Policy"
                hook_handle = module.register_forward_hook(self._get_activation_hook(clean_name, prefix))
                self.hooks.append(hook_handle)
                layer_idx += 1

    def log_action_distribution(self):
        """记录动作分布直方图。适配 Multi-Discrete 和 Discrete 动作空间。"""
        if self.writer is None:
            return
        
        # 获取当前步数
        step = getattr(self.model, 'num_timesteps', 0)
        
        with torch.no_grad():
            # 从 rollout_buffer 获取观测并采样
            obs = self.model.rollout_buffer.observations
            # 兼容处理：obs 可能有不同的形状，确保转换为 (N, C, H, W)
            flat_obs = torch.as_tensor(obs).view(-1, *obs.shape[2:]).to(self.model.device)
            
            num_samples = min(256, flat_obs.size(0))
            idx = torch.randperm(flat_obs.size(0))[:num_samples]
            obs_sample = flat_obs[idx]
            
            # 获取分布
            distribution = self.model.policy.get_distribution(obs_sample)
            
            # 适配 Multi-Discrete: distribution.distribution 是一个 Categorical 列表
            if isinstance(distribution.distribution, list):
                # 遍历每一个动作头 (Move, Defense, Skill)
                for head_idx, dist in enumerate(distribution.distribution):
                    head_name = MULTI_DISCRETE_HEAD_NAMES[head_idx] if head_idx < len(MULTI_DISCRETE_HEAD_NAMES) else f"Head_{head_idx}"
                    logits = dist.logits.detach().cpu()
                    probs = torch.softmax(logits, dim=-1)
                    
                    # 获取该动作头的标签
                    head_labels = MULTI_DISCRETE_LABELS[head_idx] if head_idx < len(MULTI_DISCRETE_LABELS) else []
                    
                    # 记录该动作头下每个选项的分布
                    for i in range(logits.shape[1]):
                        action_label = head_labels[i] if i < len(head_labels) else f"Action_{i}"
                        tag_prefix = f"Action_Dist/{head_name}"
                        self.writer.add_histogram(f"{tag_prefix}/Logits_{action_label}", logits[:, i], step)
                        self.writer.add_histogram(f"{tag_prefix}/Probs_{action_label}", probs[:, i], step)
            else:
                # 兼容旧的 Discrete 动作空间
                logits = distribution.distribution.logits.detach().cpu()
                probs = torch.softmax(logits, dim=-1)
                
                for i in range(logits.shape[1]):
                    label = ACTION_LABELS[i] if i < len(ACTION_LABELS) else f"Action_{i}"
                    self.writer.add_histogram(f"Action_Logits/{label}", logits[:, i], step)
                    self.writer.add_histogram(f"Action_Probs/{label}", probs[:, i], step)

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