import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from src.tasks.sekiro.mdp.actions import MULTI_DISCRETE_LABELS, MULTI_DISCRETE_HEAD_NAMES

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
            step = getattr(self.model, 'num_timesteps', 0)
            if step <= 0 or step % self.log_interval != 0:
                return
            if module.training:
                return
            try:
                real_output = output[0] if isinstance(output, (tuple, list)) else output
                output_cpu = real_output.detach().cpu()
                hist_tag = f"Activations/{clean_name}"
                self.writer.add_histogram(hist_tag, output_cpu, step)
                tag_name = f"Features/{clean_name}"
                if output_cpu.ndim == 4:
                    img = output_cpu[0]
                    mn, mx = img.min(), img.max()
                    img = (img - mn) / (mx - mn + 1e-9)
                    if img.shape[0] > 64: img = img[:64]
                    grid = torchvision.utils.make_grid(img.unsqueeze(1), nrow=8, padding=2)
                    self.writer.add_image(tag_name, grid, step)
                elif output_cpu.ndim == 2:
                    feat = output_cpu[0]
                    d = feat.shape[0]
                    w = int(np.sqrt(d))
                    if w == 0: w = 1
                    h = int(np.ceil(d / w))
                    if d < w * h:
                        padding = torch.zeros(w * h - d)
                        feat = torch.cat([feat, padding])
                    img = feat.view(1, h, w)
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
        if hasattr(self.model, 'policy'):
            target_module = self.model.policy
        else:
            target_module = self.model
        layer_idx = 1
        for name, module in target_module.named_modules():
            if len(list(module.children())) == 0:
                clean_name = f"{layer_idx:02d}_{name.replace('.', '_')}"
                hook_handle = module.register_forward_hook(self._get_activation_hook(clean_name, "Policy"))
                self.hooks.append(hook_handle)
                layer_idx += 1

    def remove_hooks(self):
        """移除所有已注册的 Hook。"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def log_action_distribution(self):
        """记录动作分布直方图。适配 Multi-Discrete 和 Discrete 动作空间。"""
        if self.writer is None:
            return
        step = getattr(self.model, 'num_timesteps', 0)
        with torch.no_grad():
            obs = self.model.rollout_buffer.observations
            flat_obs = torch.as_tensor(obs).view(-1, *obs.shape[2:]).to(self.model.device)
            num_samples = min(256, flat_obs.size(0))
            idx = torch.randperm(flat_obs.size(0))[:num_samples]
            obs_sample = flat_obs[idx]
            distribution = self.model.policy.get_distribution(obs_sample)
            if isinstance(distribution.distribution, list):
                for head_idx, dist in enumerate(distribution.distribution):
                    head_name = MULTI_DISCRETE_HEAD_NAMES[head_idx] if head_idx < len(MULTI_DISCRETE_HEAD_NAMES) else f"Head_{head_idx}"
                    logits = dist.logits.detach().cpu()
                    probs = torch.softmax(logits, dim=-1)
                    head_labels = MULTI_DISCRETE_LABELS[head_idx] if head_idx < len(MULTI_DISCRETE_LABELS) else []
                    for i in range(logits.shape[1]):
                        action_label = head_labels[i] if i < len(head_labels) else f"Action_{i}"
                        tag_prefix = f"Action_Dist/{head_name}"
                        self.writer.add_histogram(f"{tag_prefix}/Logits_{action_label}", logits[:, i], step)
                        self.writer.add_histogram(f"{tag_prefix}/Probs_{action_label}", probs[:, i], step)
            else:
                logits = distribution.distribution.logits.detach().cpu()
                probs = torch.softmax(logits, dim=-1)
                for i in range(logits.shape[1]):
                    action_label = f"Action_{i}"
                    tag_prefix = "Action_Dist"
                    self.writer.add_histogram(f"{tag_prefix}/Logits_{action_label}", logits[:, i], step)
                    self.writer.add_histogram(f"{tag_prefix}/Probs_{action_label}", probs[:, i], step)
