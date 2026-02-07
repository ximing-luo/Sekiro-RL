import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback
from .tensorboard import TensorboardHookManager

class SekiroCombinedCallback(BaseCallback):
    """
    全能回调类：
    1. 集成 Isaac Lab 风格打印（含奖励分量）。
    2. 集成卷积层特征图可视化。
    3. 实时特征相似度监控（诊断特征坍缩）。
    """
    def __init__(self, verbose=0, log_interval=1000):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.hook_manager = None
        self.iteration = 0
        self.reward_buffer = {}
        self.cum_player_deaths = 0
        self.cum_enemy_deaths = 0
        self.last_sim_metrics = {
            "synthetic_avg": 0.0,
            "experience_avg": 0.0
        }

    def _on_training_start(self):
        from stable_baselines3.common.logger import TensorBoardOutputFormat
        writer = None
        if self.model.logger is not None:
            for output_format in self.model.logger.output_formats:
                if isinstance(output_format, TensorBoardOutputFormat):
                    writer = output_format.writer
                    break
        
        if writer is not None:
            self.hook_manager = TensorboardHookManager(self.model, writer, log_interval=self.log_interval)
            self.hook_manager.register_hooks()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "reward_components" in info:
                components = info["reward_components"]
                for name, val in components.items():
                    if name not in self.reward_buffer:
                        self.reward_buffer[name] = []
                    self.reward_buffer[name].append(val)
            if "events" in info:
                events = info["events"]
                if 0 in events: 
                    self.cum_player_deaths += 1
                if 1 in events: 
                    self.cum_enemy_deaths += 1
        return True

    def _on_rollout_start(self):
        pass

    def _analyze_feature_similarity(self):
        device = self.model.device
        extractor = self.model.policy.features_extractor
        extractor.eval()
        with torch.no_grad():
            noise = torch.randn(1, 3, 135, 240).to(device)
            black = torch.zeros(1, 3, 135, 240).to(device)
            checker = torch.ones(1, 3, 135, 240).to(device)
            checker[:, :, ::2, ::2] = 0
            syn_inputs = [noise, black, checker]
            syn_feats = [extractor(img) for img in syn_inputs]
            syn_sims = []
            for i in range(len(syn_feats)):
                for j in range(i + 1, len(syn_feats)):
                    sim = F.cosine_similarity(syn_feats[i], syn_feats[j]).item()
                    syn_sims.append(sim)
            self.last_sim_metrics["synthetic_avg"] = sum(syn_sims) / len(syn_sims) if syn_sims else 1.0
            if hasattr(self.model, "rollout_buffer") and self.num_timesteps > 0:
                obs = self.model.rollout_buffer.observations
                flat_obs = torch.as_tensor(obs).view(-1, *obs.shape[2:])
                num_samples = min(8, flat_obs.size(0))
                idx = torch.randperm(flat_obs.size(0))[:num_samples]
                exp_samples = flat_obs[idx].to(device)
                self._save_debug_images(exp_samples)
                if exp_samples.dtype == torch.uint8:
                    exp_samples = exp_samples.float() / 255.0
                exp_feats = extractor(exp_samples)
                norm_exp_feats = F.normalize(exp_feats, dim=1)
                sim_matrix = torch.matmul(norm_exp_feats, norm_exp_feats.T)
                n = exp_feats.size(0)
                mask = torch.eye(n, device=device).bool()
                exp_sims = sim_matrix[~mask]
                if exp_sims.numel() > 0:
                    self.last_sim_metrics["experience_avg"] = exp_sims.mean().item()
        extractor.train()

    def _save_debug_images(self, samples):
        debug_dir = os.path.join("logs", "data", "debug_samples")
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        for i in range(samples.size(0)):
            img = samples[i].cpu().numpy().transpose(1, 2, 0)
            if img.max() <= 1.0: img = (img * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(debug_dir, f"sample_{i}.png"), img)
