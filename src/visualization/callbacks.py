import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback
from src.envs.mdp.actions import ACTION_LABELS
from src.visualization.tensorboard_utils import TensorboardHookManager

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
        # 奖励分量缓冲区
        self.reward_buffer = {}
        # 累计死亡统计 (训练开始至今)
        self.cum_player_deaths = 0
        self.cum_enemy_deaths = 0
        # 特征相似度指标
        self.last_sim_metrics = {
            "synthetic_avg": 0.0,
            "experience_avg": 0.0
        }

    def _on_training_start(self):
        # 初始化特征图 Hook
        from stable_baselines3.common.logger import TensorBoardOutputFormat
        writer = None
        for output_format in self.model.logger.output_formats:
            if isinstance(output_format, TensorBoardOutputFormat):
                writer = output_format.writer
                break
        
        # if writer is not None:
        #     self.hook_manager = TensorboardHookManager(self.model, writer, log_interval=self.log_interval)
        #     self.hook_manager.register_hooks()

    def _on_step(self) -> bool:
        # 1. 从 infos 提取奖励分量
        infos = self.locals.get("infos", [])
        for info in infos:
            if "reward_components" in info:
                components = info["reward_components"]
                for name, val in components.items():
                    if name not in self.reward_buffer:
                        self.reward_buffer[name] = []
                    self.reward_buffer[name].append(val)
            
            # 2. 统计死亡事件
            if "events" in info:
                events = info["events"]
                if 0 in events: 
                    self.cum_player_deaths += 1
                if 1 in events: 
                    self.cum_enemy_deaths += 1
        return True

    def _on_rollout_start(self):
        """每轮 Rollout 开始前。"""
        pass

    def _analyze_feature_similarity(self):
        """计算特征提取器对不同输入的敏感度（余弦相似度）。"""
        device = self.model.device
        extractor = self.model.policy.features_extractor
        extractor.eval()

        with torch.no_grad():
            # --- 1. 合成图片分析 ---
            # 随机噪声, 全黑, 棋盘格
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

            # --- 2. 经验图片分析 ---
            # 只要不是第一轮（已有数据采集），就从 buffer 中提取样本
            if hasattr(self.model, "rollout_buffer") and self.num_timesteps > 0:
                obs = self.model.rollout_buffer.observations # (n_steps, n_envs, C, H, W)
                # 随机采样 8 张图
                flat_obs = torch.as_tensor(obs).view(-1, *obs.shape[2:])
                num_samples = min(8, flat_obs.size(0))
                idx = torch.randperm(flat_obs.size(0))[:num_samples]
                exp_samples = flat_obs[idx].to(device)
                
                # --- 调试：保存采样的 8 张图到根目录 ---
                self._save_debug_images(exp_samples)
                
                # 确保归一化 (如果是 uint8)
                if exp_samples.dtype == torch.uint8:
                    exp_samples = exp_samples.float() / 255.0
                
                exp_feats = extractor(exp_samples) # (num_samples, features_dim)
                
                # 优化相似度计算：使用矩阵运算，更高效且不容易出错
                norm_exp_feats = F.normalize(exp_feats, dim=1)
                sim_matrix = torch.matmul(norm_exp_feats, norm_exp_feats.T)
                n = exp_feats.size(0)
                mask = torch.eye(n, device=device).bool()
                exp_sims = sim_matrix[~mask]
                
                if exp_sims.numel() > 0:
                    self.last_sim_metrics["experience_avg"] = exp_sims.mean().item()
            
        extractor.train()

    def _save_debug_images(self, samples):
        """将采样张量保存为本地图片。"""
        # 确保目录存在
        debug_dir = "debug_samples"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
            
        for i, img_tensor in enumerate(samples):
            # (C, H, W) -> (H, W, C)
            img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
            # 如果是 float 0-1，转回 0-255
            if img_np.dtype != np.uint8:
                img_np = (img_np * 255).astype(np.uint8)
            # RGB -> BGR (OpenCV)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(debug_dir, f"sample_{i}.png"), img_bgr)

    def _on_rollout_end(self):
        self.iteration += 1
        print(f"\n[Step {self.num_timesteps}] 采集完成，正在运行诊断并开始训练...")
        
        # 0. 运行特征相似度诊断 (此时 rollout_buffer 已满，数据最全)
        self._analyze_feature_similarity()
        
        # 1. 记录动作分布直方图到 TensorBoard
        self._log_action_distribution()

        # 2. 记录特征相似度到 TensorBoard
        self.logger.record("Diagnostic/Avg_Synthetic_Similarity", self.last_sim_metrics["synthetic_avg"])
        self.logger.record("Diagnostic/Avg_Experience_Similarity", self.last_sim_metrics["experience_avg"])

        # 3. 计算本轮奖励分量均值
        avg_rewards = {}
        rollout_mean_reward = 0.0
        for name, vals in self.reward_buffer.items():
            if len(vals) > 0:
                avg_val = sum(vals) / len(vals)
                avg_rewards[name] = avg_val
                rollout_mean_reward += avg_val
        self.reward_buffer = {} # 清空缓冲区

        # 4. Isaac Lab 风格打印
        metrics = self.logger.name_to_value
        print("\n" + "="*50)
        print(f"Iteration {self.iteration: <3} | Total Steps: {self.num_timesteps: <8}")
        print("-" * 50)
        
        # 打印特征敏感度
        syn_sim = self.last_sim_metrics["synthetic_avg"]
        exp_sim = self.last_sim_metrics["experience_avg"]
        syn_color = "\033[91m" if syn_sim > 0.99 else "\033[92m"
        exp_color = "\033[91m" if exp_sim > 0.99 else "\033[92m"
        
        print("特征敏感度 (余弦相似度):")
        print(f"  - 合成图 (噪声/全黑/格栅): {syn_color}{syn_sim:.6f}\033[0m")
        print(f"  - 经验图 (真实观测采样):   {exp_color}{exp_sim:.6f}\033[0m")
        if exp_sim > 0.999:
            print("  \033[91m[!!!] 警告: 检测到特征坍缩 (Feature Collapse) [!!!]\033[0m")
        print("-" * 50)

        # 打印各分量 (固定字母顺序排序，方便观察)
        print("奖励分量 (每步平均):")
        for name, val in sorted(avg_rewards.items()):
            color = "\033[92m" if val >= 0 else "\033[91m" # 绿色正分，红色负分
            print(f"  - {name: <15}: {color}{val: .4f}\033[0m")
        
        # 打印累计汇总 (本轮)
        print("-" * 50)
        color = "\033[92m" if rollout_mean_reward >= 0 else "\033[91m"
        print(f"本轮奖励均值: {color}{rollout_mean_reward:.4f}\033[0m")
        print(f"累计死亡统计 -> 玩家: {self.cum_player_deaths}, 敌人: {self.cum_enemy_deaths}")
        if "train/loss" in metrics:
            print(f"训练损失:       {metrics['train/loss']:.4f}")
        print("="*50 + "\n")

    def _log_action_distribution(self):
        """从 rollout_buffer 采样并记录动作 Logits 与 Probabilities 直方图。"""
        if self.hook_manager is None or self.hook_manager.writer is None:
            return
        
        writer = self.hook_manager.writer
        # 获取本轮采集的观测值 (n_steps, n_envs, C, H, W)
        obs = self.model.rollout_buffer.observations
        
        with torch.no_grad():
            # 展平并采样数据
            flat_obs = torch.as_tensor(obs).view(-1, *obs.shape[2:]).to(self.model.device)
            # 限制采样数量以防计算过慢
            num_samples = min(256, flat_obs.size(0))
            idx = torch.randperm(flat_obs.size(0))[:num_samples]
            obs_sample = flat_obs[idx]
            
            # 获取分布
            distribution = self.model.policy.get_distribution(obs_sample)
            logits = distribution.distribution.logits  # (batch, action_dim)
            probs = torch.softmax(logits, dim=-1)     # (batch, action_dim)
            
            # 记录到 TensorBoard
            for i in range(logits.shape[1]):
                label = ACTION_LABELS[i] if i < len(ACTION_LABELS) else f"Action_{i}"
                # 记录 Logits (观察网络输出强度)
                writer.add_histogram(f"Action_Dist/Logits_{label}", logits[:, i], self.num_timesteps)
                # 记录 Probabilities (观察最终概率分布)
                writer.add_histogram(f"Action_Dist/Probs_{label}", probs[:, i], self.num_timesteps)

    def _on_training_end(self):
        if self.hook_manager:
            self.hook_manager.remove_hooks()
