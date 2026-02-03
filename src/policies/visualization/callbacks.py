import torch
import torch.nn.functional as F
import cv2
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from src.envs.mdp.actions import ACTION_LABELS
from src.policies.visualization.tensorboard_utils import TensorboardHookManager

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
        
        if writer is not None:
            self.hook_manager = TensorboardHookManager(self.model, writer, log_interval=self.log_interval)
            self.hook_manager.register_hooks()
            
            # 建立模型与回调的直接绑定 (握手)，方便 PPO.train 直接调用
            self.model.diagnostics = self

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
        """每轮 Rollout 开始前运行。"""
        pass

    def _analyze_feature_similarity(self):
        """计算特征提取器对不同输入的敏感度（余弦相似度）。"""
        device = self.model.device
        extractor = self.model.policy.features_extractor
        extractor.eval()

        # 核心修复：从 Policy 的观测空间获取维度（12通道），而不是环境（3通道）
        # 这样生成的合成图片才能适配 MADS 特征提取器的帧差逻辑
        if hasattr(self.model.policy, "observation_space"):
            obs_space = self.model.policy.observation_space
        else:
            obs_space = self.model.observation_space
            
        n_channels = obs_space.shape[0]
        h, w = obs_space.shape[1], obs_space.shape[2]

        with torch.no_grad():
            # --- 1. 合成图片分析 ---
            # 随机噪声 (归一化到 [0, 1]), 全黑, 棋盘格
            noise = torch.rand(1, n_channels, h, w).to(device)
            black = torch.zeros(1, n_channels, h, w).to(device)
            checker = torch.ones(1, n_channels, h, w).to(device)
            checker[:, :, ::2, ::2] = 0
            
            syn_inputs = torch.cat([noise, black, checker], dim=0) # (3, C, H, W)
            syn_feats = extractor(syn_inputs) # (3, 512)
            
            # 计算合成图之间的余弦相似度 (矩阵化)
            syn_feats_norm = F.normalize(syn_feats, p=2, dim=1)
            syn_sim_matrix = torch.mm(syn_feats_norm, syn_feats_norm.t())
            
            # 提取上三角 (不含对角线)
            mask_syn = torch.triu(torch.ones_like(syn_sim_matrix), diagonal=1).bool()
            syn_sims = syn_sim_matrix[mask_syn]
            
            self.last_sim_metrics["synthetic_avg"] = syn_sims.mean().item() if syn_sims.numel() > 0 else 1.0

            # --- 2. 经验图片分析 ---
            # 只要不是第一轮（已有数据采集），就从 buffer 中提取样本
            if hasattr(self.model, "rollout_buffer") and self.num_timesteps > 0:
                buffer = self.model.rollout_buffer
                n_steps, n_envs = buffer.observations.shape[:2]
                total_samples = n_steps * n_envs
                
                # 增加采样数到 32 以提高统计稳定性
                num_samples = min(32, total_samples)
                indices = np.random.choice(total_samples, num_samples, replace=False)
                
                sampled_obs = []
                # 检查是否是支持堆叠的自定义 Buffer
                is_custom_buffer = hasattr(buffer, "_stack_frames")
                
                for idx in indices:
                    s_idx, e_idx = divmod(idx, n_envs)
                    if is_custom_buffer:
                        # 核心修复：使用堆叠帧进行诊断，对齐模型 12 通道输入要求
                        # 使用与训练一致的 n_stack=4, skip=3
                        stacked = buffer._stack_frames(s_idx, e_idx, buffer.observations, n_stack=4, skip=3)
                        sampled_obs.append(stacked)
                    else:
                        sampled_obs.append(buffer.observations[s_idx, e_idx])
                
                exp_samples = torch.as_tensor(np.array(sampled_obs)).to(device)
                
                # 确保归一化 (如果是 uint8)
                if exp_samples.dtype == torch.uint8:
                    exp_samples = exp_samples.float() / 255.0
                
                # --- [DEBUG] 可视化采样图片 ---
                # print(f"  [Debug] 正在准备可视化采样图... (采样数: {num_samples}, 输入形状: {exp_samples.shape})")
                try:
                    # 1. 准备 Numpy 数据并检测范围
                    max_val = exp_samples.max().item()
                    vis_scale = 255.0 if max_val <= 1.05 else 1.0
                    vis_data = (exp_samples.detach().cpu().numpy() * vis_scale).astype(np.uint8)
                    
                    # 2. 统一转为 (N, H, W, C)
                    if vis_data.ndim == 4: # (N, C, H, W)
                        vis_data = vis_data.transpose(0, 2, 3, 1)
                    
                    # 3. 提取显示用的通道
                    N, H, W, C = vis_data.shape
                    if C > 3: # 处理帧堆叠
                        if C % 3 == 0: vis_data = vis_data[..., -3:] # RGB 堆叠取最后一帧
                        else: vis_data = vis_data[..., -1:] # 灰度堆叠取最后一帧
                    
                    # 4. 预分配网格画布 (4行8列)
                    target_size = (240, 135) # (w, h) 保持 16:9
                    rows = (num_samples + 7) // 8
                    grid = np.zeros((rows * target_size[1], 8 * target_size[0], 3), dtype=np.uint8)
                    
                    for i in range(num_samples):
                        img = vis_data[i]
                        # 处理单通道灰度图转 BGR
                        if img.shape[-1] == 1:
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        
                        # 缩放并放入网格 (不再进行颜色反转，假设采集已经是 BGR)
                        img = cv2.resize(img, target_size)
                        r, c = i // 8, i % 8
                        grid[r*target_size[1]:(r+1)*target_size[1], c*target_size[0]:(c+1)*target_size[0]] = img
                    
                    if max_val < 1e-5:
                        print("  [WARNING] 检测到采样画面全黑（全0）！请检查环境是否正常输出。")
                    
                    # 保存到本地根目录供调试
                    cv2.imwrite("debug_experience_grid.png", grid)
                except Exception as e:
                    print(f"  [Warning] 可视化保存失败: {e}")
                
                # 提取特征
                exp_feats = extractor(exp_samples) # (num_samples, features_dim)
                exp_norms = torch.norm(exp_feats, dim=1).mean().item()
                
                # 矩阵化计算余弦相似度
                exp_feats_norm = F.normalize(exp_feats, p=2, dim=1)
                exp_sim_matrix = torch.mm(exp_feats_norm, exp_feats_norm.t())
                
                mask_exp = torch.triu(torch.ones_like(exp_sim_matrix), diagonal=1).bool()
                exp_sims = exp_sim_matrix[mask_exp]
                
                if exp_sims.numel() > 0:
                    self.last_sim_metrics["experience_avg"] = exp_sims.mean().item()
                    # 检查输入多样性：如果输入图片本身高度重合，相似度高是正常的
                    input_flat = exp_samples.view(num_samples, -1)
                    input_norm = F.normalize(input_flat, p=2, dim=1)
                    input_sim = torch.mm(input_norm, input_norm.t())[mask_exp].mean().item()
                    
                    if input_sim > 0.98:
                        # 输入本身没分化，标记为有效性低
                        print(f"  [Note] 经验图输入多样性不足 (Sim={input_sim:.4f})，诊断结果仅供参考")
                    
                    # print(f"  [Debug] 经验图诊断: 采样数={num_samples}, 平均范数={exp_norms:.4f}, 相似度均值={self.last_sim_metrics['experience_avg']:.4f}")
            else:
                self.last_sim_metrics["experience_avg"] = -1.0 # 标记为数据不足
            
        extractor.train()

    def _on_rollout_end(self):
        # 0. 在模型更新前分析当前采集到的特征相似度
        self._analyze_feature_similarity()
        
        self.iteration += 1
        print(f"\n[Step {self.num_timesteps}] 采集完成，正在开始反向传播训练...")
        
        # 1. 记录动作分布直方图到 TensorBoard
        self._log_action_distribution()

        # 2. 记录特征相似度到 TensorBoard
        self.logger.record("Diagnostic/Avg_Synthetic_Similarity", self.last_sim_metrics["synthetic_avg"])
        if self.last_sim_metrics["experience_avg"] >= 0:
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
        
        # 颜色逻辑：1.0 左右为坍缩(红)，0.0 左右为正常分化(绿)
        syn_color = "\033[91m" if syn_sim > 0.99 else "\033[92m"
        exp_color = "\033[91m" if exp_sim > 0.99 else "\033[92m"
        
        exp_sim_str = f"{exp_sim:.6f}" if exp_sim >= 0 else "\033[90mWaiting for buffer...\033[0m"
        
        print("特征敏感度 (余弦相似度):")
        print(f"  - 合成图 (噪声/全黑/格栅): {syn_color}{syn_sim:.6f}\033[0m")
        print(f"  - 经验图 (真实观测采样):   {exp_color if exp_sim >= 0 else ''}{exp_sim_str}\033[0m")
        
        if exp_sim > 0.999:
            print("  \033[91m[!!!] 警告: 检测到特征坍缩 (Feature Collapse) [!!!]\033[0m")
        elif exp_sim >= 0 and exp_sim < 0.01:
             print("  \033[94m[Note] 特征极度分化，请检查 Norm 是否过小\033[0m")
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

    def log_gradients(self, step=None):
        """外部调用：记录当前梯度。"""
        if self.hook_manager:
            log_step = step if step is not None else self.num_timesteps
            self.hook_manager.log_gradients(log_step)

    def log_weights(self, step=None):
        """外部调用：记录当前权重。"""
        if self.hook_manager:
            log_step = step if step is not None else self.num_timesteps
            self.hook_manager.log_weights(log_step)

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
        
        # 关键：在保存模型前解绑，避免 Pickle 序列化错误
        if hasattr(self.model, "diagnostics"):
            # 使用 del 或设置为 None
            self.model.diagnostics = None
