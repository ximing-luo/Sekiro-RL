import torch as th
import torch.nn.functional as F
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

class AuxPPO(PPO):
    """
    带有辅助损失的 PPO 算法。
    辅助损失：计算特征提取器输出的特征向量在 Batch 内的余弦相似度，
    并将其作为损失项，强制模型对不同输入产生区分度。
    """
    def __init__(self, *args, aux_coef=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_coef = aux_coef # 辅助损失权重


    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        clip_fractions_vf = [] # 记录价值网络截断比例
        aux_losses = [] # 记录辅助损失
        grad_norms = [] # 记录梯度范数

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()
                elif isinstance(self.action_space, spaces.MultiDiscrete):
                    # MultiDiscrete 动作通常已经是正确的 shape (batch_size, n_dims)
                    actions = rollout_data.actions.long()

                # 1. 拆解 evaluate_actions 以复用 features，避免重复运行重型 CNN (ResNet)
                features = self.policy.extract_features(rollout_data.observations)
                latent_pi, latent_vf = self.policy.mlp_extractor(features)
                distribution = self.policy._get_action_dist_from_latent(latent_pi)
                log_prob = distribution.log_prob(actions)
                values = self.policy.value_net(latent_vf).flatten()
                entropy = distribution.entropy()

                # 2. 计算优势函数
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    value_loss = F.mse_loss(rollout_data.returns, values)
                else:
                    # 裁剪后的预测值 (用于计算被截断后的损失)
                    values_pred_clipped = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                    # 未裁剪的损失 (保证在退步时有梯度)
                    value_loss_unclipped = (rollout_data.returns - values) ** 2
                    # 裁剪后的损失 (在进步过快时梯度为0)
                    value_loss_clipped = (rollout_data.returns - values_pred_clipped) ** 2
                    # 取二者最大值，是 PPO 保证价值网络在被扰动后仍能找回方向的关键
                    value_loss = th.max(value_loss_unclipped, value_loss_clipped).mean()
                    
                    # 记录价值网络被截断的比例
                    clip_fraction_vf = th.mean((th.abs(values - rollout_data.old_values) > clip_range_vf).float()).item()
                    clip_fractions_vf.append(clip_fraction_vf)
                value_losses.append(value_loss.item())

                # Entropy loss
                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                # --- 3. 辅助损失：使用已经提取好的 features，不再重复计算 ---
                norm_features = F.normalize(features, dim=1)
                sim_matrix = th.matmul(norm_features, norm_features.T)
                n = features.size(0)
                mask = th.eye(n, device=self.device).bool()
                aux_loss = sim_matrix[~mask].mean()
                aux_losses.append(aux_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.aux_coef * aux_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm 并捕获裁剪前的范数
                grad_norm = th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                grad_norms.append(grad_norm.item())
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/aux_loss", np.mean(aux_losses)) # 记录辅助损失
        self.logger.record("train/grad_norm", np.mean(grad_norms))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        if len(clip_fractions_vf) > 0:
            self.logger.record("train/clip_fraction_vf", np.mean(clip_fractions_vf))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
