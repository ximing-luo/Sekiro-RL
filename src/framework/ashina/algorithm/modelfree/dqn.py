import torch
import torch.nn as nn
import copy
from typing import Any, Dict, Optional, Union, List
from ..base import Policy, Algorithm
from ..net.discrete import QNetwork
from ...data.batch import Batch

class DQNPolicy(Policy):
    """
    DQN 策略实现 (Data Plane)
    职责：接收观测值，输出包含动作和 Q 值的 Batch。
    """
    def __init__(
        self, 
        model: nn.Module, 
        action_dim: int, 
        device: Union[str, torch.device] = "cuda"
    ):
        super().__init__(action_dim, device)
        self.model = model.to(self.device)
        self.eps = 0.0

    def forward(
        self,
        batch: Batch,
        state: Optional[Any] = None,
        **kwargs: Any
    ) -> Batch:
        obs = torch.as_tensor(batch.obs, device=self.device, dtype=torch.float32)
        q_values = self.model(obs)

        if self.training and torch.rand(1).item() < self.eps:
            act = torch.randint(0, self.action_dim, (q_values.shape[0],), device=self.device)
        else:
            act = q_values.argmax(dim=-1)

        return Batch(q_values=q_values, act=act, state=state)

class DQNAlgorithm(Algorithm):
    def __init__(
        self,
        action_dim: int,
        input_dim: Optional[int] = None,
        model: Optional[nn.Module] = None,
        lr: float = 1e-4,
        gamma: float = 0.99,
        target_update_freq: int = 1000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        device: Union[str, torch.device] = "cuda"
    ):
        if model is None:
            assert input_dim is not None, "must provide input_dim when model is None"
            model = QNetwork(input_dim=input_dim, action_dim=action_dim)
        policy = DQNPolicy(model=model, action_dim=action_dim, device=device)
        super().__init__(policy, device)
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self._epsilon = epsilon_start

        self.target_net = copy.deepcopy(self.policy.model).to(self.device)
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy.model.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss(reduction='none')
        self.optimize_count = 0

    def pre_collect(self, step: int) -> None:
        self._epsilon = max(self.epsilon_end, self._epsilon * self.epsilon_decay)
        self.policy.eps = self._epsilon

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, Any]:
        """
        DQN 学习步骤。
        """
        batch = batch.to_torch(self.device)

        obs = batch.obs
        act = batch.act
        rew = batch.rew
        next_obs = batch.obs_next
        done = batch.done
        
        weights = batch.weight
        if weights is None:
            weights = torch.ones_like(rew)

        q_values = self.policy.model(obs)
        q_sa = q_values.gather(1, act.view(-1, 1).long()).squeeze(1)

        with torch.no_grad():
            next_q_eval = self.policy.model(next_obs)
            next_act = next_q_eval.argmax(1, keepdim=True)

            next_q_tgt = self.target_net(next_obs)
            next_q_sa = next_q_tgt.gather(1, next_act).squeeze(1)
            
            target = rew.flatten() + self.gamma * (1.0 - done.to(torch.float32).flatten()) * next_q_sa.flatten()

        td_errors = (q_sa - target).abs().detach()
        loss = (weights.flatten() * self.criterion(q_sa, target)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.optimize_count += 1
        if self.optimize_count % self.target_update_freq == 0:
            self.sync_target()
            
        return {
            "loss": loss.item(),
            "td_errors": td_errors.cpu().numpy(),
            "q_avg": q_sa.mean().item()
        }

    def learn_batch(self, batches: List[Batch], **kwargs: Any) -> Dict[str, Any]:
        """
        并行处理多个 batch。
        batches: List[Batch]，每个 Batch 包含 batch_size 个样本
        返回合并后的统计信息
        """
        merged_batch = Batch.cat(batches)
        result = self.learn(merged_batch, **kwargs)
        result["batch_count"] = len(batches)
        return result

    def sync_target(self) -> None:
        """硬同步目标网络。"""
        self.target_net.load_state_dict(self.policy.model.state_dict())
