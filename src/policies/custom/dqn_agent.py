import os
import torch
from .base.agent import BaseAgent
from .dqn import DQN as DQN_Algorithm
from src.model import ddqn_res18, ddqn_simple
import configs.config as config

class DQNAgent(BaseAgent):
    """
    Sekiro 项目专用的 DQN 代理应用类。
    它使用底层框架提供的 DQN 算法，并适配项目的特定模型（ResNet18）。
    """
    def __init__(self, img_width, img_height, action_dim, buffer, model_file=None, n_step_rewards: int = 1):
        # 1. 确定设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(action_dim, device)
        
        self.model_file = model_file or config.MODEL_PATH
        
        # 2. 定义模型构建函数 (适配 ResNet18)
        def model_fn():
            return ddqn_simple(in_channels=config.FRAME_HISTORY_LEN, num_actions=action_dim)
        
        # 3. 实例化框架算法
        self.algorithm = DQN_Algorithm(
            model_fn=model_fn,
            action_dim=action_dim,
            buffer=buffer,
            device=device,
            lr=config.LR,
            gamma=config.GAMMA,
            target_update_freq=config.TARGET_UPDATE_FREQ,
            n_step_rewards=n_step_rewards
        )

    def act(self, state, epsilon=0.0):
        # 如果 state 是包含 batch 维度的 tensor，需要处理
        if isinstance(state, torch.Tensor):
            if state.ndim == 4: # (B, C, H, W)
                state = state.squeeze(0).cpu().numpy()
            else:
                state = state.cpu().numpy()
        return self.algorithm.act(state, epsilon)

    def record(self, state, action, reward, next_state, done):
        self.algorithm.record(state, action, reward, next_state, done)

    def learn(self):
        return self.algorithm.learn(batch_size=config.BATCH_SIZE)

    def save(self, path=None):
        self.algorithm.save(path or self.model_file)

    def load(self, path=None):
        self.algorithm.load(path or self.model_file)

    def train(self):
        """切换到训练模式。"""
        if hasattr(self.algorithm, 'eval_net'):
            self.algorithm.eval_net.train()

    def eval(self):
        """切换到评估模式。"""
        if hasattr(self.algorithm, 'eval_net'):
            self.algorithm.eval_net.eval()

    @property
    def optimize_count(self):
        return self.algorithm.optimize_count
    
    @property
    def last_loss(self):
        return self.algorithm.last_loss

    @property
    def last_q(self):
        return getattr(self.algorithm, '_last_q', None)
