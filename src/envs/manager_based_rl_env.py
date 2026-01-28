import time
import torch
from .manager_based_env import ManagerBasedEnv
from .manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from src.policies.buffers.replay_buffer import ReplayBuffer
from src.managers import (
    ActionManager,
    ObservationManager,
    RewardManager,
    TerminationManager,
    LogManager
)

class ManagerBasedRLEnv(ManagerBasedEnv):
    """
    基于管理器的强化学习环境基类。
    实现了标准 RL 循环：step(), reset()。
    """
    def __init__(self, cfg: ManagerBasedRLEnvCfg):
        super().__init__(cfg)
        
        # 1. 初始化 RL 特有的资源：回放缓冲
        self.replay_buffer = ReplayBuffer(
            size=cfg.buffer_size,
            frame_history_len=cfg.frame_history_len,
            obs_shape=(3, cfg.scene.observation_h, cfg.scene.observation_w)
        )
        
        # 2. 实例化管理器
        self.action_manager = ActionManager()
        self.observation_manager = ObservationManager(self.replay_buffer)
        self.reward_manager = RewardManager()
        self.termination_manager = TerminationManager()
        self.log_manager = LogManager()
        
        # 3. 启动场景
        self._setup_managers()
        
        # 状态记录
        self.last_metrics = None
        self.over = False
        self.last_total_reward_raw = 0.0
        self.last_events = []
        self.last_events_feedback = []

    def step(self, action):
        """标准 RL 步进逻辑。"""
        # 1. 执行动作
        self.action_manager.apply_action(action)

        # 2. 获取新观测指标
        next_metrics = self.observation_manager.compute_observations()
        if self.last_metrics is None:
            self.last_metrics = next_metrics

        # 3. 检测事件并计算奖励
        events = self.reward_manager.detect_events(self.last_metrics, next_metrics)
        reward, components = self.reward_manager.compute_reward(self.last_metrics, next_metrics, action, events)

        # 4. 判断终止条件
        done = self.termination_manager.check_termination(self.last_metrics, next_metrics, events)

        # 5. 更新状态记录
        self.last_total_reward_raw = float(reward)
        self.last_events = list(events)
        self.last_events_feedback = [[int(e), float(components.get(int(e), 0.0))] for e in events]
        self.last_metrics = next_metrics
        
        # 6. 写入回放缓冲
        self.replay_buffer.store_effect(action, float(reward), done)

        return float(reward)

    def reset(self):
        """重置环境状态。"""
        self.last_metrics = self.observation_manager.compute_observations()
        self.termination_manager.reset()
        self.over = False
        return self.last_metrics
