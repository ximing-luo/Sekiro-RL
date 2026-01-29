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
    基于管理器的强化学习环境类。
    遵循 Gymnasium 接口标准：step(), reset()。
    通过配置驱动所有的管理器。
    """
    def __init__(self, cfg: ManagerBasedRLEnvCfg):
        super().__init__(cfg)
        
        # 1. 初始化回放缓冲
        self.replay_buffer = ReplayBuffer(
            size=cfg.buffer_size,
            frame_history_len=cfg.frame_history_len,
            obs_shape=(3, cfg.scene.observation_h, cfg.scene.observation_w)
        )
        
        # 2. 实例化管理器 (传入配置中的 Terms)
        self.action_manager = ActionManager(cfg.actions)
        self.observation_manager = ObservationManager(cfg.observations)
        self.reward_manager = RewardManager(cfg.rewards)
        self.termination_manager = TerminationManager(cfg.terminations)
        self.log_manager = LogManager()
        
        # 3. 启动场景
        self._setup_managers()
        
        # 状态记录
        self.last_metrics = None
        self.over = False
        self.last_total_reward_raw = 0.0
        self.last_events = []
        self.last_events_feedback = []

    @property
    def action_space(self):
        """返回符合 Gymnasium 标准的动作空间（由子类或任务定义）。"""
        # 目前返回离散空间大小，未来可集成 gymnasium.spaces.Discrete
        return self.action_manager.get_action_dim()

    @property
    def observation_space(self):
        """返回符合 Gymnasium 标准的观测空间。"""
        # 目前返回图像维度，未来可集成 gymnasium.spaces.Box
        return (3, self.cfg.scene.observation_h, self.cfg.scene.observation_w)

    @property
    def action_dim(self):
        """兼容性属性：返回动作空间维度。"""
        return self.action_manager.get_action_dim()

    def step(self, action):
        """
        标准 RL 步进逻辑。
        返回: (obs, reward, terminated, truncated, info)
        """
        # 1. 执行动作
        self.action_manager.apply_action(self, action)

        # 2. 获取新观测指标
        next_metrics = self.observation_manager.compute_observations(self)
        if self.last_metrics is None:
            self.last_metrics = next_metrics

        # 3. 检测事件并计算奖励
        events = self.reward_manager.detect_events(self.last_metrics, next_metrics)
        reward, components = self.reward_manager.compute_reward(self, self.last_metrics, next_metrics, action, events)

        # 4. 判断终止条件
        terminated = self.termination_manager.check_termination(self, self.last_metrics, next_metrics, events)
        truncated = False # 目前暂无超时截断逻辑

        # 5. 更新状态记录
        self.last_total_reward_raw = float(reward)
        self.last_events = list(events)
        self.last_events_feedback = [[name, float(val)] for name, val in components.items()]
        
        # 6. 写入回放缓冲 (如果观测项中包含 'policy' 图像)
        frame = next_metrics.get('policy')
        if frame is not None:
            self.replay_buffer.add(frame, action, float(reward), terminated)
            
        self.last_metrics = next_metrics

        # 7. 构造 info 字典
        info = {
            "metrics": next_metrics,
            "events": events,
            "reward_components": components
        }

        return next_metrics, float(reward), terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        重置环境状态。
        返回: (obs, info)
        """
        self.last_metrics = self.observation_manager.compute_observations(self)
        frame = self.last_metrics.get('policy')
        if frame is not None:
            self.replay_buffer.add(frame, 0, 0.0, False)
            
        self.termination_manager.reset()
        self.over = False
        
        info = {"metrics": self.last_metrics}
        return self.last_metrics, info
