"""
模块用途：重构后的 Sekiro 环境类，采用 Isaac Lab 风格的管理器架构。
将 IO、动作、观测、奖励与终止逻辑拆分为独立的管理器，实现解耦。
"""
import time
import os
import sys

# 将项目根目录添加到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.policies.buffers.replay_buffer import ReplayBuffer
from src.envs.managers import (
    SceneManager,
    ActionManager,
    ObservationManager,
    RewardManager,
    TerminationManager,
    LogManager
)
import configs.config as config

class Sekiro:
    """
    Sekiro 环境主类：作为各个管理器的协调者（Orchestrator）。
    遵循 Isaac Lab 的 Manager-Based Env 设计理念。
    """
    def __init__(self, observation_w, observation_h, action_dim=None, pos="offscreen", debug_vis_fps=0, capture_fps=60, n_step_rewards: int = 20):
        # 基础配置
        self.width = observation_w
        self.height = observation_h
        self.n_step_rewards = int(max(1, n_step_rewards))
        
        # 1. 初始化回放缓冲 (共享资源)
        # 增加容量，50 帧太小了，改为配置中的容量或默认值
        buffer_size = getattr(config, 'BUFFER_SIZE', 50000)
        self.replay_buffer = ReplayBuffer(
            size=buffer_size, 
            frame_history_len=config.FRAME_HISTORY_LEN,
            obs_shape=(3, observation_h, observation_w)
        )
        
        # 2. 初始化各个管理器
        self.scene_manager = SceneManager(observation_w, observation_h, self.replay_buffer, pos, capture_fps)
        self.action_manager = ActionManager(action_dim)
        self.observation_manager = ObservationManager(self.replay_buffer)
        self.reward_manager = RewardManager()
        self.termination_manager = TerminationManager()
        self.log_manager = LogManager()
        
        # 启动场景 (窗口、采集)
        self.scene_manager.setup()
        
        if debug_vis_fps > 0:
            self.scene_manager.start_debug_visualization(debug_vis_fps)

        # 状态记录 (用于兼容 train.py 和 dashboard.py)
        self.last_metrics = self.observation_manager.compute_observations()
        self.last_time = time.time()
        self._step_counter = 0
        self.over = False
        
        # 外部访问接口兼容
        self.last_events = []
        self.last_events_feedback = []
        self.last_total_reward_raw = 0.0
        self.last_noop_count_1s = 0

    @property
    def action_dim(self):
        return self.action_manager.get_action_dim()

    def update_debug_visual_input(self, seq_np):
        self.scene_manager.update_debug_visualization(seq_np)

    def step(self, action):
        """环境交互主循环：动作 -> 观测 -> 奖励 -> 终止。"""
        # 1. 执行动作
        self.action_manager.apply_action(action)

        # 2. 获取新观测
        next_metrics = self.observation_manager.compute_observations()

        # 3. 检测事件并计算奖励
        events = self.reward_manager.detect_events(self.last_metrics, next_metrics)
        reward, components = self.reward_manager.compute_reward(self.last_metrics, next_metrics, action, events)

        # 4. 判断终止条件
        done = self.termination_manager.check_termination(self.last_metrics, next_metrics, events)

        # 5. 更新状态记录
        self._update_compat_stats(reward, events, components, next_metrics)
        
        # 6. 写入回放缓冲
        self.replay_buffer.store_effect(action, float(reward), done)

        # 调试打印
        self._step_counter += 1
        if self._step_counter % 10 == 0:
            print(f"狼血量: {next_metrics['self_blood']} Boss血量: {next_metrics['boss_blood']}")
            print(f"狼架势: {next_metrics['self_stamina']} Boss架势: {next_metrics['boss_stamina']}")
        
        # print(f"Step Time: {time.time() - self.last_time:.4f}s")
        self.last_time = time.time()
        
        return float(reward)

    def _update_compat_stats(self, reward, events, components, next_metrics):
        """更新用于兼容旧代码的统计变量。"""
        self.last_total_reward_raw = float(reward)
        self.last_events = list(events)
        try:
            self.last_events_feedback = [[int(e), float(components.get(int(e), 0.0))] for e in events]
        except Exception:
            self.last_events_feedback = []
            
        self.last_metrics = next_metrics
        # 兼容旧代码直接访问属性的需求
        self.self_blood = next_metrics['self_blood']
        self.boss_blood = next_metrics['boss_blood']
        self.self_stamina = next_metrics['self_stamina']
        self.boss_stamina = next_metrics['boss_stamina']

    def reset(self):
        """重置环境。"""
        self.last_metrics = self.observation_manager.compute_observations()
        self.termination_manager.reset()
        self._update_compat_stats(0.0, [], {}, self.last_metrics)

    def pause_game(self, paused):
        """调用场景管理器处理暂停。"""
        should_over, paused = self.scene_manager.check_pause(paused)
        if should_over:
            self.over = True
        return paused

if __name__ == '__main__':
    # 简单的冒烟测试
    env = Sekiro(640, 360)
    print("Env initialized with managers.")
