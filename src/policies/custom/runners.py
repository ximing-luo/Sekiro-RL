import time
import numpy as np
from collections import deque

class Runner:
    """
    训练执行器：负责环境与代理的交互循环。
    """
    def __init__(self, env, agent, config):
        self.env = env
        self.agent = agent
        self.config = config
        self.recent_rewards = deque(maxlen=100)

    def run(self, total_steps):
        state = self.env.reset()
        for step in range(total_steps):
            # 1. 选择动作
            # 注意：在 Sekiro 任务中，state 通常由 ObservationManager 提供堆叠帧
            stacked_obs = self.env.observation_manager.get_latest_stacked_frames()
            epsilon = self._get_epsilon(step)
            action = self.agent.act(stacked_obs, epsilon=epsilon)

            # 2. 环境步进
            reward = self.env.step(action)
            next_stacked_obs = self.env.observation_manager.get_latest_stacked_frames()
            done = self.env.termination_manager.check_termination(None, None, self.env.last_events)

            # 3. 记录经验
            self.agent.record(stacked_obs, action, reward, next_stacked_obs, done)

            # 4. 训练优化
            if step > self.config.get('start_learning_steps', 1000):
                loss = self.agent.learn(batch_size=self.config.get('batch_size', 32))
            
            self.recent_rewards.append(reward)
            
            if done:
                self.env.reset()
            
            if step % self.config.get('save_freq', 1000) == 0:
                self.agent.save(self.config.get('model_path', 'model.pth'))

    def _get_epsilon(self, step):
        # 简单的线性衰减
        start = self.config.get('epsilon_start', 1.0)
        end = self.config.get('epsilon_end', 0.1)
        decay = self.config.get('epsilon_decay_steps', 10000)
        return max(end, start - (start - end) * step / decay)
