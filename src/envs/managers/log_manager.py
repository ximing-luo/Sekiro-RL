import os
from src.visualization.logger import write_json, write_csv
import configs.config as config

class LogManager:
    """
    日志管理器：负责收集训练度量并写入文件。
    遵循 Isaac Lab 的管理器设计理念，将日志记录逻辑从主环境类中解耦。
    """
    def __init__(self, log_dir=None):
        self.log_dir = log_dir or config.LOG_DIR
        os.makedirs(self.log_dir, exist_ok=True)
        
    def log_step(self, agent, env, step, episode, action, reward, recent_rewards, fps, epsilon):
        """
        记录一步的训练数据。
        """
        events_feedback = getattr(env, "last_events_feedback", [])
        events = getattr(env, 'last_events', None)
        raw_reward = getattr(env, 'last_total_reward_raw', None)
        
        # 计算近期奖励均值
        reward_avg_recent = (sum(recent_rewards) / len(recent_rewards)) if len(recent_rewards) > 0 else 0.0
        
        # 写入 JSON (用于 Dashboard 实时显示)
        write_json(
            self.log_dir,
            agent.run_id,
            step,
            episode,
            action,
            events_feedback,
            reward,
            agent._last_q,
            agent._last_q_mod,
            fps=fps,
            events=events,
            raw_reward=raw_reward,
            reward_avg_recent=reward_avg_recent,
            epsilon=epsilon
        )
        
        # 写入 CSV (用于长期数据分析)
        write_csv(
            self.log_dir,
            agent.run_id,
            step,
            episode,
            action,
            events_feedback,
            reward,
            agent._last_q,
            agent._last_q_mod,
            raw_reward=raw_reward,
            reward_avg_recent=reward_avg_recent,
            epsilon=epsilon
        )
