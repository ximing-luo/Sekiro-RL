from typing import Dict, List, Any
from src.envs.manager_based_env_cfg import RewardTermCfg

class RewardManager:
    """
    奖励管理器：实现基于术语（Term-based）的奖励计算。
    遵循 Isaac Lab 风格，不再包含硬编码逻辑。
    """
    def __init__(self, cfg: Dict[str, RewardTermCfg]):
        self.cfg = cfg

    def detect_events(self, prev_metrics, next_metrics):
        """
        根据状态变化检测发生的事件。
        注意：为了彻底解耦，这里的硬编码逻辑未来可以迁移到单独的 EventManager。
        """
        events = []
        sb, nsb = prev_metrics.get('self_blood', 0), next_metrics.get('self_blood', 0)
        bb, nbb = prev_metrics.get('boss_blood', 0), next_metrics.get('boss_blood', 0)
        ss, nss = prev_metrics.get('self_stamina', 0), next_metrics.get('self_stamina', 0)
        bs, nbs = prev_metrics.get('boss_stamina', 0), next_metrics.get('boss_stamina', 0)

        # 0: 自身死亡, 1: Boss死亡, 2: 自身掉血, 3: 自身回血, 4: 自身血量过低, 
        # 5: Boss掉血, 6: 自身架势上升, 7: Boss架势上升, 8: Boss架势过低
        if sb < 100 and nsb - sb > 500: events.append(0)
        if nbb == 0 and bb - nbb > 50 and nbs > 400: events.append(1)
        if nsb - sb < -2: events.append(2)
        if 100 <= sb < 300 and nsb - sb >= 100: events.append(3)
        if nsb <= 200: events.append(4)
        if nbb - bb <= -5: events.append(5)
        if nss - ss >= 2: events.append(6)
        if nbs - bs >= 2: events.append(7)
        if nbs <= 20: events.append(8)
            
        return events

    def compute_reward(self, env, prev_metrics, next_metrics, action, events):
        """遍历配置中的所有奖励项并累加。"""
        total_reward = 0.0
        components = {}

        for name, term_cfg in self.cfg.items():
            # 调用 term 函数
            val = term_cfg.func(
                env=env,
                prev_metrics=prev_metrics,
                next_metrics=next_metrics,
                action=action,
                events=events,
                **term_cfg.params
            )
            # 应用权重
            weighted_val = val * term_cfg.weight
            total_reward += weighted_val
            components[name] = weighted_val

        return total_reward, components
