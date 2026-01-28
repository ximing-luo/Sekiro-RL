from src.envs.mdp.rewards import design_event_rewards

class RewardManager:
    """
    奖励管理器：负责事件检测和奖励分量计算。
    对应 Isaac Lab 中的 RewardManager。
    """
    def __init__(self):
        pass

    def detect_events(self, prev_metrics, next_metrics):
        """
        根据状态变化检测发生的事件。
        逻辑从 env.py 的 detect_events 迁移而来。
        """
        events = []
        
        sb = prev_metrics['self_blood']
        nsb = next_metrics['self_blood']
        bb = prev_metrics['boss_blood']
        nbb = next_metrics['boss_blood']
        ss = prev_metrics['self_stamina']
        nss = next_metrics['self_stamina']
        bs = prev_metrics['boss_stamina']
        nbs = next_metrics['boss_stamina']

        # [0] 自身死亡
        if sb < 100 and nsb - sb > 500:
            return [0]
        # [1] Boss死亡
        if nbb == 0 and bb - nbb > 50 and nbs > 400:
            return [1]
        
        # [2] 自身掉血
        if nsb - sb < -2:
            events.append(2)
        # [3] 自身回血
        if 100 <= sb < 300 and nsb - sb >= 100:
            events.append(3)
        # [4] 自身血量过低
        if nsb <= 200:
            events.append(4)
        # [5] Boss掉血
        if nbb - bb <= -5:
            events.append(5)
        # [6] 自身架势上升
        if nss - ss >= 2:
            events.append(6)
        # [7] Boss架势上升
        if nbs - bs >= 2:
            events.append(7)
        # [8] Boss架势过低
        if nbs <= 20:
            events.append(8)
            
        return events

    def compute_reward(self, prev_metrics, next_metrics, action, events):
        """调用奖励计算纯函数。"""
        return design_event_rewards(
            prev_metrics['boss_blood'], next_metrics['boss_blood'],
            prev_metrics['self_blood'], next_metrics['self_blood'],
            prev_metrics['boss_stamina'], next_metrics['boss_stamina'],
            prev_metrics['self_stamina'], next_metrics['self_stamina'],
            action,
            events
        )
