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
        """
        events = []
        # 从嵌套的 telemetry 字典中提取指标
        prev_tel = prev_metrics.get('telemetry', {})
        next_tel = next_metrics.get('telemetry', {})
        
        sb, nsb = prev_tel.get('self_blood', 0), next_tel.get('self_blood', 0)
        bb, nbb = prev_tel.get('boss_blood', 0), next_tel.get('boss_blood', 0)
        ss, nss = prev_tel.get('self_stamina', 0), next_tel.get('self_stamina', 0)
        bs, nbs = prev_tel.get('boss_stamina', 0), next_tel.get('boss_stamina', 0)

        if bb - nbb > 1000: nbs = bs
        bss = prev_tel.get('boss_stamina_max', 0)
        # 0: 自身死亡, 1: Boss死亡, 2: 自身掉血, 3: 自身回血, 4: 自身血量过低, 
        # 5: Boss掉血, 6: 自身架势恶化(数值减小), 7: Boss架势恶化(数值减小), 8: Boss架势过低(可忍杀)
        
        # 自身死亡：血量瞬间从低位跳变到高位（重生）
        if sb < 400 and nsb > 600: events.append(0)
        # Boss死亡：血量归零且架势条重置
        if nbb == 0 and bb > 0: events.append(1)
        # 自身掉血
        if nsb < sb: events.append(2)
        # 自身血量过低（危险信号）
        if nsb <= 200: events.append(4)
        # Boss掉血（有效攻击）
        if nbb < bb: events.append(5)
        # 自身架势恶化：只狼架势条是向下扣的，数值减小代表架势条变长/变黄
        if nss < ss: events.append(6)
        # Boss架势恶化：数值减小代表 Boss 快被破防了
        if nbs < bs: events.append(7)
        # 即使数值没变，如果架势条维持在低位（被持续压制），也给一个微弱的持续奖励
        if nbs < bss - 100: events.append(9)
            
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
