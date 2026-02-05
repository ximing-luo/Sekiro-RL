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
        pd, npd = prev_tel.get('player_deaths', 0), next_tel.get('player_deaths', 0)
        ed, ned = prev_tel.get('enemy_deaths', 0), next_tel.get('enemy_deaths', 0)
        sbm = prev_tel.get('self_blood_max', 0)
        bbm = prev_tel.get('boss_blood_max', 0)
        ssm = prev_tel.get('self_stamina_max', 0)
        bsm = prev_tel.get('boss_stamina_max', 0)
        
        # 0: 自身死亡, 1: Boss死亡, 2: 自身掉血, 3: 自身回血, 4: 自身血量过低, 
        # 5: Boss掉血, 6: 自身架势恶化(数值减小), 7: Boss架势恶化(数值减小), 8: Boss架势过低(可忍杀)
        
        # 自身死亡：死亡计数增加
        if npd - pd == 1:
            events.append(0)
            print(f"\033[91m自身死亡检测：死亡计数从 {pd} 增加到 {npd}\033[0m")
        # Boss死亡：死亡计数增加
        if ned - ed == 1:
            events.append(1)
            print(f"\033[91mBoss死亡检测：死亡计数从 {ed} 增加到 {ned}\033[0m")
        
        # 自身掉血
        if nsb < sb: events.append(2)
        # Boss掉血（有效攻击）
        if nbb != bb: events.append(5)
        # 自身架势恶化：只狼架势条是向下扣的，数值减小代表架势条变长/变黄
        if nss != ss: events.append(6)
        # 自身架势崩溃
        if ss <= 30 and nss == ssm: events.append(8)
        # Boss架势变化：数值减小代表 Boss 快被破防了
        if nbs != bs: events.append(7)
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
