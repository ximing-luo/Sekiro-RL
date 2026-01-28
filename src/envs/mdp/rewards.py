"""
模块用途：奖励与事件的规则化计算。
"""

def design_event_rewards(
    boss_blood, next_boss_blood, self_blood, next_self_blood,
    boss_stamina, next_boss_stamina, self_stamina, next_self_stamina,
    action,
    events,
    self_blood_gamma=0.6, boss_blood_gamma=0.4, self_stamina_gamma=0.5, boss_stamina_gamma=0.5,
):
    """
    根据事件与前后状态，设计每事件奖励分量并返回总奖励与事件分量字典。
    - 返回：
      total_reward: float，总奖励（含权重）
      components: Dict[int, float]，键为事件ID，值为应用权重后的分量
    """
    components = {}
    # [0] 自身死亡
    if 0 in events:
        components[0] = float(-200)
    # [1] boos死亡
    if 1 in events:
        components[1] = float(200)
    # [2] 自身掉血
    if 2 in events:
        a = self_blood - next_self_blood
        val = -1 * a * 0.05
        val = max(val, -20)
        components[2] = float(self_blood_gamma * val)
    # [3] 自身回血
    if 3 in events:
        components[3] = float(self_blood_gamma * 10)
    # [4] 自身血量过低
    if 4 in events:
        components[4] = float(-0.5)
    # [5] Boss掉血
    if 5 in events:
        a = boss_blood - next_boss_blood
        val = a * 0.5
        components[5] = float(boss_blood_gamma * val)
    # [6] 自身架势上升
    if 6 in events:
        a = next_self_stamina - self_stamina
        val = -3 * a * 0.02
        if action == 2:
            val = 0
        components[6] = float(self_stamina_gamma * val)
    # [7] Boss架势上升
    if 7 in events:
        val = (next_boss_stamina - boss_stamina) / 5
        if action == 5:
            val = max(5, val)
        if action in [2, 4]:
            val = max(20, val)
        components[7] = float(boss_stamina_gamma * val)
    # [8] Boss架势过低
    if 8 in events:
        components[8] = float(-0.5)

    total_reward = sum(components.values())
    return float(total_reward), components
