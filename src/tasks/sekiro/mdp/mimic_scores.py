"""
模块用途：计算动作模仿得分。
用于模仿学习中，评估 AI 动作与专家动作的一致性。
"""

def compute_action_similarity(ai_action, expert_action):
    """
    计算 AI 动作与专家动作的相似度。
    简单的实现是直接对比，也可以使用更复杂的余弦相似度（如果是连续动作）。
    """
    if ai_action == expert_action:
        return 1.0
    return 0.0

def compute_state_proximity(ai_state, expert_state):
    """
    评估 AI 当前状态与专家轨迹中对应状态的接近程度。
    用于引导 AI 走在专家的路线上。
    """
    # TODO: 实现状态距离计算逻辑 (例如 L2 距离)
    return 0.0
