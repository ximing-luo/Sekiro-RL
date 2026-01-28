from src.envs.mdp.terminations import check_simple_termination

class TerminationManager:
    """
    终止管理器：负责判断回合是否结束。
    对应 Isaac Lab 中的 TerminationManager。
    """
    def __init__(self):
        self.emergence_break = 0

    def check_termination(self, prev_metrics, next_metrics, events):
        """判断是否达到终止条件（如死亡、胜利）。"""
        done = 0
        
        # 使用 MDP 中的判定逻辑
        if check_simple_termination(events):
            if self.emergence_break < 1:
                done = 1
                self.emergence_break += 1
            else:
                done = 1
                self.emergence_break = 100
        
        return done

    def reset(self):
        self.emergence_break = 0
