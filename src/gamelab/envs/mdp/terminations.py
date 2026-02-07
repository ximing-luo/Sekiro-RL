"""
通用原子终止条件库。
"""

def time_out_termination(env, **kwargs) -> bool:
    """
    基于步数的超时终止。
    """
    if hasattr(env, "step_count") and hasattr(env.cfg, "max_episode_steps"):
        return env.step_count >= env.cfg.max_episode_steps
    return False

def illegal_state_termination(env, **kwargs) -> bool:
    """
    检测到非法状态（如数值异常）时强制终止。
    """
    return False
