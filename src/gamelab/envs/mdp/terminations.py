"""
通用原子终止条件库。
"""

def time_out_termination(env, **kwargs) -> bool:
    """基于步数的超时终止。
    
    基于强契约：环境必须具备步数和配置。
    """
    return env.common_step_counter >= env.cfg.max_episode_steps

def illegal_state_termination(env, **kwargs) -> bool:
    """
    检测到非法状态（如数值异常）时强制终止。
    """
    return False
