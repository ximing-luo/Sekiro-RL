"""
原子记录器函数定义。
"""

def sekiro_recorder(env, obs, action, reward, next_obs, info, **kwargs):
    """
    基础记录器：记录一步的完整转换数据。
    """
    return {
        "action": action,
        "reward": reward,
        "obs": obs,
        "next_obs": next_obs,
    }
