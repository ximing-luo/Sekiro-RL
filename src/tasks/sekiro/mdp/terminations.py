"""
模块用途：原子终止条件定义（Termination Terms）。
"""
import torch

def player_dead_termination(env, env_ids=None, **kwargs) -> torch.Tensor:
    """自身死亡导致回合结束。"""
    # 假设事件管理器已经更新了某个状态，或者直接从资产读取
    # 这里我们通过 env.scene.assets 检查 player 的死亡状态
    return env.scene.sekiro.status.player_deaths > env.scene.sekiro.status.prev_player_deaths

def boss_dead_termination(env, env_ids=None, **kwargs) -> torch.Tensor:
    """Boss 死亡导致回合结束。"""
    return env.scene.sekiro.status.enemy_deaths > env.scene.sekiro.status.prev_enemy_deaths

def time_out_termination(env, env_ids=None, max_episode_steps: int = 2048, **kwargs) -> torch.Tensor:
    """超时终止。
    
    需配合 ManagerBasedEnv.episode_length_buf 使用。
    可以通过在配置中自定义 max_episode_steps 参数来覆盖默认值。
    """   
    return env.episode_length_buf >= max_episode_steps
