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
