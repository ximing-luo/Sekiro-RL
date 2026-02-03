import os
import sys
import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# 将项目根目录添加到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model.ppo.dual_stream import SekiroMADSExtractor
from src.policies.sb3.sekiro_ppo import SekiroPPO
from src.train import SekiroCustomPolicy
from src.envs.managers.observation_manager import ObservationManager
from stable_baselines3.common.vec_env import DummyVecEnv

def test_pipeline():
    print("--- 启动 MADS 架构维度验证测试 ---")
    
    # 1. 模拟原始环境
    class MockEnv(gym.Env):
        def __init__(self):
            # 模拟单帧输入 (C, H, W)
            self.observation_space = spaces.Box(low=0, high=255, shape=(3, 270, 480), dtype=np.uint8)
            self.action_space = spaces.Discrete(10)
            # 注入 ObservationManager 模拟
            from src.envs.manager_based_env_cfg import ObservationTermCfg
            self.observation_manager = ObservationManager(cfg={})

        def reset(self, **kwargs):
            obs = np.zeros((3, 270, 480), dtype=np.uint8)
            # 填充管理器缓冲区
            for _ in range(12):
                self.observation_manager.frame_buffer.append(obs)
            return obs, {}

        def step(self, action):
            obs = np.zeros((3, 270, 480), dtype=np.uint8)
            self.observation_manager.frame_buffer.append(obs)
            # 模拟奖励分量以验证回调同步逻辑
            info = {
                "reward_components": {
                    "Mock_Damage": 0.5,
                    "Mock_Posture": 0.2
                }
            }
            return obs, 0.7, False, False, info

    # 使用 DummyVecEnv 包装以对齐真实训练环境
    def make_env():
        return MockEnv()
    env = DummyVecEnv([make_env])
    print(f"环境单帧维度 (VecEnv 包装后): {env.observation_space.shape}")

    # 2. 初始化 MADS 特征提取器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # MADS 提取器内部处理 12 通道输入
    stacked_observation_space = spaces.Box(low=0, high=255, shape=(12, 270, 480), dtype=np.uint8)
    extractor = SekiroMADSExtractor(stacked_observation_space, features_dim=512).to(device)
    print("MADS 特征提取器初始化成功。")

    # 3. 初始化 SekiroPPO
    policy_kwargs = dict(
        features_extractor_class=SekiroMADSExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[512, 256], vf=[512, 256])
    )
    
    # 注意：PPO 初始化时传入的是 VecEnv
    model = SekiroPPO(
        SekiroCustomPolicy, 
        env, 
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=64,
        batch_size=16,
        n_epochs=1,
        vicreg_coef=0.1,
        inv_dyn_coef=0.1,
        vf_coef=0.2,
        clip_range_vf=0.2,
        device=device
    )
    # 初始化 SB3 日志系统
    from stable_baselines3.common.logger import configure
    model.set_logger(configure(None, ["stdout"]))
    
    print("SekiroPPO 初始化成功。")

    # 4. 测试 collect_rollouts (这是之前失败的地方)
    print("\n尝试模拟 collect_rollouts (采集阶段)...")
    try:
        from stable_baselines3.common.callbacks import CallbackList
        from src.visualization.callbacks import SekiroCombinedCallback
        
        # 使用真实回调进行压力测试
        sekiro_callback = SekiroCombinedCallback(log_interval=10)
        callbacks = CallbackList([sekiro_callback])
        callbacks.init_callback(model)
        
        model.collect_rollouts(
            env, 
            callback=callbacks, 
            rollout_buffer=model.rollout_buffer, 
            n_rollout_steps=model.n_steps
        )
        print("✅ collect_rollouts 运行成功！回调诊断逻辑已通过。")
    except Exception as e:
        print(f"❌ collect_rollouts 失败！错误信息: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 5. 测试前向传播 (12 通道输入)
    print("\n尝试前向传播 (12通道)...")
    sample_obs = torch.as_tensor(np.zeros((1, 12, 270, 480), dtype=np.float32)).to(device)
    try:
        features = extractor(sample_obs)
        print(f"✅ 前向传播成功！输出维度: {features.shape}")
    except Exception as e:
        print(f"❌ 前向传播失败！错误信息: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 6. 测试训练一轮 (验证辅助损失逻辑)
    print("\n尝试模拟一次训练更新 (验证 VICReg + InvDyn Loss)...")
    try:
        model.train()
        print("✅ 训练更新逻辑运行成功！")
    except Exception as e:
        print(f"❌ 训练更新逻辑失败！错误信息: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n--- 所有验证通过！架构稳健性已确认 ---")

if __name__ == "__main__":
    test_pipeline()
