import os
import sys
import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure

# 1. 确保项目根目录在路径中
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. 导入适配后的架构组件
from src.model.ppo.spatial import SekiroStableExtractor, SekiroMultiInputExtractor
from src.framework.sb3.ppo_aux import AuxPPO as SekiroPPO
from src.tasks.sekiro.utils import SekiroCombinedCallback

def test_pipeline():
    print("--- 启动 SekiroStableExtractor (via MultiInput) 架构维度验证测试 ---")
    
    # 1. 模拟原始环境 (多输入以适配 Callback)
    class MockEnv(gym.Env):
        def __init__(self):
            # 模拟多输入: 图像 + 遥测
            self.observation_space = spaces.Dict({
                "policy": spaces.Box(low=0, high=255, shape=(3, 270, 480), dtype=np.uint8),
                "telemetry": spaces.Box(low=0, high=np.inf, shape=(10,), dtype=np.float32)
            })
            self.action_space = spaces.Discrete(10)

        def reset(self, seed=None, options=None):
            # 随机初始化避免全0导致 GroupNorm 可能的数值不稳定
            obs = {
                "policy": np.random.randint(0, 256, (3, 270, 480), dtype=np.uint8),
                "telemetry": np.random.randn(10).astype(np.float32)
            }
            return obs, {}

        def step(self, action):
            obs = {
                "policy": np.random.randint(0, 256, (3, 270, 480), dtype=np.uint8),
                "telemetry": np.random.randn(10).astype(np.float32)
            }
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

    # 2. 初始化特征提取器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 验证 Extractor 初始化
    try:
        # 我们先验证核心的 StableExtractor
        # 注意：这里需要传入图像部分的 space
        img_space = env.observation_space["policy"]
        extractor = SekiroStableExtractor(img_space, features_dim=512).to(device)
        print("SekiroStableExtractor (核心组件) 初始化成功。")
    except Exception as e:
        print(f"❌ SekiroStableExtractor 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. 初始化 SekiroPPO (实际上是 AuxPPO)
    policy_kwargs = dict(
        features_extractor_class=SekiroMultiInputExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[512, 256], vf=[512, 256])
    )
    
    try:
        # 使用 "MultiInputPolicy" 配合 MultiInput 提取器
        model = SekiroPPO(
            "MultiInputPolicy", 
            env, 
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=64,
            batch_size=16,
            n_epochs=1,
            aux_coef=0.1,  # AuxPPO 特有参数
            vf_coef=0.2,
            clip_range_vf=0.2,
            device=device,
            verbose=1
        )
        # 初始化 SB3 日志系统
        model.set_logger(configure(None, ["stdout"]))
        print("SekiroPPO (AuxPPO) 初始化成功。")
    except Exception as e:
        print(f"❌ SekiroPPO 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. 测试 collect_rollouts
    print("\n尝试模拟 collect_rollouts (采集阶段)...")
    try:
        from stable_baselines3.common.callbacks import CallbackList
        
        # 使用真实回调进行压力测试
        sekiro_callback = SekiroCombinedCallback(log_interval=10)
        callbacks = CallbackList([sekiro_callback])
        callbacks.init_callback(model)
        
        # 手动初始化 _last_obs 和 buffers
        model._last_obs = env.reset()
        from collections import deque
        model.ep_info_buffer = deque(maxlen=100)
        model.ep_success_buffer = deque(maxlen=100)
        
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

    # 5. 测试前向传播 (多输入)
    print("\n尝试前向传播 (多输入)...")
    sample_obs = {
        "policy": torch.as_tensor(np.zeros((1, 3, 136, 240), dtype=np.float32)).to(device),
        "telemetry": torch.as_tensor(np.zeros((1, 10), dtype=np.float32)).to(device)
    }
    try:
        # 使用模型中的提取器
        features = model.policy.features_extractor(sample_obs)
        print(f"✅ 前向传播成功！输出维度: {features.shape}")
    except Exception as e:
        print(f"❌ 前向传播失败！错误信息: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 6. 测试训练一轮
    print("\n尝试模拟一次训练更新 (验证 Aux Loss)...")
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
