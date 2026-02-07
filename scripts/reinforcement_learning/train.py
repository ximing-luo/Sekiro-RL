import os
import sys
import argparse
import shutil
import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# 1. 确保项目根目录在路径中
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. 导入新架构组件 (Isaac Lab 风格)
from src.gamelab.app.app_launcher import AppLauncher
from src.framework.sb3.ppo_aux import AuxPPO
from src.models.ppo_models import SekiroMultiInputExtractor
from src.gamelab.utils.rl.sb3 import SekiroCombinedCallback
import configs.config as config

def main():
    # 1. 初始化启动器
    # AppLauncher 负责解析 --task, --num_envs, --device, --headless 等核心参数
    launcher = AppLauncher()
    
    # 2. 添加训练特定参数
    parser = argparse.ArgumentParser(description="Sekiro-RL PPO 训练入口 (Isaac Lab 风格)", add_help=False)
    train_group = parser.add_argument_group("训练特定参数")
    train_group.add_argument("--steps", type=int, default=100000, help="总训练时间步数")
    train_group.add_argument("--save_freq", type=int, default=10000, help="模型保存频率 (steps)")
    train_group.add_argument("--lr", type=float, default=1e-5, help="学习率")
    train_group.add_argument("--batch_size", type=int, default=64, help="批大小")
    train_group.add_argument("--n_steps", type=int, default=2048, help="PPO 采集步数")
    train_group.add_argument("--aux_coef", type=float, default=0.05, help="特征相似度辅助损失权重")
    train_group.add_argument("--checkpoint", type=str, default=None, help="断点模型路径")
    
    # 将训练参数合并到 launcher.args 中
    args, _ = parser.parse_known_args(namespace=launcher.args)

    # 3. 创建环境
    # create_env 会自动根据 launcher.args 初始化任务注册表并返回环境实例
    env, env_cfg = launcher.create_env()
    
    # 将单环境包装为矢量化环境 (VecEnv)，这是使用 VecNormalize 的前提
    env = DummyVecEnv([lambda: env])
    
    # 包装 VecNormalize 以进行奖励归一化
    # norm_obs=False: 图像输入不建议在 VecNormalize 中做归一化
    # norm_reward=True: 归一化奖励，防止梯度爆炸
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.)
    
    # 4. 配置神经网络架构
    # 由于环境现在返回 Dict 观测空间 (image + telemetry)，必须使用 MultiInputPolicy
    policy_kwargs = dict(
        features_extractor_class=SekiroMultiInputExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[512, 256], vf=[512, 256]),
        activation_fn=nn.ReLU
    )

    # 5. 配置日志系统
    tb_log_path = os.path.join(config.cfg.path.log_dir, "ppo_sekiro")
    if os.path.exists(tb_log_path):
        print(f"[INFO] 正在清理旧日志目录: {tb_log_path}")
        shutil.rmtree(tb_log_path)
    
    # 创建统一的 SB3 日志器
    new_logger = configure(tb_log_path, ["stdout", "csv", "tensorboard"])
    
    # 6. 初始化或加载 PPO 模型
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"[INFO] 正在从断点加载模型: {args.checkpoint}")
        stats_path = args.checkpoint.replace(".zip", "_vec_normalize.pkl")
        if os.path.exists(stats_path):
            print(f"[INFO] 正在加载归一化统计量: {stats_path}")
            env = VecNormalize.load(stats_path, env)
            
        model = AuxPPO.load(
            args.checkpoint, 
            env=env, 
            device=launcher.device,
            custom_objects={
                "learning_rate": args.lr,
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "aux_coef": args.aux_coef
            }
        )
    else:
        print(f"[INFO] 正在初始化新模型 (设备: {launcher.device})...")
        model = AuxPPO(
            "MultiInputPolicy", # 适配 Dict 观测空间
            env, 
            policy_kwargs=policy_kwargs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            gamma=0.95,
            n_epochs=5,
            target_kl=0.05,
            ent_coef=0.01,
            clip_range=0.5,
            clip_range_vf=0.5,
            max_grad_norm=5,
            aux_coef=args.aux_coef,
            vf_coef=0.8,
            verbose=1,
            device=launcher.device
        )
    
    # 统一设置日志器
    model.set_logger(new_logger)

    # 7. 配置回调函数
    callbacks = []
    
    # A. 保存断点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path="outputs/models/ppo_checkpoints/",
        name_prefix="sekiro_ppo"
    )
    callbacks.append(checkpoint_callback)
    
    # B. 重构后的全能回调 (含奖励分量可视化、特征相似度分析等)
    sekiro_callback = SekiroCombinedCallback(log_interval=config.cfg.path.tb_log_interval)
    callbacks.append(sekiro_callback)
    
    # C. 从配置加载动态回调 (可选)
    if hasattr(env_cfg, "callbacks"):
        from src.utils.import_utils import import_class
        for cb_cfg in env_cfg.callbacks:
            try:
                cb_class = import_class(cb_cfg["class"])
                cb_instance = cb_class(**cb_cfg.get("params", {}))
                callbacks.append(cb_instance)
            except Exception as e:
                print(f"[WARN] 加载自定义回调失败: {e}")

    # 8. 开始训练
    print(f"[INFO] 训练开始。Tensorboard 日志目录: {tb_log_path}")
    try:
        model.learn(
            total_timesteps=args.steps, 
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("[WARN] 训练被手动中断。")
    finally:
        # 9. 保存最终结果
        os.makedirs("outputs/models", exist_ok=True)
        model.save("outputs/models/sekiro_ppo_final")
        env.save("outputs/models/sekiro_vec_normalize_final.pkl")
        env.close()
        print("[INFO] 训练结束，环境已关闭，模型已保存。")

if __name__ == "__main__":
    main()
