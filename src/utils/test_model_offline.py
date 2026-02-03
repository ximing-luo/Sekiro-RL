import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 将项目根目录添加到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model.ppo_models import SekiroMADSExtractor
from stable_baselines3 import PPO
from src.train import SekiroCustomPolicy

def test_offline(data_path="logs/data/benchmark_obs.npy", model_path="models/sekiro_ppo_final.zip"):
    """
    加载采集好的数据，离线分析模型的特征分化能力。
    """
    # 修正相对路径为绝对路径
    if not os.path.isabs(data_path):
        data_path = os.path.join(project_root, data_path)
    if not os.path.isabs(model_path):
        model_path = os.path.join(project_root, model_path)

    print(f"\n" + "="*50)
    print(f"正在启动离线模型诊断...")
    print("="*50)
    
    # 1. 加载数据
    if not os.path.exists(data_path):
        print(f"[错误] 未找到基准数据文件: {data_path}")
        print("请先运行 src/utils/collect_benchmark_data.py")
        return
        
    obs_data = np.load(data_path) # (N, 12, 135, 240)
    num_frames = obs_data.shape[0]
    print(f"成功加载数据: {num_frames} 帧")
    
    # 2. 加载或初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"运行设备: {device}")
    
    if os.path.exists(model_path):
        print(f"\033[94m[模型] 检测到训练好的模型: {model_path}，正在加载...\033[0m")
        try:
            # 直接使用官方 PPO 类加载，会自动恢复特征提取器
            model = PPO.load(model_path, device=device)
            extractor = model.policy.features_extractor
            print("\033[92m[成功] 模型权重加载完成！使用训练后的模型进行特征分化分析。\033[0m")
        except Exception as e:
            print(f"\033[91m[错误] 加载模型失败: {e}\033[0m")
            print("回退到随机初始化模式...")
            obs_space = MockSpace(obs_data.shape[1:])
            extractor = SekiroMADSExtractor(obs_space, features_dim=512).to(device)
    else:
        print(f"\033[93m[提示] 未找到训练模型 {model_path}。\033[0m")
        print("正在使用随机初始化的 SekiroMADSExtractor 进行基准分析。")
        # 模拟 SB3 的 observation_space
        class MockSpace:
            def __init__(self, shape):
                self.shape = shape
                
        obs_space = MockSpace(obs_data.shape[1:])
        extractor = SekiroMADSExtractor(obs_space, features_dim=512).to(device)
    
    extractor.eval()
    
    # 3. 提取特征
    print(f"正在提取特征...")
    features = []
    
    with torch.no_grad():
        # 分批处理以防 OOM
        batch_size = 64
        for i in range(0, num_frames, batch_size):
            batch_obs = obs_data[i:i+batch_size]
            batch_tensor = torch.from_numpy(batch_obs).to(device).float() / 255.0
            feat = extractor(batch_tensor)
            features.append(feat.cpu())
            
    features = torch.cat(features, dim=0) # (512, 512)
    
    # 4. 分析特征范数 (Norm)
    norms = torch.norm(features, dim=1)
    avg_norm = norms.mean().item()
    std_norm = norms.std().item()
    
    print(f"\n特征强度分析:")
    print(f"  - 平均范数 (Avg Norm): {avg_norm:.6f}")
    print(f"  - 范数标准差 (Std Norm): {std_norm:.6f}")
    
    if avg_norm < 1e-7:
        print("\033[91m[!!!] 警告: 特征输出近乎全零，模型信号已断路 [!!!]\033[0m")
        return
        
    # 5. 分析相似度矩阵
    print(f"\n正在计算相似度矩阵 (采样 128x128 对)...")
    # 随机采样一部分进行互相似度计算，避免 512x512 计算过慢
    sample_idx = torch.randperm(num_frames)[:128]
    sampled_feats = features[sample_idx]
    
    # 归一化特征向量
    normalized_feats = F.normalize(sampled_feats, p=2, dim=1)
    sim_matrix = torch.mm(normalized_feats, normalized_feats.t())
    
    # 提取上三角部分（不含对角线）
    mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
    triu_sims = sim_matrix[mask]
    
    avg_sim = triu_sims.mean().item()
    max_sim = triu_sims.max().item()
    min_sim = triu_sims.min().item()
    
    print(f"特征分化分析 (余弦相似度):")
    print(f"  - 平均相似度 (Avg Sim): {avg_sim:.6f}")
    print(f"  - 最大相似度 (Max Sim): {max_sim:.6f}")
    print(f"  - 最小相似度 (Min Sim): {min_sim:.6f}")
    
    if avg_sim > 0.99:
        print("\033[91m[!!!] 诊断结论: 检测到特征坍缩 (Feature Collapse)。即使不训练，模型对不同画面的反应也几乎一样 [!!!]\033[0m")
    elif avg_sim < 0.9:
        print("\033[92m[✓] 诊断结论: 特征分化良好。模型能够有效区分不同的游戏画面 [✓]\033[0m")
    else:
        print("\033[93m[!] 诊断结论: 特征存在一定程度的趋同，建议优化初始化或架构 [!]\033[0m")

    # 6. 可视化分布 (保存为图片)
    plt.figure(figsize=(12, 5))
    
    def safe_hist(data, title, xlabel, color, default_val=None):
        """安全绘制直方图，处理数据范围过窄的情况"""
        data_np = data.numpy() if hasattr(data, 'numpy') else data
        # 增加判定阈值到 1e-3，并增加 try-except 保护
        try:
            if np.ptp(data_np) > 1e-3:
                plt.hist(data_np, bins=50, alpha=0.75, color=color, edgecolor='black')
            else:
                raise ValueError("Data range too small")
        except Exception:
            # 数据几乎恒定时，画一条垂直线
            val = default_val if default_val is not None else data_np.mean()
            plt.axvline(x=val, color=color, linestyle='-', linewidth=2)
            plt.text(val, 0.5, f' Constant: {val:.4f}', color=color, transform=plt.gca().get_yaxis_transform())
            if "Norm" in title:
                print(f"\033[94m[信息] 特征范数极其恒定 (Std={std_norm:.2e})，这是 RMSNorm 的正常表现。\033[0m")
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Frequency")
        plt.grid(True)

    # 子图 1: 相似度分布
    plt.subplot(1, 2, 1)
    safe_hist(triu_sims, f"Feature Similarity (Avg: {avg_sim:.4f})", "Cosine Similarity", 'blue')
    
    # 子图 2: 范数分布
    plt.subplot(1, 2, 2)
    safe_hist(norms, f"Feature Norm (Avg: {avg_norm:.4f})", "L2 Norm", 'green', default_val=avg_norm)
    
    plt.tight_layout()
    save_path = os.path.join(project_root, "logs", "data", "offline_diagnostic.png")
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"\n离线诊断图已保存至: {save_path}")
    print("="*50 + "\n")

if __name__ == "__main__":
    test_offline()
