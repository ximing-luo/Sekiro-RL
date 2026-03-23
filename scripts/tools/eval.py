"""
Sekiro-RL 评估工具 (Model Evaluator)
加载已保存的观测数据，对模型进行离线推理性能和激活值分布测试。

使用方法:
python scripts/tools/eval.py
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from dataclasses import dataclass

# 项目根目录注入
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model.ppo.spatial import SekiroStableExtractor, SekiroMultiInputExtractor
from stable_baselines3 import PPO

@dataclass
class TestConfig:
    data_path: str = os.path.join(project_root, "logs/data/benchmark_obs.npy")
    model_path: str = os.path.join(project_root, "models/sekiro_ppo_final.zip")
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size: int = 64

class DataValidator:
    """数据完整性校验器：拒绝模糊，拥抱确定性"""
    
    @staticmethod
    def load_and_validate(path: str) -> np.ndarray:
        if not os.path.exists(path):
            raise FileNotFoundError(f"基准数据未找到: {path}。请先运行数据采集脚本。")
            
        data = np.load(path)
        # 维度契约：(N, C, H, W)
        if data.ndim == 5 and data.shape[1] == 1:
            data = data.squeeze(1)
            
        DataValidator._audit_data_variance(data)
        return data

    @staticmethod
    def _audit_data_variance(data: np.ndarray):
        """审计数据方差，确保样本具有多样性"""
        if len(data) < 2:
            return

        # 计算帧间差异 (L1 Distance)
        diffs = np.abs(data[1:] - data[:-1]).mean(axis=(1, 2, 3))
        mean_diff = diffs.mean()
        zero_diff_count = (diffs < 1e-6).sum()
        
        print(f"\n[数据审计] 样本总数: {len(data)}")
        print(f"[数据审计] 平均帧间差异: {mean_diff:.6f}")
        print(f"[数据审计] 完全静止帧对: {zero_diff_count} / {len(diffs)}")
        
        if mean_diff < 1e-6:
            print("\033[91m[严重警告] 数据集似乎完全静止！请检查采集逻辑。\033[0m")
        elif zero_diff_count > len(diffs) * 0.5:
            print("\033[93m[警告] 超过 50% 的帧是重复的。\033[0m")
        else:
            print("\033[92m[通过] 数据具备合理的时序差异。\033[0m")

class ModelLoader:
    """模型加载器：强契约，不试探"""
    
    @staticmethod
    def load(path: str, device: torch.device, obs_shape: Tuple[int, ...]) -> torch.nn.Module:
        if os.path.exists(path):
            print(f"[模型] 加载训练权重: {path}")
            try:
                model = PPO.load(path, device=device)
                extractor = model.policy.features_extractor
                # 契约检查：如果是多输入，提取视觉部分
                if isinstance(extractor, SekiroMultiInputExtractor):
                    return extractor.image_extractor
                return extractor
            except Exception as e:
                print(f"[错误] 模型加载失败: {e}")
                print("[回退] 使用随机初始化模型")
        else:
            print(f"[提示] 未找到模型文件，使用随机初始化: {path}")

        # Mock Space 用于初始化
        class MockSpace:
            def __init__(self, shape): self.shape = shape
            
        return SekiroStableExtractor(MockSpace(obs_shape), features_dim=512).to(device)

class SyntheticGenerator:
    """合成数据生成器：纯粹的逻辑映射"""
    
    @staticmethod
    def generate(shape: Tuple[int, ...], pattern: str) -> torch.Tensor:
        """生成单帧基准数据 (1, C, H, W)"""
        # 确保只生成一帧
        single_frame_shape = (1, *shape[1:])
        
        if pattern == "black":
            return torch.zeros(single_frame_shape)
        elif pattern == "white":
            return torch.ones(single_frame_shape) * 255.0
        elif pattern == "noise":
            return torch.randint(0, 256, single_frame_shape).float()
        elif pattern == "checkerboard":
            return SyntheticGenerator._make_checkerboard(single_frame_shape)
        else:
            raise ValueError(f"Unknown pattern: {pattern}")

    @staticmethod
    def _make_checkerboard(shape: Tuple[int, ...]) -> torch.Tensor:
        _, C, H, W = shape
        data = torch.zeros(shape)
        grid_size = 32 # 像素
        
        # 向量化棋盘生成
        y_grid = torch.arange(H).view(-1, 1) // grid_size
        x_grid = torch.arange(W).view(1, -1) // grid_size
        mask = (y_grid + x_grid) % 2 == 1
        
        data[..., mask] = 255.0
        return data

class Analyzer:
    """分析器：只负责计算，不负责决策"""
    
    @staticmethod
    def extract_features(model: torch.nn.Module, data: torch.Tensor, device: torch.device) -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            # 归一化预处理：[0, 255] -> [0, 1]
            if data.max() > 1.0:
                data = data.float() / 255.0
            
            # 确保在设备上
            data = data.to(device)
            return model(data)

    @staticmethod
    def compute_similarity_matrix(features_map: Dict[str, torch.Tensor]) -> Tuple[List[str], np.ndarray]:
        names = list(features_map.keys())
        n = len(names)
        matrix = np.zeros((n, n))
        
        # 1. 计算每个类别的中心向量 (Centroid) 并归一化
        centroids = {}
        intra_class_sim = {}  # 类内相似度 (多样性指标)

        for name, feats in features_map.items():
            # feats: (N, Dim)
            # 计算质心: mean -> (1, Dim) -> normalize
            mean_feat = feats.mean(dim=0, keepdim=True)
            centroid = F.normalize(mean_feat, p=2, dim=1)
            centroids[name] = centroid
            
            # 计算类内相似度
            intra_class_sim[name] = Analyzer.compute_intra_class_sim(feats)

        # 2. 计算两两余弦相似度
        for i in range(n):
            for j in range(n):
                if i == j:
                    # 对角线显示类内相似度 (1.0 表示完全一致，<1.0 表示存在多样性)
                    matrix[i, j] = intra_class_sim[names[i]]
                else:
                    # 非对角线显示质心间的余弦相似度
                    sim = torch.mm(centroids[names[i]], centroids[names[j]].t()).item()
                    matrix[i, j] = sim
                
        return names, matrix

    @staticmethod
    def compute_intra_class_sim(feats: torch.Tensor) -> float:
        """计算特征集合的内部一致性 (平均余弦相似度)"""
        if feats.size(0) <= 1:
            return 1.0
        
        # 归一化特征
        feats_norm = F.normalize(feats, p=2, dim=1)
        
        # 计算两两相似度矩阵 (N, N)
        sim_matrix = torch.matmul(feats_norm, feats_norm.T)
        
        # 排除自身 (对角线)
        n = feats.size(0)
        mask = torch.eye(n, device=feats.device).bool()
        sims = sim_matrix[~mask]
        
        return sims.mean().item()

def main():
    cfg = TestConfig()
    
    print("="*60)
    print("SEKIRO-RL 模型离线诊断系统")
    print("="*60)

    # 1. 数据加载与审计
    real_data_np = DataValidator.load_and_validate(cfg.data_path)
    real_data = torch.from_numpy(real_data_np).float() # (N, C, H, W)
    
    # 2. 模型加载
    model = ModelLoader.load(cfg.model_path, cfg.device, real_data.shape[1:])
    
    # 3. 特征提取
    features_map = {}
    
    # A. 真实数据 (Batch 处理)
    print(f"\n[分析] 提取真实数据特征 (N={len(real_data)})...")
    real_feats = []
    for i in range(0, len(real_data), cfg.batch_size):
        batch = real_data[i : i + cfg.batch_size]
        real_feats.append(Analyzer.extract_features(model, batch, cfg.device))
    features_map["Real"] = torch.cat(real_feats, dim=0).cpu()
    
    # B. 基准数据 (单帧)
    benchmarks = ["black", "white", "noise", "checkerboard"]
    for name in benchmarks:
        print(f"[分析] 生成并提取基准: {name}")
        synth_data = SyntheticGenerator.generate(real_data.shape, name)
        feat = Analyzer.extract_features(model, synth_data, cfg.device)
        features_map[name.capitalize()] = feat.cpu()

    # 4. 相似度矩阵计算与可视化
    names, matrix = Analyzer.compute_similarity_matrix(features_map)
    
    # 5. 结果输出
    print("\n" + "="*80)
    print("注意：对角线数值表示[类内一致性]，越接近1表示样本越单一；非对角线表示[质心相似度]。")
    print("-" * 80)
    print(f"{'Category':<15} | " + " | ".join([f"{n:<10}" for n in names]))
    print("-" * (15 + 13 * len(names)))
    
    for i, row_name in enumerate(names):
        row_str = f"{row_name:<15} | "
        for val in matrix[i]:
            # 高亮自身 (1.0) 和高相似度 (>0.8)
            val_str = f"{val:.4f}"
            if val > 0.99:
                row_str += f"\033[92m{val_str:<10}\033[0m | "
            elif val > 0.8:
                row_str += f"\033[93m{val_str:<10}\033[0m | "
            else:
                row_str += f"{val_str:<10} | "
        print(row_str)
    print("="*60)

    # 绘制热力图
    try:
        plt.figure(figsize=(10, 8))
        plt.imshow(matrix, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(label='Cosine Similarity')
        plt.title("Feature Representation Similarity\n(Diagonal: Intra-class Consistency, Off-diagonal: Centroid Similarity)")
        plt.xticks(range(len(names)), names, rotation=45)
        plt.yticks(range(len(names)), names)
        
        # 在格子里标数值
        for i in range(len(names)):
            for j in range(len(names)):
                plt.text(j, i, f"{matrix[i, j]:.2f}", 
                        ha="center", va="center", color="w" if matrix[i, j] < 0.7 else "k")

        save_path = os.path.join(project_root, "logs", "feature_similarity.png")
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"\n[图表] 热力图已保存至: {save_path}")
    except Exception as e:
        print(f"[警告] 绘图失败: {e}")

if __name__ == "__main__":
    main()
