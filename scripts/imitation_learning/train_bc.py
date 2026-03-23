"""
Sekiro-RL 行为克隆 (Behavior Cloning) 训练脚本
从录制好的素材中训练基础策略网络，作为 PPO 的预训练权重。

使用方法:
python scripts/imitation_learning/train_bc.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# 加入项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class SekiroBCDataset(Dataset):
    """模仿学习数据集，利用 mmap 实现高效加载。"""
    def __init__(self, record_dir="outputs/recordings"):
        self.record_dir = record_dir
        self.indices = []
        self.obs_mmaps = []
        self.act_mmaps = []
        
        if not os.path.exists(record_dir):
            return

        # 遍历所有轨迹目录 (子目录名通常为 env_x_ep_y_timestamp)
        episodes = sorted([d for d in os.listdir(record_dir) if os.path.isdir(os.path.join(record_dir, d))])
        
        for ep in episodes:
            ep_path = os.path.join(record_dir, ep)
            obs_file = os.path.join(ep_path, "obs.npy")
            act_file = os.path.join(ep_path, "action.npy")
            
            if not os.path.exists(obs_file) or not os.path.exists(act_file):
                continue
                
            obs = np.load(obs_file, mmap_mode='r')
            acts = np.load(act_file, mmap_mode='r')
            
            # 记录偏移量
            self.obs_mmaps.append(obs)
            self.act_mmaps.append(acts)
            
            for i in range(len(obs)):
                self.indices.append((len(self.obs_mmaps) - 1, i))
                
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        ep_idx, frame_idx = self.indices[idx]
        
        # 1. 归一化并转为 Tensor (零拷贝读取)
        # obs: (C, H, W) uint8 -> float32 / 255.0
        obs = torch.from_numpy(self.obs_mmaps[ep_idx][frame_idx].copy()).float() / 255.0
        # 15-bit mask -> float32
        act = torch.from_numpy(self.act_mmaps[ep_idx][frame_idx].copy()).float()
        
        return obs, act

class SekiroPolicyNet(nn.Module):
    """行为克隆 Policy 网络 (支持 15 位掩码多标签输出)。"""
    def __init__(self, input_shape=(3, 128, 240), action_dim=15):
        super().__init__()
        # CNN 特征提取器 (对标 Nature DQN 结构)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # 动态计算全连接层输入维度
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            n_flatten = self.features(dummy).shape[1]
            print(f"PolicyNet Flatten Dim: {n_flatten}")
            
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
            nn.Sigmoid() # 关键：使用 Sigmoid 实现多标签分类 (每个按键独立概率)
        )

    def forward(self, x):
        features = self.features(x)
        return self.fc(features)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n" + "="*45)
    print(f"Sekiro-RL 行为克隆训练启动")
    print(f"设备: {device}")
    print("="*45 + "\n")
    
    # 1. 加载数据集
    dataset = SekiroBCDataset()
    if len(dataset) == 0:
        print("错误: 未找到录制数据。请先运行 scripts/imitation/record.py 进行录制。")
        return
        
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    print(f"成功加载 {len(dataset)} 帧素材。")
    
    # 2. 初始化网络与优化器
    # 默认分辨率从 config.py 获取
    from configs.config import cfg as global_cfg
    input_shape = (3, global_cfg.scene.img_height, global_cfg.scene.img_width)
    
    model = SekiroPolicyNet(input_shape=input_shape, action_dim=15).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss() # 二值交叉熵
    
    # 3. 训练循环
    epochs = 20
    save_path = "outputs/models/sekiro_bc_latest.pth"
    os.makedirs("outputs/models", exist_ok=True)
    
    print(f"开始训练 ({epochs} Epochs)...")
    try:
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            progress_bar = tqdm(loader, desc=f"Epoch {epoch+1:02d}")
            
            for obs, act in progress_bar:
                obs, act = obs.to(device), act.to(device)
                
                optimizer.zero_grad()
                pred = model(obs)
                loss = criterion(pred, act)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=f"{loss.item():.6f}")
            
            # 保存检查点
            torch.save(model.state_dict(), save_path)
            
    except KeyboardInterrupt:
        print("\n训练被用户中断。")
    
    print(f"\n训练结束。模型已保存至: {save_path}")

if __name__ == "__main__":
    train()
