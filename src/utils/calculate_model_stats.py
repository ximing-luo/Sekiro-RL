import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import torch
import torch.nn as nn
from src.model.ppo_models import SekiroMADSExtractor

class MockSpace:
    def __init__(self, shape):
        self.shape = shape
        self.dtype = torch.uint8

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def analyze_memory(model, input_size, batch_size=64):
    # 模拟输入
    x = torch.randn(batch_size, *input_size)
    
    print(f"{'Layer Name':<40} | {'Params':<15} | {'Memory (MB)':<15}")
    print("-" * 75)
    
    total_params = 0
    
    # 递归分析子模块
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            params = sum(p.numel() for p in module.parameters())
            total_params += params
            # 参数显存 (FP32: 4 bytes)
            param_mem = params * 4 / (1024**2)
            print(f"{name:<40} | {params:<15,} | {param_mem:<15.2f}")

    print("-" * 75)
    print(f"Total Parameters: {total_params:,}")
    print(f"Parameter Memory (FP32): {total_params * 4 / (1024**2):.2f} MB")
    print(f"Gradient Memory (FP32): {total_params * 4 / (1024**2):.2f} MB")
    print(f"Optimizer Memory (Adam, 2x): {total_params * 8 / (1024**2):.2f} MB")
    
    # 估计激活值内存 (大致估算)
    # 这部分通常是反向传播 OOM 的主因
    print(f"\nEstimating Activation Memory (Batch Size = {batch_size})...")
    # 这里通过 forward 钩子简单记录输出尺寸
    activations = []
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            activations.append(output.numel())
    
    hooks = []
    for module in model.modules():
        hooks.append(module.register_forward_hook(hook))
    
    model(x)
    
    total_act_elements = sum(activations)
    act_mem = total_act_elements * 4 / (1024**2)
    print(f"Estimated Activation Memory: {act_mem:.2f} MB")
    print(f"Estimated Total Training Memory: {(total_params * 16 / (1024**2)) + act_mem:.2f} MB")

if __name__ == "__main__":
    extractor = SekiroMADSExtractor(MockSpace((12, 135, 240)))
    analyze_memory(extractor, (12, 135, 240), batch_size=256)
