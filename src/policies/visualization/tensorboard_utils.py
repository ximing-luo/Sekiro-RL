import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class TensorboardHookManager:
    """
    TensorBoard Hook 管理器：负责自动注册模型 Hook 并记录特征图。
    已适配 Stable-Baselines3 模型结构。
    """
    def __init__(self, model, writer: SummaryWriter, log_interval: int = 1024, max_layers: int = 5):
        self.model = model
        self.writer = writer
        self.log_interval = log_interval
        self.max_layers = max_layers
        self.hooks = []
        self.anomaly_hooks = []
        self._anomaly_detected = False

    def _get_anomaly_hook(self, name):
        """生成用于实时监测 NaN 的前向钩子。"""
        def hook(module, input, output):
            if self._anomaly_detected:
                return
            
            # 检查输出
            has_nan = False
            if torch.is_tensor(output):
                if torch.isnan(output).any() or torch.isinf(output).any():
                    has_nan = True
            elif isinstance(output, (list, tuple)):
                for o in output:
                    if torch.is_tensor(o) and (torch.isnan(o).any() or torch.isinf(o).any()):
                        has_nan = True
                        break
            
            if has_nan:
                self._anomaly_detected = True
                # 语义化层名
                pretty_name = name.replace("spatial_cnn", "[Static]").replace("temporal_cnn", "[Dynamic]")
                
                print(f"\n\033[91;1m{'='*60}\033[0m")
                print(f"\033[91;1m[FATAL ANOMALY] 在层 '{pretty_name}' 中检测到 NaN/Inf!\033[0m")
                
                # 统计输入信息
                if len(input) > 0 and torch.is_tensor(input[0]):
                    inp = input[0]
                    print(f"输入统计 | Mean: {inp.mean():.2e} | Max: {inp.max():.2e} | Min: {inp.min():.2e}")
                
                # 统计输出信息
                if torch.is_tensor(output):
                    print(f"输出统计 | Mean: {output.mean():.2e} | Max: {output.max():.2e} | Min: {output.min():.2e}")
                
                # 检查权重
                if hasattr(module, 'weight') and module.weight is not None:
                    w = module.weight
                    print(f"权重统计 | Mean: {w.mean():.2e} | Max: {w.max():.2e} | Norm: {w.norm():.2e}")

                print(f"\033[91;1m{'='*60}\033[0m")
                self.writer.flush()
                # 抛出异常中断训练
                raise RuntimeError(f"Deep Anomaly Detector: Layer '{pretty_name}' produced NaN.")
        return hook

    def register_anomaly_hooks(self):
        """为所有层注册前向异常监测钩子。"""
        self.remove_anomaly_hooks()
        self._anomaly_detected = False
        
        target = self.model.policy if hasattr(self.model, "policy") else self.model
        
        count = 0
        for name, module in target.named_modules():
            # 为所有具有参数的层或者激活层注册检查
            if len(list(module.children())) == 0: # 只给叶子节点加钩子
                h = module.register_forward_hook(self._get_anomaly_hook(name))
                self.anomaly_hooks.append(h)
                count += 1
        print(f"[DEBUG] 已注册 {count} 个前向异常监测钩子。")

    def remove_anomaly_hooks(self):
        """移除所有异常监测钩子。"""
        for h in self.anomaly_hooks:
            h.remove()
        self.anomaly_hooks = []

    def _get_activation_hook(self, clean_name, module_prefix):
        """生成用于特征图可视化的前向钩子。"""
        def hook(module, input, output):
            # 获取当前的全局步数
            step = getattr(self.model, 'num_timesteps', 0)
            
            if step % self.log_interval == 0:
                try:
                    # 1. 记录数值分布 (Histogram) - 分支优先路径
                    hist_tag = f"{module_prefix}/Activations/{clean_name}"
                    self.writer.add_histogram(hist_tag, output, step)

                    # 2. 如果是 4D 图片张量，记录图片可视化
                    tag_name = f"{module_prefix}/Features/{clean_name}"
                    if module.training:
                        tag_name += '_train'
                    
                    if output.ndim == 4:
                        img = output[0].detach().cpu()
                        # 归一化
                        mn, mx = img.min(), img.max()
                        img = (img - mn) / (mx - mn + 1e-9)
                        # 制作网格
                        if img.shape[0] > 64: img = img[:64]
                        grid = torchvision.utils.make_grid(img.unsqueeze(1), nrow=8, padding=2)
                        self.writer.add_image(tag_name, grid, step)
                    
                    # 3. 如果是 2D 特征向量
                    elif output.ndim == 2:
                        b, d = output.shape
                        side = int(np.sqrt(d))
                        if side * side == d:
                            img = output[0].view(1, side, side).detach().cpu()
                        else:
                            img = output[0].view(1, 1, -1).detach().cpu()
                        mn, mx = img.min(), img.max()
                        img = (img - mn) / (mx - mn + 1e-9)
                        self.writer.add_image(tag_name, img, step)

                    self.writer.flush()
                except Exception as e:
                    pass 
        return hook

    def register_hooks(self):
        """
        动态发现模型层并注册 Hook。
        优化策略：监控所有卷积层和线性层，实现全量特征图捕捉。
        """
        self.remove_hooks()
        
        target_modules = []
        if hasattr(self.model, 'policy'):
            # 针对 SekiroMADSExtractor 的双流结构进行精细化定位
            extractor = getattr(self.model.policy, 'features_extractor', None)
            if extractor is not None:
                # 1. 静态分支 (Spatial)
                target_modules.append(("[Static]", extractor.spatial_cnn))
                # 2. 动态分支 (Temporal)
                target_modules.append(("[Dynamic]", extractor.temporal_cnn))
                # 3. 融合与精炼层
                target_modules.append(("[Fusion]", nn.Sequential(
                    extractor.fusion, extractor.refiner
                )))
            
            # 针对 Heads (MLP Extractor)
            mlp = getattr(self.model.policy, 'mlp_extractor', None)
            if mlp is not None:
                target_modules.append(("[Heads]", mlp))
        
        if not target_modules:
            target_modules = [("Model", self.model)]

        print("\n\033[94m[DEBUG] 正在注册全量特征图监控 Hook (分支优先布局):\033[0m")
        global_layer_idx = 0
        for module_prefix, target_module in target_modules:
            count_in_module = 0
            for name, module in target_module.named_modules():
                # 只记录叶子节点的卷积和线性层
                if isinstance(module, (nn.Conv2d, nn.Linear)) and len(list(module.children())) == 0:
                    # 1. 精简命名：residual_function -> res
                    clean_name = name.replace("spatial_cnn.", "").replace("temporal_cnn.", "").replace("residual_function", "res")
                    # 2. 分支优先与物理排序：[Branch]/01_name
                    display_name = f"{global_layer_idx:02d}_{clean_name.replace('.', '_')}"
                    
                    print(f"  - 注册: {module_prefix}/{display_name: <50} | 原始路径: {name}")
                    
                    # 传入 display_name 和 module_prefix 以便 hook 组装路径
                    hook_handle = module.register_forward_hook(self._get_activation_hook(display_name, module_prefix))
                    self.hooks.append(hook_handle)
                    count_in_module += 1
                    global_layer_idx += 1
            
            if count_in_module == 0:
                # 如果没找到卷积/线性层，尝试记录模块本身
                if len(list(target_module.children())) == 0:
                    display_name = f"{global_layer_idx:02d}_Direct"
                    hook_handle = target_module.register_forward_hook(self._get_activation_hook(display_name, module_prefix))
                    self.hooks.append(hook_handle)
                    global_layer_idx += 1
                    
        print(f"\033[94m[DEBUG] 全量 Hook 注册完成。共计 {len(self.hooks)} 个监控点。\033[0m\n")

    def remove_hooks(self):
        """移除所有已注册的 Hook。"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def log_gradients(self, step: int):
        """仅用于监测梯度 NaN/临界值，不再记录到 TensorBoard。"""
        # 适配 SB3
        target = self.model.policy if hasattr(self.model, "policy") else self.model
        
        found_nan = False
        nan_info = []

        for name, param in target.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                
                # NaN 检测
                is_nan = np.isnan(grad_norm) or np.isinf(grad_norm)
                # 临界值检测：如果梯度范数超过 1000，视为异常迹象
                is_critical = grad_norm > 1000.0
                
                if is_nan or is_critical:
                    status = "NaN/Inf" if is_nan else "CRITICAL_HIGH"
                    # 终端打印语义化名称
                    pretty_name = name.replace("features_extractor.", "").replace("spatial_cnn", "[Static]").replace("temporal_cnn", "[Dynamic]").replace("residual_function", "res")
                    msg = f"[GRAD_DIAG] {status} | 层名: {pretty_name} | Norm: {grad_norm:.2e}"
                    print(f"\033[91m{msg}\033[0m")
                    if is_nan:
                        found_nan = True
                        nan_info.append(msg)
        
        if found_nan:
            if self.writer: self.writer.flush()
            raise RuntimeError(f"检测到梯度爆炸 (NaN)! \n" + "\n".join(nan_info))

    def log_weights(self, step: int):
        """仅用于监测权重 NaN，不再记录到 TensorBoard。"""
        # 适配 SB3
        target = self.model.policy if hasattr(self.model, "policy") else self.model
        
        found_nan = False

        for name, param in target.named_parameters():
            weight_norm = param.data.norm().item()
            
            # NaN 检测
            is_nan = np.isnan(weight_norm) or np.isinf(weight_norm)
            if is_nan:
                pretty_name = name.replace("features_extractor.", "").replace("spatial_cnn", "[Static]").replace("temporal_cnn", "[Dynamic]").replace("residual_function", "res")
                print(f"\033[91m[CRITICAL] 检测到权重损坏/NaN! 层名: {pretty_name} | Norm: {weight_norm:.2e}\033[0m")
                found_nan = True
        
        if found_nan:
            if self.writer: self.writer.flush()
            raise RuntimeError("检测到权重损坏 (NaN)! 训练已终止。")
        
        # 定期刷新以防崩溃时丢失数据
        if step % 10 == 0 and self.writer:
            self.writer.flush()

def register_tensorboard_hooks(agent, writer: SummaryWriter, log_interval: int = 1024):
    """便捷函数：初始化并注册 TensorBoard Hooks。"""
    manager = TensorboardHookManager(agent, writer, log_interval=log_interval)
    manager.register_hooks()
    return manager