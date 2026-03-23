import inspect
import importlib
from typing import Callable, Any, Dict

def import_class(class_path: str):
    """职责：根据字符串路径动态导入类。
    
    例如: "src.gamelab.app.runners.SekiroRunner"
    """
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def filter_kwargs(func: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """只保留 func 函数能够接收的关键字参数。
    
    支持类初始化 (__init__) 和普通函数。
    """
    # 如果是类，获取其 __init__ 方法的签名
    if inspect.isclass(func):
        sig = inspect.signature(func.__init__)
    else:
        sig = inspect.signature(func)
    
    params = sig.parameters
    
    # 如果函数有 **kwargs，则直接返回所有输入
    for param in params.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return kwargs
            
    # 否则，只保留在签名中的参数（排除 self 和 args）
    return {
        k: v for k, v in kwargs.items() 
        if k in params and params[k].kind not in [inspect.Parameter.VAR_POSITIONAL]
    }
