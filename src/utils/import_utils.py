import importlib

def import_class(class_path: str):
    """职责：根据字符串路径动态导入类。
    
    例如: "src.gamelab.rl.runners.SekiroRunner"
    """
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
