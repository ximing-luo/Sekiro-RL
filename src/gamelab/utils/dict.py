from typing import Dict, Any

def update_dict(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    """递归更新字典。"""
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def print_dict(d: Dict[str, Any], indent: int = 0):
    """漂亮地打印字典。"""
    for k, v in d.items():
        if isinstance(v, dict):
            print("  " * indent + f"{k}:")
            print_dict(v, indent + 1)
        else:
            print("  " * indent + f"{k}: {v}")
