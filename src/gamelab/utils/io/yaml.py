

import yaml
from typing import Any, Dict

def load_yaml(path: str) -> Dict[str, Any]:
    """加载 YAML 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml(path: str, data: Dict[str, Any]):
    """保存 YAML 文件。"""
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
