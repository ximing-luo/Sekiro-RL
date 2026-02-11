# Copyright (c) 2024, Sekiro-RL Project.
# All rights reserved.

import yaml
import pickle
import os
import subprocess

def dump_yaml(file_path, data):
    """将数据转储为 YAML 文件。"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

def dump_pickle(file_path, data):
    """将数据转储为 Pickle 文件。"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

def get_git_hash():
    """获取当前 git 的提交哈希值。"""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"

def get_git_diff():
    """获取当前 git 的未提交更改。"""
    try:
        return subprocess.check_output(["git", "diff"]).decode("utf-8")
    except Exception:
        return ""
