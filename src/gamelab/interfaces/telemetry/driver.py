import pymem
from typing import Optional, List

class TelemetryDriver:
    """
    跨游戏通用的遥测数据读取驱动 (Pymem 封装)。
    """
    def __init__(self, process_name: str):
        self.process_name = process_name
        self.pm = None

    def connect(self) -> bool:
        """连接到游戏进程。"""
        try:
            self.pm = pymem.Pymem(self.process_name)
            return True
        except Exception as e:
            print(f"无法连接到进程 {self.process_name}: {e}")
            return False

    def pattern_scan_all(self, pattern: bytes) -> Optional[int]:
        """全内存扫描特定签名。"""
        if not self.pm:
            return None
        results = self.pm.pattern_scan_all(pattern)
        return results

    def read_int(self, address: int) -> int:
        """读取 32 位整数。"""
        try:
            if self.pm:
                val = self.pm.read_int(address)
                return val if val is not None else 0
        except Exception:
            return 0
        return 0

    def read_float(self, address: int) -> float:
        """读取浮点数。"""
        try:
            if self.pm:
                val = self.pm.read_float(address)
                return val if val is not None else 0.0
        except Exception:
            return 0.0
        return 0.0
