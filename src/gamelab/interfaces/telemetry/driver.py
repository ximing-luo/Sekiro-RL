import pymem
from typing import Optional, List

class TelemetryDriver:
    """
    跨游戏通用的遥测数据读取驱动 (Pymem 封装)。
    """
    def __init__(self, process_name: str):
        self.process_name = process_name
        self._pm = None

    @property
    def pm(self) -> pymem.Pymem:
        """获取 Pymem 实例。如果未连接则抛出异常。"""
        if self._pm is None:
            raise RuntimeError(f"[TelemetryDriver] 未连接到进程 {self.process_name}，请先调用 connect()")
        return self._pm

    def connect(self) -> bool:
        """连接到游戏进程。"""
        try:
            self._pm = pymem.Pymem(self.process_name)
            return True
        except Exception as e:
            print(f"无法连接到进程 {self.process_name}: {e}")
            return False

    def pattern_scan_all(self, pattern: bytes) -> Optional[int]:
        """全内存扫描特定签名。"""
        results = self.pm.pattern_scan_all(pattern)
        return results

    def read_int(self, address: int) -> int:
        """读取 32 位整数。"""
        return self.pm.read_int(address)

    def read_float(self, address: int) -> float:
        """读取浮点数。"""
        return self.pm.read_float(address)

    def read_bytes(self, address: int, size: int) -> bytes:
        """读取字节块。"""
        return self.pm.read_bytes(address, size)
