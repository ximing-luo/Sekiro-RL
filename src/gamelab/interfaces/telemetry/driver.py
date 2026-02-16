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

    def connect(self, pid: Optional[int] = None) -> bool:
        """连接到游戏进程。
        
        Args:
            pid: 进程 ID。如果提供，则连接到指定 PID；否则连接到第一个匹配 process_name 的进程。
        """
        try:
            if pid:
                self._pm = pymem.Pymem()
                self._pm.open_process_from_id(pid)
            else:
                self._pm = pymem.Pymem(self.process_name)
            return True
        except Exception as e:
            target = f"PID={pid}" if pid else self.process_name
            print(f"无法连接到进程 {target}: {e}")
            return False

    @staticmethod
    def find_pids_by_name(process_name: str) -> List[int]:
        """查找指定名称的所有进程 PID。"""
        import pymem.process
        pids = []
        try:
            for entry in pymem.process.list_processes():
                # 使用 errors='ignore' 避免因其他进程名包含非UTF-8字符(如GBK)导致崩溃
                # 只要目标进程名(Sekiro.exe)是ASCII，这种处理就是安全的
                if entry.szExeFile.decode('utf-8', errors='ignore').lower() == process_name.lower():
                    pids.append(entry.th32ProcessID)
        except Exception as e:
            print(f"[TelemetryDriver] 扫描进程 {process_name} 时出错: {e}")
            pass
        return sorted(pids)

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
