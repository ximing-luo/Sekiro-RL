import pymem
import threading
import time
from collections import deque, Counter

class SekiroTelemetry:
    """
    Sekiro 游戏数据遥测类，通过内存读取直接获取角色与敌人的状态指标。
    解决了 pymem 初始化与签名搜索带来的延迟问题（约 1s）。
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SekiroTelemetry, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.pm = None
        self.base = None
        self.data = {
            "player_hp": 0,
            "player_hp_max": 0,
            "player_posture": 0,
            "player_posture_max": 0,
            "enemy_hp": 0,
            "enemy_hp_max": 0,
            "enemy_posture": 0,
            "enemy_posture_max": 0,
            "player_deaths": 0,
            "enemy_deaths": 0,
        }
        # 初始化数据缓冲区，用于平滑处理（取最近10次采样的众数）
        self._buffers = {key: deque(maxlen=10) for key in self.data.keys()}
        self.running = False
        self.thread = None
        self._initialized = True
        self._connect()

    def _connect(self):
        """连接游戏进程并寻找签名"""
        try:
            print("正在连接 Sekiro 进程...")
            self.pm = pymem.Pymem("sekiro.exe")
            # 搜索签名 SEKIRO_TLM
            results = self.pm.pattern_scan_all(b"SEKIRO_TLM")
            if results:
                self.base = results
                print(f"找到遥测区签名，基址: {hex(self.base)}")
            else:
                print("错误：未找到遥测区签名！请确保游戏正在运行且补丁已加载。")
        except Exception as e:
            print(f"连接失败: {e}")

    def _read_r32(self, offset):
        """读取 32 位整数，失败或为空时返回 0"""
        try:
            if self.pm and self.base:
                # 如果地址无效或读取失败，pymem 会抛出异常
                val = self.pm.read_int(self.base + offset)
                return val if val is not None else 0
        except Exception:
            # 针对游戏未完全加载或对象不存在的情况，捕获异常并返回 0
            return 0
        return 0

    def update(self):
        """刷新当前数据，并应用 10 次采样众数过滤"""
        if not self.pm or not self.base:
            self._connect()
            if not self.pm or not self.base:
                return

        # 1. 批量读取原始数据
        # 偏移说明：
        # Python 扫描到 SEKIRO_TLM 签名后的起始位置
        # SEKIRO_TLM (12字节)
        # telePlayerHP 紧随其后，开始于 +12
        raw_values = {
            "player_hp": self._read_r32(12),
            "player_hp_max": self._read_r32(16),
            "player_posture": self._read_r32(24),
            "player_posture_max": self._read_r32(28),
            "enemy_hp": self._read_r32(32),
            "enemy_hp_max": self._read_r32(36),
            "enemy_posture": self._read_r32(44),
            "enemy_posture_max": self._read_r32(48),
            "player_deaths": self._read_r32(52),
            "enemy_deaths": self._read_r32(56),
        }

        # 2. 更新缓冲区并计算众数
        for key, value in raw_values.items():
            self._buffers[key].append(value)
            # 取出现次数最多的数值作为当前值
            self.data[key] = Counter(self._buffers[key]).most_common(1)[0][0]

    def _run_loop(self):
        """后台循环刷新数据"""
        while self.running:
            self.update()
            time.sleep(0.01)  # 100Hz 刷新率

    def start(self):
        """启动后台刷新线程"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()
            print("遥测后台线程已启动")

    def stop(self):
        """停止后台刷新线程"""
        self.running = False
        if self.thread:
            self.thread.join()

    def get_metrics(self):
        """获取当前核心指标"""
        # 如果线程没开，则同步更新一次
        if not self.running:
            self.update()
        
        return self.data.copy()

if __name__ == "__main__":
    sek = SekiroTelemetry

