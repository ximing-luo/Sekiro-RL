

import time
from contextlib import ContextDecorator
from typing import Optional, Dict, ClassVar

class Timer(ContextDecorator):
    """性能测量计时器。
    对标 Isaac Lab 的 Timer。
    """
    timing_info: ClassVar[Dict[str, float]] = {}

    def __init__(self, msg: Optional[str] = None, name: Optional[str] = None):
        self._msg = msg
        self._name = name
        self._start_time = None
        self._elapsed = 0.0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        if self._msg:
            print(f"[Timer] {self._msg}: {self._elapsed:.6f}s")

    def start(self):
        self._start_time = time.perf_counter()

    def stop(self):
        if self._start_time is not None:
            self._elapsed = time.perf_counter() - self._start_time
            if self._name:
                Timer.timing_info[self._name] = self._elapsed
            self._start_time = None

    @property
    def time_elapsed(self) -> float:
        if self._start_time is not None:
            return time.perf_counter() - self._start_time
        return self._elapsed
