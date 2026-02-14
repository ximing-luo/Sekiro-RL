import threading
import time
import heapq
from typing import Callable, Any

class GhostScheduler:
    """时间轴幽灵：以最小作用量原理调度延时任务，逻辑与执行高度同构。"""
    def __init__(self):
        self._queue = []  # (deadline, func, args)
        self._lock = threading.Lock()
        self._event = threading.Event()
        threading.Thread(target=self._run, daemon=True, name="GhostScheduler").start()

    def enter(self, delay: float, func: Callable, *args: Any):
        """将意图投入时间轴。"""
        with self._lock:
            heapq.heappush(self._queue, (time.time() + delay, func, args))
            self._event.set()

    def _run(self):
        while True:
            self._event.clear()
            now = time.time()
            wait = 1.0

            with self._lock:
                while self._queue and self._queue[0][0] <= now:
                    _, func, args = heapq.heappop(self._queue)
                    func(*args)
                
                if self._queue:
                    wait = max(0, self._queue[0][0] - now)

            self._event.wait(timeout=wait)

# 单例即契约
scheduler = GhostScheduler()
