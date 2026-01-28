import threading
from src.envs.tasks.sekiro.action_map import get_action_callable, action_count, assert_config_consistency

class ActionManager:
    """
    动作管理器：负责动作映射和键盘指令的异步执行。
    对应 Isaac Lab 中的 ActionManager。
    """
    def __init__(self, action_dim=None):
        self.action_dim = int(action_dim) if action_dim is not None else int(action_count())
        assert_config_consistency(self.action_dim)

    def apply_action(self, action):
        """异步执行动作，避免阻塞主循环。"""
        # 注意：在重构后的架构中，这里也可以通过接口控制不同的执行器（如键盘、控制器）
        fn = get_action_callable(action)
        threading.Thread(target=fn, daemon=True).start()

    def get_action_dim(self):
        return self.action_dim
