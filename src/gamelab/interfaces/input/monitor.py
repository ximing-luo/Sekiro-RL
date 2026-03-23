import win32api as wapi
import win32con
import ctypes

# 预定义需要监测的按键映射 (Virtual Key Code: Label)
_KEYS_TO_MONITOR = [
    (win32con.VK_LSHIFT, 'SHIFT'),   # 垫步/冲刺
    (win32con.VK_RSHIFT, 'SHIFT'),
    (win32con.VK_LCONTROL, 'CTRL'),  # 潜行
    (win32con.VK_TAB, 'TAB'),        # 菜单/道具切换
    (win32con.VK_ESCAPE, 'ESC'),
    (win32con.VK_SPACE, 'SPACE'),    # 跳跃
    (win32con.VK_LMENU, 'ALT'),      # 慢行
    (win32con.VK_BACK, 'BACK'),
    # 鼠标按键 (模仿学习核心：攻击与格挡)
    (win32con.VK_LBUTTON, 'M_LEFT'),
    (win32con.VK_RBUTTON, 'M_RIGHT'),
    (win32con.VK_MBUTTON, 'M_MIDDLE'),
]

# 自动填充 A-Z, 0-9
for char_code in range(ord('A'), ord('Z') + 1):
    _KEYS_TO_MONITOR.append((char_code, chr(char_code)))
for num_code in range(ord('0'), ord('9') + 1):
    _KEYS_TO_MONITOR.append((num_code, chr(num_code)))

# 去重并排序
_KEYS_TO_MONITOR = sorted(list(set(_KEYS_TO_MONITOR)))

def key_check():
    """高效轮询当前按键状态。"""
    return [name for vk, name in _KEYS_TO_MONITOR if wapi.GetAsyncKeyState(vk) & 0x8000]

class InputMonitor:
    """全能输入监测器：集成按键轮询与鼠标位移检测。"""
    
    def __init__(self):
        # 初始化坐标
        self.last_x, self.last_y = wapi.GetCursorPos()

    def get_input(self):
        """获取当前输入状态：(按键列表, dx, dy)。"""
        # 1. 检测按键
        keys = key_check()
        
        # 2. 检测鼠标位移
        curr_x, curr_y = wapi.GetCursorPos()
        dx = curr_x - self.last_x
        dy = curr_y - self.last_y
        
        self.last_x, self.last_y = curr_x, curr_y
        return keys, dx, dy