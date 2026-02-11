import win32api as wapi
import win32con

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

# 自动填充 A-Z, 0-9 以及常用标点 (处理 win32con 可能缺失某些 OEM 键的情况)
for char_code in range(ord('A'), ord('Z') + 1):
    _KEYS_TO_MONITOR.append((char_code, chr(char_code)))
for num_code in range(ord('0'), ord('9') + 1):
    _KEYS_TO_MONITOR.append((num_code, chr(num_code)))

# 常用标点符号的虚拟键码 (Windows 标准)
_OEM_KEYS = [
    (0xBC, ','),  # VK_OEM_COMMA
    (0xBE, '.'),  # VK_OEM_PERIOD
    (0xBF, '/'),  # VK_OEM_2
    (0xDC, '\\'), # VK_OEM_5
    (0xC0, '`'),  # VK_OEM_3
]
for vk, label in _OEM_KEYS:
    _KEYS_TO_MONITOR.append((vk, label))

# 去重并排序以保证输出一致性
_KEYS_TO_MONITOR = sorted(list(set(_KEYS_TO_MONITOR)))

def key_check():
    """
    高效轮询当前按键状态。
    返回当前所有被按下按键的字符串标签列表。
    """
    # GetAsyncKeyState(vk) & 0x8000 判断最高位，表示按键当前是否处于按下状态
    return [name for vk, name in _KEYS_TO_MONITOR if wapi.GetAsyncKeyState(vk) & 0x8000]