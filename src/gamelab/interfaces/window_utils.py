"""
模块用途：窗口管理与前台激活、移动、置顶等操作封装。

包含：
- 函数：move_window, activate_window_by_title_contains, set_window_topmost 等

边界：
- 负责：窗口层面的控制与摆放
- 不负责：图像采集与训练逻辑
"""
import win32gui
import win32con
import win32api

def find_window_by_title_contains(title_part, strict=False):
    title_q = (str(title_part) if title_part is not None else "").strip()
    if title_q == "":
        return None

    exact_hwnd = None
    fallback_hwnd = None
    fallback_title_len = None

    def callback(hw, extra):
        nonlocal exact_hwnd, fallback_hwnd, fallback_title_len
        if not win32gui.IsWindowVisible(hw):
            return True
        t = win32gui.GetWindowText(hw)
        tl = t.strip().lower()
        ql = title_q.lower()
        if tl == ql:
            exact_hwnd = hw
            return False
        
        if not strict:
            if ql in tl:
                if fallback_hwnd is None:
                    fallback_hwnd = hw
                    fallback_title_len = len(tl)
                else:
                    if len(tl) < (fallback_title_len or 0):
                        fallback_hwnd = hw
                        fallback_title_len = len(tl)
        return True

    try:
        win32gui.EnumWindows(callback, None)
    except Exception as e:
        if "拒绝访问" in str(e) or "Access is denied" in str(e):
            print("\n" + "!"*60)
            print("错误：枚举窗口被拒绝访问。")
            print("请尝试以【管理员身份】运行 IDE 或终端。")
            print("!"*60 + "\n")
        raise e

    if exact_hwnd:
        return exact_hwnd
    
    if strict:
        return None
        
    return fallback_hwnd

def activate_window_by_title_contains(window_title_part, strict=False):
    hwnd = find_window_by_title_contains(window_title_part, strict=strict)
    if not hwnd:
        return False

    try:
        if win32gui.GetForegroundWindow() != hwnd:
            win32gui.SetForegroundWindow(hwnd)
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            print(f"Window '{window_title_part}' activated.")
        return True
    except Exception as e:
        print(f"Error activating window: {e}")
        return False

def set_window_topmost(window_title_part):
    try:
        hwnd = find_window_by_title_contains(window_title_part)
        if hwnd:
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
            print(f"Window with title containing '{window_title_part}' set to TOPMOST.")
            return True
        else:
            print(f"Window with title containing '{window_title_part}' not found.")
            return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def remove_window_topmost(window_title_part):
    try:
        hwnd = find_window_by_title_contains(window_title_part)
        if hwnd:
            win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
            print(f"Window with title containing '{window_title_part}' returned to NORMAL.")
            return True
        else:
            print(f"Window with title containing '{window_title_part}' not found.")
            return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def move_window(window_title_part, x, y=None, width=None, height=None):
    """移动并调整窗口大小。
    
    支持两种调用方式：
    1. move_window(title, x, y, width, height) - 经典 win32 风格
    2. move_window(title, pos_str, repaint=True) - 预设位置风格 (top_left, center 等)
    """
    try:
        hwnd = find_window_by_title_contains(window_title_part)
        if not hwnd:
            print(f"Window with title containing '{window_title_part}' not found.")
            return False

        # 处理预设位置字符串 (如 "top_left")
        if isinstance(x, str):
            pos_str = x
            repaint = y if y is not None else True
            
            # 获取窗口当前大小
            rect = win32gui.GetWindowRect(hwnd)
            w = width if width is not None else (rect[2] - rect[0])
            h = height if height is not None else (rect[3] - rect[1])
            
            # 获取屏幕大小
            sw = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
            sh = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
            
            # 计算坐标
            nx, ny = 0, 0
            if pos_str == "top_left":
                nx, ny = 0, 0
            elif pos_str == "top_right":
                nx, ny = sw - w, 0
            elif pos_str == "bottom_left":
                nx, ny = 0, sh - h
            elif pos_str == "bottom_right":
                nx, ny = sw - w, sh - h
            elif pos_str == "center":
                nx, ny = (sw - w) // 2, (sh - h) // 2
            elif pos_str == "offscreen":
                nx, ny = -w - 100, -h - 100
            else:
                print(f"Unknown position string: {pos_str}")
                return False
                
            win32gui.MoveWindow(hwnd, nx, ny, w, h, repaint)
            print(f"Window '{window_title_part}' moved to {pos_str} ({nx}, {ny}) with size {w}x{h}.")
            return True
        else:
            # 经典 5 参数调用
            if y is None or width is None or height is None:
                print("Error: move_window requires (x, y, width, height) when x is not a string.")
                return False
            win32gui.MoveWindow(hwnd, x, y, width, height, True)
            print(f"Window '{window_title_part}' moved to ({x}, {y}) with size {width}x{height}.")
            return True
            
    except Exception as e:
        print(f"An error occurred in move_window: {e}")
        return False
