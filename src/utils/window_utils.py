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

    win32gui.EnumWindows(callback, None)
    if exact_hwnd:
        return exact_hwnd
    
    if strict:
        return None
        
    return fallback_hwnd

def activate_window_by_title_contains(window_title_part, strict=False):
    hwnd = find_window_by_title_contains(window_title_part, strict=strict)
    if not hwnd:
        print(f"Window containing '{window_title_part}' (strict={strict}) not found.")
        return False

    try:
        win32gui.SetForegroundWindow(hwnd)
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        print(f"Window containing '{window_title_part}' activated.")
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

def place_window_above(target_window_title, reference_window_title):
    try:
        target_hwnd = find_window_by_title_contains(target_window_title)
        if not target_hwnd:
            print(f"Target window '{target_window_title}' not found.")
            return False

        reference_hwnd = find_window_by_title_contains(reference_window_title)
        if not reference_hwnd:
            print(f"Reference window '{reference_window_title}' not found.")
            return False

        win32gui.SetWindowPos(target_hwnd, reference_hwnd, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
        print(f"Successfully placed window '{target_window_title}' above '{reference_window_title}'.")
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def move_window(window_title_part, position, strict=False):
    try:
        hwnd = find_window_by_title_contains(window_title_part, strict=strict)
        if not hwnd:
            print(f"Window containing '{window_title_part}' (strict={strict}) not found.")
            return False

        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width = right - left
        height = bottom - top
        desktop_left, desktop_top, desktop_right, desktop_bottom = win32gui.GetWindowRect(win32gui.GetDesktopWindow())

        if position == "center":
            x = desktop_left + (desktop_right - desktop_left - width) // 2
            y = desktop_top + (desktop_bottom - desktop_top - height) // 2
        elif position == "offscreen":
            x = 10000
            y = 0
        elif position == "top_left":
            x = 0
            y = 0
        elif position == "top_right":
            x = desktop_right - width
            y = 0
        else:
            print(f"Unknown position: '{position}'. Use 'center' or 'offscreen'.")
            return False

        win32gui.SetWindowPos(hwnd, 0, x, y, 0, 0, win32con.SWP_NOSIZE | win32con.SWP_NOZORDER)
        print(f"Successfully moved window '{window_title_part}' to '{position}'.")
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

if __name__ == '__main__':
    activate_window_by_title_contains("Cheat Engine")
    set_window_topmost("Cheat Engine")
    