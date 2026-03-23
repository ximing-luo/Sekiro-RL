import win32gui
import win32con
import win32api

def find_window_by_title(title: str):
    """精准查找窗口。基于强硬契约：错一个字符都找不到。"""
    hwnd = win32gui.FindWindow(None, title)
    return hwnd if win32gui.GetWindowText(hwnd) == title else None

def activate_window(title: str):
    """激活指定标题的窗口。"""
    hwnd = find_window_by_title(title)
    if not hwnd: return False
    
    if win32gui.GetForegroundWindow() == hwnd: return True

    try:
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
        return True
    except Exception as e:
        # 排除 SetForegroundWindow 常见的系统权限错误：(0, 'SetForegroundWindow', 'No error message is available')
        if "SetForegroundWindow" in str(e) and (" 0 " in str(e) or "(0," in str(e)):
            return False
        print(f"[WindowUtils] 激活窗口失败: {e}")
        return False

def set_window_topmost(title: str):
    """设置窗口置顶。"""
    hwnd = find_window_by_title(title)
    if not hwnd: return False
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
    return True

def remove_window_topmost(title: str):
    """取消窗口置顶。"""
    hwnd = find_window_by_title(title)
    if not hwnd: return False
    win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
    return True

def move_window(title: str, x: str | int="top_left", y=None, width=None, height=None):
    """移动并调整窗口大小。"""
    hwnd = find_window_by_title(title)
    if not hwnd: return False
    
    try:
        # 1. 物理采样：获取本体维度与屏幕边界
        rect = win32gui.GetWindowRect(hwnd)
        sw, sh = win32api.GetSystemMetrics(win32con.SM_CXSCREEN), win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
        
        # 必然性赋值：固化宽度与高度
        w, h = width or (rect[2] - rect[0]), height or (rect[3] - rect[1])

        # 2. 坐标决策：映射逻辑意图
        if isinstance(x, str):
            nx, ny = {
                "top_left":     (0, 0),
                "top_right":    (sw - w, 0),
                "bottom_left":  (0, sh - h),
                "bottom_right": (sw - w, sh - h),
                "center":       ((sw - w) // 2, (sh - h) // 2),
                "offscreen":    (-w - 100, -h - 100)
            }[x]
        else:
            nx, ny = x, y

        # 3. 物理执行：触及底层接口 (bRepaint 始终为 True)
        win32gui.MoveWindow(hwnd, nx, ny, w, h, True)
        return True
    except Exception as e:
        print(f"[WindowUtils] 移动窗口失败: {e}")
        return False

if __name__ == "__main__":
    # 测试代码
    title = "Sekiro"
    move_window(title, "top_right")
    activate_window(title)
    set_window_topmost(title)
    remove_window_topmost(title)
