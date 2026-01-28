"""
模块用途：底层按键发送与键盘事件封装。

包含：
- 函数：用于发送虚拟键码的工具方法

边界：
- 负责：系统层面的按键注入
- 不负责：动作语义、环境逻辑与训练
"""
'''
Description: 
Version: 2.0
Autor: Zhang
Date: 2021-11-14 15:08:58
LastEditors: Zhang
LastEditTime: 2021-12-04 15:02:57
'''

import ctypes
import threading

SendInput = ctypes.windll.user32.SendInput
_LOCK = threading.Lock()
_KEY_RC = {}


W = 0x11
A = 0x1E
S = 0x1F
D = 0x20

M = 0x32
J = 0x24
K = 0x25
LSHIFT = 0x2A
R = 0x13#用R代替识破
V = 0x2F

Q = 0x10
I = 0x17
O = 0x18
P = 0x19
C = 0x2E
F = 0x21
G = 0x22
T = 0x14
Y = 0x15
SPACE = 0x39

up = 0xC8
down = 0xD0
left = 0xCB
right = 0xCD

esc = 0x01

# C struct redefinitions 
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

def _send_press(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
def _send_release(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
def PressKey(hexKeyCode):
    with _LOCK:
        s = _KEY_RC.get(hexKeyCode)
        if s is None:
            s = {"rc": 0, "pressed": False}
            _KEY_RC[hexKeyCode] = s
        s["rc"] += 1
        if not s["pressed"]:
            _send_press(hexKeyCode)
            s["pressed"] = True

def ReleaseKey(hexKeyCode):
    with _LOCK:
        s = _KEY_RC.get(hexKeyCode)
        if s is None:
            s = {"rc": 0, "pressed": False}
            _KEY_RC[hexKeyCode] = s
        if s["rc"] > 0:
            s["rc"] -= 1
        if s["rc"] == 0 and s["pressed"]:
            _send_release(hexKeyCode)
            s["pressed"] = False
def ForceReleaseKey(hexKeyCode):
    with _LOCK:
        s = _KEY_RC.get(hexKeyCode)
        if s is None:
            s = {"rc": 0, "pressed": False}
            _KEY_RC[hexKeyCode] = s
        s["rc"] = 0
        if s["pressed"]:
            _send_release(hexKeyCode)
            s["pressed"] = False
    
    
    
