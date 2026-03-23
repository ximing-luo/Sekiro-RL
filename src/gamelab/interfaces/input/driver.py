import ctypes
import threading

SendInput = ctypes.windll.user32.SendInput
_LOCK = threading.Lock()
_KEY_RC = {}


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
    
    
    
