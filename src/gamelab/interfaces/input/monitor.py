import win32api as wapi
import time

# 扩展 keyList 以包含常用控制键
keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
    keyList.append(char)
# 添加 Alt 键支持 (VK_MENU)
ALT_KEY = 0x12

def key_check():
    keys = []
    # 检查标准键
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    # 检查 Alt 键
    if wapi.GetAsyncKeyState(ALT_KEY):
        keys.append('ALT')
    return keys