"""
模块用途：键盘输入检测，提供当前按键集合。

包含：
- 函数：key_check()

边界：
- 负责：读取键盘输入状态
- 不负责：动作执行、训练控制
"""


import win32api as wapi
import time

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
    keyList.append(char)

def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys