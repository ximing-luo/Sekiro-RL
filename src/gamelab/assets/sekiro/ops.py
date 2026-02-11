import time

from src.gamelab.interfaces.input.driver import PressKey, ReleaseKey
from .keys import *

fps = 60
t = 1/fps

def no_op():
    """无动作：延时ts"""               
    # time.sleep(1/fps)
    pass
    
def attack():   #攻击
    PressKey(J)
    time.sleep(t)
    ReleaseKey(J)

def defense():   #格挡
    PressKey(G)
    time.sleep(2*t)
    ReleaseKey(G)
    
def jump():  #跳跃
    PressKey(SPACE)
    time.sleep(t)
    ReleaseKey(SPACE)
    
def ninja_attack(): #忍义手攻击
    PressKey(F)
    time.sleep(t)
    ReleaseKey(F)

def skill_attack(): #技能
    PressKey(G)
    PressKey(J)
    time.sleep(t)
    ReleaseKey(G)
    ReleaseKey(J)

def F_go():  #钩爪
    PressKey(Q)
    time.sleep(t)
    ReleaseKey(Q)

t_go = t * 6 * 2

def go_forward(): #前进
    PressKey(W)
    time.sleep(t_go)
    ReleaseKey(W)
    
def go_back(): #后退
    PressKey(S)
    time.sleep(t_go)
    ReleaseKey(S)
    
def go_left(): #左
    PressKey(A)
    time.sleep(t_go)
    ReleaseKey(A)
    
def go_right(): #右
    PressKey(D)
    time.sleep(t_go)
    ReleaseKey(D)

def dodge_forward(): #闪避
    PressKey(LSHIFT)
    time.sleep(t)
    ReleaseKey(LSHIFT)

def use_items(): #使用道具
    PressKey(R)
    time.sleep(t)
    ReleaseKey(R)
   
def lock_vision(): #锁定敌人
    PressKey(Y)
    time.sleep(t)
    ReleaseKey(Y)
    
def turn_left(): #视角向左
    PressKey(left)
    time.sleep(t)
    ReleaseKey(left)
    
def turn_up(): #视角向上
    PressKey(up)
    time.sleep(t)
    ReleaseKey(up)
    
def turn_right(): #视角向右
    PressKey(right)
    time.sleep(t)
    ReleaseKey(right)

def turn_down(): #视角向下
    PressKey(down)
    time.sleep(t)
    ReleaseKey(down)


if __name__ == '__main__':
    time.sleep(0.1)
    skill_attack()
    turn_left(0.8)
    attack()
    time.sleep(0.1)
    defense()
    time.sleep(0.1)
    jump()
    time.sleep(0.1)
    dodge_forward()
    time.sleep(0.1)
