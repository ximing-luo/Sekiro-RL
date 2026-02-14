import time

from src.gamelab.interfaces.input.driver import PressKey, ReleaseKey
from src.gamelab.interfaces.input.scheduler import scheduler as s
from .keys import *

fps = 60
t = 1/fps

def no_op():
    """无动作：延时ts"""               
    # time.sleep(1/fps)
    pass
    
def attack():   #攻击
    PressKey(J)
    s.enter(t, ReleaseKey, J)

def defense():   #格挡
    PressKey(G)
    s.enter(2*t, ReleaseKey, G)

def jump():  #跳跃
    PressKey(SPACE)
    s.enter(t, ReleaseKey, SPACE)
    
def ninja_attack(): #忍义手攻击
    PressKey(F)
    s.enter(t, ReleaseKey, F)

def skill_attack(): #技能
    PressKey(G)
    PressKey(J)
    s.enter(t, ReleaseKey, G)
    s.enter(t, ReleaseKey, J)

def F_go():  #钩爪
    PressKey(Q)
    s.enter(t, ReleaseKey, Q)

t_go = t * 6 * 2

def go_forward(): #前进
    PressKey(W)
    s.enter(t_go, ReleaseKey, W)
    
def go_back(): #后退
    PressKey(S)
    s.enter(t_go, ReleaseKey, S)
    
def go_left(): #左
    PressKey(A)
    s.enter(t_go, ReleaseKey, A)
    
def go_right(): #右
    PressKey(D)
    s.enter(t_go, ReleaseKey, D)

def dodge_forward(): #闪避
    PressKey(LSHIFT)
    s.enter(t, ReleaseKey, LSHIFT)

def use_items(): #使用道具
    PressKey(R)
    s.enter(t, ReleaseKey, R)
   
def lock_vision(): #锁定敌人
    PressKey(Y)
    s.enter(t, ReleaseKey, Y)
    
def turn_left(): #视角向左
    PressKey(left)
    s.enter(t, ReleaseKey, left)
    
def turn_up(): #视角向上
    PressKey(up)
    s.enter(t, ReleaseKey, up)
    
def turn_right(): #视角向右
    PressKey(right)
    s.enter(t, ReleaseKey, right)

def turn_down(): #视角向下
    PressKey(down)
    s.enter(t, ReleaseKey, down)


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
