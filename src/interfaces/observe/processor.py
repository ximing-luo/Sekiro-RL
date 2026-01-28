'''
Description: 
Version: 2.0
Autor: Zhang
Date: 2021-11-14 17:52:15
LastEditors: Zhang
LastEditTime: 2021-12-04 16:43:10
'''

import torch
import numpy as np
import cv2
import time

def self_blood_count(self_bgr):
    # 计算自身血量
    # 将NumPy数组转换为PyTorch张量，按比例选择靠近底部的行，避免不同分辨率越界
    h = self_bgr.shape[0]
    row_idx = min(h - 1, int(h * (893 / 907)))
    # 复制成可写连续内存，避免 PyTorch 关于不可写数组的警告
    row_np = np.ascontiguousarray(self_bgr[row_idx].copy())
    middle_row_tensor = torch.from_numpy(row_np).float()

    # 调整后的红色的BGR颜色范围
    lower_red_tensor = torch.tensor([40, 40, 90], dtype=torch.float32)
    upper_red_tensor = torch.tensor([100, 100, 190], dtype=torch.float32)
    lower_white_tensor = torch.tensor([130, 130, 130], dtype=torch.float32)
    upper_white_tensor = torch.tensor([170, 170, 170], dtype=torch.float32)
    white_mask = (middle_row_tensor >= lower_white_tensor) & (middle_row_tensor <= upper_white_tensor)
    white_idx = torch.nonzero(white_mask.all(dim=-1), as_tuple=False).view(-1).cpu().numpy()
    if white_idx.size >= 2:
        segs = []
        start = int(white_idx[0])
        prev = start
        for x in white_idx[1:]:
            xi = int(x)
            if xi - prev <= 100:
                prev = xi
            else:
                segs.append((start, prev))
                start = xi
                prev = xi
        segs.append((start, prev))
        if len(segs) >= 2:
            a = max(0, segs[0][1])
            b = min(middle_row_tensor.shape[0] - 1, segs[1][0])
            if b >= a:
                row_slice = middle_row_tensor[a:b+1]
                mask = (row_slice >= lower_red_tensor) & (row_slice <= upper_red_tensor)
                self_blood = torch.sum(mask.all(dim=-1)).item()
                return self_blood
    mask = (middle_row_tensor >= lower_red_tensor) & (middle_row_tensor <= upper_red_tensor)
    self_blood = torch.sum(mask.all(dim=-1)).item()
    return self_blood

def boss_blood_count(boss_bgr):
    # 计算boss血量
    # 将NumPy数组转换为PyTorch张量，按比例选择靠近顶部的行，避免不同分辨率越界
    h = boss_bgr.shape[0]
    row_idx = min(h - 1, int(h * (5 / 907)))
    row_np = np.ascontiguousarray(boss_bgr[row_idx].copy())
    middle_row_tensor = torch.from_numpy(row_np).float()
    # 调试可视：如需查看该行，可取消注释
    # row_img = np.repeat(row_np[np.newaxis, :, :], 40, axis=0)
    # cv2.imshow('middle_row', row_img)
    # 调整后的红色的BGR颜色范围
    lower_red_tensor = torch.tensor([0, 0, 80], dtype=torch.float32)
    upper_red_tensor = torch.tensor([45, 45, 255], dtype=torch.float32)
    lower_white_tensor = torch.tensor([130, 130, 130], dtype=torch.float32)
    upper_white_tensor = torch.tensor([170, 170, 170], dtype=torch.float32)
    white_mask = (middle_row_tensor >= lower_white_tensor) & (middle_row_tensor <= upper_white_tensor)
    white_idx = torch.nonzero(white_mask.all(dim=-1), as_tuple=False).view(-1).cpu().numpy()
    if white_idx.size >= 2:
        segs = []
        start = int(white_idx[0])
        prev = start
        for x in white_idx[1:]:
            xi = int(x)
            if xi - prev <= 100:
                prev = xi
            else:
                segs.append((start, prev))
                start = xi
                prev = xi
        segs.append((start, prev))
        if len(segs) >= 2:
            a = max(0, segs[0][1])
            b = min(middle_row_tensor.shape[0] - 1, segs[1][0])
            if b >= a:
                row_slice = middle_row_tensor[a:b+1]
                mask = (row_slice >= lower_red_tensor) & (row_slice <= upper_red_tensor)
                boss_blood = torch.sum(mask.all(dim=-1)).item()
                return boss_blood
    mask = (middle_row_tensor >= lower_red_tensor) & (middle_row_tensor <= upper_red_tensor)
    boss_blood = torch.sum(mask.all(dim=-1)).item()
    return boss_blood

def self_stamina_count(self_bgr):
    # 计算自身架势条
    # 将NumPy数组转换为PyTorch张量，按比例选择靠近底部的行，避免不同分辨率越界
    h = self_bgr.shape[0]
    row_idx = min(h - 1, int(h * (885 / 900)))
    row_np = np.ascontiguousarray(self_bgr[row_idx].copy())
    middle_row_tensor = torch.from_numpy(row_np).float()
    # row_img = np.repeat(row_np[np.newaxis, :, :], 40, axis=0)
    # cv2.imshow('middle_row', row_img)

    # 调整后的橙黄色BGR颜色范围
    lower_orange_yellow_tensor = torch.tensor([0, 70, 110], dtype=torch.float32)
    upper_orange_yellow_tensor = torch.tensor([40, 140, 255], dtype=torch.float32)

    # 使用张量运算进行范围检查
    mask = (middle_row_tensor >= lower_orange_yellow_tensor) & \
           (middle_row_tensor <= upper_orange_yellow_tensor)

    # mask的形状是 (width, 3)，我们需要检查所有3个颜色通道都在范围内
    # 所以我们需要对最后一个维度进行all操作
    self_stamina = torch.sum(mask.all(dim=-1)).item()
    return self_stamina

def boss_stamina_count(boss_bgr):
    # 计算boss架势条
    # 将NumPy数组转换为PyTorch张量，按比例选择靠近顶部的行，避免不同分辨率越界
    h = boss_bgr.shape[0]  # 获取图像高度（像素行数），用于后续按比例定位行
    row_idx = min(h - 1, int(h * (10 / 900)))  # 按高度的 10/900 比例取接近顶部的一行，并用 min 防止越界
    row_np = np.ascontiguousarray(boss_bgr[row_idx].copy())  # 复制该行并转为连续内存，确保可写且方便与 PyTorch 共享
    middle_row_tensor = torch.from_numpy(row_np).float()  # 将该行的 BGR 值转换为 float32 张量，便于数值比较与阈值运算
    # row_img = np.repeat(row_np[np.newaxis, :, :], 40, axis=0)
    # cv2.imshow('middle_row1', row_img)
    # 调整后的橙黄色BGR颜色范围
    lower_orange_yellow_tensor = torch.tensor([0, 70, 140], dtype=torch.float32)
    upper_orange_yellow_tensor = torch.tensor([20, 140, 255], dtype=torch.float32)

    # 使用张量运算进行范围检查
    mask = (middle_row_tensor >= lower_orange_yellow_tensor) & \
           (middle_row_tensor <= upper_orange_yellow_tensor)

    # mask的形状是 (width, 3)，我们需要检查所有3个颜色通道都在范围内
    # 所以我们需要对最后一个维度进行all操作
    boss_stamina = torch.sum(mask.all(dim=-1)).item()
    return boss_stamina

# def process_observation(width, height):
#     '''
#     处理观察到的图像，将其转换为模型输入的格式
#     Args:
#         width: 图像宽度
#         height: 图像高度
#     Returns:
#         obs: 处理后的图像，形状为 (height, width, 3)
#     '''
#     # 抓取窗口图像（BGR），调整尺寸并转换为 RGB
#     obs_bgr = grabscreen.grab_window_screen('Sekiro')
#     obs_resize = cv2.resize(obs_bgr, (width, height))
#     obs_rgb = cv2.cvtColor(obs_resize, cv2.COLOR_BGR2RGB)
#     # 输出形状为 (height, width, 3)
#     obs = np.array(obs_rgb).reshape(-1, height, width, 3)[0]
#     return obs

def crop_image(image, window_list):
    """
    根据窗口列表对图像进行裁剪
    :param image: 输入图像 (numpy array)
    :param window_list: 窗口列表，每个元素为 (x, y, width, height) 四元组
    :return: 裁剪后的图像
    """
    cropped_images = []
    for (x, y, width, height) in window_list:
        cropped = image[y:y+height, x:x+width]
        cropped_images.append(cropped)
    return cropped_images

if __name__ == '__main__':

    wait_time = 1

    blood_window = (110,90,625,907)
    stamina_window = (586,54,750,900)

    for i in list(range(wait_time))[::-1]:
        print(i+1)
        time.sleep(1)

    import ctypes
    import cv2

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()
    
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError('camera read failed')
    h, w = frame.shape[:2]

    last_time = time.time()
    printed_size = False
    flag = 0

    _step_counter = 0
    while(True):
        ret, observe = cap.read()
        if not ret or observe is None:
            continue
        
        if not printed_size:
            h, w = observe.shape[:2]
            print(f'input size: {w}x{h}')
            printed_size = True
        observe_resize = cv2.resize(observe, (960, 540), interpolation=cv2.INTER_AREA)
        obs = np.array(observe_resize).reshape(-1, 540, 960, 3)[0]

        blood_bgr = crop_image(observe, [blood_window])[0]
        self_blood = self_blood_count(blood_bgr)
        boss_blood = boss_blood_count(blood_bgr)

        stamina_bgr = crop_image(observe, [stamina_window])[0]
        self_stamina = self_stamina_count(stamina_bgr)
        boss_stamina = boss_stamina_count(stamina_bgr)
        _step_counter += 1
        if _step_counter % 40 == 0:
            print("狼的血量:",self_blood,"boos的血量:",boss_blood)
            print("狼的架势条:",self_stamina,"boos的架势条:",boss_stamina,"\n")

        flag += 1
        if flag % 30 == 0:
            print(f'self_blood: {self_blood}, boss_blood: {boss_blood}, self_stamina: {self_stamina}, boss_stamina: {boss_stamina}')
            print('loop took {} seconds'.format(time.time()-last_time))
        
        cv2.imshow('window3', blood_bgr)



        last_time = time.time()
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.waitKey()
    cv2.destroyAllWindows()
"""
模块用途：图像裁剪与指标识别的基础处理方法。

包含：
- 函数：crop_image, self_blood_count, boss_blood_count, self_stamina_count, boss_stamina_count 等

边界：
- 负责：图像预处理与指标提取算法
- 不负责：采集管理、奖励计算、训练逻辑
"""
