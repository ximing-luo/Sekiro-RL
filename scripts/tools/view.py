"""
Sekiro-RL 录制素材检查工具 (Data Viewer)
使用 mmap 高性能加载 .npy 素材，并可视化播放以验证按键同步性。

使用方法:
python scripts/tools/view.py
"""

import os
import sys
import cv2
import numpy as np
import time

# 加入项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 15 位动作位掩码映射 (用于可视化标签)
ACTION_LABELS = [
    'W', 'A', 'S', 'D', 'SPACE', 'J', 'G', 'Q',
    'UP', 'DOWN', 'LEFT', 'RIGHT', 'SHIFT', 'R', 'X'
]

def main():
    record_dir = "outputs/recordings"
    if not os.path.exists(record_dir):
        print(f"未找到录制目录: {record_dir}")
        return
        
    # 列出所有子目录 (按时间排序)
    episodes = sorted([d for d in os.listdir(record_dir) if os.path.isdir(os.path.join(record_dir, d))])
    if not episodes:
        print("未找到录制素材。")
        return
        
    print("\n" + "="*45)
    print("Sekiro-RL 录制素材检查工具")
    print("="*45)
    for i, ep in enumerate(episodes):
        print(f"[{i:2d}] {ep}")
    print("="*45 + "\n")
    
    choice = input("请选择要查看的轨迹索引 (回车默认最后一条): ")
    if not choice:
        ep_name = episodes[-1]
    else:
        try:
            ep_name = episodes[int(choice)]
        except:
            print("输入无效。")
            return
        
    ep_path = os.path.join(record_dir, ep_name)
    
    # 1. 使用 mmap 加载数据 (高性能秒开，零内存占用)
    obs_path = os.path.join(ep_path, "obs.npy")
    act_path = os.path.join(ep_path, "action.npy")
    rew_path = os.path.join(ep_path, "reward.npy")
    
    try:
        obs = np.load(obs_path, mmap_mode='r')
        actions = np.load(act_path, mmap_mode='r')
        rewards = np.load(rew_path, mmap_mode='r')
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    print(f"\n加载成功: {ep_name}")
    print(f"形状: Obs {obs.shape} | Action {actions.shape} | Reward {rewards.shape}")
    print(f"提示: 按 'q' 键退出预览，按 'SPACE' 暂停/继续。")
    
    # 动态获取分辨率以维持 16:9 或原始比例
    h, w = obs.shape[2], obs.shape[3]
    vis_w = 480
    vis_h = int(vis_w * (h / w))
    
    cv2.namedWindow("Sekiro Data Inspector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sekiro Data Inspector", vis_w, vis_h)
    
    paused = False
    idx = 0
    while idx < len(obs):
        if not paused:
            # 2. 准备画面 (C, H, W) -> (H, W, C)
            frame = obs[idx].transpose(1, 2, 0)
            # RGB -> BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # 放大画面方便观看 (使用最近邻插值保持像素边缘)
            frame_vis = cv2.resize(frame_bgr, (vis_w, vis_h), interpolation=cv2.INTER_NEAREST)
            
            # 3. 绘制按键与信息
            act = actions[idx]
            active_keys = [ACTION_LABELS[i] for i, v in enumerate(act) if v > 0.5]
            rew = rewards[idx]
            
            # 动态 UI 缩放 (基于窗口宽度)
            scale = vis_w / 960.0
            rect_h = int(150 * scale)
            font_scale = 0.8 * scale
            thickness = max(1, int(2 * scale))
            line_spacing = int(45 * scale)
            margin_x = int(20 * scale)
            
            # 绘制信息栏
            overlay = frame_vis.copy()
            cv2.rectangle(overlay, (0, 0), (vis_w, rect_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame_vis, 0.4, 0, frame_vis)
            
            cv2.putText(frame_vis, f"Frame: {idx}/{len(obs)}", (margin_x, int(40 * scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            cv2.putText(frame_vis, f"Keys: {active_keys}", (margin_x, int(85 * scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.9, (0, 255, 0), thickness)
            cv2.putText(frame_vis, f"Rew: {rew:.4f}", (margin_x, int(130 * scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.9, (0, 0, 255), thickness)
            
            cv2.imshow("Sekiro Data Inspector", frame_vis)
            idx += 1
            
        key = cv2.waitKey(33) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        
        # 允许通过 'a' 和 'd' 手动控制进度 (暂停模式下)
        if paused:
            if key == ord('a'): idx = max(0, idx - 1)
            if key == ord('d'): idx = min(len(obs) - 1, idx + 1)
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
