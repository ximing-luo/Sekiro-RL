import os
import sys
import time

# 将项目根目录添加到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.interfaces.observe.telemetry import SekiroTelemetry

def test_telemetry():
    print("开始测试 Sekiro 遥测功能...")
    telemetry = SekiroTelemetry()
    
    # 启动后台线程
    telemetry.start()
    
    try:
        print("等待数据刷新 (持续 5 秒)...")
        for i in range(5000):
            # 获取核心指标
            metrics = telemetry.get_metrics()
            # 获取所有详细数据
            all_data = telemetry.data
            
            # 预格式化指标字符串，确保即便为 0 也能完整显示 x/y 格式
            p_hp = f"{all_data['player_hp']}/{all_data['player_hp_max']}"
            e_hp = f"{all_data['enemy_hp']}/{all_data['enemy_hp_max']}"
            p_ps = f"{all_data['player_posture']}/{all_data['player_posture_max']}"
            e_ps = f"{all_data['enemy_posture']}/{all_data['enemy_posture_max']}"
            p_deaths = f"{all_data['player_deaths']}"
            e_deaths = f"{all_data['enemy_deaths']}"
            
            # 使用更短的标签并统一宽度，防止终端自动换行或截断
            output = (f"\r[帧 {i+1:02d}] "
                      f"P-HP: {p_hp:<9s} | "
                      f"E-HP: {e_hp:<11s} | "
                      f"P-PS: {p_ps:<9s} | "
                      f"E-PS: {e_ps:<9s} | "
                      f"P-D: {p_deaths:<2s} | "
                      f"E-D: {e_deaths:<2s} ")
            print(output, end="", flush=True)
            
            time.sleep(0.1)
        print("\n测试完成。")
    except KeyboardInterrupt:
        print("\n测试被用户中断。")
    finally:
        telemetry.stop()
        print("遥测线程已停止。")

if __name__ == "__main__":
    test_telemetry()
