"""
Sekiro-RL 遥测探测工具 (Telemetry Probe)
连接游戏进程并探测遥测数据基址，验证数据读取是否正常。

使用方法:
python scripts/tools/telemetry.py
"""

import os
import sys
import time
import struct

# 将项目根目录添加到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.gamelab.interfaces.telemetry.driver import TelemetryDriver

class SekiroTelemetry:
    """
    为了测试脚本适配的简单的遥测封装类。
    模拟原 SekiroTelemetry 的接口，但使用当前的 TelemetryDriver。
    """
    def __init__(self, process_name="Sekiro.exe"):
        self.driver = TelemetryDriver(process_name)
        self.base_address = None
        self.running = False
        self.data = {
            'player_hp': 0, 'player_hp_max': 0,
            'player_posture': 0, 'player_posture_max': 0,
            'enemy_hp': 0, 'enemy_hp_max': 0,
            'enemy_posture': 0, 'enemy_posture_max': 0,
            'player_deaths': 0, 'enemy_deaths': 0
        }
        # 特征码，参考 SekiroAssetCfg
        self.signature = b"SEKIRO_TLM"

    def start(self):
        print(f"正在连接到进程 {self.driver.process_name}...")
        pids = TelemetryDriver.find_pids_by_name(self.driver.process_name)
        if not pids:
            print(f"未找到进程 {self.driver.process_name}")
            return
        
        # 连接第一个进程
        pid = pids[0]
        if self.driver.connect(pid=pid):
            print(f"已连接到 PID: {pid}")
            self.base_address = self.driver.pattern_scan_all(self.signature)
            if self.base_address:
                print(f"找到遥测基址: {hex(self.base_address)}")
                self.running = True
            else:
                print("未找到遥测特征码。")
        else:
            print("连接失败。")

    def stop(self):
        self.running = False

    def get_metrics(self):
        if not self.running or not self.base_address:
            return {}
        
        try:
            # 参考 SekiroAsset.update 的读取逻辑
            # 范围：从偏移 12 到 60 (共 48 字节)
            raw_data = self.driver.read_bytes(self.base_address + 12, 48)
            # 解析: <II4xIIII4xIIII (I=uint32, 4x=padding)
            values = struct.unpack("<II4xIIII4xIIII", raw_data)
            
            self.data['player_hp'] = values[0]
            self.data['player_hp_max'] = values[1]
            self.data['player_posture'] = values[2]
            self.data['player_posture_max'] = values[3]
            self.data['enemy_hp'] = values[4]
            self.data['enemy_hp_max'] = values[5]
            self.data['enemy_posture'] = values[6]
            self.data['enemy_posture_max'] = values[7]
            self.data['player_deaths'] = values[8]
            self.data['enemy_deaths'] = values[9]
            
            return self.data
        except Exception as e:
            # print(f"读取错误: {e}")
            return {}
        
    def write(self):
        self.driver.write_int(self.base_address + 60, 1)

def run_probe():
    print("开始测试 Sekiro 遥测功能...")
    telemetry = SekiroTelemetry()
    
    # 启动 (这里不是线程，只是初始化连接)
    telemetry.start()
    
    if not telemetry.running:
        print("遥测初始化失败，退出。")
        return

    try:
        print("等待数据刷新 (持续 5 秒)...s")
        telemetry.write()
        for i in range(5000):
            # 获取数据
            all_data = telemetry.get_metrics()
            
            if not all_data:
                time.sleep(0.1)
                continue

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
        print("遥测已停止。")

if __name__ == "__main__":
    run_probe()
