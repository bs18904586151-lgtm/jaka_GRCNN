"""
DH-Robotics PGEA 夹爪控制模块
硬件：/dev/ttyUSB0，115200，无校验，1停止位
位置：0=关闭/夹住，1000=打开

关键发现：
  - 必须用 0x10 功能码一次性写 力值+位置+速度，分开发无效
  - 每次连接串口后必须重新初始化
  - 初始化流程：0xA5（等8秒）→ 夹爪打开位置=1000
"""

import serial
import time

_CMD_INIT_FULL    = bytes([0x01,0x06,0x01,0x00,0x00,0xA5,0x48,0x4D])
_READ_GRIP_STATUS = bytes([0x01,0x03,0x02,0x01,0x00,0x01,0xD4,0x72])
_READ_POS         = bytes([0x01,0x03,0x02,0x02,0x00,0x01,0x24,0x72])


def _crc16(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


def _build_cmd(pos: int, force: int = 100, speed: int = 30) -> bytes:
    pos   = max(0,  min(1000, pos))
    force = max(20, min(100,  force))
    speed = max(1,  min(100,  speed))
    payload = bytes([
        0x01, 0x10,
        0x01, 0x00,
        0x00, 0x05,
        0x0A,
        0x00, 0x00,
        0x00, force,
        0x00, 0x00,
        (pos >> 8) & 0xFF, pos & 0xFF,
        0x00, speed,
    ])
    crc = _crc16(payload)
    return payload + bytes([crc & 0xFF, (crc >> 8) & 0xFF])


def _read_reg(ser, cmd: bytes):
    ser.reset_input_buffer()
    ser.write(cmd)
    time.sleep(0.3)
    resp = ser.read(ser.in_waiting or 8)
    if resp and len(resp) >= 5:
        return (resp[3] << 8) | resp[4]
    return None


def _send(ser, cmd: bytes):
    ser.reset_input_buffer()
    ser.write(cmd)
    time.sleep(0.3)
    return ser.read(ser.in_waiting or 8)


class Gripper:
    def __init__(self, port="/dev/ttyUSB0", baud=115200):
        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(0.5)
        print(f"[Gripper] 已连接 {port}")

    def initialize(self):
        """每次连接串口后必须调用。耗时约8秒。完成后夹爪打开（位置=1000）。"""
        print("[Gripper] 0xA5 完整初始化（等8秒，夹爪先闭后开）...")
        _send(self.ser, _CMD_INIT_FULL)
        time.sleep(8)
        pos = _read_reg(self.ser, _READ_POS)
        ok = pos is not None and pos > 800
        print(f"[Gripper] 初始化完成，位置={pos}/1000  {'✓ 打开' if ok else '⚠ 异常'}")
        return ok

    def open(self, force: int = 100, speed: int = 30):
        """打开夹爪到位置1000。"""
        print(f"[Gripper] 打开 force={force}% speed={speed}%")
        cmd = _build_cmd(1000, force, speed)
        _send(self.ser, cmd)
        time.sleep(3)
        pos = _read_reg(self.ser, _READ_POS)
        ok = pos is not None and pos > 800
        print(f"[Gripper] 位置={pos}/1000  {'✓ 已打开' if ok else '⚠ 未完全打开'}")
        return ok

    def close(self, force: int = 100, speed: int = 30):
        """夹住物体（位置=0）。返回True=夹住，False=未夹到。"""
        print(f"[Gripper] 夹取 force={force}% speed={speed}%")
        cmd = _build_cmd(0, force, speed)
        _send(self.ser, cmd)
        time.sleep(3)
        grip = _read_reg(self.ser, _READ_GRIP_STATUS)
        pos  = _read_reg(self.ser, _READ_POS)
        ok = grip == 2
        print(f"[Gripper] 夹持状态={grip}，位置={pos}/1000  {'✓ 夹住' if ok else '⚠ 未夹到'}")
        return ok

    def move_to(self, pos: int, force: int = 100, speed: int = 30):
        """移动到任意位置（0-1000）"""
        print(f"[Gripper] 移动到 pos={pos} force={force}% speed={speed}%")
        cmd = _build_cmd(pos, force, speed)
        _send(self.ser, cmd)
        time.sleep(3)
        cur = _read_reg(self.ser, _READ_POS)
        print(f"[Gripper] 当前位置={cur}/1000 ({cur/10:.1f}%)")
        return cur

    def get_position(self) -> int:
        """读取当前位置（0-1000）"""
        return _read_reg(self.ser, _READ_POS)

    def get_grip_status(self) -> int:
        """0=运动中, 1=到位, 2=夹住物体, 3=物体掉落"""
        return _read_reg(self.ser, _READ_GRIP_STATUS)

    def interactive(self):
        """
        终端交互控制：
        输入位置百分比和速度，实时控制夹爪。
        输入 q 退出。
        """
        print("\n" + "="*40)
        print("夹爪交互控制模式")
        print("位置: 0=完全关闭  100=完全打开")
        print("速度: 1-100%")
        print("输入 q 退出")
        print("="*40 + "\n")

        while True:
            try:
                # 输入位置
                p_str = input("位置百分比(0-100): ").strip()
                if p_str.lower() == 'q':
                    break
                p = float(p_str)
                if not 0 <= p <= 100:
                    print("⚠ 位置必须在 0-100 之间")
                    continue

                # 输入速度
                sp_str = input("速度百分比(1-100): ").strip()
                if sp_str.lower() == 'q':
                    break
                sp = float(sp_str)
                if not 1 <= sp <= 100:
                    print("⚠ 速度必须在 1-100 之间")
                    continue

                # 可选输入力值（直接回车用默认100%）
                f_str = input("力值百分比(20-100, 回车默认100): ").strip()
                if f_str.lower() == 'q':
                    break
                f = float(f_str) if f_str else 100.0
                if not 20 <= f <= 100:
                    print("⚠ 力值必须在 20-100 之间")
                    continue

                pos   = int(p * 10)   # 0-100% → 0-1000
                speed = int(sp)
                force = int(f)

                print(f"→ 移动到 {p}% (pos={pos}) 速度={speed}% 力值={force}%")
                self.move_to(pos, force=force, speed=speed)
                print("-"*40)

            except ValueError:
                print("⚠ 请输入有效数字")
            except KeyboardInterrupt:
                break

        print("退出交互模式")

    def disconnect(self):
        self.ser.close()
        print("[Gripper] 串口已关闭")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.disconnect()


# ── 主程序 ───────────────────────────────────────────────
if __name__ == "__main__":
    with Gripper("/dev/ttyUSB0") as g:
        g.initialize()       # 约8秒，完成后夹爪打开

        # 进入终端交互控制
        g.interactive()

        print("完成。")