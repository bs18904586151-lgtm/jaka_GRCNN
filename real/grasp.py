#!/usr/bin/env python3
# coding=utf8
"""
grasp.py  —  点击图像目标 → 机械臂自动移动 → 下降 → 交互夹取 → 回位
使用：python3 real/grasp.py

流程：
  1. 左键点击彩色图中目标物体
  2. 机械臂自动移到目标上方（安全高度）
  3. 按 Enter → 机械臂下降
  4. 终端交互控制夹爪开合/速度/力值，确认夹住
  5. 输入 ok → 机械臂上升回位
"""

import cv2
import numpy as np
import time
import threading
import rclpy
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.join(current_dir, "..")
sys.path.append(parent_dir)

from Jaka_Robot import Jaka_Robot
from gripper import Gripper

# =========================================================
# 参数配置
# =========================================================

FX, FY = 607.879, 607.348
CX, CY = 325.14,  244.014

CAMERA_POSE = np.array([
    [-0.00710473,  0.67043268, -0.17731711,  0.16152366],
    [ 0.66803191,  0.00590239, -0.64112425, -0.2800614 ],
    [ 0.,          0.,          0.,          0.        ],
    [ 0.,          0.,          0.,          1.        ]
])

X_OFFSET = -3.0
Y_OFFSET = 0.0

HOME_JOINTS  = [1.6120, 1.6171, 1.7846, 2.8815, 1.6120, 4.7124]
GRASP_Z      = 115.0
SAFE_Z       = 340.0
GRASP_ROT    = [2.5273, 1.5708, 0.9565]
ROBOT_IP     = '192.168.1.100'
GRIPPER_PORT = '/dev/ttyUSB0'
TABLE_DEPTH_MM = 692.0

# =========================================================

def cam2robot_xy(u, v):
    z = TABLE_DEPTH_MM * 0.001
    x = (u - CX) * z / FX
    y = (v - CY) * z / FY
    cam_pt   = np.array([x, y, z, 1.0])
    robot_pt = CAMERA_POSE @ cam_pt
    tx = robot_pt[0] * 1000.0 + X_OFFSET
    ty = robot_pt[1] * 1000.0 + Y_OFFSET
    return tx, ty


# ── 全局变量 ──────────────────────────────────────────────
click_point = None
color_img   = None
depth_img   = None
robot       = None
gripper     = None
busy        = False


def mouse_callback(event, x, y, flags, param):
    global click_point, busy
    if event == cv2.EVENT_LBUTTONDOWN and not busy:
        click_point = (x, y)
        print(f'\n[点击] 像素坐标: ({x}, {y})')


def gripper_interactive():
    print("\n" + "="*45)
    print("夹爪控制模式")
    print("  位置: 0=完全关闭  100=完全打开")
    print("  速度: 1-100%")
    print("  力值: 20-100%")
    print("  输入 ok → 确认夹住，开始回位")
    print("  输入 q  → 取消放弃")
    print("="*45)

    while True:
        try:
            cmd = input("\n位置%(0-100) / ok / q: ").strip().lower()
            if cmd == 'ok':
                return True
            if cmd == 'q':
                return False

            p = float(cmd)
            if not 0 <= p <= 100:
                print("⚠ 位置必须在 0-100 之间")
                continue

            sp_str = input("速度%(1-100, 回车默认30): ").strip()
            sp = float(sp_str) if sp_str else 30.0
            if not 1 <= sp <= 100:
                print("⚠ 速度必须在 1-100 之间")
                continue

            f_str = input("力值%(20-100, 回车默认100): ").strip()
            f = float(f_str) if f_str else 100.0
            if not 20 <= f <= 100:
                print("⚠ 力值必须在 20-100 之间")
                continue

            pos   = int(p * 10)
            speed = int(sp)
            force = int(f)
            print(f"→ 夹爪移动到 {p}% (pos={pos}) 速度={speed}% 力值={force}%")
            gripper.move_to(pos, force=force, speed=speed)

            grip = gripper.get_grip_status()
            cur  = gripper.get_position()
            labels = {0:'运动中', 1:'到位', 2:'✓夹住物体', 3:'⚠物体掉落'}
            print(f"  夹持状态: {labels.get(grip, grip)}  当前位置: {cur}/1000")

        except ValueError:
            print("⚠ 请输入有效数字、ok 或 q")
        except KeyboardInterrupt:
            return False


def do_grasp_thread(u, v):
    global busy

    busy = True
    try:
        tx, ty = cam2robot_xy(u, v)
        print(f'\n[目标] 机器人坐标: x={tx:.1f} mm, y={ty:.1f} mm')

        # ── Step1: 打开夹爪 ──────────────────────────────
        print('[夹爪] 打开...')
        gripper.open()

        # ── Step2: 移到目标上方安全高度 ──────────────────
        print('[机械臂] 移到目标上方...')
        robot.move_j_p([
            tx / 1000.0, ty / 1000.0, SAFE_Z / 1000.0,
            GRASP_ROT[0], GRASP_ROT[1], GRASP_ROT[2]
        ])
        time.sleep(0.5)

        # ── Step3: 确认下降 ──────────────────────────────
        confirm = input('\n[确认] 按 Enter 下降 / 输入 q 取消: ').strip().lower()
        if confirm == 'q':
            print('[取消] 已取消')
            robot.move_j(HOME_JOINTS)
            return

        # ── Step4: 下降到抓取高度 ────────────────────────
        print('[机械臂] 下降到抓取高度...')
        robot.move_j_p([
            tx / 1000.0, ty / 1000.0, GRASP_Z / 1000.0,
            GRASP_ROT[0], GRASP_ROT[1], GRASP_ROT[2]
        ])
        time.sleep(0.5)
        print('[机械臂] 已到达抓取位置')

        # ── Step5: 夹爪交互控制 ──────────────────────────
        ok = gripper_interactive()
        if not ok:
            print('[取消] 放开夹爪，取消抓取...')
            gripper.open()
            print('[机械臂] 上升回安全高度...')
            robot.move_j_p([
                tx / 1000.0, ty / 1000.0, SAFE_Z / 1000.0,
                GRASP_ROT[0], GRASP_ROT[1], GRASP_ROT[2]
            ])
            robot.move_j(HOME_JOINTS)
            return

        # ── Step6: 上升 ──────────────────────────────────
        print('\n[机械臂] 上升到安全高度...')
        robot.move_j_p([
            tx / 1000.0, ty / 1000.0, SAFE_Z / 1000.0,
            GRASP_ROT[0], GRASP_ROT[1], GRASP_ROT[2]
        ])
        time.sleep(0.5)

        # ── Step7: 回初始位 ──────────────────────────────
        go_home = input('[确认] 按 Enter 回初始位 / 输入 n 保持当前: ').strip().lower()
        if go_home != 'n':
            print('[机械臂] 回初始位...')
            robot.move_j(HOME_JOINTS)

        print('\n[完成] 抓取完成，可点击下一个目标\n')

    except Exception as e:
        print(f'[错误] {e}')
    finally:
        busy = False


def main():
    global color_img, depth_img, robot, gripper, click_point

    print('初始化机械臂...')
    rclpy.init()
    robot = Jaka_Robot(tcp_host_ip=ROBOT_IP)
    robot.joint_acc = 1.0
    robot.joint_vel = 0.8

    print('初始化夹爪...')
    gripper = Gripper(GRIPPER_PORT)
    gripper.initialize()

    print('机械臂移到初始位...')
    robot.move_j(HOME_JOINTS)

    cv2.namedWindow('color')
    cv2.namedWindow('depth')
    cv2.setMouseCallback('color', mouse_callback)

    print('\n' + '='*45)
    print('操作说明：')
    print('  1. 左键点击彩色图中的目标物体')
    print('  2. 机械臂自动移到目标上方')
    print('  3. 按 Enter → 机械臂下降')
    print('  4. 终端控制夹爪开合，确认夹住')
    print('  5. 输入 ok → 上升回位')
    print('  按 q 退出程序')
    print('='*45 + '\n')

    while True:
        color_img, depth_img = robot.get_camera_data()
        if color_img is None:
            continue

        display = color_img.copy()

        if click_point is not None and not busy:
            u, v = click_point
            click_point = None
            cv2.circle(display, (u, v), 8, (0, 0, 255), 2)
            cv2.putText(display, f'({u},{v})', (u+10, v-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            t = threading.Thread(target=do_grasp_thread, args=(u, v), daemon=True)
            t.start()

        if busy:
            cv2.putText(display, 'GRASPING...', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        depth_vis = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_img, alpha=0.05),
            cv2.COLORMAP_JET
        )

        cv2.imshow('color', display)
        cv2.imshow('depth', depth_vis)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    gripper.disconnect()
    rclpy.shutdown()
    print('退出。')


if __name__ == '__main__':
    main()