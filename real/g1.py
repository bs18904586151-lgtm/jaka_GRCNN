#!/usr/bin/env python3
# coding=utf8
"""
g1.py  —  点击图像目标 → 机械臂自动移动到计算坐标 → 观察标定误差
使用：python3 real/g1.py

流程：
  1. 左键点击彩色图中目标物体
  2. 机械臂自动移动到计算出的 xy 坐标（安全高度）
  3. 观察实际位置与目标的偏差
  4. 按 Enter 回初始位，继续下一次测试
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
    [-0.02162035, -0.99975419, -0.00491181,  0.10121101],
    [-0.99893998,  0.02140252,  0.04075350, -0.63737169],
    [-0.04063835,  0.00578771, -0.99915716, -0.11962301],
    [ 0.,          0.,          0.,           1.        ]
])

HOME_JOINTS    = [1.6120, 1.6171, 1.7846, 2.8815, 1.6120, 4.7124]
SAFE_Z         = 340.0
GRASP_ROT      = [2.5273, 1.5708, 0.9565]
ROBOT_IP       = '192.168.1.100'
GRIPPER_PORT   = '/dev/ttyUSB0'
TABLE_DEPTH_MM = 692.0
X_OFFSET       = -138.5
Y_OFFSET       =  505.3

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


def do_move_thread(u, v):
    global busy

    busy = True
    try:
        tx, ty = cam2robot_xy(u, v)
        print(f'[计算] 目标坐标: x={tx:.1f} mm, y={ty:.1f} mm')
        print('[机械臂] 移动到目标上方（安全高度）...')

        robot.move_j_p([
            tx / 1000.0, ty / 1000.0, SAFE_Z / 1000.0,
            GRASP_ROT[0], GRASP_ROT[1], GRASP_ROT[2]
        ])
        time.sleep(0.5)

        # 读取实际到达位置
        actual = robot.get_tool_pose()
        ax = actual[0] * 1000.0
        ay = actual[1] * 1000.0

        print(f'\n[结果] 计算目标: x={tx:.1f} mm, y={ty:.1f} mm')
        print(f'[结果] 实际到达: x={ax:.1f} mm, y={ay:.1f} mm')
        print(f'[误差] dx={ax-tx:.1f} mm, dy={ay-ty:.1f} mm')
        print(f'[误差] 总偏差={((ax-tx)**2+(ay-ty)**2)**0.5:.1f} mm')
        print('\n[提示] 观察机械臂末端与目标物体的实际偏差')

        input('\n按 Enter 回初始位...')
        print('[机械臂] 回初始位...')
        robot.move_j(HOME_JOINTS)
        print('[完成] 可点击下一个目标继续测试\n')

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
    print('手眼标定误差测试')
    print('  左键点击目标 → 机械臂自动移到计算坐标')
    print('  终端显示计算坐标、实际坐标、误差')
    print('  按 Enter 回初始位，继续下一次')
    print('  按 q 退出')
    print('='*45 + '\n')

    while True:
        color_img, depth_img = robot.get_camera_data()
        if color_img is None:
            continue

        display = color_img.copy()

        if click_point is not None and not busy:
            u, v = click_point
            click_point = None
            cv2.circle(display, (u, v), 8, (0, 255, 0), 2)
            cv2.putText(display, f'({u},{v})', (u+10, v-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            t = threading.Thread(target=do_move_thread, args=(u, v), daemon=True)
            t.start()

        if busy:
            cv2.putText(display, 'MOVING...', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

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