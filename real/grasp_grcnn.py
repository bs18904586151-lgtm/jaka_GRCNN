##!/usr/bin/env python3
# coding=utf8
"""
grasp_grcnn.py  —  点击物体位置 + GRCNN预测抓取角度

流程：
  1. 左键点击物体 → 确定位置(x,y)
  2. 以点击位置为中心裁剪224×224 → GRCNN预测抓取角度
  3. 机械臂自动移动到目标上方
  4. 下降 → 夹爪交互 → 上升回位

按键：
  左键   点击物体，触发推理+抓取
  q      退出
"""

import cv2
import numpy as np
import time
import threading
import rclpy
import sys
import os
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.join(current_dir, "..")
sys.path.append(parent_dir)

from Jaka_Robot import Jaka_Robot
from gripper import Gripper

GRCNN_DIR = os.path.join(os.path.expanduser('~'), 'GRCNN')
sys.path.insert(0, GRCNN_DIR)

from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.dataset_processing.grasp import detect_grasps
from utils.dataset_processing import image as ds_image

# =========================================================
# 参数配置
# =========================================================
FX, FY = 607.879, 607.348
CX, CY = 325.14,  244.014

CAMERA_POSE = np.array([
    [-0.00621701,  0.67079471, -0.09046704,  0.10327271],
    [ 0.67005468,  0.00554793, -0.54316818, -0.34623102],
    [ 0.,          0.,          0.,          0.        ],
    [ 0.,          0.,          0.,          1.        ]
])
X_OFFSET = -5.0
Y_OFFSET = -3.0

MODEL_PATH  = os.path.join(os.path.expanduser('~'), 'GRCNN', 'trained-models',
              'cornell-randsplit-rgbd-grconvnet3-drop1-ch32', 'epoch_19_iou_0.98')
MODEL_INPUT = 224     # 模型输入尺寸
CROP_HALF   = 112     # 以点击点为中心裁剪224×224

HOME_JOINTS  = [1.6120, 1.6171, 1.7846, 2.8815, 1.6120, 4.7124]
GRASP_Z      = 115.0
SAFE_Z       = 340.0
GRASP_RX     = 2.5273
GRASP_RY     = 1.5708
ROBOT_IP     = '192.168.1.100'
GRIPPER_PORT = '/dev/ttyUSB0'
TABLE_DEPTH_MM = 692.0

# ── 全局变量 ──────────────────────────────────────────────
color_img = None
depth_img = None
robot     = None
gripper   = None
model     = None
cam_data  = None
device    = None
busy      = False
click_pt  = None


# =========================================================
def cam2robot_xy(u, v, depth_img):
    iy, ix = int(round(v)), int(round(u))
    patch  = depth_img[max(0,iy-3):iy+4, max(0,ix-3):ix+4].flatten()
    valid  = patch[patch > 0]
    z_mm   = float(np.median(valid)) if len(valid) > 0 else TABLE_DEPTH_MM
    if len(valid) == 0:
        print(f"[警告] 像素({u:.0f},{v:.0f}) 深度为零，使用固定值")
    z = z_mm * 0.001
    x = (u - CX) * z / FX
    y = (v - CY) * z / FY
    cam_pt   = np.array([x, y, z, 1.0])
    robot_pt = CAMERA_POSE @ cam_pt
    return robot_pt[0] * 1000.0 + X_OFFSET, robot_pt[1] * 1000.0 + Y_OFFSET


def predict_angle(u, v):
    """以像素(u,v)为中心裁剪224×224，预测抓取角度（rad）"""
    H, W = color_img.shape[:2]
    x1 = max(0, int(u) - CROP_HALF)
    y1 = max(0, int(v) - CROP_HALF)
    x2 = min(W, x1 + MODEL_INPUT)
    y2 = min(H, y1 + MODEL_INPUT)
    if x2 - x1 < MODEL_INPUT: x1 = max(0, x2 - MODEL_INPUT)
    if y2 - y1 < MODEL_INPUT: y1 = max(0, y2 - MODEL_INPUT)

    crop_rgb   = cv2.resize(color_img[y1:y2, x1:x2], (MODEL_INPUT, MODEL_INPUT))
    crop_depth = cv2.resize(depth_img[y1:y2, x1:x2], (MODEL_INPUT, MODEL_INPUT),
                            interpolation=cv2.INTER_NEAREST)

    x, _, _ = cam_data.get_data(rgb=crop_rgb, depth=crop_depth)
    with torch.no_grad():
        pred = model.predict(x.to(device).float())

    q_img, ang_img, width_img = post_process_output(
        pred['pos'], pred['cos'], pred['sin'], pred['width'])
    grasps = detect_grasps(q_img, ang_img, width_img)

    if not grasps:
        print("[GRCNN] 未检测到角度，使用默认0°")
        return 0.0

    angle = grasps[0].angle
    conf  = q_img[grasps[0].center[0], grasps[0].center[1]]
    print(f"[GRCNN] 预测角度={np.degrees(angle):.1f}°  置信度={conf:.3f}")
    return angle


def angle_cam2robot(a):
    return a + np.pi / 2.0


def mouse_callback(event, x, y, flags, param):
    global click_pt, busy
    if event == cv2.EVENT_LBUTTONDOWN and not busy:
        click_pt = (x, y)
        print(f'\n[点击] 像素坐标: ({x}, {y})')


def gripper_interactive():
    print("\n" + "="*45)
    print("夹爪控制  ok=确认夹住  q=取消")
    print("="*45)
    while True:
        try:
            cmd = input("\n位置%(0-100) / ok / q: ").strip().lower()
            if cmd == 'ok': return True
            if cmd == 'q':  return False
            p = float(cmd)
            if not 0 <= p <= 100: print("⚠ 0-100"); continue
            sp = float(input("速度%(默认30): ").strip() or 30)
            f  = float(input("力值%(默认100): ").strip() or 100)
            gripper.move_to(int(p*10), force=int(f), speed=int(sp))
            labels = {0:'运动中',1:'到位',2:'✓夹住',3:'⚠掉落'}
            print(f"  {labels.get(gripper.get_grip_status(),'?')}  位置:{gripper.get_position()}/1000")
        except (ValueError, KeyboardInterrupt):
            return False


def do_grasp_thread(u, v):
    global busy
    busy = True
    try:
        angle = predict_angle(u, v)
        rz    = angle_cam2robot(angle)
        tx, ty = cam2robot_xy(u, v, depth_img)
        print(f'[目标] X={tx:.1f}mm  Y={ty:.1f}mm  Rz={np.degrees(rz):.1f}°')

        print('[夹爪] 打开...')
        gripper.open();  time.sleep(0.5)

        print('[机械臂] 移动到目标上方...')
        robot.move_j_p([tx/1000, ty/1000, SAFE_Z/1000, GRASP_RX, GRASP_RY, rz])

        if input('\n[确认] 按 Enter 下降 / q 取消: ').strip().lower() == 'q':
            robot.move_j(HOME_JOINTS);  return

        print('[机械臂] 下降...')
        robot.move_j_p([tx/1000, ty/1000, GRASP_Z/1000, GRASP_RX, GRASP_RY, rz])
        time.sleep(0.5)

        if not gripper_interactive():
            gripper.open()
            robot.move_j_p([tx/1000, ty/1000, SAFE_Z/1000, GRASP_RX, GRASP_RY, rz])
            robot.move_j(HOME_JOINTS);  return

        print('[机械臂] 上升...')
        robot.move_j_p([tx/1000, ty/1000, SAFE_Z/1000, GRASP_RX, GRASP_RY, rz])
        time.sleep(0.5)

        if input('[确认] 按 Enter 回初始位 / n 保持: ').strip().lower() != 'n':
            robot.move_j(HOME_JOINTS)
        print('[完成]\n')

    except Exception as e:
        print(f'[错误] {e}')
    finally:
        busy = False


def main():
    global color_img, depth_img, robot, gripper, model, cam_data, device, click_pt

    print('初始化机械臂...')
    rclpy.init()
    robot = Jaka_Robot(tcp_host_ip=ROBOT_IP)
    robot.joint_acc = 1.0;  robot.joint_vel = 0.8

    print('初始化夹爪...')
    gripper = Gripper(GRIPPER_PORT)
    gripper.initialize()

    print('加载GRCNN模型...')
    model = torch.load(MODEL_PATH, weights_only=False, map_location=torch.device('cpu'))
    model.eval()
    from hardware.device import get_device
    device = get_device(force_cpu=True)
    model  = model.to(device)

    cam_data = CameraData(width=MODEL_INPUT, height=MODEL_INPUT,
                          output_size=MODEL_INPUT, include_depth=True, include_rgb=True)
    def patched_get_depth(img):
        di = ds_image.DepthImage(img)
        di.crop(bottom_right=cam_data.bottom_right, top_left=cam_data.top_left)
        di.normalise()
        return np.expand_dims(di.img, 0)
    cam_data.get_depth = patched_get_depth
    print('模型加载完成')

    print('机械臂移到初始位...')
    robot.move_j(HOME_JOINTS)

    cv2.namedWindow('color')
    cv2.setMouseCallback('color', mouse_callback)

    print('\n左键点击物体 → 预测角度+执行抓取   q=退出\n')

    last_click = None

    while True:
        color_img, depth_img = robot.get_camera_data()
        if color_img is None: continue

        display = color_img.copy()
        if last_click:
            u, v = last_click
            cv2.circle(display, (int(u), int(v)), 10, (0,255,0), 2)
            # 显示裁剪框
            x1 = max(0, int(u)-CROP_HALF);  y1 = max(0, int(v)-CROP_HALF)
            x2 = min(display.shape[1], x1+MODEL_INPUT)
            y2 = min(display.shape[0], y1+MODEL_INPUT)
            cv2.rectangle(display, (x1,y1), (x2,y2), (255,200,0), 1)

        state = 'GRASPING...' if busy else 'Left-click: grasp   Q: quit'
        col   = (0,0,255) if busy else (180,180,180)
        cv2.putText(display, state, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        cv2.imshow('color', display)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), ord('Q')): break

        if click_pt is not None and not busy:
            u, v = click_pt;  click_pt = None;  last_click = (u, v)
            threading.Thread(target=do_grasp_thread, args=(u,v), daemon=True).start()

    cv2.destroyAllWindows()
    gripper.disconnect()
    rclpy.shutdown()
    print('退出。')


if __name__ == '__main__':
    main()