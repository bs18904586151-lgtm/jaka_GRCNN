#!/usr/bin/env python
# coding=utf8
"""
calibrate.py  —  eye-to-hand 平面标定
相机固定外部俯视桌面，棋盘格装在 Jaka A5 末端法兰随臂运动。

棋盘格参数（已确认）：
  内角点：5×5
  格间距：25 mm
  棋盘中心到法兰圆心：225 mm，沿法兰 Y 轴负方向
  棋盘格面到法兰：     40  mm，沿法兰 Z 轴负方向

保留原 calibrate.py 所有变量名。
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import os
import sys
import rclpy
from scipy import optimize
from mpl_toolkits.mplot3d import Axes3D
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.join(current_dir, "..")
sys.path.append(parent_dir)

from Jaka_Robot import Jaka_Robot





# =========================================================
# User options（根据实际情况修改这里）
# =========================================================
tcp_host_ip  = '192.168.1.100'
tcp_port     = 10000

# 工作空间（机器人基坐标系，单位 m）
# 根据你已知的当前位姿 x=0.106 y=-0.261 z=0.266 来设置中心，向外扩展
workspace_limits = np.asarray([
    [0.056, 0.146],   # x: 缩小正方向，最大到 146mm
    [-0.321, -0.201], # y: 扩展到4列
    [0.265, 0.265],
])
calib_grid_step = 0.03   # 步长3cm  # 标定网格步长 5 cm，产生 4×4=16 个点

# 棋盘中心相对法兰的偏移 [x, y, z]（单位 m）
# Y 轴负方向 225 mm，Z 轴负方向 40 mm
checkerboard_offset_from_tool = [0, -0.225, -0.04]

# 末端姿态固定（RPY，单位 rad）
# 当前实测姿态：rx=180° ry=0° rz=-90° → 法兰朝下，棋盘水平朝相机
tool_orientation = [-3.1416, 0.0, -1.5708]  # 旋转向量（与 get_fk 输出一致，已验证）

# 棋盘格参数
checkerboard_size   = (5, 5)    # 内角点数
checkerboard_square = 0.025     # 格间距 0.025 m
# =========================================================


# ---------------------------------------------------------
# 构造标定网格（XY 平面，Z 固定一层）
# ---------------------------------------------------------
gridspace_x = np.linspace(
    workspace_limits[0][0], workspace_limits[0][1],
    int(1 + round((workspace_limits[0][1] - workspace_limits[0][0]) / calib_grid_step))
)
gridspace_y = np.linspace(
    workspace_limits[1][0], workspace_limits[1][1],
    int(1 + round((workspace_limits[1][1] - workspace_limits[1][0]) / calib_grid_step))
)
gridspace_z = np.array([workspace_limits[2][0]])   # 平面标定只有一层

calib_grid_x, calib_grid_y, calib_grid_z = np.meshgrid(
    gridspace_x, gridspace_y, gridspace_z
)
num_calib_grid_pts = calib_grid_x.size

calib_grid_x.shape = (num_calib_grid_pts, 1)
calib_grid_y.shape = (num_calib_grid_pts, 1)
calib_grid_z.shape = (num_calib_grid_pts, 1)
calib_grid_pts = np.concatenate(
    (calib_grid_x, calib_grid_y, calib_grid_z), axis=1
)
print(f'标定网格点数：{num_calib_grid_pts}  '
      f'({len(gridspace_x)} × {len(gridspace_y)})')

# 数据容器（变量名与原 calibrate.py 完全一致）
measured_pts = []
observed_pts = []
observed_pix = []
world2camera = np.eye(4)


# ---------------------------------------------------------
# 初始化 ROS2 & 机器人
# ---------------------------------------------------------
rclpy.init()
print('连接机器人...')
robot = Jaka_Robot(
    tcp_host_ip=tcp_host_ip,
    tcp_port=tcp_port,
    workspace_limits=workspace_limits,
    is_use_robotiq85=False
)

robot.joint_acc = 1.0
robot.joint_vel = 0.8

# 初始位关节角（rad）—— 棋盘格位于相机视野中心的位置
home_joint_config = [
    2.3784762170003013,
    1.677231626492662,
    1.9535248117331943,
    4.223225061774143,
    1.5707963,
    2.378476217029981,
]
print('运动到初始位...')
robot.move_j(home_joint_config)

refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
os.makedirs('calib_imgs', exist_ok=True)


# ---------------------------------------------------------
# 遍历标定点
# ---------------------------------------------------------
print('开始采集数据...')
for calib_pt_idx in range(num_calib_grid_pts):

    tool_position = calib_grid_pts[calib_pt_idx, :]   # [x, y, z] 单位 m
    tool_config = [
        tool_position[0], tool_position[1], tool_position[2],
        tool_orientation[0], tool_orientation[1], tool_orientation[2]
    ]
    tool_config1 = list(tool_config)   # 保留原代码同名变量
    print(f'[{calib_pt_idx+1}/{num_calib_grid_pts}] 移动到: {tool_config1}')

    robot.move_j_p(tool_config)

    # ---- 等待机械臂稳定，读取实际位姿 ----
    time.sleep(2)   # 等待震动消散，相机帧稳定

    # 读取机械臂当前实际位姿（而不是规划目标值）
    actual_pose = robot.get_tool_pose()   # [x,y,z,rx,ry,rz] 单位 m+rad
    actual_position = np.array(actual_pose[:3])
    print(f'  实际位姿(m): x={actual_position[0]:.4f}  y={actual_position[1]:.4f}  z={actual_position[2]:.4f}')

    # ---- 获取图像 ----
    camera_color_img, camera_depth_img = robot.get_camera_data()
    if camera_color_img is None:
        print('  相机获取失败，跳过此点。')
        continue

    # realsenseD435 返回 BGR，转灰度检测角点
    bgr_color_data = camera_color_img   # 已是 BGR
    gray_data = cv2.cvtColor(bgr_color_data, cv2.COLOR_BGR2GRAY)

    checkerboard_found, corners = cv2.findChessboardCorners(
        gray_data, checkerboard_size, None,
        cv2.CALIB_CB_ADAPTIVE_THRESH
    )
    print(f'  棋盘格检测: {checkerboard_found}')

    if not checkerboard_found:
        print('  未检测到棋盘格，跳过此点。')
        continue

    corners_refined = cv2.cornerSubPix(
        gray_data, corners, (5, 5), (-1, -1), refine_criteria
    )

    # 中心角点（5×5=25 个角点，中心索引=12，与原代码一致）
    checkerboard_pix = np.round(corners_refined[12, 0, :]).astype(int)

    # 深度值（uint16，单位 mm）
    checkerboard_z = camera_depth_img[checkerboard_pix[1]][checkerboard_pix[0]]
    if checkerboard_z == 0:
        print('  深度为 0，跳过。')
        continue

    # 像素 → 相机坐标（单位 mm，与原代码公式完全一致）
    checkerboard_x = np.multiply(
        checkerboard_pix[0] - robot.cam_intrinsics[0][2],
        checkerboard_z / robot.cam_intrinsics[0][0]
    )
    checkerboard_y = np.multiply(
        checkerboard_pix[1] - robot.cam_intrinsics[1][2],
        checkerboard_z / robot.cam_intrinsics[1][1]
    )

    # 保存观测点（相机坐标系，mm）
    observed_pts.append([checkerboard_x, checkerboard_y, checkerboard_z])

    # 保存测量点（机器人基坐标系，m）：实际位姿 + 棋盘中心偏移
    measured_position = actual_position + np.array(checkerboard_offset_from_tool)
    measured_pts.append(measured_position)
    observed_pix.append(checkerboard_pix)

    print(f'  已采集 {len(measured_pts)} 个有效点，棋盘中心像素: {checkerboard_pix}，深度: {checkerboard_z} mm')

    # 可视化保存（与原代码一致）
    vis = cv2.drawChessboardCorners(
        bgr_color_data.copy(), (1, 1),
        corners_refined[12, :, :], checkerboard_found
    )
    img_path = '%06d.png' % len(measured_pts)
    cv2.imwrite(img_path, vis)
    cv2.imshow('Calibration', vis)
    cv2.waitKey(1000)



measured_pts = np.asarray(measured_pts)
observed_pts = np.asarray(observed_pts)
observed_pix = np.asarray(observed_pix)

print(f'\n有效标定点数：{len(measured_pts)}')
if len(measured_pts) < 6:
    print('有效点数不足（< 6），请检查棋盘格可见性或工作空间设置。')
    rclpy.shutdown()
    exit(1)


# =========================================================
# 标定计算（与原 calibrate.py 完全相同，一字未改）
# =========================================================

def get_rigid_transform(A, B):
    assert len(A) == len(B)
    N = A.shape[0]
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))
    H  = np.dot(np.transpose(AA), BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = np.dot(-R, centroid_A.T) + centroid_B.T
    return R, t


def get_rigid_transform_error(z_scale):
    global measured_pts, observed_pts, observed_pix, world2camera

    observed_z = observed_pts[:, 2:] * z_scale
    observed_x = np.multiply(
        observed_pix[:, [0]] - robot.cam_intrinsics[0][2],
        observed_z / robot.cam_intrinsics[0][0]
    )
    observed_y = np.multiply(
        observed_pix[:, [1]] - robot.cam_intrinsics[1][2],
        observed_z / robot.cam_intrinsics[1][1]
    )
    new_observed_pts = np.concatenate((observed_x, observed_y, observed_z), axis=1)

    R, t = get_rigid_transform(
        np.asarray(measured_pts), np.asarray(new_observed_pts)
    )
    t.shape = (3, 1)
    world2camera = np.concatenate(
        (np.concatenate((R, t), axis=1), np.array([[0, 0, 0, 1]])),
        axis=0
    )

    registered_pts = np.dot(R, np.transpose(measured_pts)) + \
                     np.tile(t, (1, measured_pts.shape[0]))
    error = np.transpose(registered_pts) - new_observed_pts
    error = np.sum(np.multiply(error, error))
    rmse  = np.sqrt(error / measured_pts.shape[0])
    return rmse


# 优化 z_scale（与原代码完全一致）
print('Calibrating...')
z_scale_init = 1
optim_result = optimize.minimize(
    get_rigid_transform_error,
    np.asarray(z_scale_init),
    method='Nelder-Mead'
)
camera_depth_offset = optim_result.x

# 保存（文件名与原代码完全一致）
print('Saving...')
_script_dir = os.path.dirname(os.path.abspath(__file__))
np.savetxt(os.path.join(_script_dir, 'camera_depth_scale.txt'), camera_depth_offset, delimiter=' ')
get_rigid_transform_error(camera_depth_offset)
camera_pose = np.linalg.inv(world2camera)
np.savetxt(os.path.join(_script_dir, 'camera_pose.txt'), camera_pose, delimiter=' ')
print('Done.')

print(f'\nRMSE = {get_rigid_transform_error(camera_depth_offset):.4f} mm')
print('camera_pose (相机→机器人基坐标系):\n', camera_pose)

rclpy.shutdown()


# =========================================================
# DEBUG（与原代码一致）
# =========================================================
# np.savetxt('measured_pts.txt', np.asarray(measured_pts), delimiter=' ')
# np.savetxt('observed_pts.txt', np.asarray(observed_pts), delimiter=' ')
# np.savetxt('observed_pix.txt', np.asarray(observed_pix), delimiter=' ')
# measured_pts = np.loadtxt('measured_pts.txt', delimiter=' ')
# observed_pts = np.loadtxt('observed_pts.txt', delimiter=' ')
# observed_pix = np.loadtxt('observed_pix.txt', delimiter=' ')
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(measured_pts[:,0],measured_pts[:,1],measured_pts[:,2], c='blue')
# plt.show()