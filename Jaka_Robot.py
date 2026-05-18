##!/usr/bin/env python
# coding=utf8
"""
Jaka_Robot.py  —  替换 UR_Robot.py
适配 Jaka A5 + jaka_ros2 驱动 + RealSense D435 (eye-to-hand)
"""

import time
import math
import numpy as np

import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TwistStamped
from jaka_msgs.srv import Move, GetIK

from real.realsenseD435 import RealsenseD435


class Jaka_Robot(Node):

    def __init__(self,
                 tcp_host_ip='192.168.1.100',
                 tcp_port=10000,
                 workspace_limits=None,
                 is_use_robotiq85=False,
                 is_use_camera=True):

        super().__init__('jaka_robot_node')

        self.tcp_host_ip = tcp_host_ip
        self.tcp_port    = tcp_port
        self.workspace_limits = workspace_limits if workspace_limits is not None \
            else [[-0.7, 0.7], [-0.7, 0.7], [0.0, 0.6]]

        self.joint_acc = 1.4
        self.joint_vel = 1.05
        self.tool_acc  = 0.5
        self.tool_vel  = 0.2
        self.joint_tolerance     = 0.01
        self.tool_pose_tolerance = [0.002, 0.002, 0.002, 0.01, 0.01, 0.01]

        # 服务客户端
        self._cli_joint_move  = self.create_client(Move,   '/jaka_driver/joint_move')
        self._cli_linear_move = self.create_client(Move,   '/jaka_driver/linear_move')
        self._cli_stop        = self.create_client(Empty,  '/jaka_driver/stop_move')
        self._cli_get_ik      = self.create_client(GetIK,  '/jaka_driver/get_ik')

        for cli in [self._cli_joint_move, self._cli_get_ik]:
            while not cli.wait_for_service(timeout_sec=3.0):
                self.get_logger().warn(f'等待服务 {cli.srv_name} ...')

        # 订阅当前位姿
        self._current_tool_pos = None
        self._current_joints   = None
        self.create_subscription(TwistStamped, '/jaka_driver/tool_position', self._cb_tool_pos, 10)
        self.create_subscription(JointState,   '/jaka_driver/joint_position', self._cb_joints, 10)

        self.get_logger().info('等待机器人状态话题...')
        deadline = time.time() + 5.0
        while (self._current_joints is None) and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self._current_joints is None:
            self.get_logger().warn('未收到关节状态，请确认 jaka_ros2 驱动已启动。')

        # 相机
        if is_use_camera:
            self.camera = RealsenseD435()

        self.cam_intrinsics = np.array([
            607.879, 0, 325.14, 0, 607.348, 244.014, 0, 0, 1
        ]).reshape(3, 3)

        # 加载标定文件
        import os as _os
        _script_dir = _os.path.dirname(_os.path.abspath(__file__))
        _pose_file  = _os.path.join(_script_dir, 'camera_pose.txt')
        _scale_file = _os.path.join(_script_dir, 'camera_depth_scale.txt')
        try:
            self.cam_pose        = np.loadtxt(_pose_file,  delimiter=' ')
            self.cam_depth_scale = np.loadtxt(_scale_file, delimiter=' ')
            self.get_logger().info('已加载标定文件。')
        except (FileNotFoundError, OSError):
            self.get_logger().info('未找到标定文件，首次标定时正常。')
            self.cam_pose        = np.eye(4)
            self.cam_depth_scale = np.array([1.0])

        self.get_logger().info('Jaka_Robot 初始化完成。')

    def _cb_tool_pos(self, msg):
        self._current_tool_pos = msg

    def _cb_joints(self, msg):
        self._current_joints = list(msg.position)

    def _spin_once(self):
        rclpy.spin_once(self, timeout_sec=0.05)

    def get_tool_pose(self):
        for _ in range(20):
            self._spin_once()
            if self._current_tool_pos is not None:
                break
        t = self._current_tool_pos.twist
        return [
            t.linear.x / 1000.0, t.linear.y / 1000.0, t.linear.z / 1000.0,
            math.radians(t.angular.x), math.radians(t.angular.y), math.radians(t.angular.z),
        ]

    def get_joint_positions(self):
        for _ in range(20):
            self._spin_once()
            if self._current_joints is not None:
                break
        return list(self._current_joints)

    def move_j(self, joint_configuration, k_acc=1, k_vel=1, t=0, r=0):
        req = Move.Request()
        req.pose       = list(joint_configuration)
        req.mvvelo     = float(k_vel * self.joint_vel)
        req.mvacc      = float(k_acc * self.joint_acc)
        req.mvtime     = float(t)
        req.mvradii    = float(r)
        req.coord_mode = 0
        req.index      = 0
        req.has_ref    = False
        req.ref_joint  = []

        future = self._cli_joint_move.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
        if future.result() is not None:
            self.get_logger().info(f'move_j ret={future.result().ret}: {future.result().message}')
        self._wait_joint_stop(joint_configuration)

    def move_j_p(self, tool_configuration, k_acc=1, k_vel=1, t=0, r=0, ref_joints=None):
        """
        笛卡尔目标运动：get_ik 求关节角 → joint_move 执行
        入参：[x,y,z,rx,ry,rz]，单位 m + rad（旋转向量）
        ref_joints：IK参考关节角，不传则用当前关节角
        """
        x, y, z   = tool_configuration[0:3]
        rx, ry, rz = tool_configuration[3:6]

        # get_ik
        if ref_joints is not None:
            ik_ref = ref_joints
        else:
            ik_ref = self.get_joint_positions()

        ik_req = GetIK.Request()
        ik_req.ref_joint      = [float(j) for j in ik_ref]
        ik_req.cartesian_pose = [
            float(x  * 1000.0), float(y  * 1000.0), float(z  * 1000.0),
            float(rx), float(ry), float(rz),
        ]

        self.get_logger().info(
            f'move_j_p get_ik 目标(mm/rad): {[round(v,4) for v in ik_req.cartesian_pose]}'
        )
        ik_future = self._cli_get_ik.call_async(ik_req)
        rclpy.spin_until_future_complete(self, ik_future, timeout_sec=10.0)

        if ik_future.result() is None or 'error' in ik_future.result().message.lower():
            self.get_logger().error(f'get_ik 失败: {ik_future.result()}')
            return

        target_joints = list(ik_future.result().joint)
        self.get_logger().info(f'move_j_p IK解: {[round(j,4) for j in target_joints]}')

        jm_req = Move.Request()
        jm_req.pose       = [float(j) for j in target_joints]
        jm_req.mvvelo     = float(k_vel * self.joint_vel)
        jm_req.mvacc      = float(k_acc * self.joint_acc)
        jm_req.mvtime     = float(t)
        jm_req.mvradii    = float(r)
        jm_req.coord_mode = 0
        jm_req.index      = 0
        jm_req.has_ref    = False
        jm_req.ref_joint  = []

        future = self._cli_joint_move.call_async(jm_req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
        if future.result() is not None:
            self.get_logger().info(
                f'move_j_p joint_move ret={future.result().ret}: {future.result().message}'
            )
        self._wait_joint_stop(target_joints)
        time.sleep(0.5)

    def move_l(self, tool_configuration, k_acc=1, k_vel=1, t=0, r=0):
        x, y, z   = tool_configuration[0:3]
        rx, ry, rz = tool_configuration[3:6]
        rv = self._rpy_to_rotvec(rx, ry, rz)

        req = Move.Request()
        req.pose = [
            float(x*1000.0), float(y*1000.0), float(z*1000.0),
            float(rv[0]), float(rv[1]), float(rv[2]),
        ]
        req.mvvelo     = float(k_vel * self.tool_vel * 1000.0)
        req.mvacc      = float(k_acc * self.tool_acc * 1000.0)
        req.mvtime     = float(t)
        req.mvradii    = float(r)
        req.coord_mode = 0
        req.index      = 0
        req.has_ref    = False
        req.ref_joint  = []

        future = self._cli_linear_move.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
        self._wait_cartesian_stop(tool_configuration[:3])

    def _wait_joint_stop(self, target_joints, timeout=30.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            self._spin_once()
            if self._current_joints is None:
                continue
            if all(abs(self._current_joints[i] - target_joints[i]) < self.joint_tolerance
                   for i in range(6)):
                return
            time.sleep(0.05)
        self.get_logger().warn('move_j 等待超时，继续执行。')

    def _wait_cartesian_stop(self, target_xyz_m, timeout=30.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            self._spin_once()
            if self._current_tool_pos is None:
                continue
            t = self._current_tool_pos.twist
            if all(abs(getattr(t.linear, ax) / 1000.0 - target_xyz_m[i]) < 0.003
                   for i, ax in enumerate(['x', 'y', 'z'])):
                return
            time.sleep(0.05)
        self.get_logger().warn('move_j_p/move_l 等待超时，继续执行。')

    def get_camera_data(self, retries=5):
        for attempt in range(retries):
            try:
                color_bgr, depth = self.camera.get_data()
                if color_bgr is not None:
                    return color_bgr, depth
            except RuntimeError as e:
                self.get_logger().warn(f'相机获取帧失败 (第{attempt+1}次): {e}，重试...')
                time.sleep(1.0)
        self.get_logger().error('相机重试全部失败')
        return None, None

    @staticmethod
    def _rpy_to_rotvec(rx, ry, rz):
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        R = np.array([
            [cz*cy,  cz*sy*sx - sz*cx,  cz*sy*cx + sz*sx],
            [sz*cy,  sz*sy*sx + cz*cx,  sz*sy*cx - cz*sx],
            [-sy,    cy*sx,              cy*cx            ]
        ])
        theta = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
        if abs(theta) < 1e-6:
            return np.array([0.0, 0.0, 0.0])
        return theta / (2 * math.sin(theta)) * np.array([
            R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]
        ])

    def R2rpy(self, R):
        sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
        if sy > 1e-6:
            x = math.atan2(R[2,1], R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else:
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0.0
        return np.array([x, y, z])


if __name__ == '__main__':
    rclpy.init()
    robot = Jaka_Robot(tcp_host_ip='192.168.1.100')
    pose = robot.get_tool_pose()
    print(f'当前末端位姿 (m/rad): x={pose[0]:.4f} y={pose[1]:.4f} z={pose[2]:.4f}')
    print(f'当前关节角 (deg): {[round(math.degrees(a),2) for a in robot.get_joint_positions()]}')
    rclpy.shutdown()