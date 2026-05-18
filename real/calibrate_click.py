#!/usr/bin/env python3
# coding=utf8
"""
calibrate_click.py  --  eye-to-hand 2D 平面标定（纯手动点击版）

操作说明：
  滚轮          放大 / 缩小（以鼠标为中心）
  中键拖拽      平移画面
  左键点击      按 P1→P12 顺序点击各靶标圆心
  Z             撤销最后一个点
  C             12个点全部确认后计算 CAMERA_POSE
  Q / ESC       退出

输出：camera_pose_new.npy  和  camera_pose_new.py（粘贴进 grasp.py 替换即可）
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import os

# ── 相机内参（与 grasp.py 一致）────────────────────────
FX, FY         = 607.879, 607.348
CX, CY         = 325.14,  244.014
TABLE_DEPTH_MM = 692.0

# ── 12个标定点机械臂坐标（mm → m）─────────────────────
ROBOT_PTS_M = np.array([
    # 纸面 P1~P4：最下行（Y=45），纸面X从左到右对应机械臂Y -357→-633，机械臂X=-95
    [ -95.0, -357.505],   # P1  纸面(70, 45)
    [ -95.0, -449.505],   # P2  纸面(163,45)
    [ -95.0, -541.505],   # P3  纸面(257,45)
    [ -95.0, -633.505],   # P4  纸面(350,45)
    # 纸面 P5~P8：中间行（Y=148），机械臂X=7.5
    [   7.5, -357.505],   # P5  纸面(70, 148)
    [   7.5, -449.505],   # P6  纸面(163,148)
    [   7.5, -541.505],   # P7  纸面(257,148)
    [   7.5, -633.505],   # P8  纸面(350,148)
    # 纸面 P9~P12：最上行（Y=252），机械臂X=110
    [ 110.0, -357.505],   # P9  纸面(70, 252)
    [ 110.0, -449.505],   # P10 纸面(163,252)
    [ 110.0, -541.505],   # P11 纸面(257,252)
    [ 110.0, -633.505],   # P12 纸面(350,252)
], dtype=np.float64) / 1000.0

N_PTS = 12

C_DOT  = (  0, 200, 255)   # 已确认点
C_NEXT = (  0,   0, 255)   # 当前待确认提示色
C_OK   = (  0, 255, 180)
C_BG   = ( 20,  20,  20)
C_WHITE= (255, 255, 255)


# ── 标定求解 ────────────────────────────────────────────
def solve_camera_pose(confirmed):
    """
    直接最小二乘：12对点对应关系解转换矩阵
    robot_X = coef_x @ [cx, cy, cz, 1]
    robot_Y = coef_y @ [cx, cy, cz, 1]
    """
    cam_pts = np.array([[(u-CX)*z/FX, (v-CY)*z/FY, z] for u,v,z in confirmed])
    A = np.hstack([cam_pts, np.ones((N_PTS, 1))])   # (12, 4)

    robot_m = ROBOT_PTS_M                            # 单位 m
    coef_x, _, _, _ = np.linalg.lstsq(A, robot_m[:, 0], rcond=None)
    coef_y, _, _, _ = np.linalg.lstsq(A, robot_m[:, 1], rcond=None)

    # 组成 4×4 矩阵，与 grasp.py 的 CAMERA_POSE @ [cx,cy,cz,1] 接口一致
    cp = np.array([
        [coef_x[0], coef_x[1], coef_x[2], coef_x[3]],
        [coef_y[0], coef_y[1], coef_y[2], coef_y[3]],
        [0,         0,         0,         0        ],
        [0,         0,         0,         1        ],
    ])

    print("\n─── 各点重投影误差 ───")
    errors = []
    for i, (u, v, z) in enumerate(confirmed):
        rp    = cp @ np.array([(u-CX)*z/FX, (v-CY)*z/FY, z, 1.0])
        tx,ty = rp[0]*1000, rp[1]*1000
        rx,ry = ROBOT_PTS_M[i] * 1000
        e     = np.hypot(tx-rx, ty-ry)
        errors.append(e)
        print(f"  P{i+1:02d}: 预测({tx:7.2f},{ty:7.2f})  "
              f"真值({rx:7.2f},{ry:7.2f})  误差={e:.2f}mm  深度={z*1000:.1f}mm")
    rmse = np.sqrt(np.mean(np.array(errors)**2))
    print(f"  RMSE = {rmse:.3f} mm")
    return cp, rmse, errors


# ── 视图变换（缩放/平移）────────────────────────────────
class View:
    def __init__(self, iw, ih, ww=1440, wh=900):
        self.ww, self.wh = ww, wh
        s = min(ww/iw, wh/ih)
        self.scale = s
        self.ox = (ww - iw*s) / 2
        self.oy = (wh - ih*s) / 2

    def win2img(self, wx, wy):
        return (wx - self.ox) / self.scale, (wy - self.oy) / self.scale

    def zoom(self, factor, cx, cy):
        ix, iy = self.win2img(cx, cy)
        self.scale = max(0.2, min(15.0, self.scale * factor))
        self.ox = cx - ix * self.scale
        self.oy = cy - iy * self.scale

    def pan(self, dx, dy):
        self.ox += dx;  self.oy += dy

    def render(self, img):
        H, W = img.shape[:2]
        canvas = np.zeros((self.wh, self.ww, 3), np.uint8)
        x0 = max(0., -self.ox / self.scale)
        y0 = max(0., -self.oy / self.scale)
        x1 = min(W,  (self.ww - self.ox) / self.scale)
        y1 = min(H,  (self.wh - self.oy) / self.scale)
        if x1 <= x0 or y1 <= y0:
            return canvas
        dx0 = max(0, int(x0*self.scale + self.ox))
        dy0 = max(0, int(y0*self.scale + self.oy))
        dx1 = min(self.ww, int(x1*self.scale + self.ox))
        dy1 = min(self.wh, int(y1*self.scale + self.oy))
        src = img[int(y0):int(y1), int(x0):int(x1)]
        if src.size == 0 or dx1 <= dx0 or dy1 <= dy0:
            return canvas
        canvas[dy0:dy1, dx0:dx1] = cv2.resize(src, (dx1-dx0, dy1-dy0))
        return canvas


def put_label(img, text, pos, fg=C_WHITE, bg=C_BG, scale=0.55, thick=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    x, y = int(pos[0]), int(pos[1])
    cv2.rectangle(img, (x-2, y-th-2), (x+tw+2, y+bl+1), bg, -1)
    cv2.putText(img, text, (x, y), font, scale, fg, thick, cv2.LINE_AA)


def draw_state(base, view, confirmed, frozen):
    anno = base.copy()

    # 已确认的点：原图坐标系画十字+序号
    for i, (u, v, *_) in enumerate(confirmed):
        iu, iv = int(u), int(v)
        cv2.drawMarker(anno, (iu, iv), C_DOT,
                       cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
        put_label(anno, f"P{i+1}", (iu+12, iv-8), fg=C_DOT)

    canvas = view.render(anno)

    # 状态栏
    n = len(confirmed)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 48), C_BG, -1)
    if not frozen:
        txt = "按 D 冻结画面"
        col = (180, 180, 180)
    elif n < N_PTS:
        txt = f"请点击 P{n+1} 的圆心   ({n}/{N_PTS} 完成)"
        col = C_NEXT if n == 0 else C_DOT
    else:
        txt = "全部完成！按 C 计算   Z 撤销"
        col = C_OK
    cv2.putText(canvas, txt, (14, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, col, 2, cv2.LINE_AA)

    # 右下角缩放比例
    pct = f"{view.scale*100:.0f}%"
    cv2.putText(canvas, pct, (canvas.shape[1]-70, canvas.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1)

    # 底部操作提示
    hint = "P:放大  W:缩小  中键拖拽:平移  D:冻结  Z:撤销  C:计算  Q:退出"
    cv2.putText(canvas, hint, (14, canvas.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (110,110,110), 1, cv2.LINE_AA)

    return canvas


# ── 主程序 ──────────────────────────────────────────────
def main():
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16,  30)
    pipeline.start(cfg)
    align = rs.align(rs.stream.color)
    for _ in range(10):
        pipeline.wait_for_frames()

    print("相机已启动 (1280×720)")
    print("  D      冻结当前帧")
    print("  滚轮   缩放（放大后再点击，更准）")
    print("  中键   平移画面")
    print("  左键   按 P1→P12 顺序点击各圆心")
    print("  Z      撤销最后一个点")
    print("  C      计算 CAMERA_POSE")
    print("  Q      退出\n")

    IMG_H, IMG_W = 720, 1280
    view = View(IMG_W, IMG_H, 1440, 900)

    frozen       = False
    frozen_frame = None
    frozen_depth = None
    live_frame   = None
    live_depth   = None
    confirmed    = []   # [(u, v, z_m), ...]

    pan_active = False
    pan_last   = (0, 0)
    mouse_win  = (0, 0)

    WIN = "Eye-to-Hand Calibration"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1440, 900)

    def on_mouse(event, wx, wy, flags, param):
        nonlocal pan_active, pan_last, confirmed, mouse_win

        mouse_win = (wx, wy)

        # 滚轮缩放
        if event == cv2.EVENT_MOUSEWHEEL:
            view.zoom(1.15 if flags > 0 else 1/1.15, wx, wy)
            return

        # 中键平移
        if event == cv2.EVENT_MBUTTONDOWN:
            pan_active = True;  pan_last = (wx, wy);  return
        if event == cv2.EVENT_MBUTTONUP:
            pan_active = False;  return
        if event == cv2.EVENT_MOUSEMOVE and pan_active:
            view.pan(wx - pan_last[0], wy - pan_last[1])
            pan_last = (wx, wy);  return

        # 左键：确认点击位置
        if event == cv2.EVENT_LBUTTONDOWN:
            if not frozen:
                print("请先按 D 冻结画面");  return
            if len(confirmed) >= N_PTS:
                print("已满12个点，按 C 计算 或 Z 撤销");  return

            # 窗口坐标 → 原图坐标
            ix, iy = view.win2img(wx, wy)
            ix = max(0, min(IMG_W-1, ix))
            iy = max(0, min(IMG_H-1, iy))

            # 读实测深度（7×7邻域中位数）
            icx, icy = int(round(ix)), int(round(iy))
            patch = frozen_depth[max(0,icy-3):icy+4,
                                  max(0,icx-3):icx+4].flatten()
            valid = patch[patch > 0]
            if len(valid) == 0:
                print(f"  P{len(confirmed)+1} 处深度为零，请换一帧重试（重按 D）")
                return
            z_mm = float(np.median(valid))
            confirmed.append((float(ix), float(iy), z_mm * 0.001))
            n = len(confirmed)
            rx, ry = ROBOT_PTS_M[n-1] * 1000
            print(f"  P{n:02d} <- 像素({ix:.1f},{iy:.1f})  "
                  f"深度={z_mm:.1f}mm  机械臂({rx:.1f},{ry:.3f})mm")

    cv2.setMouseCallback(WIN, on_mouse)

    try:
        while True:
            if not frozen:
                frames  = pipeline.wait_for_frames()
                aligned = align.process(frames)
                cf = aligned.get_color_frame()
                df = aligned.get_depth_frame()
                if not cf or not df:
                    continue
                live_frame = np.asanyarray(cf.get_data())
                live_depth = np.asanyarray(df.get_data())

            base   = frozen_frame if frozen else live_frame
            canvas = draw_state(base, view, confirmed, frozen)
            cv2.imshow(WIN, canvas)
            key = cv2.waitKey(16) & 0xFF

            # P / W：放大 / 缩小（以画面中心为基准）
            if key in (ord('p'), ord('P')):
                cx, cy = view.ww // 2, view.wh // 2
                view.zoom(1.20, cx, cy)
            elif key in (ord('w'), ord('W')):
                cx, cy = view.ww // 2, view.wh // 2
                view.zoom(1/1.20, cx, cy)

            # D：冻结
            elif key in (ord('d'), ord('D')):
                if not frozen:
                    frozen_frame = live_frame.copy()
                    frozen_depth = live_depth.copy()
                    frozen = True
                    print("\n画面已冻结，请放大后按 P1→P12 顺序点击各圆心")
                else:
                    # 再次按D：用新帧重新冻结（重拍）
                    frozen = False
                    confirmed.clear()
                    print("重新冻结，已清空所有点，请重新点击")

            # Z：撤销
            elif key in (ord('z'), ord('Z')):
                if confirmed:
                    u, v, z = confirmed.pop()
                    print(f"  撤销 P{len(confirmed)+1}  像素({u:.1f},{v:.1f})")
                else:
                    print("  没有可撤销的点")

            # C：计算
            elif key in (ord('c'), ord('C')):
                if len(confirmed) < N_PTS:
                    print(f"  还差 {N_PTS - len(confirmed)} 个点");  continue
                try:
                    cp, rmse, errors = solve_camera_pose(confirmed)
                    print(f"\n═══ 标定完成  RMSE = {rmse:.3f} mm ═══")
                    print("CAMERA_POSE =\n", cp)

                    out_dir  = os.path.dirname(os.path.abspath(__file__))
                    npy_path = os.path.join(out_dir, "camera_pose_new.npy")
                    py_path  = os.path.join(out_dir, "camera_pose_new.py")
                    np.save(npy_path, cp)
                    with open(py_path, "w") as f:
                        f.write(f"# RMSE = {rmse:.3f} mm\n")
                        f.write("# 替换 grasp.py 中对应参数\n\n")
                        rows = ["    [" + ", ".join(f"{v: .8f}" for v in row) + "]"
                                for row in cp]
                        f.write("CAMERA_POSE = np.array([\n" +
                                ",\n".join(rows) + "\n])\n\n")
                        f.write("X_OFFSET = 0.0\nY_OFFSET = 0.0\n")
                    print(f"已保存：{npy_path}")
                    print(f"已保存：{py_path}")
                    print(">>> 将 camera_pose_new.py 内容粘贴进 grasp.py 即可\n")

                    # 结果叠加显示
                    result = frozen_frame.copy()
                    for i, (u, v, z) in enumerate(confirmed):
                        rp = cp @ np.array([(u-CX)*z/FX, (v-CY)*z/FY, z, 1.0])
                        e  = np.hypot(rp[0]*1000 - ROBOT_PTS_M[i,0]*1000,
                                      rp[1]*1000 - ROBOT_PTS_M[i,1]*1000)
                        col = C_OK if e < 5 else (0, 80, 255)
                        cv2.drawMarker(result, (int(u), int(v)), col,
                                       cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
                        put_label(result, f"P{i+1} {e:.1f}mm",
                                  (u+12, v-8), fg=col)
                    rc = view.render(result)
                    cv2.rectangle(rc, (0,0), (rc.shape[1], 48), C_BG, -1)
                    cv2.putText(rc, f"RMSE={rmse:.2f}mm  任意键退出",
                                (14, 32), cv2.FONT_HERSHEY_SIMPLEX,
                                0.85, C_OK, 2, cv2.LINE_AA)
                    cv2.imshow(WIN, rc)
                    cv2.waitKey(0)
                    break

                except Exception as e:
                    print(f"  求解失败：{e}")

            elif key in (ord('q'), ord('Q'), 27):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("退出。")


if __name__ == "__main__":
    main()