#python3 -c "
from real.realsenseD435 import RealsenseD435
import cv2, numpy as np

cam = RealsenseD435()
color, depth = cam.get_data()
gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
found, corners = cv2.findChessboardCorners(gray, (5,5), None)
if found:
    pix = np.round(corners[12, 0, :]).astype(int)
    # 采样中心5x5区域过滤零值
    region = depth[pix[1]-5:pix[1]+5, pix[0]-5:pix[0]+5]
    valid = region[region > 0]
    raw = float(np.median(valid))
    print(f'深度原始值: {raw}')
    print(f'现在用卷尺量标定板到相机镜头的直线距离，告诉我是多少cm')
else:
    print('未找到棋盘格')
cam.pipeline.stop()
