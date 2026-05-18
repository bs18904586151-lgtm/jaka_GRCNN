import os
import random
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
import cv2

# -------------------------
# 1. 加载模型 (支持 DataParallel)
# -------------------------
def load_model(model_path, device):
    print(f"Loading model from: {model_path}")
    # 加载整个模型对象
    net = torch.load(model_path, map_location=device)
    # 如果模型是用 DataParallel 训练的，取其 .module
    if hasattr(net, 'module'):
        net = net.module
    net.eval()
    return net

# -------------------------
# 2. 图像预处理 (缩放到模型输入的 300x300)
# -------------------------
def preprocess_image(img, input_size=300):
    img_resized = img.resize((input_size, input_size))
    img_np = np.array(img_resized).astype(np.float32) / 255.0

    # 转换为灰度图 (RGB -> Gray)
    if img_np.ndim == 3:
        img_np = 0.2989 * img_np[:, :, 0] + 0.5870 * img_np[:, :, 1] + 0.1140 * img_np[:, :, 2]

    img_np = np.expand_dims(img_np, axis=0)  # (1, H, W)
    img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # (1, 1, H, W)
    return img_tensor

# -------------------------
# 3. 预测 Quality Map
# -------------------------
def predict_q_map(net, img_tensor, device):
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        pred = net(img_tensor)
        
        # 处理多输出情况 (pos, cos, sin, width)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
            
        q_map = pred.squeeze().cpu().numpy()
    return q_map

# -------------------------
# 4. 从 Q-map 提取抓取参数 (含坐标缩放)
# -------------------------
def extract_grasp(q_map, original_size):
    """
    original_size: (width, height) 原始图片的尺寸
    """
    # 高斯滤波平滑噪声
    q_smooth = gaussian_filter(q_map, sigma=2)
    
    # 找到最大响应点
    y, x = np.unravel_index(np.argmax(q_smooth), q_smooth.shape)
    
    # 映射回原始图像坐标
    scale_y = original_size[1] / q_map.shape[0]
    scale_x = original_size[0] / q_map.shape[1]
    
    orig_x = x * scale_x
    orig_y = y * scale_y

    # 计算旋转角度 (基于局部梯度)
    gy, gx = np.gradient(q_smooth)
    angle = np.arctan2(gy[y, x], gx[y, x])
    
    # 估计宽度 (基于局部响应均值)
    patch_size = 11
    half = patch_size // 2
    patch = q_smooth[max(0, y-half):y+half, max(0, x-half):x+half]
    width = int(np.clip(np.mean(patch) * 150, 20, 100) * scale_x)

    return orig_x, orig_y, angle, width

# -------------------------
# 5. 绘制抓取框
# -------------------------
def draw_grasp(ax, img, grasp, q_map=None):
    x, y, angle, width = grasp
    
    # 矩形框定义: ((中心x, 中心y), (长, 宽), 角度)
    # 这里长度固定为 25 (会随原图缩放), 宽度由模型估算
    rect_len = 30 * (img.shape[1] / 300)
    rect = ((x, y), (rect_len, width), np.degrees(angle))

    box = cv2.boxPoints(rect)
    box = np.int0(box)

    ax.imshow(img)
    # 画抓取矩形
    ax.plot(*zip(*(list(box) + [box[0]])), color='red', linewidth=2)
    # 画中心点
    ax.scatter([x], [y], color='yellow', s=20)
    ax.axis('off')

# -------------------------
# 6. 主程序
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', required=True)
    parser.add_argument('--dataset-path', required=True)
    parser.add_argument('--num-samples', type=int, default=6)
    parser.add_argument('--use-cpu', action='store_true')
    parser.add_argument('--output', default='results_vis/jacquard_batch.png')
    args = parser.parse_args()

    device = torch.device('cpu' if args.use_cpu else 'cuda')

    # 加载模型
    net = load_model(args.network, device)

    # 搜索数据集
    samples = []
    for root, _, files in os.walk(args.dataset_path):
        for f in files:
            if f.endswith("_RGB.png"):
                samples.append(os.path.join(root, f))
    
    if not samples:
        print(f"Error: No _RGB.png files found in {args.dataset_path}")
        return

    selected = random.sample(samples, min(len(samples), args.num_samples))

    # 准备画布
    cols = 3
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for i, img_path in enumerate(selected):
        # 读取原图
        raw_img = Image.open(img_path).convert('RGB')
        
        # 预处理并推理
        img_tensor = preprocess_image(raw_img)
        q_map = predict_q_map(net, img_tensor, device)
        
        # 提取抓取参数 (传入原图尺寸用于坐标转换)
        grasp = extract_grasp(q_map, raw_img.size)
        
        # 绘图
        draw_grasp(axes[i], np.array(raw_img), grasp, q_map)
        axes[i].set_title(os.path.basename(img_path))

    # 隐藏多余的子图
    for j in range(len(selected), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(args.output, dpi=300)
    plt.close()
    print(f"Visualization saved to: {args.output}")

if __name__ == '__main__':
    main()