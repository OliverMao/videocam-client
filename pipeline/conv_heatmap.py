#!/usr/bin/env python3
"""
卷积 Embedding 热力图模块
========================
轻量 CNN 提取特征向量，生成可视化热力图
"""
from __future__ import annotations

import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T

# ----- 全局状态 -----
_CNN_MODEL = None
_CNN_TRANSFORM = None
_EMBED_DIM = 2000
_HEATMAP_H = 512
_HEATMAP_W = 512


class ConvEmbedding(nn.Module):
    """轻量卷积网络，输出低分辨率特征图由 OpenCV 上采样"""

    def __init__(self, embed_dim: int = 2000, heatmap_h: int = 512, heatmap_w: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self.heatmap_h = heatmap_h
        self.heatmap_w = heatmap_w
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = self.proj(feat).squeeze(1)
        return feat


def load_cnn_model(device: torch.device):
    """加载轻量卷积 embedding 模型"""
    global _CNN_MODEL, _CNN_TRANSFORM
    if _CNN_MODEL is not None:
        return _CNN_MODEL, _CNN_TRANSFORM

    print(f"🔄 初始化卷积 embedding 模型 (device={device}, dim={_EMBED_DIM}) ...")
    _CNN_MODEL = ConvEmbedding(
        embed_dim=_EMBED_DIM,
        heatmap_h=_HEATMAP_H,
        heatmap_w=_HEATMAP_W,
    ).eval().to(device)
    _CNN_TRANSFORM = T.Compose([
        T.ToPILImage(),
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    print(f"✅ 卷积 embedding 模型就绪 (device={device}, 轻量CNN+OpenCV上采样)")
    return _CNN_MODEL, _CNN_TRANSFORM


def compute_heatmap(frame_bgr: np.ndarray,
                    device: torch.device = torch.device("cpu")) -> np.ndarray:
    """
    计算卷积 embedding 热力图，颜色反转（高值=蓝，低值=红）。
    返回 BGR 图像。
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    model, transform = load_cnn_model(device)
    img_tensor = transform(rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(img_tensor)
        heat = feat.squeeze().cpu().numpy()

    heat_min = heat.min()
    heat_max = heat.max()
    heat_norm = (heat - heat_min) / (heat_max - heat_min + 1e-8) * 255.0
    heat_u8 = heat_norm.astype(np.uint8)

    h, w = frame_bgr.shape[:2]
    heat_resized = cv2.resize(heat_u8, (w, h), interpolation=cv2.INTER_CUBIC)

    k = max(5, (min(h, w) // 80) | 1)
    heat_smooth = cv2.GaussianBlur(heat_resized, (k, k), 0)

    heat_color = cv2.applyColorMap(255 - heat_smooth, cv2.COLORMAP_JET)
    return heat_color
