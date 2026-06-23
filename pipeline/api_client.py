#!/usr/bin/env python3
"""
API 调用模块
============
远程 API 重建图像处理
"""
from __future__ import annotations

import asyncio
from io import BytesIO
import os
import cv2
import httpx
import numpy as np
from PIL import Image

GPU_HOST = os.environ.get("GPU_HOST", "")  # GPU 服务器地址
MAE_PORT = os.environ.get("MAE_PORT", "30852")  # MAE 端口

# ----- API 配置 -----
API_URL_1 = f"http://{GPU_HOST}:{MAE_PORT}/process"
API_URL_2 = f"http://{GPU_HOST}:31344/reconstruct"  # 不用了
_async_client: httpx.AsyncClient | None = None


async def get_client() -> httpx.AsyncClient:
    """获取或创建异步 HTTP 客户端"""
    global _async_client
    if _async_client is None:
        _async_client = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=20),
            timeout=httpx.Timeout(5.0, connect=3.0),
        )
    return _async_client


async def call_api(frame: np.ndarray, url: str) -> np.ndarray:
    """调用远程 API 进行图像重建"""
    success, jpg_bytes = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90]
    )
    if not success:
        return frame
    client = await get_client()
    for attempt in range(3):
        try:
            resp = await client.post(
                url, files={"file": ("frame.jpg", jpg_bytes.tobytes(), "image/jpeg")}
            )
            resp.raise_for_status()
            break
        except Exception as e:
            if attempt == 2:
                print(f"⚠️ API 调用失败: {url} -> {e}", flush=True)
                return frame
            await asyncio.sleep(0.1)
    try:
        rec_img = Image.open(BytesIO(resp.content)).convert("RGB")
        out = cv2.cvtColor(np.array(rec_img), cv2.COLOR_RGB2BGR)
        if out.shape[:2] != frame.shape[:2]:
            out = cv2.resize(out, (frame.shape[1], frame.shape[0]))
        return out
    except Exception as e:
        print(f"⚠️ 解析返回图像失败: {url} -> {e}", flush=True)
        return frame


async def action_1_frame(frame: np.ndarray) -> np.ndarray:
    """调用 API 1 进行重建"""
    return await call_api(frame, API_URL_1)


async def action_2_frame(frame: np.ndarray) -> np.ndarray:
    """调用 API 2 进行重建"""
    return await call_api(frame, API_URL_2)


async def cleanup():
    """清理 HTTP 客户端"""
    global _async_client
    if _async_client:
        await _async_client.aclose()
        _async_client = None
