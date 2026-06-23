"""
FastAPI 服务：双模型隐私重建 + 卷积 heatmap + 四路合并

模型运行在独立子进程中（HTTP通信），命名空间完全隔离。

启动方式（在项目根目录下）：
    python http_server.py

POST /original      → 使用 original 模型重建（cuda:6, 端口28010）
POST /noise         → 使用 noise 模型重建（cuda:7, 端口28011）
POST /process       → 同时调用 two 模型 + heatmap，返回四路合并结果
GET  /health        → 健康检查
"""

from __future__ import annotations

import asyncio
import functools
import io
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Tuple, Optional

import httpx
import torch
import torch.nn as nn
import torchvision.transforms as T
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

FONT_CANDIDATES = [SCRIPT_DIR / "NotoSansCJKsc-Regular.otf"]
_CN_FONT: Optional[ImageFont.FreeTypeFont] = None
_CN_FONT_PATH: Optional[Path] = None

_cnn_model = None
_cnn_transform = None

ORIGINAL_URL = "http://127.0.0.1:28010"
NOISE_URL = "http://127.0.0.1:28011"
http_client: httpx.AsyncClient | None = None


class ConvEmbedding(nn.Module):
    def __init__(self, embed_dim: int = 2000, heatmap_h: int = 512, heatmap_w: int = 512):
        super().__init__()
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
        return self.proj(feat).squeeze(1)


def _load_cn_font(size: int = 28) -> ImageFont.FreeTypeFont:
    global _CN_FONT, _CN_FONT_PATH
    if _CN_FONT is not None and _CN_FONT_PATH is not None:
        try:
            return _CN_FONT.font_variant(size=size) if hasattr(_CN_FONT, "font_variant") else ImageFont.truetype(str(_CN_FONT_PATH), size=size)
        except Exception:
            pass
    for p in FONT_CANDIDATES:
        if p.exists():
            try:
                _CN_FONT = ImageFont.truetype(str(p), size=size)
                _CN_FONT_PATH = p
                return _CN_FONT
            except Exception:
                continue
    system_candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "C:/Windows/Fonts/msyh.ttc",
    ]
    for p in system_candidates:
        if Path(p).exists():
            try:
                _CN_FONT = ImageFont.truetype(p, size=size)
                _CN_FONT_PATH = Path(p)
                return _CN_FONT
            except Exception:
                continue
    raise RuntimeError("未找到可用的中文字体！请将 NotoSansCJKsc-Regular.otf 放到脚本同目录。")


def _draw_text_bg(draw: ImageDraw.Draw, text: str, xy: Tuple[int, int],
                  font: ImageFont.FreeTypeFont, bg_color: Tuple[int, int, int],
                  color: Tuple[int, int, int] = (255, 255, 255),
                  pad: Tuple[int, int] = (8, 4)):
    has_newline = "\n" in text
    if has_newline:
        bbox = draw.multiline_textbbox(xy, text, font=font, spacing=4)
    else:
        bbox = draw.textbbox(xy, text, font=font)
    x0, y0, x1, y1 = bbox
    draw.rectangle((x0 - pad[0], y0 - pad[1], x1 + pad[0], y1 + pad[1]), fill=bg_color)
    if has_newline:
        draw.multiline_text(xy, text, font=font, fill=color, spacing=4)
    else:
        draw.text(xy, text, font=font, fill=color)


def _compute_heatmap(frame_bgr: np.ndarray, device: torch.device) -> np.ndarray:
    global _cnn_model, _cnn_transform
    if _cnn_model is None:
        print(f"Initializing CNN model on {device}...", flush=True)
        _cnn_model = ConvEmbedding().eval().to(device)
        _cnn_transform = T.Compose([
            T.ToPILImage(),
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("CNN model ready.", flush=True)

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = _cnn_transform(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = _cnn_model(img_tensor)
        heat = feat.squeeze().cpu().numpy()
    heat_norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8) * 255.0
    heat_u8 = heat_norm.astype(np.uint8)
    h, w = frame_bgr.shape[:2]
    heat_resized = cv2.resize(heat_u8, (w, h), interpolation=cv2.INTER_CUBIC)
    k = max(5, (min(h, w) // 80) | 1)
    heat_smooth = cv2.GaussianBlur(heat_resized, (k, k), 0)
    return cv2.applyColorMap(255 - heat_smooth, cv2.COLORMAP_JET)

async def _call_worker(url: str, contents: bytes) -> bytes:
    files = {"file": ("image.png", contents, "image/png")}
    r = await http_client.post(f"{url}/reconstruct", files=files, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"Worker error {r.status_code}: {r.text}")
    return r.content


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(proxy=None)

    # await asyncio.gather(
    #     _wait_for_worker(ORIGINAL_URL, "original"),
    #     _wait_for_worker(NOISE_URL, "noise"),
    # )

    yield

    await http_client.aclose()
    torch.cuda.empty_cache()


app = FastAPI(title="Privacy Reconstruction API", lifespan=lifespan)


@app.get("/health")
async def health():
    results = {}
    for name, url in [("original", ORIGINAL_URL), ("noise", NOISE_URL)]:
        try:
            r = await http_client.get(f"{url}/health", timeout=3)
            results[name] = r.status_code == 200 and r.json().get("ready", False)
        except Exception:
            results[name] = False
    return {"status": "ok", **results}


async def _run_worker(url: str, file: UploadFile):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="仅支持图片文件")

    contents = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        original_size = pil_image.size
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法解析图片: {e}")

    try:
        result_bytes = await _call_worker(url, contents)
        result_image = Image.open(io.BytesIO(result_bytes)).convert("RGB")
        if result_image.size != original_size:
            result_image = result_image.resize(original_size, Image.LANCZOS)
        return cv2.imencode(
            '.jpg',
            cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )[1].tobytes()
    except Exception as e:
        print(f"[Error] worker failed: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"重建失败: {e}")


@app.post("/original")
async def reconstruct_original(file: UploadFile = File(...)):
    image_bytes = await _run_worker(ORIGINAL_URL, file)
    return Response(content=image_bytes, media_type="image/jpeg")


@app.post("/noise")
async def reconstruct_noise(file: UploadFile = File(...)):
    image_bytes = await _run_worker(NOISE_URL, file)
    return Response(content=image_bytes, media_type="image/jpeg")


@app.post("/process")
async def process_combined(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="仅支持图片文件")

    contents = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        original_size = pil_image.size
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法解析图片: {e}")

    w, h = original_size
    bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    
    loop = asyncio.get_running_loop()
    device = torch.device("cuda:6")

    result1_bytes, result2_bytes, heatmap = await asyncio.gather(
        _call_worker(ORIGINAL_URL, contents),
        _call_worker(NOISE_URL, contents),
        loop.run_in_executor(None, functools.partial(_compute_heatmap, bgr, device)),
    )

    result1 = cv2.resize(
        cv2.cvtColor(np.array(Image.open(io.BytesIO(result1_bytes)).convert("RGB")), cv2.COLOR_RGB2BGR),
        (w, h),
    )
    result2 = cv2.resize(
        cv2.cvtColor(np.array(Image.open(io.BytesIO(result2_bytes)).convert("RGB")), cv2.COLOR_RGB2BGR),
        (w, h),
    )
    heatmap = cv2.resize(heatmap, (w, h))

    combined = np.empty((h * 2, w * 2, 3), dtype=np.uint8)
    combined[:h, :w] = bgr
    combined[:h, w:] = heatmap
    combined[h:, :w] = result1
    combined[h:, w:] = result2

    rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    font = _load_cn_font(96)

    _draw_text_bg(draw, "原始视频（云端无法获取，仅用于展厅展示）", (10, -5), font, (180, 0, 0))
    _draw_text_bg(draw, "隐私相机传递给云端的词元流", (w + 10, -5), font, (180, 0, 0))
    _draw_text_bg(draw, "窃听者基于词元流重建的画面\n（无隐私保护机制）", (10, h - 5), font, (180, 0, 0))
    _draw_text_bg(draw, "窃听者基于词元流重建的画面\n（有隐私保护机制）", (w + 10, h - 5), font, (180, 0, 0))

    combined_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    image_bytes = cv2.imencode('.jpg', combined_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
    return Response(content=image_bytes, media_type="image/jpeg")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=28000)
