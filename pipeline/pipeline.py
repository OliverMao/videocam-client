#!/usr/bin/env python3
"""
视频四路同步处理 → 推流 / 输出文件 (含卷积 embedding 热力图)
============================================================
两种运行模式:
  python pipeline.py file  <input.mp4> [--fps 10] [--no-heatmap]
  python pipeline.py stream [--fps 10]                                实时 RTSP 推流

文件模式下右上角显示卷积 embedding 热力图：
  - 使用轻量 CNN 提取 10000 维特征向量
  - reshape 为 100x100 热力图，归一化到 0-255 全色谱

action_1 与 action_2 均调用远程 API: http://116.238.240.2:32101/reconstruct

依赖: opencv-python, httpx, pillow, torch, torchvision, numpy
"""
from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torchvision.transforms as T

# ═══════════════════════════════════════════════════════════
# 路径与常量
SCRIPT_DIR = Path(__file__).resolve().parent

FONT_CANDIDATES = [
    SCRIPT_DIR / "NotoSansCJKsc-Regular.otf",
]

FORCE_FPS: float | None = None
API_URL_1 = "http://116.238.240.2:32101/reconstruct"
API_URL_2 = "http://116.238.240.2:30586/reconstruct"
_async_client: httpx.AsyncClient | None = None

# ----- 卷积 Embedding 模型 -----
_CNN_MODEL = None
_CNN_TRANSFORM = None
_EMBED_DIM = 2000  # 2000维特征向量
_HEATMAP_H = 512    # 热力图高度
_HEATMAP_W = 512    # 热力图宽度 (512*512=262144像素点，每点2000维特征，约等于10000维全局特征)

# ----- 字体全局单例 -----
_CN_FONT: Optional[ImageFont.FreeTypeFont] = None
_CN_FONT_PATH: Optional[Path] = None


# ═══════════════════════════════════════════════════════════
# 卷积 Embedding 模型定义
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
        feat = self.features(x)                    # [B, 64, 28, 28]
        feat = self.proj(feat).squeeze(1)          # [B, 28, 28]
        return feat


# ═══════════════════════════════════════════════════════════
# 字体加载
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
                print(f"🔤 使用字体: {p.name}")
                return _CN_FONT
            except Exception as e:
                print(f"⚠️ 字体加载失败 {p}: {e}")
                continue

    system_candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "C:/Windows/Fonts/msyh.ttc",
    ]
    for p in system_candidates:
        if Path(p).exists():
            try:
                _CN_FONT = ImageFont.truetype(p, size=size)
                _CN_FONT_PATH = Path(p)
                print(f"🔤 使用系统字体: {p}")
                return _CN_FONT
            except Exception:
                continue

    raise RuntimeError(
        "❌ 未找到可用的中文字体！\n"
        "请将 NotoSansCJKsc-Regular.otf 放到脚本同目录。"
    )


def _put_text_cn(img_bgr: np.ndarray, text: str, xy: Tuple[int, int],
                 size: int = 28, color=(255, 255, 255),
                 bg_color: Optional[Tuple[int, int, int]] = None,
                 pad: Tuple[int, int] = (8, 4)) -> np.ndarray:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    font = _load_cn_font(size)

    has_newline = "\n" in text
    if has_newline:
        bbox = draw.multiline_textbbox(xy, text, font=font, spacing=4)
        x0, y0, x1, y1 = bbox
        if bg_color is not None:
            draw.rectangle(
                (x0 - pad[0], y0 - pad[1], x1 + pad[0], y1 + pad[1]),
                fill=bg_color,
            )
        draw.multiline_text(xy, text, font=font, fill=(color[2], color[1], color[0]), spacing=4)
    else:
        bbox = draw.textbbox(xy, text, font=font)
        x0, y0, x1, y1 = bbox
        if bg_color is not None:
            draw.rectangle(
                (x0 - pad[0], y0 - pad[1], x1 + pad[0], y1 + pad[1]),
                fill=bg_color,
            )
        draw.text(xy, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ═══════════════════════════════════════════════════════════
# 卷积 Embedding 热力图
def _load_cnn_model(device: torch.device):
    """加载轻量卷积 embedding 模型，使用反卷积上采样输出平滑热力图"""
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


def _conv_heatmap(frame_bgr: np.ndarray,
                  device: torch.device = torch.device("cpu")) -> np.ndarray:
    """
    计算卷积 embedding 热力图，颜色反转（高值=蓝，低值=红）。
    返回 BGR 图像。
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    model, transform = _load_cnn_model(device)
    img_tensor = transform(rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(img_tensor)  # [B, 28, 28]
        heat = feat.squeeze().cpu().numpy()

    # 归一化到 0-255
    heat_min = heat.min()
    heat_max = heat.max()
    heat_norm = (heat - heat_min) / (heat_max - heat_min + 1e-8) * 255.0
    heat_u8 = heat_norm.astype(np.uint8)

    # OpenCV 上采样到原图尺寸（双三次插值，天然平滑）
    h, w = frame_bgr.shape[:2]
    heat_resized = cv2.resize(heat_u8, (w, h), interpolation=cv2.INTER_CUBIC)

    # 单次轻量高斯模糊，平滑残留锯齿
    k = max(5, (min(h, w) // 80) | 1)
    heat_smooth = cv2.GaussianBlur(heat_resized, (k, k), 0)

    # 颜色反转：高值→蓝（冷色），低值→红（暖色）
    heat_color = cv2.applyColorMap(255 - heat_smooth, cv2.COLORMAP_JET)
    return heat_color


# ----- 通用 HTTP 辅助 -----
async def get_client() -> httpx.AsyncClient:
    global _async_client
    if _async_client is None:
        _async_client = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=10),
            timeout=httpx.Timeout(3.0, connect=5.0),
        )
    return _async_client


async def _call_api(frame: np.ndarray, url: str) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    success, jpg_bytes = cv2.imencode(
        ".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 90]
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
            await asyncio.sleep(0.3)
    try:
        rec_img = Image.open(BytesIO(resp.content)).convert("RGB")
        out = np.array(rec_img)
        if out.shape[:2] != frame.shape[:2]:
            out = cv2.resize(out, (frame.shape[1], frame.shape[0]))
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"⚠️ 解析返回图像失败: {url} -> {e}", flush=True)
        return frame


async def action_1_frame(frame: np.ndarray) -> np.ndarray:
    return await _call_api(frame, API_URL_1)


async def action_2_frame(frame: np.ndarray) -> np.ndarray:
    return await _call_api(frame, API_URL_2)


# ----- ffmpeg writer -----
def _make_writer(output: str, w: int, h: int, fps: float) -> "_FFmpegWriter":
    is_stream = output.startswith(("rtmp://", "rtsp://", "srt://", "udp://"))
    if is_stream:
        return _FFmpegWriter(
            output, w, h, fps,
            preset="ultrafast", tune="zerolatency",
            crf=23, gop=2, container="flv",
        )
    return _FFmpegWriter(
        output, w, h, fps,
        preset="medium", tune=None,
        crf=18, gop=None, container=None,
    )


class _FFmpegWriter:
    def __init__(self, output: str, w: int, h: int, fps: float, *,
                 preset: str, tune: Optional[str], crf: int,
                 gop: Optional[int], container: Optional[str]):
        cmd = [
            "ffmpeg", "-y", "-loglevel", "warning",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.3f}", "-i", "-",
            "-c:v", "libx264", "-preset", preset,
        ]
        if tune:
            cmd += ["-tune", tune]
        if gop:
            cmd += ["-g", str(gop), "-keyint_min", str(gop)]
        cmd += ["-crf", str(crf), "-pix_fmt", "yuv420p",
                "-max_muxing_queue_size", "1024"]
        if container:
            cmd += ["-f", container]
        else:
            cmd += ["-movflags", "+faststart"]
        cmd += [output]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._alive = True

    def write(self, frame: np.ndarray):
        if not self._alive:
            return
        try:
            self.proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError):
            self._alive = False

    def close(self):
        if self.proc.stdin:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=3)


# ----- 流模式 -----
def _open_rtsp(url: str) -> cv2.VideoCapture:
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap

async def _read_frame(cap: cv2.VideoCapture, timeout: float = 15.0) -> tuple[bool, np.ndarray | None]:
    """带超时的 cap.read() 封装"""
    return await asyncio.wait_for(
        asyncio.to_thread(cap.read),
        timeout=timeout,
    )


async def _stream_once(rtsp_url: str, rtmp_base: str,
                      enable_heatmap: bool = True):
    cap = _open_rtsp(rtsp_url)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开 RTSP: {rtsp_url}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_fps = FORCE_FPS if FORCE_FPS else src_fps
    frame_interval = 1.0 / target_fps
    if w == 0 or h == 0:
        for _ in range(10):
            ok, tmp = cap.read()
            if ok and tmp is not None:
                w, h = tmp.shape[1], tmp.shape[0]
                break

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if enable_heatmap:
        _load_cnn_model(device)
        print(f"🧠 卷积 embedding 设备: {device}, 轻量CNN+OpenCV双三次上采样")

    writers = {
        "orig": _make_writer(f"{rtmp_base}/original", w, h, target_fps),
        "action1": _make_writer(f"{rtmp_base}/action1", w, h, target_fps),
        "action2": _make_writer(f"{rtmp_base}/action2", w, h, target_fps),
        "combined": _make_writer(f"{rtmp_base}/combined", w * 2, h * 2, target_fps),
    }
    t_window_start = time.perf_counter()
    frames_in_window = 0
    connect_at = time.perf_counter()
    last_push_time = connect_at - frame_interval
    try:
        while True:
            try:
                ok, frame = await _read_frame(cap, timeout=15.0)
            except asyncio.TimeoutError:
                raise RuntimeError("RTSP 读帧超时 (15s)")
            if not ok:
                raise RuntimeError("RTSP 读帧失败")
            now = time.perf_counter()
            if now - last_push_time < frame_interval:
                continue

            if enable_heatmap:
                result1, result2, heatmap = await asyncio.gather(
                    action_1_frame(frame),
                    action_2_frame(frame),
                    asyncio.to_thread(_conv_heatmap, frame, device),
                )
                top_row = np.hstack([frame, heatmap])
            else:
                result1, result2 = await asyncio.gather(
                    action_1_frame(frame),
                    action_2_frame(frame),
                )
                pad_l = np.zeros((h, w // 2, 3), dtype=np.uint8)
                pad_r = np.zeros((h, w - w // 2, 3), dtype=np.uint8)
                top_row = np.hstack([pad_l, frame, pad_r])

            bot_row = np.hstack([result1, result2])
            combined = np.vstack([top_row, bot_row])

            if enable_heatmap:
                combined = _put_text_cn(combined, "原始视频（云端无法获取，仅用于展厅展示）", (10, -10), size=128, bg_color=(180, 0, 0))
                combined = _put_text_cn(combined, "隐私相机传递给云端的词元流", (w + 10, -10), size=128, bg_color=(180, 0, 0))
            else:
                combined = _put_text_cn(combined, "原始视频（云端无法获取，仅用于展厅展示）", (w // 2 + 10, -10), size=128, bg_color=(180, 0, 0))
            combined = _put_text_cn(combined, "窃听者基于词元流重建的画面\n（无隐私保护机制）", (10, h - 10), size=128, bg_color=(180, 0, 0))
            combined = _put_text_cn(combined, "窃听者基于词元流重建的画面\n（有隐私保护机制）", (w + 10, h - 10), size=128, bg_color=(180, 0, 0))

            await asyncio.gather(
                asyncio.to_thread(writers["orig"].write, frame),
                asyncio.to_thread(writers["action1"].write, result1),
                asyncio.to_thread(writers["action2"].write, result2),
                asyncio.to_thread(writers["combined"].write, combined),
            )
            if not all(w._alive for w in writers.values()):
                print("⚠️ ffmpeg 推流断开，准备重连")
                break
            last_push_time = time.perf_counter()
            frames_in_window += 1
            elapsed = last_push_time - t_window_start
            if elapsed >= 5.0:
                actual_fps = frames_in_window / elapsed
                print(f"   {time.perf_counter() - connect_at:6.1f}s | "
                      f"{frames_in_window} 帧, 实际 {actual_fps:.1f}fps")
                t_window_start = last_push_time
                frames_in_window = 0
    finally:
        cap.release()
        for w in writers.values():
            w.close()


async def run_stream(rtsp_url: str, rtmp_base: str,
                     enable_heatmap: bool = True):
    backoff = 2.0
    while True:
        try:
            await _stream_once(rtsp_url, rtmp_base,
                               enable_heatmap=enable_heatmap)
            backoff = 2.0
        except KeyboardInterrupt:
            print("\n中断")
            return
        except Exception as e:
            print(f"⚠️ 流中断: {e}, {backoff}s 后重连")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


# ----- 文件模式 -----
async def run_file(input_path: str, output_dir: str,
                   enable_heatmap: bool = True):
    inp = Path(input_path)
    if not inp.exists():
        print(f"❌ 输入文件不存在: {inp}")
        sys.exit(1)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = inp.stem

    cap = cv2.VideoCapture(str(inp))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_fps = FORCE_FPS if FORCE_FPS else src_fps
    skip = round(src_fps / target_fps) if (FORCE_FPS and FORCE_FPS < src_fps) else 1

    selected_frames, selected_indices = [], []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if (idx - 1) % skip == 0:
            selected_frames.append(frame.copy())
            selected_indices.append(idx)
    cap.release()
    written_frames = len(selected_frames)
    if written_frames == 0:
        print("❌ 没有输出帧，退出。")
        return

    print(f"📥 输入: {inp}")
    print(f"   {w}x{h}  源帧率: {src_fps:.2f}fps  输出帧率: {target_fps:.2f}fps")
    if skip > 1:
        print(f"   采样间隔: 每 {skip} 帧输出 1 帧")
    print(f"   源总帧数: {n_frames}, 输出帧数: {written_frames}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if enable_heatmap:
        _load_cnn_model(device)
        print(f"🧠 卷积 embedding 设备: {device}, 轻量CNN+OpenCV双三次上采样")
    else:
        print("ℹ️  热力图已禁用")

    try:
        _load_cn_font(36)
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)

    paths = {
        "orig":     out_dir / f"{stem}_1_original.mp4",
        "action1":  out_dir / f"{stem}_2_action1.mp4",
        "action2":  out_dir / f"{stem}_3_action2.mp4",
        "combined": out_dir / f"{stem}_4_combined.mp4",
    }
    cw, ch = (w * 2, h * 2)
    writers = {
        "orig":     _make_writer(str(paths["orig"]), w, h, target_fps),
        "action1":  _make_writer(str(paths["action1"]), w, h, target_fps),
        "action2":  _make_writer(str(paths["action2"]), w, h, target_fps),
        "combined": _make_writer(str(paths["combined"]), cw, ch, target_fps),
    }

    frame_times = []
    print("🚀 开始逐帧处理 + API 调用 ...")
    for i, (frame, gidx) in enumerate(zip(selected_frames, selected_indices)):
        t0 = time.perf_counter()

        if enable_heatmap:
            r1, r2, heatmap = await asyncio.gather(
                action_1_frame(frame),
                action_2_frame(frame),
                asyncio.to_thread(_conv_heatmap, frame, device),
            )
            top_row = np.hstack([frame, heatmap])
        else:
            r1, r2 = await asyncio.gather(
                action_1_frame(frame),
                action_2_frame(frame),
            )
            pad_l = np.zeros((h, w // 2, 3), dtype=np.uint8)
            pad_r = np.zeros((h, w - w // 2, 3), dtype=np.uint8)
            top_row = np.hstack([pad_l, frame, pad_r])

        bot_row = np.hstack([r1, r2])
        combined = np.vstack([top_row, bot_row])

        if enable_heatmap:
            combined = _put_text_cn(combined, "原始视频（云端无法获取，仅用于展厅展示）", (20, 10), size=48, bg_color=(180, 0, 0))
            combined = _put_text_cn(combined, "隐私相机传递给云端的词元流", (w + 20, 10), size=48, bg_color=(180, 0, 0))
        else:
            combined = _put_text_cn(combined, "原始视频（云端无法获取，仅用于展厅展示）", (w // 2 + 20, 10), size=48, bg_color=(180, 0, 0))
        combined = _put_text_cn(combined, "窃听者基于词元流重建的画面\n（无隐私保护机制）", (20, h + 10), size=48, bg_color=(180, 0, 0))
        combined = _put_text_cn(combined, "窃听者基于词元流重建的画面\n（有隐私保护机制）", (w + 20, h + 10), size=48, bg_color=(180, 0, 0))

        await asyncio.gather(
            asyncio.to_thread(writers["orig"].write, frame),
            asyncio.to_thread(writers["action1"].write, r1),
            asyncio.to_thread(writers["action2"].write, r2),
            asyncio.to_thread(writers["combined"].write, combined),
        )
        frame_times.append(time.perf_counter() - t0)
        if i % max(1, written_frames // 10) == 0:
            print(f"\r   {i + 1}/{written_frames} "
                  f"({(i + 1) / written_frames * 100:.1f}%)",
                  end="", flush=True)
    print()
    for w in writers.values():
        w.close()

    if frame_times:
        ts = np.array(frame_times)
        avg_ms = ts.mean() * 1000
        p50 = np.percentile(ts, 50) * 1000
        p95 = np.percentile(ts, 95) * 1000
        target_duration = written_frames / target_fps if target_fps > 0 else 0.0
        print(f"\n📊 性能统计:")
        print(f"   输出帧数:       {written_frames}")
        print(f"   输出视频时长:   {target_duration:.2f}s")
        print(f"   每帧平均耗时:   {avg_ms:.1f}ms")
        print(f"   帧延迟:         p50={p50:.1f}ms  p95={p95:.1f}ms")
    print("📁 输出文件:")
    for p in paths.values():
        if p.exists():
            size_mb = p.stat().st_size / 1024 / 1024
            print(f"   • {p.name:40s}  {size_mb:8.2f} MB")


async def _cleanup():
    global _async_client
    if _async_client:
        await _async_client.aclose()
        _async_client = None


def main():
    global FORCE_FPS
    p = argparse.ArgumentParser(
        description="视频四路处理 (含卷积 embedding 热力图)"
    )
    sp = p.add_subparsers(dest="mode", required=True)

    pf = sp.add_parser("file")
    pf.add_argument("input")
    pf.add_argument("-o", "--output", default="./output")
    pf.add_argument("--fps", type=float)
    pf.add_argument("--no-heatmap", action="store_true")

    ps = sp.add_parser("stream")
    ps.add_argument("--rtsp", default=os.environ.get("RTSP_URL", "rtmp://srs:1935/live/livestream"))
    ps.add_argument("--rtmp-base", default=os.environ.get("RTMP_BASE", "rtmp://srs:1935/live"))
    ps.add_argument("--fps", default=1, type=float)
    ps.add_argument("--no-heatmap", action="store_true")

    args = p.parse_args()
    FORCE_FPS = args.fps

    async def run():
        try:
            if args.mode == "file":
                await run_file(
                    args.input, args.output,
                    enable_heatmap=not args.no_heatmap,
                )
            else:
                if not args.rtsp:
                    print("❌ stream 模式需要 --rtsp", file=sys.stderr)
                    sys.exit(1)
                await run_stream(args.rtsp, args.rtmp_base,
                                 enable_heatmap=not args.no_heatmap)
        finally:
            await _cleanup()

    asyncio.run(run())


if __name__ == "__main__":
    main()
