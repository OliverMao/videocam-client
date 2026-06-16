#!/usr/bin/env python3
"""
视频四路同步处理 → 推流 (含卷积 embedding 热力图)
============================================================
实时 RTSP 推流

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
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch

from conv_heatmap import load_cnn_model, compute_heatmap
from api_client import action_1_frame, action_2_frame, cleanup as api_cleanup

# ═══════════════════════════════════════════════════════════
# 路径与常量
SCRIPT_DIR = Path(__file__).resolve().parent

FONT_CANDIDATES = [
    SCRIPT_DIR / "NotoSansCJKsc-Regular.otf",
]

FORCE_FPS: float | None = None

# ----- 字体全局单例 -----
_CN_FONT: Optional[ImageFont.FreeTypeFont] = None
_CN_FONT_PATH: Optional[Path] = None


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
    draw.rectangle(
        (x0 - pad[0], y0 - pad[1], x1 + pad[0], y1 + pad[1]),
        fill=bg_color,
    )
    if has_newline:
        draw.multiline_text(xy, text, font=font, fill=color, spacing=4)
    else:
        draw.text(xy, text, font=font, fill=color)


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
        load_cnn_model(device)
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

            t_perf = time.perf_counter()

            async def _timed_api(func, frame):
                t0 = time.perf_counter()
                r = await func(frame)
                return r, time.perf_counter() - t0

            def _timed_heatmap():
                t0 = time.perf_counter()
                h = compute_heatmap(frame, device)
                return h, time.perf_counter() - t0

            if enable_heatmap:
                (result1, t1), (result2, t2), (heatmap, t_heat) = await asyncio.gather(
                    _timed_api(action_1_frame, frame),
                    _timed_api(action_2_frame, frame),
                    asyncio.to_thread(_timed_heatmap),
                )
                t_api = time.perf_counter() - t_perf

                combined = np.empty((h * 2, w * 2, 3), dtype=np.uint8)
                combined[:h, :w] = frame
                combined[:h, w:] = heatmap
                combined[h:, :w] = result1
                combined[h:, w:] = result2
            else:
                (result1, t1), (result2, t2) = await asyncio.gather(
                    _timed_api(action_1_frame, frame),
                    _timed_api(action_2_frame, frame),
                )
                t_heat = 0
                t_api = time.perf_counter() - t_perf

                combined = np.empty((h * 2, w * 2, 3), dtype=np.uint8)
                combined[:h, w//2:w//2+w] = frame
                combined[h:, :w] = result1
                combined[h:, w:] = result2

            t_hstack = time.perf_counter()
            rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            draw = ImageDraw.Draw(pil)
            font = _load_cn_font(96)

            if enable_heatmap:
                _draw_text_bg(draw, "原始视频（云端无法获取，仅用于展厅展示）", (10, -10), font, (180, 0, 0))
                _draw_text_bg(draw, "隐私相机传递给云端的词元流", (w + 10, -10), font, (180, 0, 0))
            else:
                _draw_text_bg(draw, "原始视频（云端无法获取，仅用于展厅展示）", (w // 2 + 10, -10), font, (180, 0, 0))
            _draw_text_bg(draw, "窃听者基于词元流重建的画面\n（无隐私保护机制）", (10, h - 10), font, (180, 0, 0))
            _draw_text_bg(draw, "窃听者基于词元流重建的画面\n（有隐私保护机制）", (w + 10, h - 10), font, (180, 0, 0))
            combined = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            t_stitch = time.perf_counter() - t_hstack

            await asyncio.gather(
                asyncio.to_thread(writers["orig"].write, frame),
                asyncio.to_thread(writers["action1"].write, result1),
                asyncio.to_thread(writers["action2"].write, result2),
                asyncio.to_thread(writers["combined"].write, combined),
            )
            t_write = time.perf_counter() - t_stitch - t_hstack
            if not all(w._alive for w in writers.values()):
                print("⚠️ ffmpeg 推流断开，准备重连")
                break
            last_push_time = time.perf_counter()
            frames_in_window += 1
            elapsed = last_push_time - t_window_start
            if elapsed >= 5.0:
                actual_fps = frames_in_window / elapsed
                print(f"   {time.perf_counter() - connect_at:6.1f}s | "
                      f"{frames_in_window} 帧, 实际 {actual_fps:.1f}fps | "
                      f"API1:{t1*1000:.0f}ms API2:{t2*1000:.0f}ms Heat:{t_heat*1000:.0f}ms "
                      f"拼接:{t_stitch*1000:.0f}ms 写入:{t_write*1000:.0f}ms")
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


async def _cleanup():
    await api_cleanup()


def main():
    global FORCE_FPS
    p = argparse.ArgumentParser(
        description="视频四路推流处理 (含卷积 embedding 热力图)"
    )
    p.add_argument("--rtsp", default=os.environ.get("RTSP_URL", "rtmp://srs:1935/live/livestream"))
    p.add_argument("--rtmp-base", default=os.environ.get("RTMP_BASE", "rtmp://srs:1935/live"))
    p.add_argument("--fps", default=1, type=float)
    p.add_argument("--no-heatmap", action="store_true")

    args = p.parse_args()
    FORCE_FPS = args.fps

    async def run():
        try:
            if not args.rtsp:
                print("❌ 需要 --rtsp", file=sys.stderr)
                sys.exit(1)
            await run_stream(args.rtsp, args.rtmp_base,
                             enable_heatmap=not args.no_heatmap)
        finally:
            await _cleanup()

    asyncio.run(run())


if __name__ == "__main__":
    main()
