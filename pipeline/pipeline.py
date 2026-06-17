#!/usr/bin/env python3
"""
视频处理 → 调用 action1 推流
"""
from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from api_client import action_1_frame, cleanup as api_cleanup

# 路径与常量
FORCE_FPS: float | None = None


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


def _open_rtsp(url: str) -> cv2.VideoCapture:
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


async def _read_frame(cap: cv2.VideoCapture, timeout: float = 15.0) -> tuple[bool, np.ndarray | None]:
    return await asyncio.wait_for(
        asyncio.to_thread(cap.read),
        timeout=timeout,
    )


async def _stream_once(rtsp_url: str, rtmp_base: str):
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

    writer = _FFmpegWriter(
        f"{rtmp_base}/combined", w, h, target_fps,
        preset="ultrafast", tune="zerolatency",
        crf=23, gop=2, container="flv",
    )

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
            result1 = await action_1_frame(frame)
            t_api = time.perf_counter() - t_perf

            await asyncio.to_thread(writer.write, result1)

            if not writer._alive:
                print("⚠️ ffmpeg 推流断开，准备重连")
                break

            last_push_time = time.perf_counter()
            frames_in_window += 1
            elapsed = last_push_time - t_window_start
            if elapsed >= 5.0:
                actual_fps = frames_in_window / elapsed
                print(f"   {time.perf_counter() - connect_at:6.1f}s | "
                      f"{frames_in_window} 帧, 实际 {actual_fps:.1f}fps | "
                      f"API1:{t_api*1000:.0f}ms")
                t_window_start = last_push_time
                frames_in_window = 0
    finally:
        cap.release()
        writer.close()


async def run_stream(rtsp_url: str, rtmp_base: str):
    backoff = 2.0
    while True:
        try:
            await _stream_once(rtsp_url, rtmp_base)
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
    p = argparse.ArgumentParser(description="视频处理 → 调用 action1 推流")
    p.add_argument("--rtsp", default=os.environ.get("RTSP_URL", "rtsp://127.0.0.1:8554/cam_main"))
    p.add_argument("--rtmp-base", default=os.environ.get("RTMP_BASE", "rtmp://127.0.0.1:1935/live "))
    p.add_argument("--fps", default=1, type=float)
    args = p.parse_args()
    FORCE_FPS = args.fps

    async def run():
        try:
            if not args.rtsp:
                print("❌ 需要 --rtsp", file=sys.stderr)
                sys.exit(1)
            await run_stream(args.rtsp, args.rtmp_base)
        finally:
            await _cleanup()

    asyncio.run(run())


if __name__ == "__main__":
    main()
