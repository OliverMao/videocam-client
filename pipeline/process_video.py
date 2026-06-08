#!/usr/bin/env python3
"""
实时视频流四路同步处理 → 推送到 SRS
====================================

两种运行模式:
  python pipeline.py file  <input.mp4>             离线文件模式
  python pipeline.py stream                        RTSP → 4 路 RTMP 推流

文件模式输出:
  out_1_original.mp4 / out_2_blur.mp4 / out_3_noise.mp4 / out_4_combined.mp4
  + 每帧平均耗时 / p50/p95/p99 / 实时率统计

流模式输出(RTMP):
  {RTMP_BASE}/original / blur / noise / combined
  + 断流自动重连(指数退避) + 周期性打点

action 可替换
-------------
Section 1 是你要改的地方。两个接口:

    帧级:
        def action_1_frame(frame: np.ndarray, kernel: int = 31) -> np.ndarray
        def action_2_frame(frame: np.ndarray, strength: int = 30) -> np.ndarray

    视频级(可独立调用,文件模式用):
        def action_1(input_video: str, output_video: str, kernel: int = 31) -> None
        def action_2(input_video: str, output_video: str, strength: int = 30) -> None
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


# ╔════════════════════════════════════════════════════════════════════════╗
# ║  Section 1: action 函数区 —— 后续要替换就改这里                       ║
# ╚════════════════════════════════════════════════════════════════════════╝

def action_1_frame(frame: np.ndarray, kernel: int = 31) -> np.ndarray:
    """action_1 帧级:高斯模糊(OpenCV)。"""
    if kernel < 1:
        kernel = 1
    if kernel % 2 == 0:
        kernel += 1
    return cv2.GaussianBlur(frame, (kernel, kernel), sigmaX=0)


def action_2_frame(frame: np.ndarray, strength: int = 30) -> np.ndarray:
    """action_2 帧级:高斯噪声(OpenCV)。"""
    if strength < 1:
        strength = 1
    noise = np.random.normal(0, strength, frame.shape).astype(np.int16)
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def action_1(input_video: str, output_video: str, kernel: int = 31) -> None:
    """action_1 视频级(独立可调用)。"""
    _process_video_solo(input_video, output_video,
                        lambda f: action_1_frame(f, kernel))


def action_2(input_video: str, output_video: str, strength: int = 30) -> None:
    """action_2 视频级(独立可调用)。"""
    _process_video_solo(input_video, output_video,
                        lambda f: action_2_frame(f, strength))


def _process_video_solo(input_video, output_video, frame_fn):
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = _make_writer(output_video, w, h, fps)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame_fn(frame))
    finally:
        cap.release()
        writer.close()


# ╔════════════════════════════════════════════════════════════════════════╗
# ║  Section 2: ffmpeg writer(支持文件 / RTMP / RTSP)                      ║
# ╚════════════════════════════════════════════════════════════════════════╝

def _make_writer(output: str, w: int, h: int, fps: float) -> "_FFmpegWriter":
    """
    根据 output 协议自动选择编码参数:
      - rtmp://... / rtsp://...  → 流式(ultrafast + zerolatency + flv)
      - 其他                     → 文件(medium + crf 18 + faststart)
    """
    is_stream = output.startswith(("rtmp://", "rtsp://", "srt://", "udp://"))
    if is_stream:
        return _FFmpegWriter(
            output, w, h, fps,
            preset="ultrafast",
            tune="zerolatency",
            crf=23,
            gop=30,                       # GOP = 1s(假设 30fps)
            container="flv",
        )
    return _FFmpegWriter(
        output, w, h, fps,
        preset="medium",
        tune=None,
        crf=18,
        gop=None,
        container=None,
    )


class _FFmpegWriter:
    """rawvideo BGR24 -> ffmpeg 子进程 -> mp4/flv。"""

    def __init__(
        self, output: str, w: int, h: int, fps: float, *,
        preset: str, tune: str | None, crf: int,
        gop: int | None, container: str | None,
    ):
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}",
            "-r", f"{fps:.3f}",
            "-i", "-",
            "-c:v", "libx264",
            "-preset", preset,
        ]
        if tune:
            cmd += ["-tune", tune]
        if gop:
            cmd += ["-g", str(gop), "-keyint_min", str(gop)]
        cmd += [
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
        ]
        if container:
            cmd += ["-f", container]
        else:
            cmd += ["-movflags", "+faststart"]
        cmd += [output]

        # 重要:stderr 必须走 DEVNULL,不然 ffmpeg 进度把 pipe buffer 撑满
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self.proc.stdin:
            try:
                self.proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        try:
            ret = self.proc.wait(timeout=10)
            if ret != 0:
                raise RuntimeError(f"ffmpeg 退出码 {ret}")
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)
            raise RuntimeError("ffmpeg 收尾超时,已 kill")


# ╔════════════════════════════════════════════════════════════════════════╗
# ║  Section 3: 实时流 pipeline(RTSP → 4 路 RTMP)                          ║
# ╚════════════════════════════════════════════════════════════════════════╝

def _open_rtsp(url: str) -> cv2.VideoCapture:
    """打开 RTSP,优先走 FFMPEG 后端 + TCP + 1 帧缓冲(降延迟)。"""
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


def run_stream(
    rtsp_url: str,
    rtmp_base: str,
    blur_kernel: int = 31,
    noise_strength: int = 30,
    reconnect_initial: float = 2.0,
    reconnect_max: float = 30.0,
) -> None:
    """
    主循环:RTSP 拉流 → 逐帧处理 → 4 路 RTMP 推送。
    断流自动指数退避重连,Ctrl+C 退出。
    """
    backoff = reconnect_initial
    while True:
        try:
            _stream_once(rtsp_url, rtmp_base, blur_kernel, noise_strength)
            backoff = reconnect_initial  # 成功运行过后重置退避
        except KeyboardInterrupt:
            print("\n👋 用户中断,退出", flush=True)
            return
        except Exception as e:
            print(f"⚠️  流中断: {type(e).__name__}: {e}", flush=True)
            print(f"   {backoff:.1f}s 后重连 ...", flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, reconnect_max)


def _stream_once(rtsp_url, rtmp_base, blur_kernel, noise_strength):
    cap = _open_rtsp(rtsp_url)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开 RTSP: {rtsp_url}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 容错:某些 RTSP 服务器第一帧返回 0x0,等几帧再读
    if w == 0 or h == 0:
        for _ in range(10):
            ok, tmp = cap.read()
            if ok and tmp is not None:
                w = tmp.shape[1]
                h = tmp.shape[0]
                break
        if w == 0 or h == 0:
            cap.release()
            raise RuntimeError("RTSP 无法读到有效分辨率")

    print(f"📥 RTSP 已连接: {rtsp_url}", flush=True)
    print(f"   {w}x{h}  {fps:.2f}fps", flush=True)
    print(f"🚀 启动 4 路推流 → {rtmp_base}/", flush=True)

    writers = {
        "orig":     _make_writer(f"{rtmp_base}/original", w, h,   fps),
        "blur":     _make_writer(f"{rtmp_base}/blur",     w, h,   fps),
        "noise":    _make_writer(f"{rtmp_base}/noise",    w, h,   fps),
        "combined": _make_writer(f"{rtmp_base}/combined", w, h*3, fps),
    }

    # 性能打点
    t_window_start = time.perf_counter()
    frames_in_window = 0
    connect_at = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("RTSP 读帧失败(可能断流)")

            blur_f   = action_1_frame(frame, blur_kernel)
            noise_f  = action_2_frame(frame, noise_strength)
            combined = np.vstack([frame, blur_f, noise_f])

            writers["orig"].write(frame)
            writers["blur"].write(blur_f)
            writers["noise"].write(noise_f)
            writers["combined"].write(combined)

            frames_in_window += 1
            now = time.perf_counter()
            elapsed_window = now - t_window_start
            if elapsed_window >= 5.0:
                actual_fps = frames_in_window / elapsed_window
                rt = actual_fps / fps if fps > 0 else 0
                rt_pct = rt * 100
                runtime = now - connect_at
                print(
                    f"   ⏱  {runtime:6.1f}s | "
                    f"5s 内 {frames_in_window} 帧, "
                    f"实际 {actual_fps:5.1f}fps / 源 {fps:.1f}fps "
                    f"({rt_pct:5.1f}% realtime)",
                    flush=True,
                )
                t_window_start = now
                frames_in_window = 0
    finally:
        cap.release()
        for w_ in writers.values():
            try:
                w_.close()
            except Exception:
                pass


# ╔════════════════════════════════════════════════════════════════════════╗
# ║  Section 4: 文件模式 pipeline(离线 / 调试用)                           ║
# ╚════════════════════════════════════════════════════════════════════════╝

def run_file(input_path: str, output_dir: str,
             blur_kernel: int = 31, noise_strength: int = 30) -> None:
    inp = Path(input_path)
    if not inp.exists():
        print(f"❌ 输入文件不存在: {inp}", file=sys.stderr)
        sys.exit(1)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = inp.stem

    paths = {
        "orig":     out_dir / f"{stem}_1_original.mp4",
        "blur":     out_dir / f"{stem}_2_blur.mp4",
        "noise":    out_dir / f"{stem}_3_noise.mp4",
        "combined": out_dir / f"{stem}_4_combined.mp4",
    }

    cap = cv2.VideoCapture(str(inp))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_duration = n_frames / fps if fps > 0 else 0.0

    print(f"📥 输入: {inp}")
    print(f"   {w}x{h}  {fps:.2f}fps  {n_frames} 帧  {src_duration:.2f}s")
    print()

    writers = {
        "orig":     _make_writer(str(paths["orig"]),     w, h,   fps),
        "blur":     _make_writer(str(paths["blur"]),     w, h,   fps),
        "noise":    _make_writer(str(paths["noise"]),    w, h,   fps),
        "combined": _make_writer(str(paths["combined"]), w, h*3, fps),
    }

    frame_times: list[float] = []
    t_wall = time.perf_counter()
    frame_idx = 0
    log_step = max(1, n_frames // 20)

    print(f"🚀 逐帧处理中 (blur_kernel={blur_kernel}, noise_strength={noise_strength}) ...")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            blur_f   = action_1_frame(frame, blur_kernel)
            noise_f  = action_2_frame(frame, noise_strength)
            combined = np.vstack([frame, blur_f, noise_f])
            writers["orig"].write(frame)
            writers["blur"].write(blur_f)
            writers["noise"].write(noise_f)
            writers["combined"].write(combined)
            frame_times.append(time.perf_counter() - t0)
            frame_idx += 1
            if frame_idx % log_step == 0 or frame_idx == n_frames:
                pct = frame_idx / max(1, n_frames) * 100
                print(f"\r   {frame_idx}/{n_frames} ({pct:5.1f}%)", end="", flush=True)
    finally:
        cap.release()
        for w_ in writers.values():
            w_.close()
    t_wall = time.perf_counter() - t_wall
    print("\n")

    if frame_times:
        times = np.asarray(frame_times)
        avg_ms = float(times.mean() * 1000)
        p50 = float(np.percentile(times, 50) * 1000)
        p95 = float(np.percentile(times, 95) * 1000)
        p99 = float(np.percentile(times, 99) * 1000)
        real_time_ratio = src_duration / t_wall if t_wall > 0 else 0.0
        print("📊 性能统计:")
        print(f"   处理帧数:       {frame_idx}")
        print(f"   源时长:         {src_duration:.2f}s")
        print(f"   墙钟时间:       {t_wall:.2f}s")
        print(f"   每帧平均耗时:   {avg_ms:.2f}ms")
        print(f"   帧延迟分位:     p50={p50:.2f}ms  p95={p95:.2f}ms  p99={p99:.2f}ms")
        print(f"   实时率:         {real_time_ratio:.2f}x  "
              f"({'超实时' if real_time_ratio >= 1 else '低于实时'})")
        print()

    print("📁 输出文件:")
    for p in paths.values():
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"   • {p.name:40s}  {size_mb:8.2f} MB")


# ╔════════════════════════════════════════════════════════════════════════╗
# ║  Section 5: 入口                                                       ║
# ╚════════════════════════════════════════════════════════════════════════╝

def main() -> None:
    p = argparse.ArgumentParser(
        description="视频四路同步处理(文件模式 / 实时流模式)",
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # ---- file 子命令 ----
    pf = sub.add_parser("file", help="离线处理文件,输出 4 个 mp4")
    pf.add_argument("input", help="输入视频文件路径")
    pf.add_argument("-o", "--output", default="./output")
    pf.add_argument("--blur", type=int, default=31,
                    help="高斯模糊 kernel size(奇数,默认 31)")
    pf.add_argument("--noise", type=int, default=30,
                    help="高斯噪声标准差(默认 30)")

    # ---- stream 子命令 ----
    ps = sub.add_parser("stream", help="实时模式:RTSP 拉流,4 路 RTMP 推送")
    ps.add_argument("--rtsp", default=os.environ.get("RTSP_URL"),
                    help="RTSP 输入 URL(默认读 RTSP_URL 环境变量)")
    ps.add_argument("--rtmp-base",
                    default=os.environ.get("RTMP_BASE", "rtmp://srs:1935/live"),
                    help="RTMP 推流基础 URL(默认读 RTMP_BASE 环境变量)")
    ps.add_argument("--blur", type=int, default=31)
    ps.add_argument("--noise", type=int, default=30)

    args = p.parse_args()

    if args.mode == "file":
        run_file(args.input, args.output, args.blur, args.noise)
    elif args.mode == "stream":
        if not args.rtsp:
            print("❌ stream 模式需要 --rtsp 或环境变量 RTSP_URL", file=sys.stderr)
            sys.exit(1)
        run_stream(args.rtsp, args.rtmp_base, args.blur, args.noise)


if __name__ == "__main__":
    main()
