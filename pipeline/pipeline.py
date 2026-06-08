#!/usr/bin/env python3
"""
实时视频流四路同步处理 → 推流 / 输出文件
========================================

两种运行模式:
  python pipeline.py file  <input.mp4> [--fps 10]          离线文件模式（可强制帧率）
  python pipeline.py stream [--fps 10]                      RTSP → 4 路 RTMP 推流（可强制帧率）

当指定 --fps 时，输出帧率固定为此值。文件模式通过均匀跳帧实现；
流模式通过控制推送时间间隔实现（若源帧率更高则丢帧，实时维持目标帧率）。

action_1 调用远程 API: http://116.238.240.2:32101/reconstruct
action_2 调用远程 API: http://116.238.240.2:32101/action2   ← 请按实际端点修改
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
from typing import Callable

import cv2
import httpx
import numpy as np
from PIL import Image


# ╔════════════════════════════════════════════════════════════════════════╗
# ║  全局强制帧率（两种模式均生效）                                      ║
# ╚════════════════════════════════════════════════════════════════════════╝
FORCE_FPS: float | None = None


# ╔════════════════════════════════════════════════════════════════════════╗
# ║  Section 1: action 函数区（异步 API 调用，两个 action 完全一致）     ║
# ╚════════════════════════════════════════════════════════════════════════╝

API_URL_1 = "http://116.238.240.2:32101/reconstruct"
API_URL_2 = "http://116.238.240.2:32101/reconstruct"  # 请按实际端点修改

_async_client: httpx.AsyncClient | None = None


async def get_client() -> httpx.AsyncClient:
    global _async_client
    if _async_client is None:
        _async_client = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=10),
            timeout=httpx.Timeout(3.0, connect=5.0),
        )
    return _async_client


async def _call_api(frame: np.ndarray, url: str) -> np.ndarray:
    """通用 API 调用：BGR -> JPEG -> POST -> 解码 -> BGR"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    success, jpg_bytes = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        return frame

    client = await get_client()
    for attempt in range(3):
        try:
            resp = await client.post(
                url,
                files={"file": ("frame.jpg", jpg_bytes.tobytes(), "image/jpeg")},
            )
            resp.raise_for_status()
            break
        except Exception as e:
            if attempt == 2:
                print(f"⚠️ API 调用失败（已重试 3 次）: {url} -> {e}", flush=True)
                return frame
            await asyncio.sleep(0.3)

    try:
        rec_img = Image.open(BytesIO(resp.content)).convert("RGB")
        out = np.array(rec_img)
        if out.shape[:2] != frame.shape[:2]:
            out = cv2.resize(out, (frame.shape[1], frame.shape[0]))
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"⚠️ 无法解析 API 返回图像: {url} -> {e}", flush=True)
        return frame


async def action_1_frame(frame: np.ndarray) -> np.ndarray:
    return await _call_api(frame, API_URL_1)


async def action_2_frame(frame: np.ndarray) -> np.ndarray:
    return await _call_api(frame, API_URL_2)


async def action_1(input_video: str, output_video: str) -> None:
    await _process_video_solo(input_video, output_video, action_1_frame)


async def action_2(input_video: str, output_video: str) -> None:
    await _process_video_solo(input_video, output_video, action_2_frame)


async def _process_video_solo(input_video: str, output_video: str, frame_fn: Callable):
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = _make_writer(output_video, w, h, fps)
    try:
        while True:
            ok, frame = await asyncio.to_thread(cap.read)
            if not ok:
                break
            result = await frame_fn(frame)
            await asyncio.to_thread(writer.write, result)
    finally:
        cap.release()
        writer.close()


# ╔════════════════════════════════════════════════════════════════════════╗
# ║  Section 2: ffmpeg writer                                            ║
# ╚════════════════════════════════════════════════════════════════════════╝

def _make_writer(output: str, w: int, h: int, fps: float) -> "_FFmpegWriter":
    is_stream = output.startswith(("rtmp://", "rtsp://", "srt://", "udp://"))
    if is_stream:
        return _FFmpegWriter(
            output, w, h, fps,
            preset="ultrafast",
            tune="zerolatency",
            crf=23,
            gop=30,
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
    def __init__(self, output: str, w: int, h: int, fps: float, *,
                 preset: str, tune: str | None, crf: int,
                 gop: int | None, container: str | None):
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
# ║  Section 3: 实时流 pipeline（异步 + 强制帧率）                        ║
# ╚════════════════════════════════════════════════════════════════════════╝

def _open_rtsp(url: str) -> cv2.VideoCapture:
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


async def _stream_once(rtsp_url: str, rtmp_base: str) -> None:
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
            ok, tmp = await asyncio.to_thread(cap.read)
            if ok and tmp is not None:
                w, h = tmp.shape[1], tmp.shape[0]
                break
        if w == 0 or h == 0:
            cap.release()
            raise RuntimeError("RTSP 无法读到有效分辨率")

    print(f"📥 RTSP 已连接: {rtsp_url}")
    print(f"   {w}x{h}  源帧率: {src_fps:.2f}fps  输出帧率: {target_fps:.2f}fps")
    print(f"🚀 启动 4 路推流 → {rtmp_base}/")

    writers = {
        "orig":     _make_writer(f"{rtmp_base}/original", w, h,   target_fps),
        "action1":  _make_writer(f"{rtmp_base}/action1",  w, h,   target_fps),
        "action2":  _make_writer(f"{rtmp_base}/action2",  w, h,   target_fps),
        "combined": _make_writer(f"{rtmp_base}/combined", w*2, h*2, target_fps),
    }

    t_window_start = time.perf_counter()
    frames_in_window = 0
    connect_at = time.perf_counter()
    last_push_time = connect_at - frame_interval

    try:
        while True:
            ok, frame = await asyncio.to_thread(cap.read)
            if not ok:
                raise RuntimeError("RTSP 读帧失败")

            now = time.perf_counter()
            if now - last_push_time < frame_interval:
                continue

            # 调用两个 API
            start_time = time.time()
            result1 = await action_1_frame(frame)
            result2 = await action_2_frame(frame)
            cost_time = time.time() - start_time
            print(f"   API 调用耗时: {cost_time*1000:.1f}ms", flush=True)

            # 构建 combined 画面
            pad_l = np.zeros((h, w // 2, 3), dtype=np.uint8)
            pad_r = np.zeros((h, w - w // 2, 3), dtype=np.uint8)
            top = np.hstack([pad_l, frame, pad_r])
            bot = np.hstack([result1, result2])
            combined = np.vstack([top, bot])

            await asyncio.gather(
                asyncio.to_thread(writers["orig"].write, frame),
                asyncio.to_thread(writers["action1"].write, result1),
                asyncio.to_thread(writers["action2"].write, result2),
                asyncio.to_thread(writers["combined"].write, combined),
            )

            last_push_time = time.perf_counter()
            frames_in_window += 1

            elapsed_window = last_push_time - t_window_start
            if elapsed_window >= 5.0:
                actual_fps = frames_in_window / elapsed_window
                rt = actual_fps / target_fps if target_fps > 0 else 0
                runtime = last_push_time - connect_at
                print(f"   ⏱ {runtime:6.1f}s | 5s内 {frames_in_window} 帧, "
                      f"实际 {actual_fps:.1f}fps / 目标 {target_fps:.1f}fps "
                      f"({rt*100:.1f}% 实时)")
                t_window_start = last_push_time
                frames_in_window = 0
    finally:
        cap.release()
        for w in writers.values():
            try:
                w.close()
            except Exception:
                pass


async def run_stream(rtsp_url: str, rtmp_base: str,
                     reconnect_initial: float = 2.0,
                     reconnect_max: float = 30.0) -> None:
    backoff = reconnect_initial
    while True:
        try:
            await _stream_once(rtsp_url, rtmp_base)
            backoff = reconnect_initial
        except KeyboardInterrupt:
            print("\n👋 用户中断,退出")
            return
        except Exception as e:
            print(f"⚠️ 流中断: {type(e).__name__}: {e}")
            print(f"   {backoff:.1f}s 后重连 ...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, reconnect_max)


# ╔════════════════════════════════════════════════════════════════════════╗
# ║  Section 4: 文件模式 pipeline（异步 + 基于帧索引的强制帧率）        ║
# ╚════════════════════════════════════════════════════════════════════════╝

def _draw_label(img: np.ndarray, text: str, x: int, y: int) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (255, 255, 255), 2, cv2.LINE_AA)


async def run_file(input_path: str, output_dir: str) -> None:
    inp = Path(input_path)
    if not inp.exists():
        print(f"❌ 输入文件不存在: {inp}", file=sys.stderr)
        sys.exit(1)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = inp.stem

    paths = {
        "orig":     out_dir / f"{stem}_1_original.mp4",
        "action1":  out_dir / f"{stem}_2_action1.mp4",
        "action2":  out_dir / f"{stem}_3_action2.mp4",
        "combined": out_dir / f"{stem}_4_combined.mp4",
    }

    cap = cv2.VideoCapture(str(inp))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_duration = n_frames / src_fps if src_fps > 0 else 0.0

    target_fps = FORCE_FPS if FORCE_FPS else src_fps
    if FORCE_FPS and FORCE_FPS < src_fps:
        skip = round(src_fps / target_fps)
    else:
        skip = 1

    print(f"📥 输入: {inp}")
    print(f"   {w}x{h}  源帧率: {src_fps:.2f}fps  输出帧率: {target_fps:.2f}fps")
    if skip > 1:
        print(f"   采样间隔: 每 {skip} 帧输出 1 帧")
    print(f"   源总帧数: {n_frames}  源时长: {src_duration:.2f}s")
    print()

    writers = {
        "orig":     _make_writer(str(paths["orig"]),     w, h,   target_fps),
        "action1":  _make_writer(str(paths["action1"]),  w, h,   target_fps),
        "action2":  _make_writer(str(paths["action2"]),  w, h,   target_fps),
        "combined": _make_writer(str(paths["combined"]), w*2, h*2, target_fps),
    }

    frame_times: list[float] = []
    t_wall_start = time.perf_counter()
    frame_idx = 0
    written_frames = 0
    log_step = max(1, n_frames // 20)

    cost_time_list = []

    print("🚀 逐帧处理中 ...")
    try:
        while True:
            ok, frame = await asyncio.to_thread(cap.read)
            if not ok:
                break
            frame_idx += 1

            if (frame_idx - 1) % skip != 0:
                if frame_idx % 100 == 0:
                    pct = frame_idx / max(1, n_frames) * 100
                    print(f"\r   {frame_idx}/{n_frames} ({pct:5.1f}%) - 跳帧中...", end="", flush=True)
                continue

            t0 = time.perf_counter()
            start_time = time.time()
            result1 = await action_1_frame(frame)
            api_end1 = time.time()
            print(f"\r   {frame_idx}/{n_frames} ({(frame_idx/n_frames)*100:5.1f}%) - API1 耗时: {(api_end1 - start_time)*1000:.1f}ms", end="", flush=True)
            result2 = await action_2_frame(frame)
            api_end2 = time.time()
            print(f"\r   {frame_idx}/{n_frames} ({(frame_idx/n_frames)*100:5.1f}%) - API2 耗时: {(api_end2 - api_end1)*1000:.1f}ms", end="", flush=True)

            cost_time_list.append((api_end1 - start_time))
            cost_time_list.append((api_end2 - api_end1))

            pad_l = np.zeros((h, w // 2, 3), dtype=np.uint8)
            pad_r = np.zeros((h, w - w // 2, 3), dtype=np.uint8)
            top = np.hstack([pad_l, frame, pad_r])
            bot = np.hstack([result1, result2])
            combined = np.vstack([top, bot])

            _draw_label(combined, "Original",      w // 2 + 20, 40)
            _draw_label(combined, "Action1",       20,           h + 40)
            _draw_label(combined, "Action2",       w + 20,       h + 40)

            await asyncio.gather(
                asyncio.to_thread(writers["orig"].write, frame),
                asyncio.to_thread(writers["action1"].write, result1),
                asyncio.to_thread(writers["action2"].write, result2),
                asyncio.to_thread(writers["combined"].write, combined),
            )

            frame_times.append(time.perf_counter() - t0)
            written_frames += 1

            if frame_idx % log_step == 0 or frame_idx == n_frames:
                pct = frame_idx / max(1, n_frames) * 100
                print(f"\r   {frame_idx}/{n_frames} ({pct:5.1f}%)", end="", flush=True)
    finally:
        cap.release()
        for w in writers.values():
            w.close()
    t_wall = time.perf_counter() - t_wall_start
    print("\n")

    if frame_times:
        times = np.asarray(frame_times)
        avg_ms = float(times.mean() * 1000)
        p50 = float(np.percentile(times, 50) * 1000)
        p95 = float(np.percentile(times, 95) * 1000)
        p99 = float(np.percentile(times, 99) * 1000)
        target_duration = written_frames / target_fps if target_fps > 0 else 0.0
        real_time_ratio = target_duration / t_wall if t_wall > 0 else 0.0
        print("📊 性能统计:")
        print(f"   读取源帧数:     {frame_idx}")
        print(f"   实际输出帧数:   {written_frames}")
        print(f"   输出视频时长:   {target_duration:.2f}s")
        print(f"   墙钟处理时间:   {t_wall:.2f}s")
        print(f"   每帧平均耗时:   {avg_ms:.2f}ms (仅输出帧)")
        print(f"   帧延迟分位:     p50={p50:.2f}ms  p95={p95:.2f}ms  p99={p99:.2f}ms")
        print(f"   实时率:         {real_time_ratio:.2f}x  "
              f"({'超实时' if real_time_ratio >= 1 else '低于实时'})")
        print()
        print(f"平均每帧 API 调用耗时: {np.mean(cost_time_list)*1000:.1f}ms")

    print("📁 输出文件:")
    for p in paths.values():
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"   • {p.name:40s}  {size_mb:8.2f} MB")


# ╔════════════════════════════════════════════════════════════════════════╗
# ║  Section 5: 入口                                                     ║
# ╚════════════════════════════════════════════════════════════════════════╝

async def _cleanup():
    global _async_client
    if _async_client is not None:
        await _async_client.aclose()
        _async_client = None


def main() -> None:
    global FORCE_FPS

    p = argparse.ArgumentParser(
        description="视频四路同步处理 (action_1 与 action_2 均调用远程 API)"
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # file 子命令
    pf = sub.add_parser("file", help="离线处理文件,输出 4 个 mp4")
    pf.add_argument("input", help="输入视频文件路径")
    pf.add_argument("-o", "--output", default="./output")
    pf.add_argument("--fps", type=float, default=None,
                    help="强制输出帧率（按此帧率均匀采样源帧；不指定则保留源帧率）")

    # stream 子命令
    ps = sub.add_parser("stream", help="实时模式:RTSP 拉流,4 路 RTMP 推送")
    ps.add_argument("--rtsp", default=os.environ.get("RTSP_URL"),
                    help="RTSP 输入 URL(默认读 RTSP_URL 环境变量)")
    ps.add_argument("--rtmp-base",
                    default=os.environ.get("RTMP_BASE", "rtmp://srs:1935/live"),
                    help="RTMP 推流基础 URL")
    ps.add_argument("--fps", type=float, default=None,
                    help="强制输出帧率（高于源帧率时丢帧维持实时；不指定则使用源帧率）")

    args = p.parse_args()
    FORCE_FPS = args.fps

    async def _main_async():
        try:
            if args.mode == "file":
                await run_file(args.input, args.output)
            elif args.mode == "stream":
                if not args.rtsp:
                    print("❌ stream 模式需要 --rtsp 或环境变量 RTSP_URL", file=sys.stderr)
                    sys.exit(1)
                await run_stream(args.rtsp, args.rtmp_base)
        finally:
            await _cleanup()

    asyncio.run(_main_async())


if __name__ == "__main__":
    main()