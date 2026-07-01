#!/usr/bin/env python3
"""
视频处理 → 调用 action1 推流（paced 节奏模式 + API 任务池）

关键设计：
  - 按 --fps 固定节奏触发 API 调用，不受 RTSP 帧率 / API 延迟影响
  - 维护 in-flight 任务池（默认 10，受 httpx 池自然限流）
  - **绝不取消** in-flight 任务：drain 取最大 seq 的成功结果写 latest_result
  - **单调推进** latest_result_seq：晚到的低 seq 不会覆盖已确认的较新结果，
    保证推流帧序绝不回退（API 返回速度不一时的关键不变量）
  - 推流与 API 完全解耦：按节奏取"最新可用结果"写 RTMP，
    首次/尚无结果时用原帧占位
  - 池满时 await 背压：等任意一个完成再发新的，避免无界堆积
  - API 慢于帧间隔时：推流频率 = target_fps 不掉帧；
    显示的画面会滞后若干拍（最新可用结果），但绝不会是原帧

用法：
    python3 stream_paced.py --fps 20 --rtsp rtsp://... --rtmp-base rtmp://...
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import time
from io import BytesIO
from typing import Optional

import cv2
import httpx
import numpy as np
from PIL import Image


# ====================== CONFIG ======================
GPU_HOST = os.environ.get("GPU_HOST", "116.238.240.2")
MAE_PORT = os.environ.get("MAE_PORT", "32061")
API_URL_1 = f"http://{GPU_HOST}:{MAE_PORT}/process"
# ====================================================


# ---------- HTTP 客户端 ----------
_async_client: httpx.AsyncClient | None = None


async def get_client() -> httpx.AsyncClient:
    global _async_client
    if _async_client is None:
        _async_client = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=10),
            timeout=httpx.Timeout(2.0, connect=1.0),
        )
    return _async_client


async def cleanup() -> None:
    global _async_client
    if _async_client is not None:
        await _async_client.aclose()
        _async_client = None


# ---------- API 调用 ----------
async def call_api(frame: np.ndarray, url: str) -> np.ndarray:
    t0 = time.perf_counter()
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return frame
    client = await get_client()
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            resp = await client.post(
                url, files={"file": ("frame.jpg", jpg.tobytes(), "image/jpeg")}
            )
            resp.raise_for_status()
            try:
                rec = Image.open(BytesIO(resp.content)).convert("RGB")
                out = cv2.cvtColor(np.array(rec), cv2.COLOR_RGB2BGR)
                if out.shape[:2] != frame.shape[:2]:
                    out = cv2.resize(out, (frame.shape[1], frame.shape[0]))
                print(f"[API] {int((time.perf_counter() - t0) * 1000)}ms", flush=True)
                return out
            except Exception as e:
                print(f"[解析失败] {e}", flush=True)
                return frame
        except Exception as e:
            last_err = e
            if attempt < 2:
                await asyncio.sleep(0.1)
    print(f"[API 失败] {last_err}", flush=True)
    return frame


async def action_1_frame(frame: np.ndarray) -> np.ndarray:
    return await call_api(frame, API_URL_1)


# ---------- FFmpeg 推流 ----------
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


# ---------- RTSP + 帧缓冲 ----------
def _open_rtsp(url: str) -> cv2.VideoCapture:
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


async def _read_frame(cap: cv2.VideoCapture, timeout: float = 15.0
                      ) -> tuple[bool, Optional[np.ndarray]]:
    return await asyncio.wait_for(
        asyncio.to_thread(cap.read),
        timeout=timeout,
    )


class _LatestFrame:
    """
    容量 1 的 latest-frame 缓冲（纯同步，asyncio 单线程下安全）：
      - reader 任务持续 put，覆盖旧帧
      - main paced 循环按节奏 take_if_new；如无新帧返回 None
    """
    __slots__ = ("_frame", "_has_new")

    def __init__(self):
        self._frame: Optional[np.ndarray] = None
        self._has_new = False

    def put(self, frame: np.ndarray) -> None:
        self._frame = frame
        self._has_new = True

    def take_if_new(self) -> Optional[np.ndarray]:
        if not self._has_new:
            return None
        self._has_new = False
        return self._frame

    def has_any(self) -> bool:
        return self._frame is not None


# ---------- paced 主循环 ----------
async def _stream_once(rtsp_url: str, rtmp_base: str,
                        target_fps: float, output_width: int):
    cap = _open_rtsp(rtsp_url)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开 RTSP: {rtsp_url}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w0 == 0 or h0 == 0:
        for _ in range(10):
            ok, tmp = cap.read()
            if ok and tmp is not None:
                w0, h0 = tmp.shape[1], tmp.shape[0]
                break

    # 输出尺寸
    out_w, out_h = w0, h0
    if output_width and output_width < w0:
        out_w = output_width
        out_h = int(h0 * output_width / w0)
        out_h -= out_h % 2  # H.264 yuv420p 要求偶数

    frame_interval = 1.0 / target_fps
    latest = _LatestFrame()
    stop_event = asyncio.Event()

    # API 任务池：解耦 API 调用与推流节奏。
    # 关键点：**不取消** in-flight 任务（取消会让 _drain 收不到结果，
    #         推流就会卡在原帧上）。允许多个并行，由 httpx 连接池
    #         (max_connections=10) 自然限流；超过上限时本循环 await
    #         背压，不会无界堆积。
    MAX_INFLIGHT = 10
    inflight: dict[int, asyncio.Task] = {}   # seq -> Task
    inflight_t0: dict[int, float] = {}        # seq -> start perf_counter
    latest_result: Optional[np.ndarray] = None
    latest_result_seq: int = -1
    beats: int = 0
    api_done: int = 0
    api_failed: int = 0
    api_latency_ms: float = 0.0               # 最近一次成功的 API 耗时

    async def reader():
        """后台持续把最新帧写入 latest；RTSP 异常时设置 stop_event。"""
        while not stop_event.is_set():
            try:
                ok, frame = await _read_frame(cap, timeout=15.0)
            except asyncio.TimeoutError:
                print("⚠️ RTSP 读帧超时 (15s)", flush=True)
                stop_event.set()
                return
            if not ok:
                print("⚠️ RTSP 读帧失败", flush=True)
                stop_event.set()
                return
            latest.put(frame)

    reader_task = asyncio.create_task(reader())

    writer = _FFmpegWriter(
        f"{rtmp_base}/combined", out_w, out_h, target_fps,
        preset="ultrafast", tune="zerolatency",
        crf=23, gop=2, container="flv",
    )

    # 等 reader 拿到第一帧再开始节拍
    while not latest.has_any() and not stop_event.is_set():
        await asyncio.sleep(0.01)

    def _resize(frame: np.ndarray) -> np.ndarray:
        if out_w != w0 or out_h != h0:
            return cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        return frame

    def _drain_done() -> None:
        """
        把已完成的 task 从 inflight 搬走；取最大 seq 的成功结果写到 latest_result。

        **单调性约束**：
          late-comers (seq 较小但完成更晚) 不能覆盖已确认的较新结果。
          推流拍基于 latest_result_seq 单调递增，画面绝不打回头。
          代价：偶尔跳过 1-2 拍（被更新的结果替代），但代价是 *顺序正确*。
        """
        nonlocal latest_result, latest_result_seq, api_latency_ms
        nonlocal api_done, api_failed
        best_seq = -1
        best_out: Optional[np.ndarray] = None
        best_latency = 0.0
        for seq, task in list(inflight.items()):
            if not task.done():
                continue
            inflight.pop(seq, None)
            t0 = inflight_t0.pop(seq, time.perf_counter())
            if task.cancelled():
                api_failed += 1
                continue
            try:
                out = task.result()
            except Exception:
                api_failed += 1
                continue
            api_done += 1
            latency = (time.perf_counter() - t0) * 1000
            # 与当前 latest_result_seq 保持单调推进；
            # 跳过 mid-call 的 "用 0 初始化导致回退" 旧 bug
            if seq > best_seq and seq > latest_result_seq:
                best_seq = seq
                best_out = out
                best_latency = latency
        if best_out is not None:
            latest_result = best_out
            latest_result_seq = best_seq
            api_latency_ms = best_latency

    connect_at = time.perf_counter()
    t_window = connect_at
    frames_in_window = 0
    skipped = 0
    next_time = time.perf_counter()  # 立即触发第一拍

    try:
        while not stop_event.is_set():
            now = time.perf_counter()
            wait = next_time - now
            if wait > 0:
                # 睡到下一拍；期间 reader 协程会跑更新 latest，
                # API 任务在 httpx 池里继续推进。
                await asyncio.sleep(wait)

            beats += 1
            next_time += frame_interval
            # 漂移保护：落后 >5 个间隔就重置
            if time.perf_counter() - next_time > frame_interval * 5:
                next_time = time.perf_counter() + frame_interval

            # 1) 收已完成的 API 结果（取最新 seq）
            _drain_done()

            # 2) 取最新帧，fire-and-forget 启动 API 调用
            frame = latest.take_if_new()

            # 背压：池满则 await 至少一个完成（也顺便 drain 一次）
            if frame is not None and len(inflight) >= MAX_INFLIGHT:
                # 等任意一个完成
                done, _pending = await asyncio.wait(
                    list(inflight.values()),
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=5.0,
                )
                _drain_done()

            if frame is not None:
                small = _resize(frame)
                inflight[beats] = asyncio.create_task(action_1_frame(small))
                inflight_t0[beats] = time.perf_counter()
            else:
                skipped += 1

            # 3) 推流：永远用 latest_result；首次/尚无结果时用最新原帧
            if latest_result is not None:
                push_frame = latest_result
            elif frame is not None:
                push_frame = _resize(frame)
            else:
                push_frame = None

            if push_frame is not None:
                await asyncio.to_thread(writer.write, push_frame)
                if not writer._alive:
                    print("⚠️ ffmpeg 推流断开，准备重连", flush=True)
                    break
                frames_in_window += 1

            now = time.perf_counter()
            if now - t_window >= 5.0:
                elapsed = now - t_window
                actual_fps = frames_in_window / elapsed
                print(f"   {now - connect_at:6.1f}s | "
                      f"推 {frames_in_window} 帧, 实际 {actual_fps:.1f}fps | "
                      f"节拍 {beats}, 在飞 {len(inflight)}, "
                      f"完成 {api_done}, 失败 {api_failed} | "
                      f"最新结果 seq={latest_result_seq} | "
                      f"API:{api_latency_ms:.0f}ms | "
                      f"跳 {skipped} 拍", flush=True)
                t_window = now
                frames_in_window = 0
                skipped = 0
    finally:
        stop_event.set()
        # 取消所有在飞 API 任务
        if inflight:
            for t in inflight.values():
                if not t.done():
                    t.cancel()
            await asyncio.gather(*inflight.values(), return_exceptions=True)
        try:
            await asyncio.wait_for(reader_task, timeout=2.0)
        except asyncio.TimeoutError:
            reader_task.cancel()
        cap.release()
        writer.close()


async def run_stream(rtsp_url: str, rtmp_base: str,
                      target_fps: float, output_width: int):
    backoff = 2.0
    while True:
        try:
            await _stream_once(rtsp_url, rtmp_base, target_fps, output_width)
            backoff = 2.0
        except KeyboardInterrupt:
            print("\n中断", flush=True)
            return
        except Exception as e:
            print(f"⚠️ 流中断: {e}, {backoff:.1f}s 后重连", flush=True)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


def main():
    p = argparse.ArgumentParser(description="视频处理 → 调用 action1 推流（paced 节奏模式）")
    p.add_argument("--rtsp", default=os.environ.get("RTSP_URL", "rtsp://127.0.0.1:8554/cam_main"),
                   help="RTSP 源地址")
    p.add_argument("--rtmp-base", default=os.environ.get("RTMP_BASE", "rtmp://127.0.0.1:1935/live").rstrip("/ "),
                   help="RTMP 推流基础地址（会拼上 /combined）")
    p.add_argument("--fps", default=10, type=float,
                   help="目标节拍 FPS（每 1/--fps 秒触发一次 API 调用）。默认 10")
    p.add_argument("--width", default=int(os.environ.get("WIDTH", "0")), type=int,
                   help="输出宽度，0=不缩放；推荐 640 省编码带宽")
    args = p.parse_args()

    async def run():
        try:
            await run_stream(args.rtsp, args.rtmp_base, args.fps, args.width)
        finally:
            await cleanup()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
