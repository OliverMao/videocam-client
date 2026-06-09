#!/usr/bin/env python3
"""
视频四路同步处理 → 推流 / 输出文件 (含 DINOv2 patch 特征可视化)
================================================================
两种运行模式:
  python pipeline.py file  <input.mp4> [--fps 10] [--no-dinov2] [--query center]
                                                                     # center | tl | tr | bl | br
  python pipeline.py stream [--fps 10]                                实时 RTSP 推流

文件模式下右上角显示 DINOv2 patch 相似度热力图：
  - 把每帧切成 14x14 的 patch（vit_small）
  - 提取 patch 特征向量
  - 选 query patch（默认中心），算它与所有 patch 的余弦相似度
  - 上采样 + JET 配色叠加到原图
  通过 --no-dinov2 可关闭热力图，恢复原布局。

action_1 与 action_2 均调用远程 API: http://116.238.240.2:32101/reconstruct

依赖资源（与本脚本同目录）：
  - dinov2_vits14.pth          DINOv2 vit_small 权重
  - NotoSansCJK-Regular.otf    思源黑体 / Noto Sans CJK（免费商用）
    （或 WenQuanYiMicroHei.ttf  文泉驿微米黑，免费商用）

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
import torchvision.transforms as T

# ═══════════════════════════════════════════════════════════
# 路径与常量
SCRIPT_DIR = Path(__file__).resolve().parent
DINOV2_WEIGHTS = SCRIPT_DIR / "dinov2_vits14_pretrain.pth"

# 字体候选（按优先级查找，找到第一个就停）
FONT_CANDIDATES = [
    SCRIPT_DIR / "NotoSansCJKsc-Regular.otf",   # 思源黑体 / Noto Sans CJK SC
]

FORCE_FPS: float | None = None
API_URL_1 = "http://116.238.240.2:32101/reconstruct"
API_URL_2 = "http://116.238.240.2:32101/reconstruct"
_async_client: httpx.AsyncClient | None = None

# ----- DINOv2 全局单例 -----
_DINOV2_MODEL = None
_DINOV2_TRANSFORM = None
# ----- 字体全局单例 -----
_CN_FONT: Optional[ImageFont.FreeTypeFont] = None
_CN_FONT_PATH: Optional[Path] = None


# ═══════════════════════════════════════════════════════════
# 字体加载
def _load_cn_font(size: int = 28) -> ImageFont.FreeTypeFont:
    """
    加载与脚本同目录的免费商用中文字体。
    优先：思源黑体(NotoSansCJK) → 文泉驿微米黑 → 系统字体
    """
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

    # 兜底：尝试系统字体
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
        "请将以下任一免费商用字体放到脚本同目录：\n"
        "  • NotoSansCJK-Regular.otf  (SIL OFL, 推荐)\n"
        "  • WenQuanYiMicroHei.ttf    (GPL+exception, 轻量)\n"
        "下载：\n"
        "  https://github.com/notofonts/noto-cjk/releases  →  NotoSansCJK-Regular.otf\n"
        "  https://sourceforge.net/projects/wqy-microhei/  →  wqy-microhei.ttc\n"
    )


def _put_text_cn(img_bgr: np.ndarray, text: str, xy: Tuple[int, int],
                 size: int = 28, color=(255, 255, 255)) -> np.ndarray:
    """用 PIL + FreeType 写中文，写回 BGR 图。"""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    font = _load_cn_font(size)
    # PIL 颜色是 RGB
    draw.text(xy, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ═══════════════════════════════════════════════════════════
# DINOv2 本地加载
def _load_dinov2(device: torch.device):
    """从同目录 dinov2_vits14.pth 加载本地权重，避免联网下载。"""
    global _DINOV2_MODEL, _DINOV2_TRANSFORM
    if _DINOV2_MODEL is not None:
        return _DINOV2_MODEL, _DINOV2_TRANSFORM

    if not DINOV2_WEIGHTS.exists():
        raise FileNotFoundError(
            f"❌ 找不到 DINOv2 权重: {DINOV2_WEIGHTS}\n"
            f"请把 dinov2_vits14_pretrain.pth 放到脚本同目录。\n"
            f"下载方式: torch.hub 联网一次后, ~/.cache/torch/hub/checkpoints/ 下有现成文件, "
        )

    print(f"🔄 加载本地 DINOv2 vit_small: {DINOV2_WEIGHTS.name} ...")
    _DINOV2_MODEL = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vits14", pretrained=False
    )
    state = torch.load(str(DINOV2_WEIGHTS), map_location="cpu")
    _DINOV2_MODEL.load_state_dict(state)
    _DINOV2_MODEL.eval().to(device)
    _DINOV2_TRANSFORM = T.Compose([
        T.ToPILImage(),
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    print(f"✅ DINOv2 加载完成 (device={device})")
    return _DINOV2_MODEL, _DINOV2_TRANSFORM


def _query_index(query: str, grid: int) -> int:
    cx = grid // 2
    cy = grid // 2
    if query == "center":
        return cy * grid + cx
    if query == "tl":
        return 0
    if query == "tr":
        return grid - 1
    if query == "bl":
        return (grid - 1) * grid
    if query == "br":
        return grid * grid - 1
    raise ValueError(f"未知 query: {query}")


def _dinov2_heatmap(frame_bgr: np.ndarray,
                    query: str = "center",
                    device: torch.device = torch.device("cpu")) -> np.ndarray:
    """计算 DINOv2 patch 相似度热力图（不与原图叠加），返回 BGR。"""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    model, transform = _load_dinov2(device)
    img_tensor = transform(rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model.forward_features(img_tensor)
        patch_feats = out["x_norm_patchtokens"][0]

    num_patches = patch_feats.shape[0]
    grid = int(round(num_patches ** 0.5))
    q_idx = _query_index(query, grid)

    q_feat = patch_feats[q_idx].unsqueeze(0)
    sim = torch.nn.functional.cosine_similarity(q_feat, patch_feats, dim=-1)
    sim = sim.reshape(grid, grid).cpu().numpy()
    sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
    sim_u8 = (sim * 255).astype(np.uint8)

    h, w = frame_bgr.shape[:2]
    heat_resized = cv2.resize(sim_u8, (w, h), interpolation=cv2.INTER_CUBIC)
    heat_color = cv2.applyColorMap(heat_resized, cv2.COLORMAP_JET)

    qy = q_idx // grid
    qx = q_idx % grid
    cx_img = int((qx + 0.5) / grid * w)
    cy_img = int((qy + 0.5) / grid * h)
    cv2.circle(heat_color, (cx_img, cy_img), 8, (0, 255, 255), 2)
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
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    h, w, c = frame.shape
    result = frame.astype(np.float32).copy()
    for i in range(c):
        f = np.fft.fft2(result[:, :, i])
        fshift = np.fft.fftshift(f)
        flat = fshift.flatten()
        np.random.shuffle(flat)
        result[:, :, i] = np.abs(np.fft.ifft2(flat.reshape(fshift.shape)))

    result = (result - result.min()) / (result.max() - result.min() + 1e-8) * 255
    result = result.astype(np.uint8)

    rng = np.random.default_rng(42)
    key_array = rng.integers(0, 256, size=result.shape, dtype=np.uint8)
    encrypted = cv2.bitwise_xor(result, key_array)
    return encrypted


# ----- ffmpeg writer -----
def _make_writer(output: str, w: int, h: int, fps: float) -> "_FFmpegWriter":
    is_stream = output.startswith(("rtmp://", "rtsp://", "srt://", "udp://"))
    if is_stream:
        return _FFmpegWriter(
            output, w, h, fps,
            preset="ultrafast", tune="zerolatency",
            crf=23, gop=30, container="flv",
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
            "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.3f}", "-i", "-",
            "-c:v", "libx264", "-preset", preset,
        ]
        if tune:
            cmd += ["-tune", tune]
        if gop:
            cmd += ["-g", str(gop), "-keyint_min", str(gop)]
        cmd += ["-crf", str(crf), "-pix_fmt", "yuv420p"]
        if container:
            cmd += ["-f", container]
        else:
            cmd += ["-movflags", "+faststart"]
        cmd += [output]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def write(self, frame: np.ndarray):
        self.proc.stdin.write(frame.tobytes())

    def close(self):
        if self.proc.stdin:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
        try:
            ret = self.proc.wait(timeout=10)
            if ret != 0:
                raise RuntimeError(f"ffmpeg 退出码 {ret}")
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)
            raise RuntimeError("ffmpeg 收尾超时")


# ----- 流模式（无 DINOv2）-----
def _open_rtsp(url: str) -> cv2.VideoCapture:
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


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
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("RTSP 读帧失败")
            now = time.perf_counter()
            if now - last_push_time < frame_interval:
                continue
            result1 = await action_1_frame(frame)
            result2 = await action_2_frame(frame)
            pad_l = np.zeros((h, w // 2, 3), dtype=np.uint8)
            pad_r = np.zeros((h, w - w // 2, 3), dtype=np.uint8)
            top = np.hstack([pad_l, frame, pad_r])
            bot = np.hstack([result1, result2])
            combined = np.vstack([top, bot])

            # 改:用中文渲染
            combined = _put_text_cn(combined, "原始画面", (20, 10), size=36)
            combined = _put_text_cn(combined, "传递的信息", (w + 20, 10), size=36)
            combined = _put_text_cn(combined, "未加保护的攻击重建", (20, h + 10), size=36)
            combined = _put_text_cn(combined, "加保护的攻击重建", (w + 20, h + 10), size=36)

            await asyncio.gather(
                asyncio.to_thread(writers["orig"].write, frame),
                asyncio.to_thread(writers["action1"].write, result1),
                asyncio.to_thread(writers["action2"].write, result2),
                asyncio.to_thread(writers["combined"].write, combined),
            )
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


# ----- 文件模式（含 DINOv2 热力图）-----
async def run_file(input_path: str, output_dir: str,
                   enable_dinov2: bool = True,
                   query: str = "center"):
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
    if enable_dinov2:
        _load_dinov2(device)
        print(f"🧠 DINOv2 设备: {device}, query patch: {query}")
    else:
        print("ℹ️  DINOv2 热力图已禁用")

    # 预加载字体（避免首帧延迟）
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

        if enable_dinov2:
            r1, r2, dinov2_overlay = await asyncio.gather(
                action_1_frame(frame),
                action_2_frame(frame),
                asyncio.to_thread(_dinov2_heatmap, frame, query, device),
            )
            top_row = np.hstack([frame, dinov2_overlay])
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

        # 改:中文标签(用 PIL + FreeType 渲染)
        if enable_dinov2:
            combined = _put_text_cn(combined, "原始画面", (20, 10), size=36)
            combined = _put_text_cn(combined, "传递的信息", (w + 20, 10), size=36)
        else:
            combined = _put_text_cn(combined, "原始画面", (w // 2 + 20, 10), size=36)
        combined = _put_text_cn(combined, "未加保护的攻击重建", (20, h + 10), size=36)
        combined = _put_text_cn(combined, "加保护的攻击重建", (w + 20, h + 10), size=36)

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
        description="视频四路处理 (含 DINOv2 patch 特征可视化)"
    )
    sp = p.add_subparsers(dest="mode", required=True)

    pf = sp.add_parser("file")
    pf.add_argument("input")
    pf.add_argument("-o", "--output", default="./output")
    pf.add_argument("--fps", type=float)
    pf.add_argument("--no-dinov2", action="store_true")
    pf.add_argument(
        "--query", default="center",
        choices=["center", "tl", "tr", "bl", "br"],
        help="query patch 位置",
    )

    ps = sp.add_parser("stream")
    ps.add_argument("--rtsp", default="rtmp://srs:1935/live/livestream")
    ps.add_argument("--rtmp-base", default="rtmp://srs:1935/live")
    ps.add_argument("--fps", type=float)

    args = p.parse_args()
    FORCE_FPS = args.fps

    async def run():
        try:
            if args.mode == "file":
                await run_file(
                    args.input, args.output,
                    enable_dinov2=not args.no_dinov2,
                    query=args.query,
                )
            else:
                if not args.rtsp:
                    print("❌ stream 模式需要 --rtsp", file=sys.stderr)
                    sys.exit(1)
                await run_stream(args.rtsp, args.rtmp_base)
        finally:
            await _cleanup()

    asyncio.run(run())


if __name__ == "__main__":
    main()