"""
FastAPI 服务：单张图像隐私重建（自动缩放回原图尺寸），支持多 GPU 负载均衡。

启动方式（在项目根目录下）：
    uvicorn worker_original:app --host 0.0.0.0 --port 28010

所有可配项集中在文件顶部的 CONFIG 区域，修改后重启服务即可生效。

POST /reconstruct   → 上传图片，返回重建后的 JPEG（尺寸与输入一致）
GET  /health        → 健康检查（含各 GPU 当前 pending 数）
"""

from __future__ import annotations

import asyncio
import io
import sys
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
import os
import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

sys.path.append(str(Path(__file__).parent / "original_cam_attack_main"))
from infer_single_image import PrivacyReconstructor

# ====================== CONFIG ======================
# 使用的 GPU ID 列表，指定几个就启动几个模型实例
# 例如 [4, 5, 6, 7] 表示在 cuda:4/5/6/7 上各加载一个模型，并发推理
GPU_IDS: list[int] = [ 6 ]

# 模型权重路径
CKPT_PATH: str = "/workspace/s/yuanc/jzy/VideoCam/mae/models/original_0615.pt"

# 实验标签 / 视觉模型路径 / 隐私模型路径
EXPERIMENT_TAG: str = "native"
MODEL_PATH: str = "/workspace/s/ddn/gemini/gemini-sharedata/space/wqmu4k88unnm/Models/Qwen3.5-9B"
PRIVACY_MODEL_PATH: str = ""

# 是否启用 AMP（自动混合精度）加速
USE_AMP: bool = True

# 服务监听端口
SERVER_PORT: int = 28010
# =====================================================


def build_devices() -> list[str]:
    """将 GPU_IDS 规范化为 ["cuda:0", "cuda:1", ...]。"""
    return [f"cuda:{i}" for i in GPU_IDS]


class WorkerPool:
    """
    多 GPU Worker 池：每个 GPU 加载一个 PrivacyReconstructor 实例；
    任务按"当前在途请求数最少"分派到不同 GPU，实现负载均衡。
    同一 GPU 上的请求通过 pending 计数互斥（避免多线程争用同一 CUDA 上下文）。
    """

    def __init__(self):
        self.workers: list[PrivacyReconstructor] = []
        self.devices: list[str] = []
        self.pending: list[int] = []
        self._lock = threading.Lock()

    def add(self, worker: PrivacyReconstructor, device: str):
        self.workers.append(worker)
        self.devices.append(device)
        self.pending.append(0)

    def acquire(self) -> tuple[int, PrivacyReconstructor]:
        with self._lock:
            idx = min(range(len(self.workers)), key=lambda i: self.pending[i])
            self.pending[idx] += 1
            return idx, self.workers[idx]

    def release(self, idx: int):
        with self._lock:
            self.pending[idx] = max(0, self.pending[idx] - 1)


pool: WorkerPool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时为每张指定 GPU 加载一个模型实例，关闭时释放资源。"""
    global pool
    devices = build_devices()

    print(f"[启动] 计划加载 GPU: {devices}（共 {len(devices)} 个实例）")
    pool = WorkerPool()
    for dev in devices:
        print(f"[启动] 正在加载模型到 {dev} ... ckpt={CKPT_PATH}, amp={USE_AMP}")
        worker = PrivacyReconstructor(
            ckpt_path=CKPT_PATH,
            experiment_tag=EXPERIMENT_TAG,
            model_path=MODEL_PATH,
            privacy_model_path=PRIVACY_MODEL_PATH,
            device=dev,
            amp=USE_AMP,
        )
        pool.add(worker, dev)
        print(f"[启动] {dev} 加载完成")

    print(f"[启动] {len(devices)} 个模型实例就绪，开始接收请求。")
    yield
    pool = None
    torch.cuda.empty_cache()


app = FastAPI(title="Privacy Reconstruction API (original)", lifespan=lifespan)


@app.get("/health")
async def health():
    if pool is None or not pool.workers:
        return {"status": "loading"}
    return {
        "status": "ok",
        "model_loaded": True,
        "gpu_count": len(pool.workers),
        "devices": pool.devices,
        "pending_per_gpu": pool.pending,
    }


@app.post("/reconstruct")
async def reconstruct(file: UploadFile = File(..., description="待重建的图片（RGB）")):
    """
    接收一张图片，返回隐私重建后的 JPEG 图像。
    重建结果会自动缩放回输入图片的原始尺寸。
    """
    if pool is None or not pool.workers:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="仅支持图片文件")

    idx, worker = pool.acquire()
    temp_input_path = None
    temp_output_path = None
    try:
        contents = await file.read()
        try:
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
            original_size = pil_image.size
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"无法解析图片: {e}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_in:
            tmp_in.write(contents)
            temp_input_path = tmp_in.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_out:
            temp_output_path = tmp_out.name

        # 推理放到独立线程，避免阻塞事件循环；
        # pool 内部通过 pending 计数保证同一 worker 同一时刻只被一个线程使用。
        out_path = await asyncio.to_thread(
            worker.reconstruct, temp_input_path, temp_output_path
        )

        result_image = Image.open(out_path).convert("RGB")
        if result_image.size != original_size:
            result_image = result_image.resize(original_size, Image.LANCZOS)

        image_bytes = cv2.imencode(
            ".jpg",
            cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )[1].tobytes()
        return Response(content=image_bytes, media_type="image/jpeg")

    except HTTPException:
        raise
    except Exception as e:
        print(f"[错误] 重建失败 (gpu={pool.devices[idx]}): {e}")
        raise HTTPException(status_code=500, detail=f"重建失败: {e}")
    finally:
        for path in (temp_input_path, temp_output_path):
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass
        pool.release(idx)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT, reload=False)
