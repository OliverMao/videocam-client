import asyncio
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from vectordb.db import VectorDB
from vectordb.embeddings import compute_embedding

logger = logging.getLogger(__name__)

THUMB_WIDTH = 160
SAVE_INTERVAL = 30


class FrameCaptureService:
    def __init__(self, rtsp_url: str, db: VectorDB, storage_dir: Path):
        self.rtsp_url = rtsp_url
        self.db = db
        self.storage_dir = storage_dir
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"[Capture] 启动: {self.rtsp_url[:40]}...")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        self.db.save()
        logger.info("[Capture] 停止")

    def _capture_loop(self) -> None:
        cap: cv2.VideoCapture | None = None
        last_save = time.monotonic()

        while self._running:
            if cap is None or not cap.isOpened():
                cap = self._open_stream()
                if cap is None:
                    time.sleep(3)
                    continue

            ret, frame = cap.read()
            if not ret:
                logger.warning("[Capture] 读帧失败, 重连...")
                cap.release()
                cap = None
                time.sleep(2)
                continue

            now = datetime.now(timezone.utc)
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

            try:
                vec = compute_embedding(frame)
                thumb_path = self._save_thumbnail(frame, timestamp)
                self.db.add(vec, timestamp, thumb_path)
            except Exception as e:
                logger.warning(f"[Capture] 处理帧失败: {e}")

            if time.monotonic() - last_save > SAVE_INTERVAL:
                self.db.save()
                last_save = time.monotonic()

            time.sleep(1.0)

        if cap:
            cap.release()

    def _open_stream(self) -> cv2.VideoCapture | None:
        logger.info(f"[Capture] 连接: {self.rtsp_url[:40]}...")
        cap = cv2.VideoCapture(self.rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            logger.error("[Capture] 连接失败")
            cap.release()
            return None
        logger.info("[Capture] 连接成功")
        return cap

    def _save_thumbnail(self, frame: np.ndarray, timestamp: str) -> str:
        thumbs_dir = self.storage_dir / "thumbs"
        thumbs_dir.mkdir(parents=True, exist_ok=True)
        h, w = frame.shape[:2]
        scale = THUMB_WIDTH / w
        thumb = cv2.resize(frame, (THUMB_WIDTH, int(h * scale)))
        safe_ts = timestamp.replace(" ", "_").replace(":", "-")
        path = thumbs_dir / f"{safe_ts}.jpg"
        cv2.imwrite(str(path), thumb, [cv2.IMWRITE_JPEG_QUALITY, 60])
        return str(path.relative_to(self.storage_dir))
