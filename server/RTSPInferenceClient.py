import io
import time
import threading
import queue
from typing import Optional, Dict, Any, List

import cv2
import httpx
from PIL import Image

DEFAULT_INFER_URL = "http://116.238.240.2:31676/packet"
DEFAULT_INTERVAL_SEC = 5.0
DEFAULT_FRAME_INTERVAL_SEC = 0.3
MAX_WIDTH = 1024
MAX_FAIL_COUNT = 2


def _resize_image(image: Image.Image, max_width: int) -> Image.Image:
    width, height = image.size
    if width <= max_width:
        return image
    new_height = int(height * max_width / width)
    return image.resize((max_width, new_height), Image.Resampling.LANCZOS)


def _frame_to_png_bytes(frame_bgr) -> bytes:
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    image = _resize_image(image, MAX_WIDTH)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _frame_to_jpeg_bytes(frame_bgr, quality: int = 70) -> bytes:
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    image = _resize_image(image, MAX_WIDTH)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


class FrameBroadcaster:
    def __init__(self):
        self._subscribers: List[queue.Queue] = []
        self._lock = threading.Lock()

    def publish(self, data: bytes):
        with self._lock:
            dead = []
            for q in self._subscribers:
                try:
                    q.put_nowait(data)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self._subscribers.remove(q)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=32)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)


class RTSPInferenceClient:
    def __init__(
        self,
        rtsp_url: str,
        infer_url: str = DEFAULT_INFER_URL,
        interval_sec: float = DEFAULT_INTERVAL_SEC,
        frame_interval_sec: float = DEFAULT_FRAME_INTERVAL_SEC,
        max_fail_count: int = MAX_FAIL_COUNT,
    ):
        self.rtsp_url = rtsp_url
        self.infer_url = infer_url
        self.interval_sec = interval_sec
        self.frame_interval_sec = frame_interval_sec
        self.max_fail_count = max_fail_count

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._latest_result: Optional[Dict[str, Any]] = None
        self._latest_frame: Optional[bytes] = None
        self._lock = threading.Lock()
        self._is_healthy = False
        self._last_frame_time = 0.0

        self._broadcaster = FrameBroadcaster()

    def _open_stream(self) -> bool:
        if self._cap is not None:
            self._cap.release()
        print(f"[INFO] Opening RTSP stream: {self.rtsp_url}")
        self._cap = cv2.VideoCapture(self.rtsp_url)
        if not self._cap.isOpened():
            print("[ERROR] Failed to open RTSP stream")
            return False
        print("[INFO] RTSP stream opened successfully")
        return True

    def _inference_loop(self):
        if not self._open_stream():
            print("[FATAL] Cannot open stream, thread exiting.")
            return

        last_infer_time = 0.0
        fail_count = 0

        while not self._stop_event.is_set():
            now = time.time()

            ok, frame = self._cap.read()
            if not ok:
                fail_count += 1
                print(f"Frame read failed ({fail_count}/{self.max_fail_count}), retrying...")
                time.sleep(0.5)
                if fail_count >= self.max_fail_count:
                    print("Too many failures, trying to reconnect...")
                    if not self._open_stream():
                        print("Reconnect failed, will retry later.")
                        time.sleep(5)
                    fail_count = 0
                continue

            fail_count = 0
            self._is_healthy = True

            if now - self._last_frame_time >= self.frame_interval_sec:
                try:
                    frame_bytes = _frame_to_jpeg_bytes(frame)
                    with self._lock:
                        self._latest_frame = frame_bytes
                    self._last_frame_time = now
                    self._broadcaster.publish(frame_bytes)
                except Exception as e:
                    print(f"Failed to save frame: {e}")

            if now - last_infer_time >= self.interval_sec:
                last_infer_time = now
                try:
                    image_bytes = _frame_to_png_bytes(frame)
                    response = httpx.post(
                        self.infer_url,
                        files={"file": ("frame.png", image_bytes, "image/png")},
                        timeout=30.0,
                    )
                    response.raise_for_status()
                    result = response.json()
                    with self._lock:
                        self._latest_result = result
                except Exception as exc:
                    print(f"Inference request failed: {exc}")

            time.sleep(0.05)

        if self._cap is not None:
            self._cap.release()
            self._cap = None
        print("Inference loop stopped.")

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            print("Already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()
        print("RTSP inference client started.")

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def get_latest_result(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._latest_result

    def get_latest_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_frame

    def create_stream_queue(self) -> queue.Queue:
        return self._broadcaster.subscribe()

    def remove_stream_queue(self, q: queue.Queue):
        self._broadcaster.unsubscribe(q)

    @property
    def is_healthy(self) -> bool:
        return self._is_healthy

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
