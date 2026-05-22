import io
import time
import threading
from typing import Optional, Dict, Any
import cv2
import httpx
from PIL import Image
from pathlib import Path

from Loop.yolo import YOLODetector

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


class RTSPInferenceClient:
    def __init__(
        self,
        rtsp_url: str,
        infer_url: str = DEFAULT_INFER_URL,
        interval_sec: float = DEFAULT_INTERVAL_SEC,
        frame_interval_sec: float = DEFAULT_FRAME_INTERVAL_SEC,
        max_fail_count: int = MAX_FAIL_COUNT,
        save_dir: str = "./save",
    ):
        self.rtsp_url = rtsp_url
        self.infer_url = infer_url
        self.interval_sec = interval_sec
        self.frame_interval_sec = frame_interval_sec
        self.save_dir = Path(save_dir)
        self.max_fail_count = max_fail_count
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.detector = YOLODetector("./Loop/yolo26x.pt")

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._latest_result: Optional[Dict[str, Any]] = None
        self._person_detected = False
        self._lock = threading.Lock()
        self._is_healthy = False

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

        last_yolo_time = 0.0
        last_vlm_time = 0.0
        fail_count = 0
        yolo_interval = 0.5          # 500ms
        vlm_interval = self.interval_sec  # 默认5s
        temp_yolo_path = self.save_dir / "yolo_temp.png"

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

            # ----- YOLO 检测，每 500ms 一次 -----
            if now - last_yolo_time >= yolo_interval:
                last_yolo_time = now
                try:
                    # 保存当前帧为临时图片（BGR 格式，符合 OpenCV 读取习惯）
                    cv2.imwrite(str(temp_yolo_path), frame)
                    person_detected = self.detector.has_person(temp_yolo_path)
                except Exception as e:
                    print(f"YOLO detection error: {e}")
                    person_detected = False

                with self._lock:
                    self._person_detected = person_detected
            else:
                person_detected = self._person_detected

            # ----- VLM 推理：有人 + 间隔 >= 5s -----
            if person_detected and (now - last_vlm_time >= vlm_interval):
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
                    last_vlm_time = now
                    print("VLM inference completed.")
                except Exception as exc:
                    print(f"Inference request failed: {exc}")

            # 降低 CPU 占用
            time.sleep(0.01)

        if self._cap is not None:
            self._cap.release()
            self._cap = None
        # 清理临时文件
        try:
            temp_yolo_path.unlink(missing_ok=True)
        except Exception:
            pass
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

    def get_person_detected(self) -> bool:
        """返回最新 YOLO 检测是否有人"""
        with self._lock:
            return self._person_detected

    @property
    def is_healthy(self) -> bool:
        return self._is_healthy

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()