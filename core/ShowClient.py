import argparse
import io
import time
import queue
import threading
from typing import Optional, Dict, Any

import cv2
import httpx
from PIL import Image

DEFAULT_INFER_URL = "http://116.238.240.2:31676/packet"
DEFAULT_INTERVAL_SEC = 2.0
MAX_WIDTH = 512
# 连续读帧失败多少次后触发重连
MAX_FAIL_COUNT = 10


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


class RTSPInferenceClient:
    """
    在后台线程中持续从 RTSP 流取帧并调用推理 API，
    通过队列将结果传出，供主线程调用 get_result() 获取。
    """

    def __init__(
        self,
        rtsp_url: str,
        infer_url: str = DEFAULT_INFER_URL,
        interval_sec: float = DEFAULT_INTERVAL_SEC,
        max_fail_count: int = MAX_FAIL_COUNT,
    ):
        self.rtsp_url = rtsp_url
        self.infer_url = infer_url
        self.interval_sec = interval_sec
        self.max_fail_count = max_fail_count

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        # 队列中存放推理成功的结果
        self._result_queue: queue.Queue = queue.Queue()

    def _open_stream(self) -> bool:
        """打开 RTSP 流，返回是否成功"""
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(self.rtsp_url)
        if not self._cap.isOpened():
            return False
        return True

    def _inference_loop(self):
        """后台线程主循环：取帧、推理、结果入队，含自动重连"""
        if not self._open_stream():
            print(f"[ERROR] Cannot open RTSP stream: {self.rtsp_url}")
            return

        last_time = 0.0
        fail_count = 0

        while not self._stop_event.is_set():
            now = time.time()
            if now - last_time < self.interval_sec:
                time.sleep(0.05)
                continue

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

            # 读帧成功，重置失败计数，更新时间
            fail_count = 0
            last_time = now

            try:
                image_bytes = _frame_to_png_bytes(frame)
                response = httpx.post(
                    self.infer_url,
                    files={"file": ("frame.png", image_bytes, "image/png")},
                    timeout=30.0,
                )
                response.raise_for_status()
                result = response.json()
            except Exception as exc:
                print(f"Inference request failed: {exc}")
                continue

            # 将结果放入队列，如果服务已停止则直接丢弃
            if not self._stop_event.is_set():
                self._result_queue.put(result)

        # 退出时释放资源
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        print("Inference loop stopped.")

    def start(self):
        """启动后台推理线程"""
        if self._thread is not None and self._thread.is_alive():
            print("Already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()
        print("RTSP inference client started.")

    def stop(self):
        """停止后台线程并等待结束"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def get_result(self, block: bool = True, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        获取一次推理结果。默认阻塞直到有结果。

        返回:
            推理 API 返回的完整 JSON 字典（即原脚本中 response.json() 的内容）
        """
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError("Client is not running. Call start() first.")
        return self._result_queue.get(block=block, timeout=timeout)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def main():
    parser = argparse.ArgumentParser(description="RTSP frame inference loop (class-based)")
    parser.add_argument("--rtsp", default="rtsp://admin:qazwsx168@192.168.158.195:554/Streaming/Channels/101")
    parser.add_argument("--infer-url", default=DEFAULT_INFER_URL)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_SEC)
    args = parser.parse_args()

    client = RTSPInferenceClient(args.rtsp, args.infer_url, args.interval)
    try:
        client.start()
        while True:
            # 每次调用 get_result() 都会返回最新的推理结果
            result = client.get_result()
            print(result)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        client.stop()


if __name__ == "__main__":
    main()