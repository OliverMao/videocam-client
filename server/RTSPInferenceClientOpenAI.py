import json
import time
import threading
import base64
import cv2
from typing import Optional, Dict, Any, List
from pathlib import Path
from openai import OpenAI
import tempfile
import os

# ================= 全局模式配置 =================
USE_VIDEO_MODE = False

# 是否开启两阶段VLM复核
ENABLE_VLM_VERIFICATION = False
# ================================================
DEFAULT_OPENAI_API = "http://116.238.240.2:30630/v1"  # Jetson
# DEFAULT_OPENAI_API = "http://116.238.240.2:32726/v1"  # 4090
DEFAULT_OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", DEFAULT_OPENAI_API)
DEFAULT_OPENAI_API_KEY = "vllm"
MODEL_NAME="Qwen3.6-35B-A3B"
# MODEL_NAME = "/ddn/gemini/gemini-sharedata/space/wqmu4k88unnm/guarded_files/songhuan/Models/Qwen3.6-35B-A3B"
DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME", MODEL_NAME)

# 推理触发间隔（每 1 秒触发一次）
DEFAULT_INTERVAL_SEC = float(os.environ.get("INTERVAL_SEC", "0.5"))

# 抽帧相关：6 秒窗口 × 2 fps = 12 帧（实际取14帧）
SAMPLE_FPS = float(os.environ.get("SAMPLE_FPS", "2.0"))     # 每秒抽几帧
REQUIRED_FRAMES = int(os.environ.get("REQUIRED_FRAMES", "8"))    # 缓冲区大小：存储14帧
INFERENCE_FRAMES = int(os.environ.get("INFERENCE_FRAMES", "4"))     # 每次推理发送前4帧

MAX_WIDTH = 512
MAX_FAIL_COUNT = 2

# 推理并发上限
MAX_CONCURRENT_INFER = int(os.environ.get("MAX_CONCURRENT_INFER", "4"))
_INFER_SEM = threading.Semaphore(MAX_CONCURRENT_INFER)

DEFAULT_SYSTEM_PROMPT = """
图片中的场景是TeleAI的展厅。请详细分析视频内容，判断是否存在违规行为。违规行为包括但不限于：吸烟、打架、摔倒三类违规行为。
如果摆出类似抽烟的姿势我们也认为是抽烟，如果摆出打架动作而没有真的在打架也认为是打架。只要人坐在地上，我们也认为是摔倒。
存在，输出 {"has_person": 1, "violations": [...]}，其中 violations 数组列出所有发现的违规行为（如 ["吸烟"] 或 ["吸烟","打架"]）。如果没有任何违规行为，violations 应为空数组 []。
严格按照上述格式输出 JSON，不要添加任何多余的文本或解释，不要开启思考模式。
"""

DEFAULT_VERIFICATION_PROMPT = """
你现在要核验上一轮输出是否与当前画面内容对应。

请只根据当前提供的画面和上一轮输出结果进行判断，不要重复展开无关分析。
如果上一轮结果与画面一致，输出 {"match": 1, "reason": "简要说明", "correct_result": <上一轮结果>}。
如果上一轮结果与画面不一致，输出 {"match": 0, "reason": "简要说明", "correct_result": {"has_person": 1, "violations": [...]}}。
严格按照上述格式输出 JSON，不要添加任何多余的文本或解释。 /no_think
"""


def _frames_to_video_base64(frames: List[Any]) -> str:
    if not frames:
        return ""
    height, width = frames[0].shape[:2]
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_path = temp_file.name
    temp_file.close()

    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = max(1.0, SAMPLE_FPS)  # 用采样帧率，避免拼出来的视频看着抽搐
        out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
        for frame in frames:
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            out.write(frame)
        out.release()
        with open(temp_path, "rb") as f:
            video_bytes = f.read()
        return base64.b64encode(video_bytes).decode("utf-8")
    except Exception as e:
        print(f"Error encoding video: {e}")
        return ""
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _frame_to_base64(frame: Any) -> str:
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode("utf-8")
    except Exception as e:
        print(f"Error encoding frame to base64: {e}")
        return ""


def _strip_code_fences(content: str) -> str:
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    return content.strip()


def _parse_json_object(content: str) -> Optional[Dict[str, Any]]:
    cleaned = _strip_code_fences(content)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_bool_flag(value: Any) -> bool:
    try:
        return int(value) != 0
    except (TypeError, ValueError):
        return bool(value)


class RTSPInferenceClient:
    def __init__(
        self,
        rtsp_url: str,
        openai_api_base: str = DEFAULT_OPENAI_API_BASE,
        openai_api_key: str = DEFAULT_OPENAI_API_KEY,
        model_name: str = DEFAULT_MODEL_NAME,
        interval_sec: float = DEFAULT_INTERVAL_SEC,
        max_fail_count: int = MAX_FAIL_COUNT,
        save_dir: str = "./save",
    ):
        self.rtsp_url = rtsp_url
        self.interval_sec = interval_sec
        self.save_dir = Path(save_dir)
        self.max_fail_count = max_fail_count
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.openai_client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
        self.model_name = model_name

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._latest_result: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()
        self._is_healthy = False

        self._valid_frames_buffer: List[Any] = []

        self._llm_call_times: List[float] = []
        self._llm_call_count = 0
        self._total_llm_time_ms = 0.0
        self._cost_log_path = self.save_dir / "costtime.log"

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

    def _build_media_content(self, frames: List[Any]) -> List[Dict[str, Any]]:
        user_content: List[Dict[str, Any]] = []

        if USE_VIDEO_MODE:
            video_b64 = _frames_to_video_base64(frames)
            if not video_b64:
                raise ValueError("Video encoding returned empty result")
            user_content.append({
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}
            })
        else:
            for frame in frames:
                img_b64 = _frame_to_base64(frame)
                if img_b64:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    })

        return user_content

    def _build_vlm_messages(self, frames: List[Any]) -> tuple:
        t_start = time.time()
        user_content = self._build_media_content(frames)
        user_content.append({"type": "text", "text": "请分析这段监控内容。"})
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        return messages, (time.time() - t_start) * 1000

    def _build_verification_messages(self, frames: List[Any], initial_result_text: str) -> tuple:
        t_start = time.time()
        user_content = self._build_media_content(frames)
        user_content.append({
            "type": "text",
            "text": f"上一轮输出结果如下，请核验它是否与当前画面对应：\n{initial_result_text}\n\n请只输出核验 JSON。"
        })
        messages = [
            {"role": "system", "content": DEFAULT_VERIFICATION_PROMPT},
            {"role": "user", "content": user_content}
        ]
        return messages, (time.time() - t_start) * 1000

    def _log_llm_call_time(self, call_time_ms: float):
        self._llm_call_count += 1
        self._total_llm_time_ms += call_time_ms
        self._llm_call_times.append(call_time_ms)
        avg_time_ms = self._total_llm_time_ms / self._llm_call_count

        log_entry = f"Call #{self._llm_call_count}: {call_time_ms:.2f}ms, Average: {avg_time_ms:.2f}ms\n"
        try:
            with open(self._cost_log_path, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except Exception as e:
            print(f"[ERROR] Failed to write costtime.log: {e}")

    def _call_vlm(self, messages: List[Dict[str, Any]]) -> tuple:
        t_total_start = time.time()
        response = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=2048,
            temperature=0.0001,
            response_format={"type": "json_object"},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
        content = response.choices[0].message.content.strip()
        call_time_ms = (time.time() - t_total_start) * 1000
        self._log_llm_call_time(call_time_ms)
        return content, call_time_ms

    def _process_vlm_inference(self, frames: List[Any]):
        """后台线程执行 VLM 推理；fire-and-forget 调用"""
        if not frames:
            return
        if not _INFER_SEM.acquire(blocking=False):
            print("[VLM SKIP] Too many concurrent inferences, dropping this batch.")
            return
        try:
            self._do_inference(frames)
        finally:
            _INFER_SEM.release()

    def _do_inference(self, frames: List[Any]):
        try:
            messages, media_processing_time_ms = self._build_vlm_messages(frames)
            content, first_api_time_ms = self._call_vlm(messages)

            model_json = _parse_json_object(content)
            result_text = (
                json.dumps(model_json, ensure_ascii=False)
                if model_json is not None else _strip_code_fences(content)
            )

            verification_api_time_ms = 0.0
            verification_media_processing_time_ms = 0.0

            if ENABLE_VLM_VERIFICATION:
                verification_messages, verification_media_processing_time_ms = (
                    self._build_verification_messages(frames, result_text)
                )
                verification_content, verification_api_time_ms = self._call_vlm(verification_messages)
                print(f"[VLM] 复核响应: {verification_content}")

                verification_json = _parse_json_object(verification_content)
                verification_text = (
                    json.dumps(verification_json, ensure_ascii=False)
                    if verification_json is not None else _strip_code_fences(verification_content)
                )

                verification_match = (
                    _parse_bool_flag(verification_json.get("match", 0))
                    if verification_json is not None else False
                )
                if not verification_match:
                    print(f"[VLM MISMATCH] 初次结果与画面不一致，丢弃本次结果。verification={verification_text}")
                    return

            final_result = {
                "mode": "video" if USE_VIDEO_MODE else "image",
                "api_time_ms": first_api_time_ms,
                "total_api_time_ms": first_api_time_ms + verification_api_time_ms,
                "server_response": {"result": result_text},
                "media_processing_time_ms": media_processing_time_ms,
                "verification_media_processing_time_ms": verification_media_processing_time_ms,
                "frame_count": len(frames),
                "sample_fps": SAMPLE_FPS,
            }

            with self._lock:
                self._latest_result = final_result
            print(f"[VLM RESULT] {final_result}")

        except Exception as exc:
            empty_result = {
                "mode": "video" if USE_VIDEO_MODE else "image",
                "api_time_ms": 0.0,
                "total_api_time_ms": 0.0,
                "server_response": {"result": '{"has_person": -1, "violations": []}'},
                "media_processing_time_ms": 0.0,
                "verification_media_processing_time_ms": 0.0,
                "frame_count": 0,
                "sample_fps": SAMPLE_FPS,
            }
            with self._lock:
                self._latest_result = empty_result
            print(f"[VLM ERROR] Inference failed: {exc}")

    def _preprocess_frame(self, frame: Any) -> Any:
        """裁剪 + 缩放，节省 VLM 推理开销"""
        h, w = frame.shape[:2]
        crop_w = min(w, 2000)
        if crop_w > 0:
            frame = frame[:, 0:crop_w]
            new_h = int(h * (MAX_WIDTH / crop_w))
            frame = cv2.resize(frame, (MAX_WIDTH, new_h))
        return frame

    def _inference_loop(self):
        if not self._open_stream():
            print("[FATAL] Cannot open stream, thread exiting.")
            return

        fail_count = 0
        self._last_infer_ts = 0.0       # 第一次立刻触发
        self._last_sample_ts = 0.0      # 第一次立即采一帧
        sample_interval = 1.0 / max(SAMPLE_FPS, 0.001)  # 采样间隔：1/2 = 0.5s

        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok:
                fail_count += 1
                if fail_count >= self.max_fail_count:
                    if not self._open_stream():
                        time.sleep(5)
                    fail_count = 0
                    self._valid_frames_buffer = []
                continue

            frame = self._preprocess_frame(frame)
            fail_count = 0
            self._is_healthy = True

            now = time.time()

            # === 关键：按 SAMPLE_FPS 节奏抽帧，而不是每帧都塞 ===
            if (now - self._last_sample_ts) >= sample_interval:
                self._last_sample_ts = now
                self._valid_frames_buffer.append(frame.copy())
                # buffer 上限 = REQUIRED_FRAMES，超了就丢最老的
                if len(self._valid_frames_buffer) > REQUIRED_FRAMES:
                    self._valid_frames_buffer = self._valid_frames_buffer[-REQUIRED_FRAMES:]

            # === 定时触发推理：每 interval_sec 触发一次，buffer 不够就跳过 ===
            if (now - self._last_infer_ts) >= self.interval_sec:
                self._last_infer_ts = now
                if len(self._valid_frames_buffer) >= REQUIRED_FRAMES:
                    # FIFO队列：发送前4帧（最早的4帧，实现6秒延时）
                    frames_to_process = list(self._valid_frames_buffer[:INFERENCE_FRAMES])
                    print(f"[VLM] 发送前{INFERENCE_FRAMES}帧，缓冲区: {len(self._valid_frames_buffer)}帧")
                    threading.Thread(
                        target=self._process_vlm_inference,
                        args=(frames_to_process,),
                        daemon=True
                    ).start()
                else:
                    print(f"[VLM] 缓冲帧不足 ({len(self._valid_frames_buffer)}/{REQUIRED_FRAMES})，等待攒帧...")

            time.sleep(0.001)

        if self._cap is not None:
            self._cap.release()
        print("Inference loop stopped.")

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()
        mode = "VIDEO" if USE_VIDEO_MODE else "IMAGE"
        print(
            f"RTSP client started. "
            f"[Mode: {mode}] [Buffer: {REQUIRED_FRAMES} frames] [Infer: {INFERENCE_FRAMES} frames, FIFO] "
            f"[Infer Interval: {self.interval_sec}s, fire-and-forget]"
        )

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def get_latest_result(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._latest_result

    @property
    def is_healthy(self) -> bool:
        return self._is_healthy

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()