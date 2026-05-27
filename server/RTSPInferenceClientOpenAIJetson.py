import io
import json
import time
import threading
import base64
import cv2
from typing import Optional, Dict, Any, List
from PIL import Image
from pathlib import Path
from openai import OpenAI
import tempfile
import os



# ================= 全局模式配置 =================
USE_VIDEO_MODE = False 
# YOLO 检测频率 (Hz)
YOLO_DETECT_FPS = 3 

# 是否开启两阶段VLM复核
ENABLE_VLM_VERIFICATION = False
# ================================================

DEFAULT_OPENAI_API_BASE = "http://116.238.240.2:32726/v1"
DEFAULT_OPENAI_API_KEY = "vllm"
DEFAULT_MODEL_NAME = "/ddn/gemini/gemini-sharedata/space/wqmu4k88unnm/guarded_files/songhuan/Models/Qwen3.6-35B-A3B"
DEFAULT_INTERVAL_SEC = 3.0
MAX_WIDTH = 1024
MAX_FAIL_COUNT = 2

# 视频/图像采集配置
REQUIRED_FRAMES = 4  # 固定采集4帧有效的"有人"帧

DEFAULT_VERIFICATION_PROMPT = """
你现在要核验上一轮输出是否与当前画面内容对应。

请只根据当前提供的画面和上一轮输出结果进行判断，不要重复展开无关分析。
如果上一轮结果与画面一致，输出 {"match": 1, "reason": "简要说明", "correct_result": <上一轮结果>}。
如果上一轮结果与画面不一致，输出 {"match": 0, "reason": "简要说明", "correct_result": {"has_person": 1, "violations": [...]}}。
严格按照上述格式输出 JSON，不要添加任何多余的文本或解释。 /no_think

"""

# 因为yolo连续4帧检测到人物，所以VLM直接判断是否存在违规行为，不再让VLM判断是否有人，避免重复和矛盾的输出

DEFAULT_SYSTEM_PROMPT = """
图片中的场景是TeleAI的展厅。请详细分析视频内容，判断是否存在违规行为。违规行为包括但不限于：吸烟、打架、摔倒三类违规行为。
如果摆出类似抽烟的姿势我们也认为是抽烟，如果摆出打架动作而没有真的在打架也认为是打架。只要人坐在地上，我们也认为是摔倒。
存在，输出 {"has_person": 1, "violations": [...]}，其中 violations 数组列出所有发现的违规行为（如 ["吸烟"] 或 ["吸烟","打架"]）。如果没有任何违规行为，violations 应为空数组 []。
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
        out = cv2.VideoWriter(temp_path, fourcc, YOLO_DETECT_FPS, (width, height))
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
        flv_url: str = "http://srs:8080/live/livestream.flv",
        openai_api_base: str = DEFAULT_OPENAI_API_BASE,
        openai_api_key: str = DEFAULT_OPENAI_API_KEY,
        model_name: str = DEFAULT_MODEL_NAME,
        interval_sec: float = DEFAULT_INTERVAL_SEC,
        max_fail_count: int = MAX_FAIL_COUNT,
        save_dir: str = "./save",
    ):
        # ===== 核心修改：只接收 flv_url，删除 V4L2 相关参数 =====
        self.flv_url = flv_url
        self.interval_sec = interval_sec
        self.save_dir = Path(save_dir)
        self.max_fail_count = max_fail_count
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.detector = None
        self.openai_client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
        self.model_name = model_name

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._latest_result: Optional[Dict[str, Any]] = None
        self._person_detected = False
        self._lock = threading.Lock()
        self._is_healthy = False
        
        self._valid_frames_buffer: List[Any] = []
        self._last_yolo_time = 0.0
        self._yolo_interval = 1.0 / YOLO_DETECT_FPS if YOLO_DETECT_FPS > 0 else 0.0

    def _open_stream(self) -> bool:
        # ===== 核心修改：使用 OpenCV 直接打开 FLV 流 =====
        if self._cap is not None:
            self._cap.release()
        
        print(f"[INFO] Opening FLV stream: {self.flv_url}")
        
        # OpenCV 直接读取 HTTP-FLV 流
        self._cap = cv2.VideoCapture(self.flv_url)
        
        # 设置缓冲区大小为 1，降低延迟
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self._cap.isOpened():
            print(f"[ERROR] Failed to open FLV stream: {self.flv_url}")
            return False
        
        # 获取实际分辨率
        actual_width = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        print(f"[INFO] FLV stream opened: {actual_width:.0f}x{actual_height:.0f} @ {actual_fps:.0f}fps")
        return True

    # ===== 以下所有方法完全不变 =====
    
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

    def _build_vlm_messages(self, frames: List[Any]) -> tuple[List[Dict], float]:
        t_start = time.time()
        user_content = self._build_media_content(frames)
        user_content.append({"type": "text", "text": "请分析这段监控内容。"})
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        return messages, (time.time() - t_start) * 1000

    def _build_verification_messages(self, frames: List[Any], initial_result_text: str) -> tuple[List[Dict], float]:
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

    def _call_vlm(self, messages: List[Dict[str, Any]]) -> tuple[str, float]:
        t_total_start = time.time()
        response = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=2048,
            temperature=0.01,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content.strip()
        return content, (time.time() - t_total_start) * 1000

    def _process_vlm_inference(self, frames: List[Any]):
        """同步阻塞执行 VLM 推理，适配 QPS=1 的场景"""
        if not frames:
            return
        try:
            messages, media_processing_time_ms = self._build_vlm_messages(frames)
            content, first_api_time_ms = self._call_vlm(messages)
            print(f"[VLM] VLM完成，")
            print(f"[VLM] API响应: {content}")

            model_json = _parse_json_object(content)
            result_text = json.dumps(model_json, ensure_ascii=False) if model_json is not None else _strip_code_fences(content)

            verification_api_time_ms = 0.0
            verification_media_processing_time_ms = 0.0

            if ENABLE_VLM_VERIFICATION:
                verification_messages, verification_media_processing_time_ms = self._build_verification_messages(frames, result_text)
                verification_content, verification_api_time_ms = self._call_vlm(verification_messages)
                print(f"[VLM] 复核响应: {verification_content}")

                verification_json = _parse_json_object(verification_content)
                verification_text = json.dumps(verification_json, ensure_ascii=False) if verification_json is not None else _strip_code_fences(verification_content)

                verification_match = _parse_bool_flag(verification_json.get("match", 0)) if verification_json is not None else False
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
                "capture_fps": YOLO_DETECT_FPS
            }

            with self._lock:
                self._latest_result = final_result
            print(f"[VLM RESULT] {final_result}")
            print(f"[VLM DONE] Frames: {len(frames)}, Total: {first_api_time_ms + verification_api_time_ms:.0f}ms")
            

        except Exception as exc:
            empty_result = {
                "mode": "video" if USE_VIDEO_MODE else "image",
                "api_time_ms": 0.0,
                "total_api_time_ms": 0.0,
                "server_response": {"result": '{"has_person": -1, "violations": []}'},
                "media_processing_time_ms": 0.0,
                "verification_media_processing_time_ms": 0.0,
                "frame_count": 0,
                "capture_fps": YOLO_DETECT_FPS
            }
            with self._lock:
                self._latest_result = empty_result
            print(f"[VLM ERROR] Inference failed: {exc}")

    def _inference_loop(self):
        if not self._open_stream():
            print("[FATAL] Cannot open stream, thread exiting.")
            return

        fail_count = 0
        temp_yolo_path = self.save_dir / "yolo_temp.png"
        self._last_yolo_time = time.time() 

        while not self._stop_event.is_set():
            now = time.time()

            ok, frame = self._cap.read()
            if not ok:
                fail_count += 1
                if fail_count >= self.max_fail_count:
                    if not self._open_stream():
                        time.sleep(5)
                    fail_count = 0
                    self._valid_frames_buffer = []
                continue

            # 从左侧裁剪0-2000px，然后缩放至 MAX_WIDTH
            h, w = frame.shape[:2]
            crop_w = min(w, 2000)
            if crop_w > 0:
                frame = frame[:, 0:crop_w]
                new_h = int(h * (MAX_WIDTH / crop_w))
                frame = cv2.resize(frame, (MAX_WIDTH, new_h))

            fail_count = 0
            self._is_healthy = True

            # 1. 频率控制：未到检测时间点则跳过
            if now - self._last_yolo_time < self._yolo_interval:
                continue 
            
            # 2. 执行 YOLO 检测
            self._last_yolo_time = now
            person_detected = False
            try:
                start_time = time.time()
                cv2.imwrite(str(temp_yolo_path), frame)
                person_detected = True
                print('[YOLO] Person detected:', person_detected,'cost time:', time.time() - start_time)
            except Exception as e:
                print(f"YOLO detection error: {e}")

            with self._lock:
                self._person_detected = person_detected

            # 3. 连续性状态机
            if person_detected:
                self._valid_frames_buffer.append(frame.copy())
                
                if len(self._valid_frames_buffer) >= REQUIRED_FRAMES:
                    frames_to_process = self._valid_frames_buffer[-REQUIRED_FRAMES:]
                    self._valid_frames_buffer = []
                    
                    print(f"[验证]" f"{REQUIRED_FRAMES} 连续帧已收集，正在进行 VLM 推理...")
                    
                    self._process_vlm_inference(frames_to_process)
            else:
                if self._valid_frames_buffer:
                    self._valid_frames_buffer = []
                
                # 更新状态为无人
                empty_result = {
                    "mode": "video" if USE_VIDEO_MODE else "image",
                    "api_time_ms": 0.0,
                    "total_api_time_ms": 0.0,
                    "server_response": {"result": '{"has_person": 0, "violations": []}'},
                    "media_processing_time_ms": 0.0,
                    "verification_media_processing_time_ms": 0.0,
                    "frame_count": 0,
                    "capture_fps": YOLO_DETECT_FPS
                }
                with self._lock:
                    self._latest_result = empty_result

            time.sleep(0.001)

        if self._cap is not None:
            self._cap.release()
        try:
            temp_yolo_path.unlink(missing_ok=True)
        except Exception:
            pass
        print("Inference loop stopped.")

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()
        mode = "VIDEO" if USE_VIDEO_MODE else "IMAGE"
        print(f"FLV client started. [Mode: {mode}] [YOLO: {YOLO_DETECT_FPS}FPS] [Sync Blocking VLM]")

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def get_latest_result(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._latest_result

    def get_person_detected(self) -> bool:
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