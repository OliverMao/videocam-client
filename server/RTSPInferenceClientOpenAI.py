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

from Loop.yolo import YOLODetector

# ================= 全局模式配置 =================
USE_VIDEO_MODE = False 

# YOLO 检测频率 (Hz)
YOLO_DETECT_FPS = 3 
# ================================================

DEFAULT_OPENAI_API_BASE = "http://116.238.240.2:32726/v1"
DEFAULT_OPENAI_API_KEY = "vllm"
DEFAULT_MODEL_NAME = "/ddn/gemini/gemini-sharedata/space/wqmu4k88unnm/guarded_files/songhuan/Models/Qwen3.6-35B-A3B"
DEFAULT_INTERVAL_SEC = 5.0
MAX_WIDTH = 1024
MAX_FAIL_COUNT = 2

# 视频/图像采集配置
REQUIRED_FRAMES = 4  # 固定采集4帧有效的“有人”帧

# DEFAULT_SYSTEM_PROMPT = """
# 图片中的场景是TeleAI的展厅。请详细分析视频内容，判断是否存在违规行为。违规行为包括但不限于：吸烟、打架、着火、摔倒四类违规行为。

# 首先，统计画面中是否有人物存在。如果没有人，输出 {"has_person": 1, "violations": []}。如果有人物存在，输出 {"has_person": 1, "violations": [...]}，其中 violations 数组列出所有发现的违规行为（如 ["吸烟"] 或 ["吸烟","打架"]）。如果没有任何违规行为，violations 应为空数组 []。
# 严格按照上述格式输出 JSON，不要添加任何多余的文本或解释。 /no_think

# """

# 因为yolo连续4帧检测到人物，所以VLM直接判断是否存在违规行为，不再让VLM判断是否有人，避免重复和矛盾的输出

DEFAULT_SYSTEM_PROMPT = """
图片中的场景是TeleAI的展厅。请详细分析视频内容，判断是否存在违规行为。违规行为包括但不限于：吸烟、打架、着火、摔倒四类违规行为。

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

        self.detector = YOLODetector("./Loop/yolo26n.pt")
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
        if self._cap is not None:
            self._cap.release()
        print(f"[INFO] Opening RTSP stream: {self.rtsp_url}")
        self._cap = cv2.VideoCapture(self.rtsp_url)
        if not self._cap.isOpened():
            print("[ERROR] Failed to open RTSP stream")
            return False
        print("[INFO] RTSP stream opened successfully")
        return True

    def _build_vlm_messages(self, frames: List[Any]) -> tuple[List[Dict], float]:
        t_start = time.time()
        user_content = []

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
        
        user_content.append({"type": "text", "text": "请分析这段监控内容。"})
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        return messages, (time.time() - t_start) * 1000

    def _process_vlm_inference(self, frames: List[Any]):
        """同步阻塞执行 VLM 推理，适配 QPS=1 的场景"""
        if not frames:
            return
        try:
            t_total_start = time.time()
            messages, media_processing_time_ms = self._build_vlm_messages(frames)

            # 同步阻塞等待 API 返回
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=2048,
                temperature=0.01,
                response_format={"type": "json_object"}
            )
            print(f"[VLM] VLM完成，")
            print(f"[VLM] API响应: {response.choices[0].message.content.strip()}")  
            total_api_time_ms = (time.time() - t_total_start) * 1000

            content = response.choices[0].message.content.strip()
            if content.startswith("```json"): content = content[7:]
            if content.endswith("```"): content = content[:-3]
            
            try:
                model_json = json.loads(content.strip())
                result_text = json.dumps(model_json, ensure_ascii=False) \
                    if "description" in model_json and "violations" in model_json \
                    else model_json.get("result", content)
            except json.JSONDecodeError:
                result_text = content

            final_result = {
                "mode": "video" if USE_VIDEO_MODE else "image",
                "total_api_time_ms": total_api_time_ms,
                "server_response": {"result": result_text},
                "media_processing_time_ms": media_processing_time_ms,
                "frame_count": len(frames),
                "capture_fps": YOLO_DETECT_FPS
            }

            with self._lock:
                self._latest_result = final_result
            print(f"[VLM RESULT] {final_result}")
            print(f"[VLM DONE] Frames: {len(frames)}, Total: {total_api_time_ms:.0f}ms")
            

        except Exception as exc:
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
                person_detected = self.detector.has_person(temp_yolo_path)
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
                    
                    # print(f"[TRIGGER] {REQUIRED_FRAMES} continuous frames collected. Blocking for VLM...")  中文
                    print(f"[验证]" f"{REQUIRED_FRAMES} 连续帧已收集，正在进行 VLM 推理...")
                    
                    # 【关键修改】同步阻塞调用，不再开启新线程
                    # 在 VLM 返回结果前，整个采集循环会暂停，天然符合 QPS=1 的限制
                    self._process_vlm_inference(frames_to_process)
            else:
                if self._valid_frames_buffer:
                    self._valid_frames_buffer = []

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
        print(f"RTSP client started. [Mode: {mode}] [YOLO: {YOLO_DETECT_FPS}FPS] [Sync Blocking VLM]")

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