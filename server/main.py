import asyncio
import logging
import queue
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from qa_agent import QAAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

JETSON = False  # 是否在 Jetson 上运行
if JETSON:
    from RTSPInferenceClientOpenAIJetson import RTSPInferenceClient
    RTSP_URL = "http://srs:8080/live/livestream.flv"  # Jetson 上使用 v4l2rtspserver 输出的本地设备路径
else:
    from RTSPInferenceClientOpenAI import RTSPInferenceClient
    RTSP_URL = "rtsp://admin:qazwsx168@192.168.158.195:554/Streaming/Channels/101"


DEFAULT_OPENAI_API_BASE = "http://116.238.240.2:30630"
DEFAULT_OPENAI_API_KEY = "vllm"
DEFAULT_MODEL_NAME = "Qwen3.6-35B-A3B"


class ServerResponse(BaseModel):
    result: str
    api_time_ms: float
    inference_time_ms: float

class ShowClientResponse(BaseModel):
    vision_split_time_ms: float
    total_api_time_ms: float
    server_response: ServerResponse
    image_processing_time_ms: float

inference_client: Optional[RTSPInferenceClient] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_client
    inference_client = RTSPInferenceClient(RTSP_URL)
    inference_client.start()
    print("Inference client started.")
    yield
    if inference_client:
        inference_client.stop()
        print("Inference client stopped.")

# @asynccontextmanager
# def lifespan(app: FastAPI):
#     global inference_client
#     print("Inference client started.")
#     yield

# app = FastAPI(title="VideoCam API", lifespan=lifespan)
app = FastAPI(title="VideoCam API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# @app.get("/api/show-client", response_model=Optional[ShowClientResponse])
@app.get("/api/show-client")
async def get_show_client():
    return {
            "mode": "image",
            "total_api_time_ms": 4305.08279800415,
            "server_response": {
                "result": "\n{\"has_person\": 0, \"violations\": []}\n"
            },
            "media_processing_time_ms": 91.82310104370117,
            "frame_count": 4,
            "capture_fps": 3
            }
    if inference_client is None:
        raise HTTPException(status_code=503, detail="Client not initialized")
    if not inference_client.is_healthy:
        raise HTTPException(status_code=503, detail="Stream not ready yet")
    
    raw = inference_client.get_latest_result()
    if raw is None:
        return {
            "mode": "image",
            "total_api_time_ms": 4305.08279800415,
            "server_response": {
                "result": "\n{\"has_person\": 0, \"violations\": []}\n"
            },
            "media_processing_time_ms": 91.82310104370117,
            "frame_count": 4,
            "capture_fps": 3
            }
    return raw


# --- Anthropic 智能体 ---
qa_agent = QAAgent(
        base_url=DEFAULT_OPENAI_API_BASE,
        api_key=DEFAULT_OPENAI_API_KEY,
        model=DEFAULT_MODEL_NAME,
)


@app.post("/api/qa/ask")
async def qa_ask(request: Request):
    body = await request.json()
    question = body.get("question", "").strip()
    logger.info(f"[QA] 收到请求, question='{question}', history_len={len(body.get('history', []))}")
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    history = body.get("history", [])
    result = await qa_agent.ask(question, history)
    return JSONResponse(result)


@app.post("/api/qa/ask-stream")
async def qa_ask_stream(request: Request):
    body = await request.json()
    question = body.get("question", "").strip()
    logger.info(f"[QA][Stream] 收到请求, question='{question}'")
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    history = body.get("history", [])
    return StreamingResponse(
        qa_agent.ask_stream(question, history),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=28001)
