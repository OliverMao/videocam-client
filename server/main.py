import asyncio
import queue
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from RTSPInferenceClient import RTSPInferenceClient

RTSP_URL = "rtsp://admin:qazwsx168@192.168.158.195:554/Streaming/Channels/101"
INFER_URL = "http://116.238.240.2:31676/packet"

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
    inference_client = RTSPInferenceClient(RTSP_URL, INFER_URL)
    inference_client.start()
    print("Inference client started.")
    yield
    if inference_client:
        inference_client.stop()
        print("Inference client stopped.")

app = FastAPI(title="VideoCam API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/show-client", response_model=Optional[ShowClientResponse])
async def get_show_client():
    if inference_client is None:
        raise HTTPException(status_code=503, detail="Client not initialized")
    if not inference_client.is_healthy:
        raise HTTPException(status_code=503, detail="Stream not ready yet")
    result = inference_client.get_latest_result()
    if result is None:
        return None
    try:
        return ShowClientResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invalid result format: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=28001)
