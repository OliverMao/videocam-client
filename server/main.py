import asyncio
import queue
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

JETSON = True  # 是否在 Jetson 上运行
if JETSON:
    from RTSPInferenceClientOpenAIJetson import RTSPInferenceClient
    RTSP_URL = "http://srs:8080/live/livestream.flv"  # Jetson 上使用 v4l2rtspserver 输出的本地设备路径
else:
    from RTSPInferenceClientOpenAI import RTSPInferenceClient
    RTSP_URL = "rtsp://admin:qazwsx168@192.168.158.195:554/Streaming/Channels/101"



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

app = FastAPI(title="VideoCam API", lifespan=lifespan)
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
    try:
        # ==========================================
        # 核心修复：在这里将 raw 数据转换为 ShowClientResponse 需要的格式
        # 请根据 print(raw) 的实际输出调整下面的 key 名称
        # ==========================================
        
        # 如果 result 字段是 JSON 字符串，可能需要先解析
        result_str = raw.get("result", "{}")
        
        mapped_data = {
            # 1. 映射顶层缺失字段 (根据报错，这些字段在 raw 中不存在或名称不同)
            "vision_split_time_ms": raw.get("vision_split_time_ms", 0.0), 
            "image_processing_time_ms": raw.get("image_processing_time_ms", 0.0),
            "total_api_time_ms": raw.get("total_time_ms", 0.0),  # ⚠️ 请确认 raw 中的实际 key
            
            # 2. 构造嵌套的 server_response 对象
            "server_response": {
                "result": result_str,
                # ⚠️ 以下两个字段在 raw 中缺失，请确认实际 key 或提供默认值
                "api_time_ms": raw.get("api_time_ms", 0.0),      
                "inference_time_ms": raw.get("inference_time_ms", 0.0), 
            }
        }

        return ShowClientResponse(**mapped_data)
        
    except Exception as e:
        # 建议打印原始数据以便调试字段名
        print(f"[DEBUG] Raw data structure: {json.dumps(raw, indent=2, default=str)}")
        raise HTTPException(status_code=500, detail=f"Invalid result format: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=28001)
