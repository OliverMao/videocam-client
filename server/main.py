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

@app.websocket("/api/ws/stream")
async def websocket_stream(websocket: WebSocket):
    if inference_client is None:
        await websocket.close(code=1013, reason="Client not initialized")
        return

    await websocket.accept()

    q = inference_client.create_stream_queue()
    try:
        while True:
            try:
                chunk = q.get_nowait()
                await websocket.send_bytes(chunk)
            except queue.Empty:
                await asyncio.sleep(0.01)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        inference_client.remove_stream_queue(q)
        try:
            await websocket.close()
        except:
            pass

@app.get("/api/stream-sse")
async def stream_sse():
    if inference_client is None:
        raise HTTPException(status_code=503, detail="Client not initialized")

    async def event_generator():
        import asyncio
        last_frame = None
        while True:
            frame = inference_client.get_latest_frame()
            if frame and frame != last_frame:
                last_frame = frame
                import base64
                b64 = base64.b64encode(frame).decode()
                yield f"data: image/jpeg;base64,{b64}\n\n"
            await asyncio.sleep(0.1)

    from fastapi.responses import StreamingResponse
    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=28001)
