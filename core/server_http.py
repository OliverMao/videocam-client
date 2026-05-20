from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import uvicorn
from transformers import AutoProcessor

from Qwen3ModelSplit import Qwen3VLForSplitInference
from Server import run_split_packet_inference
import time
APP_NAME = "FastAPI Server Inference"
APP_VERSION = "0.1.0"

DEFAULT_CHECKPOINT = (
	"/workspace/s/ddn/gemini/gemini-sharedata/space/wqmu4k88unnm/guarded_files/jzy/models/Qwen3-VL-2B-Instruct"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
	print("Loading server model...")
	model = Qwen3VLForSplitInference.from_pretrained(
		DEFAULT_CHECKPOINT,
		device_map="auto",
		trust_remote_code=True,
		mode="server",
	)
	processor = AutoProcessor.from_pretrained(DEFAULT_CHECKPOINT, trust_remote_code=True)

	app.state.model = model
	app.state.processor = processor
	app.state.device = model.device

	try:
		yield
	finally:
		app.state.model = None
		app.state.processor = None
		app.state.device = None


app = FastAPI(
	title=APP_NAME,
	version=APP_VERSION,
	lifespan=lifespan,
)

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.get("/")
async def root():
	return {
		"message": "FastAPI is running",
		"name": APP_NAME,
		"version": APP_VERSION,
	}


@app.get("/health")
async def health_check():
	return {"status": "ok"}


@app.post("/infer")
async def infer_packet(file: UploadFile = File(...), max_new_tokens: int = 200):
	api_start_time = time.time()
	model = app.state.model
	processor = app.state.processor

	if model is None or processor is None:
		raise HTTPException(status_code=503, detail="Model is not ready")

	if not file.filename or not file.filename.endswith(".pt"):
		raise HTTPException(status_code=400, detail="Expected a .pt file")

	data = await file.read()
	try:
		packet = torch.load(BytesIO(data), map_location="cpu")
	except Exception as exc:
		raise HTTPException(status_code=400, detail=f"Invalid packet: {exc}")
	inference_start_time = time.time()
	result = run_split_packet_inference(model, processor, packet, max_new_tokens=max_new_tokens)
	inference_end_time = time.time()
	api_end_time = time.time()
	return {
		"result": result,
		"api_time_ms": (api_end_time - api_start_time)*1000,
		"inference_time_ms": (inference_end_time - inference_start_time)*1000
	}


if __name__ == "__main__":
	uvicorn.run(app, host="0.0.0.0", port=8000)
