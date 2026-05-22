import os
import torch
import uvicorn
import aiofiles
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoProcessor
import io # 新增：用于内存流处理
from PIL import Image # 新增：用于图像处理
import httpx  # 用于异步发送请求
from Qwen3ModelSplit import Qwen3VLForSplitInference
from Client import _pick_tensor
import time
# --- 配置 ---
APP_NAME = "FastAPI Simple Framework"
APP_VERSION = "0.1.0"
DEFAULT_PACKET_PATH = "packet.pt"
DEFAULT_CHECKPOINT = "/workspace/s/ddn/gemini/gemini-sharedata/space/wqmu4k88unnm/guarded_files/jzy/models/Qwen3-VL-8B-Instruct"
TMP_DIR = "./tmpfile"
# DEFAULT_TEXT = "图片中是否有违规行为，包括但不限于：1.吸烟 2.打架 3.赌博 4.其他违规行为。明确指出违规行为。输出如下：\n吸烟。\n如果图片中没有违规行为，输出如下：\n无。"

DEFAULT_TEXT = """
图片中的场景是TeleAI的展厅。请详细分析图片内容，判断是否存在违规行为。违规行为包括但不限于：吸烟、打架、着火、摔倒、赌博以及其他违规行为。

首先，统计画面中一共有几个人。对每一个人，描述其具体的动作、姿态、衣着特征（如颜色、款式、是否佩戴口罩等）以及所处的展厅位置，不要透露性别、年龄等隐私信息。
然后，细致地描述整个展厅的环境，包括：可见的展品、屏幕内容、灯光、空间布局、地面情况等。
将上述信息整合成一段连贯、详细的画面描述。
不描述画面左上角和右上角的文字。因为是摄像头画面会包含当前摄像头的位置，时间等信息，这些信息不需要描述。不要显示“02.算法羽真_沙盘”这几个字。

严格按照以下JSON格式输出，不要添加任何其他文字或解释：
{
  "description": "详细的画面描述",
  "violations": ["违规行为"]
}

如果存在违规行为，violations数组中列出所有发现的违规行为（如 ["吸烟"] 或 ["吸烟","打架"]）。如果没有任何违规行为，violations 应为空数组 []。
"""

# 假设你的后端地址
SERVER_URL = "http://127.0.0.1:8010/infer"
# 确保临时目录存在
os.makedirs(TMP_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading client model...")
    # client 模式只跑到 patch_embed，输出可传输的视觉 token。
    model = Qwen3VLForSplitInference.from_pretrained(
        DEFAULT_CHECKPOINT,
        device_map={"": "cuda:1"},
        trust_remote_code=True,
        mode="client",
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

@app.post("/packet")
async def build_packet(
    file: UploadFile = File(...), 
    prompt: str = DEFAULT_TEXT
):
    api_start_time = time.time()
    model = app.state.model
    processor = app.state.processor
    device = app.state.device

    if model is None or processor is None or device is None:
        raise HTTPException(status_code=503, detail="Model is not ready")

    # 1. 读取并处理图像
    image_start_time = time.time()
    save_path = os.path.join(TMP_DIR, "file.png")
    try:
        # 读取上传的文件内容
        content = await file.read()
        
        # 使用 PIL 打开图像
        image = Image.open(io.BytesIO(content)).convert("RGB")
        
        # 调整尺寸到 256*256
        # 注意：这会强制改变图像比例，如果原图不是正方形会变形
        # image = image.resize((256, 256), Image.Resampling.LANCZOS)
        
        # 保存调整后的图像
        image.save(save_path, format="PNG")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图像处理或保存失败: {str(e)}")

    image_end_time = time.time()
    print(f"Image processing time: {(image_end_time - image_start_time)*1000:.2f} ms")

    # 2. 构建 messages (后续逻辑保持不变)
    messages =[
        {
            "role": "system",
            "content": [{"type": "text", "text": prompt}],
        },
        {
            "role": "user",
            "content":[{"type": "image", "image": save_path}],
        },
    ]

    # 3. 处理输入
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")
    
    if pixel_values is None or image_grid_thw is None:
        raise HTTPException(status_code=400, detail="processor 未返回 pixel_values 或 image_grid_thw")

    # 4. 模型推理
    start_time = time.time()
    with torch.no_grad():
        split_output = model.model.visual(pixel_values, grid_thw=image_grid_thw)
    end_time = time.time()
    print(f"Vision split inference time: {(end_time - start_time)*1000:.2f} ms")

    if isinstance(split_output, dict):
        vision_hidden_states = split_output["hidden_states"]
        split_layer = int(split_output.get("start_layer", 1))
        deepstack_prefix = split_output.get("deepstack_features",[])
    else:
        vision_hidden_states = split_output
        split_layer = 0
        deepstack_prefix =[]

    # 5. 保存结果
    packet = {
        "vision_hidden_states": vision_hidden_states.detach().cpu(),
        "split_layer": split_layer,
        "deepstack_prefix": [x.detach().cpu() for x in deepstack_prefix],
        "input_ids": _pick_tensor(inputs, "input_ids"),
        "attention_mask": _pick_tensor(inputs, "attention_mask"),
        "mm_token_type_ids": _pick_tensor(inputs, "mm_token_type_ids"),
        "image_grid_thw": _pick_tensor(inputs, "image_grid_thw"),
    }

    # 6. 将 packet 序列化到内存中
    buffer = io.BytesIO()
    torch.save(packet, buffer)
    buffer.seek(0) # 将指针重置到开头

    # 7. 通过 HTTP 发送到后端
    try:
        async with httpx.AsyncClient(proxy=None, trust_env=False) as client:
            # 模拟 multipart/form-data 上传
            response = await client.post(
                SERVER_URL,
                files={"file": ("packet.pt", buffer, "application/octet-stream")},
                params={"max_new_tokens": 500},
                timeout=30.0 # 根据网络情况设置超时
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=f"后端推理失败: {response.text}")
            
            api_end_time = time.time()
            # print(f"Total API time (including backend inference): {(api_end_time - api_start_time)*1000:.2f} ms")
            # 直接返回后端的推理结果给客户端用户
            # {
            # "result": "吸烟。"
            # }
            server_response = response.json()
            total_response = {
                "vision_split_time_ms": (end_time - start_time)*1000,
                "total_api_time_ms": (api_end_time - api_start_time)*1000,
                "server_response": server_response,
                "image_processing_time_ms": (image_end_time - image_start_time)*1000
            }
            return total_response
    
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"无法连接到后端服务器: {str(exc)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)