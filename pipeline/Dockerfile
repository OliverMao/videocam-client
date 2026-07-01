FROM python:3.11-slim

# aiortc 拉 WebRTC 视频轨依赖 GStreamer 运行时 + 常用编解码
# ffmpeg 给 RTMP 推流兜底
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgstreamer1.0-0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libavdevice-dev \
    libopus0 \
    libvpx7 \
    libx264-164 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# 安装 uv 并用 uv 安装 Python 依赖
RUN pip install uv --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && uv pip install --system --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY pipeline.py .

CMD ["uv","run", "pipeline.py", "--fps", "2"]
