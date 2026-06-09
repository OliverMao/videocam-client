FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# 安装 uv 并用 uv 安装 Python 依赖
RUN pip install uv --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && uv pip install --system --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY pipeline.py .

CMD ["uv","run", "pipeline.py", "stream", "--fps", "2"]