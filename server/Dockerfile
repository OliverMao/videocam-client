FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0t64 libsm6 libxext6 libxrender1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install uv -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN uv pip install --system --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
COPY . .

EXPOSE 28001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "28001"]
