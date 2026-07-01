#!/bin/bash
# 停止视频推流相关进程
# 用法: bash stop.sh
set -e

GST_PID_FILE="/tmp/gst_v4l2_rtsp.pid"
PIPELINE_PID_FILE="/tmp/pipeline.pid"
VIDEO_DEV="${VIDEO_DEV:-/dev/video0}"

# --- 1) 用 pid 文件杀 GStreamer / pipeline(精准) ---
for pidfile in "$GST_PID_FILE" "$PIPELINE_PID_FILE"; do
    if [ -f "$pidfile" ]; then
        PID=$(cat "$pidfile")
        if kill -0 "$PID" 2>/dev/null; then
            echo "停止 $(basename "$pidfile") 记录进程 (PID $PID)..."
            kill "$PID" 2>/dev/null || true
            sleep 1
            kill -9 "$PID" 2>/dev/null || true
        fi
        rm -f "$pidfile"
    fi
done

# --- 2) 兜底:把所有还在跑的相关进程都杀一遍 ---
# 2a) 占着摄像头的进程(原来要手 fuser -v 才能看到)
if [ -e "$VIDEO_DEV" ]; then
    PIDS=$(fuser "$VIDEO_DEV" 2>/dev/null || true)
    if [ -n "$PIDS" ]; then
        echo "释放 $VIDEO_DEV,占用 PID: $PIDS"
        fuser -k "$VIDEO_DEV" 2>/dev/null || true
        sleep 1
    else
        echo "$VIDEO_DEV 当前无占用"
    fi
else
    echo "$VIDEO_DEV 不存在,跳过 fuser"
fi

# 2b) GStreamer 推流进程
if pgrep -af 'gst-launch-1\.0' >/dev/null 2>&1; then
    echo "停止残留 gst-launch 进程:"
    pgrep -af 'gst-launch-1\.0'
    pkill -TERM -f 'gst-launch-1\.0' 2>/dev/null || true
    sleep 1
    pkill -KILL -f 'gst-launch-1\.0' 2>/dev/null || true
fi

# 2c) 本仓库的 pipeline 进程(uv run pipeline.py / python pipeline.py)
if pgrep -af '(uv run pipeline\.py|python[0-9.]* .*pipeline\.py)' >/dev/null 2>&1; then
    echo "停止残留 pipeline 进程:"
    pgrep -af '(uv run pipeline\.py|python[0-9.]* .*pipeline\.py)'
    pkill -TERM -f 'pipeline\.py' 2>/dev/null || true
    sleep 1
    pkill -KILL -f 'pipeline\.py' 2>/dev/null || true
fi

# 2d) ffmpeg(防止外部 FFmpeg 卡住 /dev/video0)
if pgrep -af 'ffmpeg.*video0' >/dev/null 2>&1; then
    pkill -TERM -f 'ffmpeg.*video0' 2>/dev/null || true
    sleep 1
    pkill -KILL -f 'ffmpeg.*video0' 2>/dev/null || true
fi

# --- 3) docker-compose ---
echo "停止 docker-compose..."
docker compose down

echo "=== 全部已停止 ==="
