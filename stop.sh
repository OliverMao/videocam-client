#!/bin/bash
set -e

GST_PID_FILE="/tmp/gst_v4l2_rtsp.pid"

if [ -f "$GST_PID_FILE" ]; then
    PID=$(cat "$GST_PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "停止 GStreamer 推流 (PID $PID)..."
        kill "$PID"
        sleep 1
        kill -9 "$PID" 2>/dev/null || true
    fi
    rm -f "$GST_PID_FILE"
fi

echo "停止 docker-compose..."
docker compose down

echo "=== 全部已停止 ==="