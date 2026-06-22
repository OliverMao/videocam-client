#!/bin/bash
set -e

GST_PID_FILE="/tmp/gst_v4l2_rtsp.pid"
GST_LOG_FILE="/tmp/gst_v4l2_rtsp.log"

echo "=== 启动 docker-compose（mediamtx、app、frontend-dev 等）==="
docker compose up -d

# 等待 mediamtx 就绪（根据实际情况调整等待时间）
sleep 2

if [ -f "$GST_PID_FILE" ] && kill -0 $(cat "$GST_PID_FILE") 2>/dev/null; then
    echo "GStreamer 推流已在运行 (PID $(cat $GST_PID_FILE))"
else
    echo "=== 启动 GStreamer 摄像头推流 ==="
    nohup gst-launch-1.0 -e \
        v4l2src device=/dev/video0 do-timestamp=true \
        ! image/jpeg,width=1920,height=1080,framerate=30/1 \
        ! jpegdec \
        ! nvvidconv \
        ! "video/x-raw(memory:NVMM),format=NV12" \
        ! nvv4l2h264enc bitrate=4000000 maxperf-enable=1 insert-sps-pps=true \
        ! h264parse \
        ! rtspclientsink location=rtsp://127.0.0.1:8554/cam_main \
        > "$GST_LOG_FILE" 2>&1 &
    echo $! > "$GST_PID_FILE"
    echo "GStreamer 已启动 (PID $(cat $GST_PID_FILE))"
fi

echo "=== 全部启动完毕 ==="