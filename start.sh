#!/bin/bash
set -e

GST_PID_FILE="/tmp/gst_v4l2_rtsp.pid"
GST_LOG_FILE="/tmp/gst_v4l2_rtsp.log"
RTSP_URL="${RTSP_URL:-rtsp://127.0.0.1:8554/cam_main}"
VIDEO_DEVICE="${VIDEO_DEVICE:-/dev/video0}"
MAC_CAMERA_INDEX="${MAC_CAMERA_INDEX:-0}"
WIDTH=1920
HEIGHT=1080
FRAMERATE=30
BITRATE=4000000

detect_hardware() {
    case "$(uname -s)" in
        Darwin)
            echo "darwin"; return
            ;;
    esac

    if [ -f /etc/nv_tegra_release ] || grep -qi "tegra" /proc/device-tree/model 2>/dev/null; then
        echo "jetson"; return
    fi

    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        echo "nvidia"; return
    fi

    if lspci 2>/dev/null | grep -Ei "vga|3d|display" | grep -qi intel; then
        echo "intel"; return
    fi

    if ls /dev/dri/renderD128 >/dev/null 2>&1; then
        VENDOR=""
        for f in /sys/class/drm/renderD128/device/vendor \
                 /sys/class/drm/card0/device/vendor; do
            if [ -r "$f" ] && [ -s "$f" ]; then
                VENDOR=$(cat "$f" 2>/dev/null); break
            fi
        done

        if [ "$VENDOR" = "0x8086" ]; then
            echo "intel"; return
        fi

        if [ -z "$VENDOR" ]; then
            for lib in /usr/lib/x86_64-linux-gnu/dri /usr/lib/dri; do
                if [ -d "$lib" ] && ls "$lib"/iHD"*".so "$lib"/i965"*".so >/dev/null 2>&1; then
                    echo "intel"; return
                fi
            done

            if command -v vainfo >/dev/null 2>&1 && vainfo 2>&1 | grep -qi "intel\|iHD\|i965"; then
                echo "intel"; return
            fi

            if lsmod 2>/dev/null | grep -Eq '^i915(\s|$)'; then
                echo "intel"; return
            fi

            if [ -n "${WSL_DISTRO_NAME:-}" ] || uname -r | grep -qi "microsoft\|wsl"; then
                if [ -d /dev/dri ] && command -v gst-inspect-1.0 >/dev/null 2>&1 \
                   && gst-inspect-1.0 vaapih264enc >/dev/null 2>&1; then
                    echo "intel"; return
                fi
            fi
        fi
    fi

    echo "cpu"
}

HW=$(detect_hardware)
echo "=== 检测到硬件类型: $HW ==="

build_pipeline() {
    case "$HW" in
        darwin)
            gst-launch-1.0 -e \
                avfvideosrc device-index="$MAC_CAMERA_INDEX" do-timestamp=true \
                ! videoconvert \
                ! "video/x-raw,width=$WIDTH,height=$HEIGHT,framerate=$FRAMERATE/1,format=NV12" \
                ! vtenc_h264 bitrate=$BITRATE allow-frame-reordering=false realtime=true \
                ! h264parse \
                ! rtspclientsink location="$RTSP_URL"
            ;;
        jetson)
            gst-launch-1.0 -e \
                v4l2src device="$VIDEO_DEVICE" do-timestamp=true \
                ! "image/jpeg,width=$WIDTH,height=$HEIGHT,framerate=$FRAMERATE/1" \
                ! jpegdec \
                ! nvvidconv \
                ! "video/x-raw(memory:NVMM),format=NV12" \
                ! nvv4l2h264enc bitrate=$BITRATE maxperf-enable=1 insert-sps-pps=true \
                ! h264parse \
                ! rtspclientsink location="$RTSP_URL"
            ;;
        intel)
            gst-launch-1.0 -e \
                v4l2src device="$VIDEO_DEVICE" do-timestamp=true \
                ! "image/jpeg,width=$WIDTH,height=$HEIGHT,framerate=$FRAMERATE/1" \
                ! jpegdec \
                ! videoconvert \
                ! video/x-raw,format=NV12 \
                ! vaapih264enc bitrate=$BITRATE \
                ! h264parse \
                ! rtspclientsink location="$RTSP_URL"
            ;;
        nvidia)
            gst-launch-1.0 -e \
                v4l2src device="$VIDEO_DEVICE" do-timestamp=true \
                ! "image/jpeg,width=$WIDTH,height=$HEIGHT,framerate=$FRAMERATE/1" \
                ! jpegdec \
                ! videoconvert \
                ! video/x-raw,format=NV12 \
                ! nvh264enc bitrate=$BITRATE \
                ! h264parse \
                ! rtspclientsink location="$RTSP_URL"
            ;;
        *)
            gst-launch-1.0 -e \
                v4l2src device="$VIDEO_DEVICE" do-timestamp=true \
                ! "image/jpeg,width=$WIDTH,height=$HEIGHT,framerate=$FRAMERATE/1" \
                ! jpegdec \
                ! videoconvert \
                ! video/x-raw,format=I420 \
                ! x264enc tune=zerolatency bitrate=$BITRATE speed-preset=ultrafast \
                ! h264parse \
                ! rtspclientsink location="$RTSP_URL"
            ;;
    esac
}

echo "=== 启动 docker-compose（mediamtx、app、frontend-dev 等）==="
if [ "$HW" = "darwin" ] && ! command -v docker >/dev/null 2>&1; then
    warn "未检测到 docker，请安装 Docker Desktop for Mac"
elif [ "$HW" = "darwin" ] && ! docker info >/dev/null 2>&1; then
    warn "docker 未运行，请启动 Docker Desktop"
else
    docker compose up -d
fi

sleep 2

if [ -f "$GST_PID_FILE" ] && kill -0 $(cat "$GST_PID_FILE") 2>/dev/null; then
    echo "GStreamer 推流已在运行 (PID $(cat $GST_PID_FILE))"
else
    echo "=== 启动 GStreamer 摄像头推流 ($HW) ==="
    build_pipeline > "$GST_LOG_FILE" 2>&1 &
    echo $! > "$GST_PID_FILE"
    echo "GStreamer 已启动 (PID $(cat $GST_PID_FILE))"
fi

echo "=== 全部启动完毕 ==="
