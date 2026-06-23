#!/bin/bash
set -e

RED='\033[0;31m'
GRN='\033[0;32m'
YLW='\033[1;33m'
BLU='\033[0;34m'
NC='\033[0m'

log()  { echo -e "${BLU}[*]${NC} $*"; }
ok()   { echo -e "${GRN}[+]${NC} $*"; }
warn() { echo -e "${YLW}[!]${NC} $*"; }
err()  { echo -e "${RED}[-]${NC} $*"; }

# ---------- 1. 识别架构 ----------
detect_hw() {
    case "$(uname -s)" in
        Darwin) echo "darwin"; return ;;
    esac

    if [ -f /etc/nv_tegra_release ] || grep -qi "tegra" /proc/device-tree/model 2>/dev/null; then
        echo "jetson"; return
    fi
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        echo "nvidia"; return
    fi
    if lspci 2>/dev/null | grep -Ei "vga|3d" | grep -qi intel; then
        echo "intel"; return
    fi
    if [ -f /sys/class/drm/renderD128/device/vendor ]; then
        VENDOR=$(cat /sys/class/drm/renderD128/device/vendor 2>/dev/null || true)
        if [ "$VENDOR" = "0x8086" ]; then
            echo "intel"; return
        fi
    fi
    echo "cpu"
}

HW=$(detect_hw)
log "检测到硬件类型: ${YLW}$HW${NC}"

# ---------- 2. macOS 分支：Homebrew ----------
if [ "$HW" = "darwin" ]; then
    if ! command -v brew >/dev/null 2>&1; then
        err "macOS 需要 Homebrew：/bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi

    BREW_FORMULAE=()
    BREW_CASKS=()

    if ! command -v gst-launch-1.0 >/dev/null 2>&1; then
        warn "缺少 gst-launch-1.0"
        BREW_FORMULAE+=(gstreamer)
    fi
    if ! gst-inspect-1.0 avfvideosrc >/dev/null 2>&1; then
        warn "缺少 avfvideosrc（摄像头采集）"
        BREW_FORMULAE+=(gst-plugins-good)
    fi
    if ! gst-inspect-1.0 vtenc_h264 >/dev/null 2>&1; then
        warn "缺少 vtenc_h264（VideoToolbox H.264 硬编码）"
        BREW_FORMULAE+=(gst-plugins-bad)
    fi
    if ! gst-inspect-1.0 h264parse >/dev/null 2>&1; then
        warn "缺少 h264parse"
        BREW_FORMULAE+=(gst-plugins-base)
    fi
    if ! gst-inspect-1.0 rtspclientsink >/dev/null 2>&1; then
        warn "缺少 rtspclientsink"
        BREW_FORMULAE+=(gst-plugins-bad)
    fi

    if [ ${#BREW_FORMULAE[@]} -eq 0 ]; then
        ok "macOS 依赖已齐全，跳过安装"
        exit 0
    fi

    echo
    echo "================== 状态汇总 (macOS) =================="
    echo "  硬件类型      : darwin"
    echo "  gst-launch    : $(command -v gst-launch-1.0 >/dev/null 2>&1 && echo OK || echo MISSING)"
    echo "  avfvideosrc   : $(gst-inspect-1.0 avfvideosrc   >/dev/null 2>&1 && echo OK || echo MISSING)"
    echo "  vtenc_h264    : $(gst-inspect-1.0 vtenc_h264    >/dev/null 2>&1 && echo OK || echo MISSING)"
    echo "  h264parse     : $(gst-inspect-1.0 h264parse     >/dev/null 2>&1 && echo OK || echo MISSING)"
    echo "  rtspclientsink: $(gst-inspect-1.0 rtspclientsink >/dev/null 2>&1 && echo OK || echo MISSING)"

    echo
    echo "================== 待安装 formula =================="
    printf '  - %s\n' "${BREW_FORMULAE[@]}"
    echo "  （注：macOS 上 gst-plugins-base/good/bad 通常作为 gstreamer 的依赖自动装好）"

    echo
    read -rp "$(echo -e "${YLW}[?]${NC} 确认用 Homebrew 安装吗？[y/N] ")" CONFIRM
    case "$CONFIRM" in
        y|Y|yes|YES) ;;
        *) warn "已取消"; exit 0 ;;
    esac

    log "brew update..."
    brew update

    log "brew install..."
    brew install "${BREW_FORMULAE[@]}"

    echo
    log "安装后验证："
    gst-inspect-1.0 avfvideosrc    >/dev/null 2>&1 && ok "avfvideosrc OK"    || err "avfvideosrc 仍缺失"
    gst-inspect-1.0 vtenc_h264     >/dev/null 2>&1 && ok "vtenc_h264 OK"     || err "vtenc_h264 仍缺失"
    gst-inspect-1.0 h264parse      >/dev/null 2>&1 && ok "h264parse OK"      || err "h264parse 仍缺失"
    gst-inspect-1.0 rtspclientsink >/dev/null 2>&1 && ok "rtspclientsink OK" || err "rtspclientsink 仍缺失"

    ok "完成。可以执行：./start.sh"
    exit 0
fi

# ---------- Linux 分支：apt ----------
if [ "$EUID" -ne 0 ]; then
    err "请使用 sudo 运行：sudo $0"
    exit 1
fi

PKGS=(
    gstreamer1.0-tools
    gstreamer1.0-plugins-base
    gstreamer1.0-plugins-good
    gstreamer1.0-plugins-bad
    gstreamer1.0-plugins-ugly
)

HAS_GST=false
HAS_V4L2SRC=false
HAS_H264PARSE=false
HAS_RTSP=false

if command -v gst-launch-1.0 >/dev/null 2>&1; then HAS_GST=true; fi
if gst-inspect-1.0 v4l2src     >/dev/null 2>&1; then HAS_V4L2SRC=true; fi
if gst-inspect-1.0 h264parse   >/dev/null 2>&1; then HAS_H264PARSE=true; fi
if gst-inspect-1.0 rtspclientsink >/dev/null 2>&1; then HAS_RTSP=true; fi

[ "$HAS_GST" = false ]       && warn "缺少 gst-launch-1.0"
[ "$HAS_V4L2SRC" = false ]   && warn "缺少 v4l2src 插件（base/good）"
[ "$HAS_H264PARSE" = false ] && warn "缺少 h264parse 插件（base）"
[ "$HAS_RTSP" = false ]      && warn "缺少 rtspclientsink 插件（bad，需 gst-plugins-bad）"

case "$HW" in
    jetson)
        if ! gst-inspect-1.0 nvv4l2h264enc >/dev/null 2>&1; then
            warn "缺少 nvv4l2h264enc (Jetson 编码器)"
            PKGS+=(nvidia-l4t-gstreamer gstreamer1.0-plugins-bad)
        fi
        if ! gst-inspect-1.0 nvvidconv >/dev/null 2>&1; then
            warn "缺少 nvvidconv (Jetson 转换器)"
            PKGS+=(nvidia-l4t-gstreamer)
        fi
        ;;
    intel)
        if ! gst-inspect-1.0 vaapih264enc >/dev/null 2>&1; then
            warn "缺少 vaapih264enc"
            PKGS+=(gstreamer1.0-vaapi libva2 libva-drm2 libva-wayland2 vainfo)
        fi
        if ! command -v vainfo >/dev/null 2>&1; then
            warn "缺少 vainfo"
            PKGS+=(vainfo)
        fi
        if ! dpkg -l intel-media-va-driver 2>/dev/null | grep -q '^ii'; then
            warn "未安装 intel-media-va-driver（iHD 驱动）"
            PKGS+=(intel-media-va-driver)
        fi
        ;;
    nvidia)
        if ! gst-inspect-1.0 nvh264enc >/dev/null 2>&1; then
            warn "缺少 nvh264enc"
            PKGS+=(gstreamer1.0-plugins-bad gstreamer1.0-nvenc)
        fi
        if ! command -v nvidia-smi >/dev/null 2>&1; then
            err "未检测到 nvidia-smi：请先安装 NVIDIA 专有驱动"
        fi
        ;;
    cpu)
        if ! gst-inspect-1.0 x264enc >/dev/null 2>&1; then
            warn "缺少 x264enc"
            PKGS+=(gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly)
        fi
        ;;
esac

echo
echo "================== 状态汇总 =================="
echo "  硬件类型      : $HW"
echo "  gst-launch    : $([ "$HAS_GST" = true ] && echo OK || echo MISSING)"
echo "  v4l2src       : $([ "$HAS_V4L2SRC" = true ] && echo OK || echo MISSING)"
echo "  h264parse     : $([ "$HAS_H264PARSE" = true ] && echo OK || echo MISSING)"
echo "  rtspclientsink: $([ "$HAS_RTSP" = true ] && echo OK || echo MISSING)"

case "$HW" in
    jetson)
        echo "  nvv4l2h264enc : $(gst-inspect-1.0 nvv4l2h264enc >/dev/null 2>&1 && echo OK || echo MISSING)"
        echo "  nvvidconv     : $(gst-inspect-1.0 nvvidconv     >/dev/null 2>&1 && echo OK || echo MISSING)"
        ;;
    intel)
        echo "  vaapih264enc  : $(gst-inspect-1.0 vaapih264enc  >/dev/null 2>&1 && echo OK || echo MISSING)"
        echo "  vainfo        : $(command -v vainfo >/dev/null 2>&1 && echo OK || echo MISSING)"
        if command -v vainfo >/dev/null 2>&1; then
            if vainfo >/dev/null 2>&1; then ok "vainfo 可用（驱动 OK）"
            else warn "vainfo 运行失败，驱动可能未生效"; fi
        fi
        ;;
    nvidia)
        echo "  nvh264enc     : $(gst-inspect-1.0 nvh264enc     >/dev/null 2>&1 && echo OK || echo MISSING)"
        echo "  nvidia-smi    : $(command -v nvidia-smi >/dev/null 2>&1 && echo OK || echo MISSING)"
        ;;
    cpu)
        echo "  x264enc       : $(gst-inspect-1.0 x264enc       >/dev/null 2>&1 && echo OK || echo MISSING)"
        ;;
esac

echo
echo "================== 待安装包 =================="
if [ ${#PKGS[@]} -eq 0 ]; then
    ok "无需安装，依赖已齐全"
    exit 0
fi
printf '  - %s\n' "${PKGS[@]}"

echo
read -rp "$(echo -e "${YLW}[?]${NC} 确认安装以上包吗？[y/N] ")" CONFIRM
case "$CONFIRM" in
    y|Y|yes|YES) ;;
    *) warn "已取消"; exit 0 ;;
esac

log "更新 apt 索引..."
apt update

log "安装依赖..."
apt install -y "${PKGS[@]}"

echo
log "安装后验证："
case "$HW" in
    jetson)
        gst-inspect-1.0 nvv4l2h264enc >/dev/null 2>&1 && ok "nvv4l2h264enc OK" || err "nvv4l2h264enc 仍缺失"
        gst-inspect-1.0 nvvidconv     >/dev/null 2>&1 && ok "nvvidconv OK"     || err "nvvidconv 仍缺失"
        ;;
    intel)
        gst-inspect-1.0 vaapih264enc  >/dev/null 2>&1 && ok "vaapih264enc OK"  || err "vaapih264enc 仍缺失"
        if command -v vainfo >/dev/null 2>&1; then
            vainfo 2>&1 | head -3
        fi
        ;;
    nvidia)
        gst-inspect-1.0 nvh264enc     >/dev/null 2>&1 && ok "nvh264enc OK"     || err "nvh264enc 仍缺失"
        ;;
    cpu)
        gst-inspect-1.0 x264enc       >/dev/null 2>&1 && ok "x264enc OK"       || err "x264enc 仍缺失"
        ;;
esac

ok "完成。可以执行：./start.sh"
