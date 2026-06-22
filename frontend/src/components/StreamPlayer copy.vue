<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'

const props = defineProps({
  streamUrl: {
    type: String,
    required: true
  },
  cropRight: {
    type: Boolean,
    default: false
  },
  privacyMode: {
    type: Boolean,
    default: false
  },
  // ===== 隐私模式（2×2 网格）配置 =====
  // 输入：privacyGridCols × privacyGridRows 个 privacyCellWidth × privacyCellHeight 子视频拼接
  // 每个子视频裁剪：保留左侧 privacyCropWidth × privacyCellHeight
  // 输出：privacyGridCols × privacyGridRows 个 privacyCropWidth × privacyCellHeight 重新拼接
  privacyGridCols: {
    type: Number,
    default: 2
  },
  privacyGridRows: {
    type: Number,
    default: 2
  },
  privacyCellWidth: {
    type: Number,
    default: 2560
  },
  privacyCellHeight: {
    type: Number,
    default: 1440
  },
  privacyCropWidth: {
    type: Number,
    default: 2000
  }
})

const emit = defineEmits(['error', 'connected', 'disconnected'])

const videoRef = ref(null)
const canvasRef = ref(null)
const status = ref('idle')

// ===== 单流裁剪配置 =====
// 源：2560×1440  →  输出：2000×1440（保留左侧）
const SRC_W = 2560
const SRC_H = 1440
const DST_W = 2000
const DST_H = 1440
const KEEP = DST_W / SRC_W // ≈ 0.78125

let pc = null
let refreshTimer = null
let isRefreshing = false
let rafId = null

function cleanup() {
  if (pc) {
    try {
      pc.close()
    } catch (e) {
      console.warn('[StreamPlayer] Close RTCPeerConnection error', e)
    }
    pc = null
  }
  if (videoRef.value) {
    videoRef.value.srcObject = null
  }
  if (rafId !== null) {
    cancelAnimationFrame(rafId)
    rafId = null
  }
  status.value = 'idle'
}

function clearRefreshTimer() {
  if (refreshTimer !== null) {
    clearTimeout(refreshTimer)
    refreshTimer = null
  }
  isRefreshing = false
}

function isCanvasMode() {
  return props.cropRight || props.privacyMode
}

// ===== canvas 渲染循环 =====
function renderTick() {
  rafId = requestAnimationFrame(renderTick)
  if (!isCanvasMode()) return

  const video = videoRef.value
  const canvas = canvasRef.value
  if (!video || !canvas) return
  if (video.readyState < 2 || !video.videoWidth) return

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  if (props.privacyMode) {
    renderPrivacyGrid(ctx, video, canvas)
  } else {
    renderCropRight(ctx, video, canvas)
  }
}

// 单流：从视频帧左侧裁出 KEEP 比例
function renderCropRight(ctx, video, canvas) {
  const sw = Math.round(video.videoWidth * KEEP)
  const sh = video.videoHeight

  if (canvas.width !== DST_W) canvas.width = DST_W
  if (canvas.height !== DST_H) canvas.height = DST_H
  ctx.clearRect(0, 0, DST_W, DST_H)
  ctx.drawImage(
    video,
    0, 0, sw, sh,        // 源矩形：从左上角开始，宽 sw 高 sh（左侧裁剪）
    0, 0, DST_W, DST_H   // 目标矩形：铺满整个 canvas
  )
}

// 隐私模式：把 2×2 网格的每个子视频都做同样的左侧裁剪，再拼回 2×2
function renderPrivacyGrid(ctx, video, canvas) {
  const cols = props.privacyGridCols
  const rows = props.privacyGridRows
  const cellW = props.privacyCellWidth
  const cellH = props.privacyCellHeight
  const cropW = props.privacyCropWidth

  // 输入总尺寸 & 输出总尺寸
  const srcTotalW = cols * cellW
  const srcTotalH = rows * cellH
  const outTotalW = cols * cropW
  const outTotalH = rows * cellH

  // 按比例换算（处理实际分辨率与预期不符的情况）
  const scaleX = video.videoWidth / srcTotalW
  const scaleY = video.videoHeight / srcTotalH
  const cellSrcW = cellW * scaleX
  const cellSrcH = cellH * scaleY
  const cropSrcW = cropW * scaleX

  if (canvas.width !== outTotalW) canvas.width = outTotalW
  if (canvas.height !== outTotalH) canvas.height = outTotalH
  ctx.clearRect(0, 0, outTotalW, outTotalH)

  // 遍历网格，每个子视频做一次"取左侧、贴到对应位置"
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      ctx.drawImage(
        video,
        col * cellSrcW, row * cellSrcH,        // 源起点（对应子视频的左上角）
        cropSrcW, cellSrcH,                     // 源矩形：宽 cropSrcW（即子视频左侧）
        col * cropW, row * cellH,               // 目标起点（拼回去对应位置）
        cropW, cellH                            // 目标矩形
      )
    }
  }
}

function startRender() {
  if (rafId === null) rafId = requestAnimationFrame(renderTick)
}

function stopRender() {
  if (rafId !== null) {
    cancelAnimationFrame(rafId)
    rafId = null
  }
}

async function initPlayer() {
  cleanup()
  if (!videoRef.value || !props.streamUrl) {
    console.warn('[StreamPlayer] No video ref or stream URL')
    return
  }

  status.value = 'connecting'

  try {
    pc = new RTCPeerConnection({
      iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
    })

    pc.ontrack = (e) => {
      videoRef.value.srcObject = e.streams[0]
      if (isCanvasMode()) startRender()
    }

    pc.oniceconnectionstatechange = () => {
      console.log('[StreamPlayer] ICE state:', pc.iceConnectionState)
      if (pc.iceConnectionState === 'connected' || pc.iceConnectionState === 'completed') {
        status.value = 'playing'
        emit('connected')
      } else if (pc.iceConnectionState === 'failed' || pc.iceConnectionState === 'disconnected') {
        status.value = 'error'
        emit('error', 'WebRTC 连接断开')
        scheduleReconnect()
      }
    }

    pc.addTransceiver('video', { direction: 'recvonly' })
    pc.addTransceiver('audio', { direction: 'recvonly' })

    const offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    const res = await fetch(props.streamUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/sdp' },
      body: offer.sdp
    })

    if (!res.ok) {
      throw new Error(`WHEP 请求失败: ${res.status}`)
    }

    const answer = await res.text()
    await pc.setRemoteDescription({ type: 'answer', sdp: answer })
  } catch (e) {
    console.error('[StreamPlayer] WebRTC init error', e)
    status.value = 'error'
    emit('error', e.message || 'WebRTC 播放器初始化失败')
    scheduleReconnect()
  }
}

function scheduleReconnect() {
  if (isRefreshing) return
  isRefreshing = true
  console.log('[StreamPlayer] Scheduling reconnect in 3s...')
  refreshTimer = window.setTimeout(() => {
    isRefreshing = false
    refreshTimer = null
    initPlayer()
  }, 3000)
}

onMounted(() => {
  initPlayer()
})

onUnmounted(() => {
  clearRefreshTimer()
  cleanup()
})

watch(
  () => props.streamUrl,
  (newUrl, oldUrl) => {
    if (newUrl && newUrl !== oldUrl) {
      console.log('[StreamPlayer] URL changed, re-initializing...')
      clearRefreshTimer()
      initPlayer()
    }
  }
)

watch(
  [() => props.cropRight, () => props.privacyMode],
  ([crop, privacy]) => {
    console.log('[StreamPlayer] mode ->', { cropRight: crop, privacyMode: privacy })
    if (crop || privacy) startRender()
    else stopRender()
  }
)

defineExpose({ status })
</script>

<template>
  <div class="stream-wrapper">
    <!-- video 永远存在，负责接收 WebRTC 流 -->
    <video
      ref="videoRef"
      class="stream-video"
      :class="{ 'is-source': cropRight || privacyMode }"
      muted
      autoplay
      playsinline
    />
    <!-- canvas 在 cropRight 或 privacyMode 时显示，作为实际渲染输出 -->
    <canvas
      v-show="cropRight || privacyMode"
      ref="canvasRef"
      class="stream-canvas"
    />
  </div>
</template>

<style scoped>
.stream-wrapper {
  position: relative;
  width: 100%;
  height: 100%;
  background: #000;
  overflow: hidden;
}
.stream-video {
  width: 100%;
  height: 100%;
  object-fit: contain;
  background: #000;
}
/* canvas 模式下：把 video 缩到 1×1 但保留播放，避免部分浏览器暂停隐藏元素 */
.stream-video.is-source {
  position: absolute;
  top: 0;
  left: 0;
  width: 1px;
  height: 1px;
  opacity: 0;
  pointer-events: none;
  visibility: visible; /* 不能是 hidden */
}
.stream-canvas {
  width: 100%;
  height: 100%;
  display: block;
  background: #000;
  object-fit: contain;
}
</style>
