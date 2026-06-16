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
  }
})

const emit = defineEmits(['error', 'connected', 'disconnected'])

const videoRef = ref(null)
const canvasRef = ref(null)
const status = ref('idle')

// ===== 裁剪配置 =====
// 源：2560×1440  →  输出：2000×1440（保留左侧 2000px）
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

// ===== canvas 渲染循环 =====
// 每帧把 video 当前画面左侧 78.125% 的区域，绘制到 2000×1440 的 canvas 上
function renderTick() {
  rafId = requestAnimationFrame(renderTick)
  if (!props.cropRight) return

  const video = videoRef.value
  const canvas = canvasRef.value
  if (!video || !canvas) return
  if (video.readyState < 2 || !video.videoWidth) return

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  // 按比例从源视频里取左侧 KEEP 比例的宽度
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
      if (props.cropRight) startRender()
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
  () => props.cropRight,
  (val) => {
    console.log('[StreamPlayer] cropRight ->', val)
    if (val) startRender()
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
      :class="{ 'is-source': cropRight }"
      muted
      autoplay
      playsinline
    />
    <!-- canvas 在 cropRight 时显示，作为实际渲染输出 -->
    <canvas
      v-show="cropRight"
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
/* cropRight 模式下：把 video 缩到 1×1 但保留播放，避免部分浏览器暂停隐藏元素 */
.stream-video.is-source {
  position: absolute;
  top: 0;
  left: 0;
  width: 1px;
  height: 1px;
  opacity: 0;
  pointer-events: none;
}
.stream-canvas {
  width: 100%;
  height: 100%;
  display: block;
  background: #000;
}
</style>
