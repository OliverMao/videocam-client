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
const status = ref('idle')

let pc = null
let refreshTimer = null
let isRefreshing = false

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
  status.value = 'idle'
}

function clearRefreshTimer() {
  if (refreshTimer !== null) {
    clearTimeout(refreshTimer)
    refreshTimer = null
  }
  isRefreshing = false
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

defineExpose({ status })
</script>

<template>
  <div class="stream-wrapper">
    <video
      ref="videoRef"
      class="stream-video"
      :class="{ 'crop-right': cropRight }"
      muted
      autoplay
      playsinline
    />
  </div>
</template>

<style scoped>
.stream-wrapper {
  position: relative;
  width: 100%;
  height: 100%;
  background-color: #000;
  overflow: hidden;
}
.stream-video {
  width: 100%;
  height: 100%;
  object-fit: contain;
  background: #000;
}
.stream-video.crop-right {
  width: 128%;
  max-width: none;
  object-fit: cover;
  object-position: left;
}
</style>
