<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'
import flvjs from 'flv.js'

const props = defineProps({
  streamUrl: {
    type: String,
    required: true
  }
})

const emit = defineEmits(['error', 'connected', 'disconnected'])

const videoRef = ref(null)
const status = ref('idle')

let player = null
let refreshTimer = null
let isRefreshing = false

// 清理播放器实例
function cleanup() {
  if (player) {
    try {
      player.pause()
      player.unload()
      player.detachMediaElement()
      player.destroy()
    } catch (e) {
      console.warn('[StreamPlayer] Cleanup FLV error', e)
    }
    player = null
  }
  status.value = 'idle'
}

// 清除定时器
function clearRefreshTimer() {
  if (refreshTimer !== null) {
    clearTimeout(refreshTimer)
    refreshTimer = null
  }
  isRefreshing = false
}

// 初始化播放器
function initPlayer() {
  cleanup()
  if (!videoRef.value || !props.streamUrl) {
    console.warn('[StreamPlayer] No video ref or stream URL')
    return
  }
  if (!flvjs.isSupported()) {
    const msg = '当前浏览器不支持 FLV 播放'
    console.error(msg)
    status.value = 'error'
    emit('error', msg)
    return
  }
  status.value = 'connecting'
  try {
    player = flvjs.createPlayer(
      {
        type: 'flv',
        url: props.streamUrl,
        isLive: true,
        hasAudio: false
      },
      {
        enableWorker: false,
        enableStashBuffer: true,
        stashInitialSize: 1024,
        autoCleanupSourceBuffer: true,
        lazyLoad: false,
        lazyLoadMaxDuration: 0,
        fixAudioTimestampGap: false
      }
    )
    player.attachMediaElement(videoRef.value)
    // 错误监听
    player.on(flvjs.Events.ERROR, (errType, errDetail, errInfo) => {
      console.error('[FLV Error]', errType, errDetail, errInfo)
      if (errType === flvjs.ErrorTypes.NETWORK_ERROR) {
        status.value = 'error'
        emit('error', `网络错误: ${errDetail}`)
        scheduleReconnect()
      } else if (errType === flvjs.ErrorTypes.MEDIA_ERROR) {
        status.value = 'error'
        emit('error', `媒体错误: ${errDetail}`)
        scheduleReconnect()
      }
    })
    // 加载成功/开始播放监听
    player.on(flvjs.Events.MEDIA_INFO, (info) => {
      console.log('[FLV Media Info]', info)
      status.value = 'playing'
      emit('connected')
    })
    player.load()
    player.play().catch((e) => {
      console.warn('[StreamPlayer] Autoplay prevented:', e)
    })
  } catch (e) {
    console.error('[StreamPlayer] Init exception', e)
    status.value = 'error'
    emit('error', '播放器初始化失败')
  }
}

// 错误触发重连（带防抖）
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
  // 组件卸载时：1. 清除定时器 2. 销毁播放器
  clearRefreshTimer()
  cleanup()
})

// 支持动态切换流地址
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
</style>