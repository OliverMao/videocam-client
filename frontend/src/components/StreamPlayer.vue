<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch } from 'vue'
import flvjs from 'flv.js'

const props = defineProps<{
  streamUrl: string
}>()

const emit = defineEmits<{
  (e: 'error', error: string): void
  (e: 'connected'): void
  (e: 'disconnected'): void
}>()

const videoRef = ref<HTMLVideoElement | null>(null)
const status = ref<'connecting' | 'playing' | 'error' | 'idle'>('idle')

let player: flvjs.Player | null = null
let refreshTimer: number | null = null // 用于存储定时器ID

// 清理播放器实例
function cleanup() {
  if (player) {
    try {
      player.pause()
      player.unload()
      player.detachMediaElement()
      player.destroy()
    } catch (e) {
      console.warn('[StreamPlayer] Cleanup error', e)
    }
    player = null
  }
  status.value = 'idle'
}

// 清除定时器
function clearRefreshTimer() {
  if (refreshTimer !== null) {
    clearInterval(refreshTimer)
    refreshTimer = null
  }
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
        enableStashBuffer: false, 
        stashInitialSize: 128,
        autoCleanupSourceBuffer: true,
        lazyLoad: false,
        lazyLoadMaxDuration: 0
      }
    )

    player.attachMediaElement(videoRef.value)
    
    // 错误监听
    player.on(flvjs.Events.ERROR, (errType, errDetail, errInfo) => {
      console.error('[FLV Error]', errType, errDetail, errInfo)
      // 如果是严重错误，可以考虑在这里触发刷新，但要避免死循环
      if (errType === flvjs.ErrorTypes.NETWORK_ERROR) {
         status.value = 'error'
         emit('error', `网络错误: ${errDetail}`)
      } else if (errType === flvjs.ErrorTypes.MEDIA_ERROR) {
         status.value = 'error'
         emit('error', `媒体错误: ${errDetail}`)
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

// 强制刷新流的函数
function forceRefreshStream() {
  console.log('[StreamPlayer] Force refreshing stream...')
  // 直接重新初始化，内部会先 cleanup
  initPlayer()
}

onMounted(() => {
  initPlayer()
  
  // 启动定时器：每 60,000 毫秒 (1分钟) 刷新一次
  refreshTimer = window.setInterval(() => {
    forceRefreshStream()
  }, 60 * 1000)
})

onUnmounted(() => {
  // 组件卸载时：1. 清除定时器 2. 销毁播放器
  clearRefreshTimer()
  cleanup()
})

// 支持动态切换流地址
watch(() => props.streamUrl, (newUrl, oldUrl) => {
  if (newUrl && newUrl !== oldUrl) {
    console.log('[StreamPlayer] URL changed, re-initializing...')
    // 切换URL时，通常也需要重置定时器，从新URL开始计时
    clearRefreshTimer()
    initPlayer()
    
    // 重新启动定时器
    refreshTimer = window.setInterval(() => {
      forceRefreshStream()
    }, 60 * 1000)
  }
})

defineExpose({ status })
</script>

<template>
  <video
    ref="videoRef"
    class="stream-video"
    muted
    autoplay
    playsinline
  />
</template>

<style scoped>
.stream-video {
  width: 100%;
  height: 100%;
  object-fit: contain;
  background: #000;
}
</style>