<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch } from 'vue'
import flvjs from 'flv.js'

const props = defineProps<{
  streamUrl: string // 原 wsUrl 改为 streamUrl
}>()

const emit = defineEmits<{
  (e: 'error', error: string): void
  (e: 'connected'): void
  (e: 'disconnected'): void
}>()

const videoRef = ref<HTMLVideoElement | null>(null)
const status = ref<'connecting' | 'playing' | 'error' | 'idle'>('idle')
let player: flvjs.Player | null = null

function cleanup() {
  if (player) {
    player.pause()
    player.unload()
    player.detachMediaElement()
    player.destroy()
    player = null
  }
  status.value = 'idle'
}

function initPlayer() {
  cleanup()
  if (!videoRef.value || !props.streamUrl) {
    console.warn('[StreamPlayer] No video ref or stream URL')
    return
  }

  console.log('[StreamPlayer] Initializing with URL:', props.streamUrl)

  if (!flvjs.isSupported()) {
    const msg = '当前浏览器不支持 FLV 播放 (可能需要 MSE 支持)'
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
      enableWorker: false, // <--- 关键修改：禁用 Worker，避免打包路径错误
      enableStashBuffer: false, 
      stashInitialSize: 128,
      autoCleanupSourceBuffer: true,
      lazyLoad: false,
      lazyLoadMaxDuration: 0
    }
  )

    player.attachMediaElement(videoRef.value)
    
    // 监听 flv.js 内部事件
    player.on(flvjs.Events.ERROR, (errType, errDetail, errInfo) => {
      console.error('[FLV Error]', errType, errDetail, errInfo)
      status.value = 'error'
      emit('error', `FLV 错误: ${errType} - ${errDetail}`)
      // 不要立即 cleanup，允许重试或查看状态
    })

    player.on(flvjs.Events.MEDIA_INFO, (info) => {
      console.log('[FLV Media Info]', info)
      // 检查编码是否为 avc1 (H.264)
      if (info.videoCodec !== 7 && info.videoCodec !== 'avc1') {
         console.warn('非 H.264 编码，flv.js 可能无法播放')
      }
    })

    player.load()
    player.play().catch((e) => {
      console.warn('[StreamPlayer] Autoplay prevented:', e)
      // 浏览器策略阻止自动播放，通常需要用户交互
    })

  } catch (e) {
    console.error('[StreamPlayer] Init exception', e)
    status.value = 'error'
    emit('error', '播放器初始化失败')
  }
}

onMounted(() => {
  initPlayer()
})

onUnmounted(() => {
  cleanup()
})

// 支持动态切换流地址
watch(() => props.streamUrl, (newUrl, oldUrl) => {
  if (newUrl && newUrl !== oldUrl) {
    initPlayer()
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