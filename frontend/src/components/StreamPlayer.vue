<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'

const props = defineProps<{
  wsUrl: string
}>()

const emit = defineEmits<{
  (e: 'error', error: string): void
  (e: 'connected'): void
  (e: 'disconnected'): void
}>()

const imgRef = ref<HTMLImageElement | null>(null)
const status = ref<'connecting' | 'playing' | 'error' | 'idle'>('idle')

let ws: WebSocket | null = null
let reconnectTimer: number | null = null
let currentBlobUrl: string | null = null

function cleanup() {
  if (reconnectTimer) {
    clearTimeout(reconnectTimer)
    reconnectTimer = null
  }
  if (currentBlobUrl) {
    URL.revokeObjectURL(currentBlobUrl)
    currentBlobUrl = null
  }
  if (ws) {
    ws.onclose = null
    ws.onerror = null
    ws.close()
    ws = null
  }
}

function scheduleReconnect() {
  status.value = 'idle'
  emit('disconnected')
  reconnectTimer = window.setTimeout(() => {
    connect()
  }, 1000)
}

function connect() {
  cleanup()
  if (!imgRef.value) return

  status.value = 'connecting'

  ws = new WebSocket(props.wsUrl)
  ws.binaryType = 'arraybuffer'

  ws.onopen = () => {
    status.value = 'playing'
    emit('connected')
  }

  ws.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) {
      const blob = new Blob([event.data], { type: 'image/jpeg' })
      if (currentBlobUrl) {
        URL.revokeObjectURL(currentBlobUrl)
      }
      currentBlobUrl = URL.createObjectURL(blob)
      if (imgRef.value) {
        imgRef.value.src = currentBlobUrl
      }
    }
  }

  ws.onclose = () => {
    scheduleReconnect()
  }

  ws.onerror = () => {
    status.value = 'error'
    emit('error', 'WebSocket 连接错误')
  }
}

onMounted(() => {
  connect()
})

onUnmounted(() => {
  cleanup()
})

defineExpose({ status })
</script>

<template>
  <img
    ref="imgRef"
    class="stream-image"
  />
</template>

<style scoped>
.stream-image {
  width: 100%;
  height: 100%;
  object-fit: contain;
  background: #000;
}
</style>
