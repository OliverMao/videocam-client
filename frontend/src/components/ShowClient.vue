<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { fetchShowClientData, parseInferenceResult, getWebSocketUrl, getViolationInfo } from '../api'
import type { ShowClientData, InferenceResult } from '../api/types'
import StreamPlayer from './StreamPlayer.vue'

const data = ref<ShowClientData | null>(null)
const parsedResult = ref<InferenceResult | null>(null)
const error = ref<string | null>(null)
const loading = ref(false)
const streamStatus = ref<'connecting' | 'playing' | 'error' | 'idle'>('idle')

const detectionItems: { key: string; label: string; icon: string }[] = [
  { key: '吸烟', label: '吸烟', icon: '' },
  { key: '打架', label: '打架', icon: '' },
  { key: '着火', label: '着火', icon: '' },
  { key: '摔倒', label: '摔倒', icon: '' },
]

const violationKeys = computed(() => new Set(parsedResult.value?.violations ?? []))

let intervalId: number | null = null

function onStreamError(msg: string) {
  error.value = msg
  streamStatus.value = 'error'
}

function onStreamConnected() {
  error.value = null
  streamStatus.value = 'playing'
}

function onStreamDisconnected() {
  streamStatus.value = 'idle'
}

async function loadData() {
  try {
    loading.value = true
    error.value = null
    const result = await fetchShowClientData()
    data.value = result
    parsedResult.value = parseInferenceResult(result.server_response.result)
  } catch (e) {
    error.value = e instanceof Error ? e.message : '加载数据失败'
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  loadData()
  intervalId = window.setInterval(loadData, 2000)
})

onUnmounted(() => {
  if (intervalId) {
    clearInterval(intervalId)
  }
})
</script>

<template>
  <div class="show-client">
    <div class="header">
      <h1>AI Flow 隐私相机演示系统</h1>
      <div class="status-bar">
        <span :class="['status-dot', { active: streamStatus === 'playing' }]"></span>
        <span>{{ streamStatus === 'playing' ? '已连接' : streamStatus === 'connecting' ? '连接中...' : streamStatus === 'error' ? '连接异常' : '等待中' }}</span>
      </div>
    </div>

    <div v-if="error && streamStatus !== 'playing'" class="error-panel">{{ error }}</div>

    <div v-if="data" class="content">
      <div class="panel video-panel">
        <StreamPlayer
          :ws-url="getWebSocketUrl()"
          @error="onStreamError"
          @connected="onStreamConnected"
          @disconnected="onStreamDisconnected"
        />
      </div>

      <div class="panel description-panel">
        <h2>画面描述</h2>
        <div class="description-content">
          {{ parsedResult?.description || data.server_response.result }}
        </div>
      </div>

      <div class="panel status-panel">
        <h2>画面状态</h2>
        <div class="detection-list">
          <div
            v-for="item in detectionItems"
            :key="item.key"
            class="detection-item"
            :class="{ violated: violationKeys.has(item.key) }"
          >
            <span class="detection-icon">{{ item.icon }}</span>
            <span class="detection-label">{{ item.label }}</span>
            <span v-if="violationKeys.has(item.key)" class="detection-badge" :style="{ background: getViolationInfo(item.key).color }">异常</span>
            <span v-else class="detection-badge normal">正常</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.show-client {
  min-height: 100vh;
  background: #0f172a;
  color: #e2e8f0;
  padding: 2rem;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
}

.header h1 {
  font-size: 1.5rem;
  font-weight: 600;
  margin: 0;
}

.status-bar {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.875rem;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #ef4444;
}

.status-dot.active {
  background: #22c55e;
}

.error-panel {
  background: #7f1d1d;
  padding: 1rem;
  border-radius: 0.5rem;
  margin-bottom: 1rem;
}

.content {
  display: grid;
  grid-template-columns: 6fr 3fr 2fr;
  gap: 1rem;
  height: calc(100vh - 8rem);
}

.panel {
  background: #1e293b;
  border-radius: 0.75rem;
  padding: 1rem;
  overflow: hidden;
}

.panel h2 {
  font-size: 1rem;
  font-weight: 600;
  margin: 0 0 0.75rem 0;
  color: #94a3b8;
}

.video-panel {
  display: flex;
  align-items: center;
  justify-content: center;
  background: #000;
  padding: 0;
  overflow: hidden;
}

.description-content {
  font-size: 1.2rem;
  line-height: 1.6;
  overflow-y: auto;
  max-height: 100%;
}

.status-panel {
  display: flex;
  flex-direction: column;
}

.detection-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.detection-item {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 0.6rem 0.75rem;
  border-radius: 0.5rem;
  font-size: 1.2rem;
  background: #0f172a;
  border: 1px solid #1e293b;
}

.detection-item.violated {
  background: #7f1d1d;
  border-color: #ef4444;
}

.detection-icon {
  font-size: 1.1rem;
  line-height: 1;
}

.detection-label {
  flex: 1;
  font-weight: 500;
}

.detection-badge {
  font-size: 1rem;
  padding: 0.15rem 0.5rem;
  border-radius: 0.25rem;
  color: #fff;
  font-weight: 600;
}

.detection-badge.normal {
  background: #22c55e;
}
</style>