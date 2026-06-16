<script setup>
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { fetchShowClientData, parseInferenceResult, getWebSocketUrl } from '../api/index.js'
import StreamPlayer from './StreamPlayer.vue'
import QAZone from './QAZone.vue'
import { Cigarette, Swords as Fight, Flame as Fire, PersonStanding as Fall } from 'lucide-vue-next'

const streamUrl = import.meta.env.VITE_STREAM_URL
const combinedStreamUrl = import.meta.env.VITE_COMBINED_STREAM_URL
const is_Jetson = import.meta.env.VITE_IS_JETSON === 'true'
const loadDataInterval = parseInt(import.meta.env.VITE_LOAD_DATA_INTERVAL || '500', 10)
const typingDuration = parseInt(import.meta.env.VITE_TYPING_DURATION || '500', 10)
const freshInterval = parseInt(import.meta.env.VITE_FRESH_INTERVAL || '3000', 10)

const streamMode = ref('main')
const activeStreamUrl = computed(() =>
  streamMode.value === 'combined' ? combinedStreamUrl : streamUrl
)
function toggleStream() {
  streamMode.value = streamMode.value === 'main' ? 'combined' : 'main'
}

const data = ref(null)
const parsedResult = ref(null)
const error = ref(null)
const loading = ref(false)
const streamStatus = ref('idle')
const location = 'TeleAI-展厅'

const descriptionHistory = ref([])
let entryId = 0
const MAX_HISTORY = 20

const detectionItems = [
  { key: '吸烟', label: '吸烟监测', iconComponent: Cigarette },
  { key: '打架', label: '冲突识别', iconComponent: Fight },
  { key: '摔倒', label: '跌倒监测', iconComponent: Fall },
]

const violationKeys = computed(() => new Set(parsedResult.value?.violations ?? []))

let intervalId = null
let lastAddTime = 0
let lastHasPerson = null
let lastViolationsStr = ''

function onStreamError(msg) { error.value = msg; streamStatus.value = 'error' }
function onStreamConnected() { error.value = null; streamStatus.value = 'playing' }
function onStreamDisconnected() { streamStatus.value = 'idle' }

const ESTIMATED_ITEM_HEIGHT = 80
const RENDER_BUFFER = 20
const historyContainer = ref(null)
const scrollTop = ref(0)
const containerHeight = ref(0)
const totalHeight = computed(() => descriptionHistory.value.length * ESTIMATED_ITEM_HEIGHT)
const visibleRange = computed(() => {
  const len = descriptionHistory.value.length
  if (len === 0) return { start: 0, end: 0 }
  const start = Math.max(0, Math.floor(scrollTop.value / ESTIMATED_ITEM_HEIGHT) - RENDER_BUFFER)
  const end = Math.min(len, Math.ceil((scrollTop.value + containerHeight.value) / ESTIMATED_ITEM_HEIGHT) + RENDER_BUFFER)
  return { start, end }
})
const visibleItems = computed(() => descriptionHistory.value.slice(visibleRange.value.start, visibleRange.value.end))
const offsetY = computed(() => visibleRange.value.start * ESTIMATED_ITEM_HEIGHT)

function onScroll() {
  if (historyContainer.value) scrollTop.value = historyContainer.value.scrollTop
}
function updateContainerHeight() {
  if (historyContainer.value) containerHeight.value = historyContainer.value.clientHeight
}

async function loadData() {
  if (loading.value) return
  try {
    loading.value = true
    error.value = null
    const result = await fetchShowClientData()
    data.value = result
    parsedResult.value = parseInferenceResult(result.server_response.result)
    if (parsedResult.value?.description) {
      const fullText = parsedResult.value.description
      const currentViolationsStr = (parsedResult.value.violations || []).join(',')
      const currentHasPerson = parsedResult.value.hasPerson
      const nowMs = Date.now()
      const timeSinceLastAdd = nowMs - lastAddTime
      const isChanged = currentHasPerson !== lastHasPerson || currentViolationsStr !== lastViolationsStr
      const isTimeout = timeSinceLastAdd >= freshInterval
      if (isChanged || isTimeout) {
        lastAddTime = nowMs
        lastHasPerson = currentHasPerson
        lastViolationsStr = currentViolationsStr
        console.log('New inference result:', fullText, 'Violations:', parsedResult.value.violations)
        const now = new Date()
        const timeStr = now.toLocaleTimeString('zh-CN', { hour12: false }) + '.' + String(now.getMilliseconds()).padStart(3, '0')
        const newEntry = {
          id: entryId++,
          time: timeStr,
          description: fullText,
          displayedDescription: '█',
          violations: parsedResult.value.violations ?? [],
        }
        descriptionHistory.value.unshift(newEntry)
        if (descriptionHistory.value.length > MAX_HISTORY) descriptionHistory.value.length = MAX_HISTORY
        const targetEntry = descriptionHistory.value[0]
        const duration = typingDuration
        const totalChars = fullText.length
        if (totalChars > 0) {
          const startTime = Date.now()
          const timerId = setInterval(() => {
            const elapsed = Date.now() - startTime
            const progress = Math.min(1, elapsed / duration)
            const currentChars = Math.floor(progress * totalChars)
            if (progress >= 1) {
              targetEntry.displayedDescription = fullText
              clearInterval(timerId)
            } else {
              targetEntry.displayedDescription = fullText.slice(0, currentChars) + '█'
            }
          }, 40)
        } else {
          targetEntry.displayedDescription = fullText
        }
      }
    }
  } catch (e) {
    error.value = e instanceof Error ? e.message : '加载数据失败'
  } finally {
    loading.value = false
  }
}

watch(descriptionHistory, () => {
  nextTick(() => {
    if (historyContainer.value) {
      historyContainer.value.scrollTop = 0
      scrollTop.value = 0
    }
  })
}, { deep: true, flush: 'post' })

onMounted(() => {
  loadData()
  intervalId = window.setInterval(loadData, loadDataInterval)
  updateContainerHeight()
  window.addEventListener('resize', updateContainerHeight)
})
onUnmounted(() => {
  if (intervalId) clearInterval(intervalId)
  window.removeEventListener('resize', updateContainerHeight)
})
</script>

<template>
  <!-- 增加 :class 判断，为 Jetson 环境注入专属样式类 -->
  <div class="tech-dashboard" :class="{ 'is-jetson': is_Jetson }">
    <div class="bg-layer">
      <div class="dashboard-bg"></div>
      <div class="dashboard-grid-pulse"></div>
      <div class="corner-glow corner-glow-tl"></div>
      <div class="corner-glow corner-glow-br"></div>
      <div class="corner-glow corner-glow-tr"></div>
      <div class="light-beam"></div>
      <div class="particle particle-1"></div>
      <div class="particle particle-2"></div>
      <div class="particle particle-3"></div>
      <div class="particle particle-4"></div>
      <div class="particle particle-5"></div>
      <div class="particle particle-6"></div>
      <div class="particle particle-7"></div>
      <div class="particle particle-8"></div>
      <div class="particle particle-9"></div>
      <div class="particle particle-10"></div>
      <div class="particle particle-11"></div>
      <div class="particle particle-12"></div>
      <div class="glow-orb glow-orb-1"></div>
      <div class="glow-orb glow-orb-2"></div>
    </div>

    <header class="tech-header">
      <div style="display: flex; align-items: center; gap: 1rem; justify-content: center;">
        <div>
          <img src="/AIFlow.png" alt="AI FLOW" style="width: 53px;">
        </div>
        <div class="brand" v-if="!is_Jetson">
          <h1>TeleAI 隐私相机演示系统</h1>
        </div>
        <div class="brand" v-else>
          <h1>TeleAI 隐私相机</h1>
        </div>
      </div>
    </header>

    <div v-if="error && streamStatus !== 'playing'" class="tech-alert">
      <span>{{ error }}</span>
    </div>

    <!-- QA -->
    <QAZone />

    <div v-if="data || loading" class="content-grid">
      <section class="tech-panel video-panel" v-if="!is_Jetson">
        <div class="panel-frame">
          <div class="panel-content">
            <div class="video-top-bar">
              <span class="video-tip-text">原始视频仅用于展厅展示，云端无法获取原始视频。</span>
              <button class="stream-toggle-btn" @click="toggleStream">
                <span class="toggle-label">{{ streamMode === 'main' ? '原始画面' : '隐私处理' }}</span>
                <span class="toggle-indicator">
                  <span class="toggle-dot" :class="{ active: streamMode === 'combined' }"></span>
                </span>
              </button>
            </div>
            <StreamPlayer
              :privacy-mode="'True'" privacy-grid-cols="1" privacy-grid-rows="1" privacy-cell-width="2560"
              privacy-cell-height="1440" privacy-crop-width="2000" :streamUrl="streamUrl" @error="onStreamError"
              @connected="onStreamConnected" @disconnected="onStreamDisconnected" />
            <StreamPlayer
              :style="{ position: 'absolute', width: streamMode === 'combined' ? '100%' : '1px', height: streamMode === 'combined' ? '100%' : '1px' }"
              :privacy-mode="'True'" privacy-grid-cols="2" privacy-grid-rows="2" privacy-cell-width="2560"
              privacy-cell-height="1440" privacy-crop-width="2000" :streamUrl="combinedStreamUrl" @error="onStreamError"
              @connected="onStreamConnected" @disconnected="onStreamDisconnected" />
          </div>
        </div>
      </section>

      <section class="tech-panel desc-panel" v-if="!is_Jetson">
        <div class="panel-frame">
          <div class="panel-header">
            <div class="header-left">
              <span class="header-title">画面语义解析</span>
            </div>
            <span class="header-tag">AI INFERENCE</span>
          </div>
          <div class="panel-content desc-content" ref="historyContainer" @scroll="onScroll">
            <div v-if="descriptionHistory.length === 0" class="placeholder">等待模型推理...</div>
            <div v-else class="virtual-list" :style="{ height: totalHeight + 'px' }">
              <div class="virtual-list-inner" :style="{ transform: 'translateY(' + offsetY + 'px)' }">
                <div class="list-wrapper">
                  <div v-for="entry in visibleItems" :key="entry.id" class="history-entry"
                    :class="{ latest: entry.id === descriptionHistory[0]?.id }">
                    <div class="entry-meta">
                      <span class="entry-location">{{ location }}</span>
                      <span class="entry-time">{{ entry.time }}</span>
                    </div>
                    <p class="desc-text">{{ entry.displayedDescription !== undefined ? entry.displayedDescription :
                      entry.description }}</p>
                    <div v-if="entry.violations.length > 0" class="entry-violations">
                      <span v-for="v in entry.violations" :key="v" class="violation-tag">{{ v }}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section class="tech-panel status-panel">
        <div class="panel-frame">
          <div class="panel-header">
            <div class="header-left">
              <span class="header-title">异常行为监测</span>
            </div>
            <span class="header-tag">VIOLATION DET</span>
          </div>
          <div class="panel-content detection-grid">
            <div v-for="item in detectionItems" :key="item.key" class="detection-card"
              :class="{ active: violationKeys.has(item.key) }">
              <div class="card-top">
                <component :is="item.iconComponent" class="card-icon-svg" />
                <span class="card-label">{{ item.label }}</span>
              </div>
              <div class="card-status-badge" :class="{ active: violationKeys.has(item.key) }">
                <span class="status-dot"></span>
                <span class="status-text">
                  {{ violationKeys.has(item.key) ? '监测到异常' : '正常' }}
                </span>
              </div>
              <div v-if="violationKeys.has(item.key)" class="alert-watermark">WARN</div>
              <div v-if="violationKeys.has(item.key)" class="scan-line"></div>
            </div>
          </div>
        </div>
      </section>
    </div>
  </div>
</template>

<style scoped>
.tech-dashboard {
  --bg-deep: #05051a;
  --bg-mid: #0a0a2e;
  --bg-panel: rgba(10, 15, 50, 0.75);
  --border-tech: rgba(100, 100, 255, 0.35);
  --accent-blue: #6366f1;
  --accent-purple: #a855f7;
  --accent-cyan: #22d3ee;
  --safe: #34d399;
  --alert: #f43f5e;
  --text-primary: #f1f5f9;
  --text-secondary: #94a3b8;
  --font-tech: 'JetBrains Mono', 'Consolas', monospace;
  --font-ui: system-ui, -apple-system, sans-serif;

  position: relative;
  width: 100%;
  min-height: 100vh;
  background-color: var(--bg-deep);
  color: var(--text-primary);
  font-family: var(--font-ui);
  padding: clamp(1rem, 2vw, 1.5rem);
  overflow: visible;
  display: flex;
  flex-direction: column;
  box-sizing: border-box;
}

.bg-layer {
  position: absolute;
  inset: 0;
  pointer-events: none;
}

.dashboard-bg {
  position: absolute;
  inset: 0;
  background:
    radial-gradient(ellipse at 20% 30%, rgba(99, 102, 241, 0.15) 0%, transparent 50%),
    radial-gradient(ellipse at 80% 70%, rgba(168, 85, 247, 0.1) 0%, transparent 40%),
    repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(99, 102, 241, 0.08) 40px),
    repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(99, 102, 241, 0.08) 40px);
}

.dashboard-grid-pulse {
  position: absolute;
  inset: 0;
  pointer-events: none;
  background:
    repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(34, 211, 238, 0.08) 40px),
    repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(34, 211, 238, 0.08) 40px);
  animation: gridPulse 3s ease-in-out infinite;
}

@keyframes gridPulse {

  0%,
  100% {
    opacity: 0.4;
  }

  50% {
    opacity: 1;
  }
}

.particle {
  position: absolute;
  border-radius: 50%;
  pointer-events: none;
  box-shadow: 0 0 6px currentColor;
}

.particle-1 {
  width: 5px;
  height: 5px;
  color: var(--accent-cyan);
  background: var(--accent-cyan);
  top: 15%;
  left: 10%;
  animation: particleFloat1 8s ease-in-out infinite;
}

.particle-2 {
  width: 4px;
  height: 4px;
  color: var(--accent-blue);
  background: var(--accent-blue);
  top: 45%;
  left: 5%;
  animation: particleFloat2 11s ease-in-out infinite;
}

.particle-3 {
  width: 6px;
  height: 6px;
  color: var(--accent-purple);
  background: var(--accent-purple);
  top: 70%;
  left: 20%;
  animation: particleFloat1 9s ease-in-out infinite 1s;
}

.particle-4 {
  width: 4px;
  height: 4px;
  color: var(--accent-cyan);
  background: var(--accent-cyan);
  top: 30%;
  right: 15%;
  animation: particleFloat2 10s ease-in-out infinite 2s;
}

.particle-5 {
  width: 5px;
  height: 5px;
  color: var(--accent-purple);
  background: var(--accent-purple);
  top: 60%;
  right: 8%;
  animation: particleFloat1 7s ease-in-out infinite 0.5s;
}

.particle-6 {
  width: 3px;
  height: 3px;
  color: var(--accent-cyan);
  background: var(--accent-cyan);
  top: 85%;
  right: 25%;
  animation: particleFloat2 12s ease-in-out infinite 3s;
}

.particle-7 {
  width: 5px;
  height: 5px;
  color: var(--accent-blue);
  background: var(--accent-blue);
  top: 10%;
  left: 50%;
  animation: particleFloat1 9s ease-in-out infinite 4s;
}

.particle-8 {
  width: 4px;
  height: 4px;
  color: var(--accent-purple);
  background: var(--accent-purple);
  top: 50%;
  left: 80%;
  animation: particleFloat2 8s ease-in-out infinite 1.5s;
}

.particle-9 {
  width: 5px;
  height: 5px;
  color: var(--accent-purple);
  background: var(--accent-purple);
  top: 8%;
  left: 25%;
  animation: particleFloat1 7s ease-in-out infinite 0.5s;
}

.particle-10 {
  width: 4px;
  height: 4px;
  color: var(--accent-cyan);
  background: var(--accent-cyan);
  top: 12%;
  right: 40%;
  animation: particleFloat2 9s ease-in-out infinite 2s;
}

.particle-11 {
  width: 6px;
  height: 6px;
  color: var(--accent-blue);
  background: var(--accent-blue);
  top: 5%;
  right: 15%;
  animation: particleFloat1 8s ease-in-out infinite 1.5s;
}

.particle-12 {
  width: 3px;
  height: 3px;
  color: var(--accent-cyan);
  background: var(--accent-cyan);
  top: 16%;
  left: 75%;
  animation: particleFloat2 10s ease-in-out infinite 3s;
}

@keyframes particleFloat1 {

  0%,
  100% {
    opacity: 0;
    transform: translateY(0) translateX(0) scale(1);
  }

  20% {
    opacity: 1;
  }

  50% {
    opacity: 0.7;
    transform: translateY(-50px) translateX(25px) scale(1.8);
    box-shadow: 0 0 12px currentColor;
  }

  80% {
    opacity: 1;
  }
}

@keyframes particleFloat2 {

  0%,
  100% {
    opacity: 0;
    transform: translateY(0) translateX(0) scale(1);
  }

  20% {
    opacity: 0.8;
  }

  50% {
    opacity: 0.5;
    transform: translateY(35px) translateX(-20px) scale(1.5);
    box-shadow: 0 0 10px currentColor;
  }

  80% {
    opacity: 0.8;
  }
}

.corner-glow {
  position: absolute;
  border-radius: 50%;
  filter: blur(70px);
  pointer-events: none;
  opacity: 0.7;
}

.corner-glow-tl {
  width: 400px;
  height: 400px;
  background: radial-gradient(circle, rgba(99, 102, 241, 0.5), transparent 70%);
  top: -180px;
  left: -120px;
  animation: cornerPulse 3s ease-in-out infinite alternate;
}

.corner-glow-br {
  width: 450px;
  height: 450px;
  background: radial-gradient(circle, rgba(168, 85, 247, 0.45), transparent 70%);
  bottom: -180px;
  right: -120px;
  animation: cornerPulse 3.5s ease-in-out infinite alternate-reverse;
}

.corner-glow-tr {
  width: 350px;
  height: 350px;
  background: radial-gradient(circle, rgba(34, 211, 238, 0.35), transparent 70%);
  top: -100px;
  right: 10%;
  animation: cornerPulse 4s ease-in-out infinite alternate 1s;
}

@keyframes cornerPulse {
  0% {
    opacity: 0.3;
    transform: scale(1);
  }

  100% {
    opacity: 0.8;
    transform: scale(1.15);
  }
}

.light-beam {
  position: absolute;
  top: 0;
  left: -20%;
  width: 25%;
  height: 100%;
  pointer-events: none;
  background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.12) 30%, rgba(34, 211, 238, 0.15) 50%, rgba(168, 85, 247, 0.08) 70%, transparent);
  transform: skewX(-15deg);
  animation: beamSweep 6s ease-in-out infinite;
}

@keyframes beamSweep {
  0% {
    left: -25%;
    opacity: 0;
  }

  8% {
    opacity: 1;
  }

  45% {
    left: 110%;
    opacity: 1;
  }

  55% {
    opacity: 0;
  }
}

.glow-orb {
  position: absolute;
  border-radius: 50%;
  filter: blur(90px);
  pointer-events: none;
  animation: orbFloat 14s ease-in-out infinite alternate;
}

.glow-orb-1 {
  width: 450px;
  height: 450px;
  background: radial-gradient(circle, rgba(99, 102, 241, 0.35), transparent 70%);
  top: -10%;
  left: -5%;
}

.glow-orb-2 {
  width: 400px;
  height: 400px;
  background: radial-gradient(circle, rgba(168, 85, 247, 0.3), transparent 70%);
  bottom: -10%;
  right: -5%;
  animation-delay: -7s;
}

@keyframes orbFloat {
  0% {
    transform: translate(0, 0) scale(1);
  }

  100% {
    transform: translate(40px, 30px) scale(1.05);
  }
}

@keyframes scanMove {
  0% {
    transform: translateY(-10vh);
  }

  100% {
    transform: translateY(110vh);
  }
}

.tech-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-bottom: 1rem;
  border-bottom: 1px solid var(--border-tech);
  margin-bottom: 1.5rem;
}

.brand {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.brand-tag {
  font-family: var(--font-tech);
  font-size: 1.25rem;
  font-weight: 900;
  color: var(--accent-cyan);
  text-transform: uppercase;
}

.brand h1 {
  font-size: clamp(1.25rem, 2vw, 1.75rem);
  font-weight: 700;
  margin: 0;
  background: linear-gradient(90deg, #e0e7ff, #c4b5fd, #a78bfa);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.status-pill {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.4rem 0.8rem;
  background: rgba(10, 15, 50, 0.8);
  border: 1px solid var(--border-tech);
  border-radius: 2rem;
  font-family: var(--font-tech);
  font-size: 0.8rem;
}

.pulse-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #475569;
  transition: all 0.3s ease;
}

.pulse-dot.playing {
  background: var(--safe);
  box-shadow: 0 0 10px var(--safe);
  animation: pulse 2s infinite;
}

.pulse-dot.connecting {
  background: var(--accent-cyan);
  animation: pulse 0.8s infinite;
}

.pulse-dot.error {
  background: var(--alert);
  box-shadow: 0 0 10px var(--alert);
}

.divider {
  color: var(--border-tech);
}

.sys-id {
  color: var(--text-secondary);
}

@keyframes pulse {

  0%,
  100% {
    opacity: 1;
    transform: scale(1);
  }

  50% {
    opacity: 0.5;
    transform: scale(1.3);
  }
}

.tech-alert {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem 1rem;
  background: rgba(244, 63, 94, 0.15);
  border: 1px solid rgba(244, 63, 94, 0.4);
  border-radius: 0.5rem;
  color: #fda4af;
  font-family: var(--font-tech);
  margin-bottom: 1rem;
  animation: fadeIn 0.3s ease;
}

.alert-icon-svg {
  width: 18px;
  height: 18px;
  stroke: #fda4af;
}

.header-icon-svg {
  width: 18px;
  height: 18px;
  stroke: var(--accent-cyan);
}

.card-icon-svg {
  width: 28px;
  height: 28px;
  stroke: var(--accent-cyan);
  transition: stroke 0.3s ease, filter 0.3s ease;
}

.detection-card.active .card-icon-svg {
  stroke: var(--alert);
  filter: drop-shadow(0 0 6px rgba(244, 63, 94, 0.6));
}

.content-grid {
  display: grid;
  grid-template-columns: 2fr 1fr 1fr;
  gap: 1.25rem;
  height: calc(100vh - 8rem);
  min-height: 0;
}

.tech-panel {
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.panel-frame {
  position: relative;
  flex: 1;
  min-height: 0;
  background: var(--bg-panel);
  border: 1px solid var(--border-tech);
  border-radius: 0.75rem;
  overflow: hidden;
  backdrop-filter: blur(16px);
  display: flex;
  flex-direction: column;
}

.panel-frame::before {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: 0.75rem;
  padding: 1px;
  background: linear-gradient(135deg, transparent 35%, var(--accent-blue) 50%, var(--accent-purple) 65%, transparent 75%);
  -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  -webkit-mask-composite: xor;
  mask-composite: exclude;
  pointer-events: none;
  opacity: 0.5;
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.75rem 1rem;
  background: rgba(5, 5, 30, 0.6);
  border-bottom: 1px solid var(--border-tech);
}

.header-left {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.header-title {
  font-size: 0.9rem;
  font-weight: 600;
  letter-spacing: 0.05em;
}

.header-tag {
  font-family: var(--font-tech);
  font-size: 0.8rem;
  color: var(--accent-purple);
  opacity: 0.8;
  letter-spacing: 0.1em;
}

.panel-content {
  flex: 1;
  min-height: 0;
  padding: 1rem;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.video-panel .panel-content {
  padding: 0;
  background: #020210;
  position: relative;
  display: flex;
  flex-direction: column;
}

.video-panel :deep(video),
.video-panel :deep(canvas) {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.video-top-bar {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  z-index: 20;
  display: flex;
  align-items: center;
  justify-content: space-between;
  background-color: #ff0000;
  padding: 6px 12px;
  pointer-events: none;
}

.video-tip-text {
  color: #ffffff;
  font-size: 28px;
  font-weight: 500;
  letter-spacing: 0.5px;
}

.stream-toggle-btn {
  pointer-events: auto;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.3rem 0.8rem;
  background: rgba(5, 5, 30, 0.75);
  border: 1px solid var(--border-tech);
  border-radius: 2rem;
  color: var(--text-primary);
  font-family: var(--font-tech);
  font-size: 0.75rem;
  letter-spacing: 0.05em;
  cursor: pointer;
  backdrop-filter: blur(8px);
  transition: all 0.25s ease;
  flex-shrink: 0;
}

.stream-toggle-btn:hover {
  background: rgba(99, 102, 241, 0.25);
  border-color: var(--accent-blue);
}

.toggle-indicator {
  display: flex;
  align-items: center;
  width: 28px;
  height: 16px;
  background: rgba(99, 102, 241, 0.2);
  border-radius: 8px;
  position: relative;
  transition: background 0.25s ease;
}

.toggle-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: var(--accent-cyan);
  box-shadow: 0 0 6px var(--accent-cyan);
  position: absolute;
  left: 2px;
  transition: all 0.25s ease;
}

.toggle-dot.active {
  left: 14px;
  background: var(--accent-purple);
  box-shadow: 0 0 6px var(--accent-purple);
}

.combined-labels {
  position: absolute;
  inset: 0;
  z-index: 15;
  pointer-events: none;
}

.combined-label {
  position: absolute;
  padding: 4px 14px;
  background: rgba(0, 0, 0, 0.6);
  color: #fff;
  font-family: var(--font-ui);
  font-size: 0.85rem;
  font-weight: 600;
  letter-spacing: 0.05em;
  border-radius: 4px;
  backdrop-filter: blur(4px);
}

.label-top {
  top: 4%;
  left: 50%;
  transform: translateX(-50%);
}

.label-bl {
  top: 54%;
  left: 3%;
}

.label-br {
  top: 54%;
  left: 53%;
}

.desc-content {
  font-size: clamp(0.95rem, 1.1vw, 1.15rem);
  line-height: 1.8;
  color: var(--text-secondary);
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 1rem;
  flex: 1;
  min-height: 0;
}

.virtual-list {
  position: relative;
  flex: 1;
}

.virtual-list-inner {
  position: relative;
  will-change: transform;
}

.list-wrapper {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.history-entry {
  border-bottom: 1px solid rgba(99, 102, 241, 0.15);
  padding-bottom: 0.75rem;
  flex-shrink: 0;
}

.history-entry.latest {
  border-bottom-color: rgba(99, 102, 241, 0.35);
  animation: slideIn 0.35s cubic-bezier(0.34, 1.56, 0.64, 1);
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(-16px);
  }

  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.entry-meta {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 0.4rem;
  font-family: var(--font-tech);
  font-size: 0.75rem;
}

.entry-location {
  color: var(--accent-cyan);
  font-weight: 600;
  letter-spacing: 0.05em;
}

.entry-time {
  color: var(--text-secondary);
  opacity: 0.7;
}

.entry-violations {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin-top: 0.4rem;
}

.violation-tag {
  display: inline-block;
  padding: 0.2rem 0.6rem;
  font-size: 0.9rem;
  font-weight: 600;
  font-family: var(--font-ui);
  color: var(--alert);
  background: rgba(244, 63, 94, 0.08);
  border: 1px solid rgba(244, 63, 94, 0.3);
  border-radius: 0.25rem;
}

.desc-text {
  animation: fadeIn 0.5s ease;
  border-left: 2px solid var(--accent-blue);
  padding-left: 0.75rem;
  color: #cbd5e1;
  margin: 0;
}

.placeholder {
  color: var(--text-secondary);
  opacity: 0.5;
  font-style: italic;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(5px);
  }

  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.desc-content {
  scrollbar-width: none;
}

.desc-content::-webkit-scrollbar {
  display: none;
}

.detection-grid {
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
}

.detection-card {
  background: rgba(5, 10, 40, 0.6);
  border: 1px solid rgba(99, 102, 241, 0.2);
  border-radius: 0.75rem;
  padding: 1.25rem;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

@property --border-angle {
  syntax: "<angle>";
  inherits: true;
  initial-value: 0deg;
}

.detection-card.active {
  background:
    radial-gradient(circle at 90% 10%, rgba(244, 63, 94, 0.15) 0%, transparent 60%),
    repeating-linear-gradient(45deg, rgba(244, 63, 94, 0.03) 0, rgba(244, 63, 94, 0.03) 1px, transparent 1px, transparent 6px),
    linear-gradient(135deg, rgba(244, 63, 94, 0.1) 0%, rgba(5, 10, 40, 0.8) 100%);
  border-color: transparent;
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
}

.alert-watermark {
  position: absolute;
  right: -5px;
  bottom: -15px;
  font-size: 5rem;
  font-family: var(--font-tech);
  font-weight: 900;
  font-style: italic;
  color: rgba(244, 63, 94, 0.2);
  pointer-events: none;
  z-index: 0;
  user-select: none;
  letter-spacing: -2px;
}

@keyframes cardScan {
  0% {
    transform: translateY(-10px);
    opacity: 0;
  }

  10% {
    opacity: 1;
  }

  90% {
    opacity: 1;
  }

  100% {
    transform: translateY(120px);
    opacity: 0;
  }
}

.detection-card.active::before {
  background: var(--alert);
  opacity: 0.8;
  width: 3px;
}

.detection-card.active::after {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: inherit;
  border: 4px solid transparent;
  background: conic-gradient(from var(--border-angle), transparent 10%, rgba(244, 63, 94, 0.15) 35%, var(--alert) 50%, transparent 60%, rgba(244, 63, 94, 0.15) 85%, var(--alert) 100%) border-box;
  -webkit-mask: linear-gradient(#fff 0 0) padding-box, linear-gradient(#fff 0 0);
  -webkit-mask-composite: xor;
  mask-composite: exclude;
  animation: flowBorder 2s linear infinite;
  pointer-events: none;
}

@keyframes flowBorder {
  to {
    --border-angle: 360deg;
  }
}

.card-top {
  position: relative;
  z-index: 1;
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 0.75rem;
}

.card-label {
  font-weight: 700;
  font-size: 1.25rem;
  color: #e2e8f0;
}

.card-status-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: rgba(52, 211, 153, 0.1);
  border: 1px solid rgba(52, 211, 153, 0.3);
  border-radius: 0.375rem;
  margin-top: 0.5rem;
  transition: all 0.3s ease;
}

.card-status-badge.active {
  background: rgba(244, 63, 94, 0.08);
  border-color: rgba(244, 63, 94, 0.25);
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--safe);
  box-shadow: 0 0 6px var(--safe);
}

.card-status-badge.active .status-dot {
  background: var(--alert);
  box-shadow: 0 0 8px rgba(244, 63, 94, 0.5);
  animation: subtlePulse 2s ease-in-out infinite;
}

.card-status-badge .status-text {
  font-family: var(--font-ui);
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--safe);
  letter-spacing: 0.02em;
}

.card-status-badge.active .status-text {
  color: var(--alert);
}

@keyframes subtlePulse {

  0%,
  100% {
    transform: scale(1);
    opacity: 1;
  }

  50% {
    transform: scale(1.2);
    opacity: 0.7;
  }
}

@media (max-width: 1024px) {
  .content-grid {
    grid-template-columns: 1fr 1fr;
  }

  .video-panel {
    grid-column: span 2;
  }
}

@media (max-width: 640px) {
  .content-grid {
    grid-template-columns: 1fr;
  }

  .video-panel {
    grid-column: span 1;
  }

  .tech-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 0.5rem;
  }
}

/* =========================================
   🚀 Jetson 展厅大屏专属优化 (预警区域重构)
   ========================================= */

/* 1. 取消原本的 3 列网格，让单个预警面板占满全屏宽度 */
.tech-dashboard.is-jetson .content-grid {
  grid-template-columns: 1fr;
}

/* 2. 放大展厅大屏下的顶部主标题 */
.tech-dashboard.is-jetson .brand h1 {
  font-size: 2.25rem;
}

/* 3. 放大面板头部文字，适配远距离观看 */
.tech-dashboard.is-jetson .status-panel .panel-header {
  padding: 1.25rem 2rem;
}

.tech-dashboard.is-jetson .status-panel .header-title {
  font-size: 1.5rem;
}

.tech-dashboard.is-jetson .status-panel .header-tag {
  font-size: 1.1rem;
}

/* 4. 增加面板内部留白 */
.tech-dashboard.is-jetson .status-panel .panel-content {
  padding: 2rem 3rem;
}

/* 5. 将垂直列表重构为横向 3 列网格 (完美适配横屏电视) */
.tech-dashboard.is-jetson .detection-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 2.5rem;
  height: 100%;
}

/* 6. 卡片内容垂直居中，并增加交互反馈 */
.tech-dashboard.is-jetson .detection-card {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
  padding: 2rem;
  transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.3s ease;
}

.tech-dashboard.is-jetson .card-top {
  flex-direction: column;
  align-items: center;
  gap: 1.5rem;
  margin-bottom: 2.5rem;
}

/* 7. 图标与字体尺寸大幅升级 */
.tech-dashboard.is-jetson .card-icon-svg {
  width: 96px;
  height: 96px;
  stroke-width: 1.5;
}

.tech-dashboard.is-jetson .card-label {
  font-size: 2rem;
  font-weight: 700;
  letter-spacing: 0.05em;
}

.tech-dashboard.is-jetson .card-status-badge {
  padding: 0.85rem 2.5rem;
  font-size: 1.35rem;
  border-radius: 0.5rem;
  gap: 0.75rem;
}

.tech-dashboard.is-jetson .status-dot {
  width: 14px;
  height: 14px;
}

/* 8. 背景水印放大，增强视觉张力 */
.tech-dashboard.is-jetson .alert-watermark {
  font-size: 14rem;
  bottom: -40px;
  right: 0;
  opacity: 0.15;
}

/* 9. 异常触发时的特效强化 (放大+阴影扩散) */
.tech-dashboard.is-jetson .detection-card.active {
  transform: scale(1.03);
  box-shadow: 0 20px 50px rgba(244, 63, 94, 0.35);
}

.tech-dashboard.is-jetson .detection-card.active .card-icon-svg {
  filter: drop-shadow(0 0 15px rgba(244, 63, 94, 0.7));
}
</style>