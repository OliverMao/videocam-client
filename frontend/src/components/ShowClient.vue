<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { fetchShowClientData, parseInferenceResult, getWebSocketUrl } from '../api'
import type { ShowClientData, InferenceResult } from '../api/types'
import StreamPlayer from './StreamPlayer.vue'

const streamUrl = "http://192.168.153.50:8080/live/livestream.flv"

import {
  Cigarette,
  Swords as Fight,
  Flame as Fire,
  PersonStanding as Fall
} from 'lucide-vue-next'

const data = ref<ShowClientData | null>(null)
const parsedResult = ref<InferenceResult | null>(null)
const error = ref<string | null>(null)
const loading = ref(false)
const streamStatus = ref<'connecting' | 'playing' | 'error' | 'idle'>('idle')

const location = 'TeleAI-展厅'

interface HistoryEntry {
  id: number
  time: string
  description: string
  displayedDescription: string
  violations: string[]
}
const descriptionHistory = ref<HistoryEntry[]>([])
let entryId = 0
const MAX_HISTORY = 20

const detectionItems: { key: string; label: string; iconComponent: any }[] = [
  { key: '吸烟', label: '吸烟检测', iconComponent: Cigarette },
  { key: '打架', label: '冲突识别', iconComponent: Fight },
  { key: '着火', label: '火情预警', iconComponent: Fire },
  { key: '摔倒', label: '跌倒监测', iconComponent: Fall },
]

const violationKeys = computed(() => new Set(parsedResult.value?.violations ?? []))

const statusLabel = computed(() => {
  switch (streamStatus.value) {
    case 'playing': return '已连接'
    case 'connecting': return '链路建立中...'
    case 'error': return '信号中断'
    default: return '等待数据流'
  }
})

let intervalId: number | null = null

function onStreamError(msg: string) { error.value = msg; streamStatus.value = 'error' }
function onStreamConnected() { error.value = null; streamStatus.value = 'playing' }
function onStreamDisconnected() { streamStatus.value = 'idle' }

// --- Virtual list for description history ---
const ESTIMATED_ITEM_HEIGHT = 80
const RENDER_BUFFER = 20
const historyContainer = ref<HTMLElement | null>(null)
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

const visibleItems = computed(() =>
  descriptionHistory.value.slice(visibleRange.value.start, visibleRange.value.end)
)

const offsetY = computed(() => visibleRange.value.start * ESTIMATED_ITEM_HEIGHT)

function onScroll() {
  if (historyContainer.value) {
    scrollTop.value = historyContainer.value.scrollTop
  }
}

function updateContainerHeight() {
  if (historyContainer.value) {
    containerHeight.value = historyContainer.value.clientHeight
  }
}

// --- Data loading ---
async function loadData() {
  try {
    loading.value = true
    error.value = null
    const result = await fetchShowClientData()
    data.value = result
    parsedResult.value = parseInferenceResult(result.server_response.result)
    if (parsedResult.value?.description) {
      const fullText = parsedResult.value.description
      console.log('New inference result:', fullText, 'Violations:', parsedResult.value.violations)
      const isDuplicate = descriptionHistory.value.length > 0 && descriptionHistory.value[0].description === fullText
      // 由于现在是固定模板拼接，每次解析出的文案相同，移除 isDuplicate 拦截以保持列表持续追加
      // const isDuplicate = false 
      
      if (!isDuplicate) {
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
        
        if (descriptionHistory.value.length > MAX_HISTORY) {
          descriptionHistory.value.length = MAX_HISTORY
        }

        const targetEntry = descriptionHistory.value[0]
        const duration = 4000 // 严格保证在3秒内完全打印完
        const totalChars = fullText.length
        
        if (totalChars > 0) {
          const startTime = Date.now()
          const timerId = setInterval(() => {
            const elapsed = Date.now() - startTime
            const progress = Math.min(1, elapsed / duration)
            // 实时计算此时应该显示的字符数，确保3秒一定刚好输出完所有长度的字符
            const currentChars = Math.floor(progress * totalChars)
            
            if (progress >= 1) {
              targetEntry.displayedDescription = fullText
              clearInterval(timerId)
            } else {
              targetEntry.displayedDescription = fullText.slice(0, currentChars) + '█'
            }
          }, 40) // 使用肉眼感到平滑的短间隔高频刷新
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
  intervalId = window.setInterval(loadData, 5000)
  updateContainerHeight()
  window.addEventListener('resize', updateContainerHeight)
})
onUnmounted(() => {
  if (intervalId) clearInterval(intervalId)
  window.removeEventListener('resize', updateContainerHeight)
})
</script>

<template>
  <div class="tech-dashboard">
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
        <div class="brand">
          <!-- <span class="brand-tag">AI FLOW</span> -->
          <h1>TeleAI 隐私相机演示系统</h1>
        </div>
      </div>
      <!-- <div class="status-pill">
        <span class="pulse-dot" :class="streamStatus"></span>
        <span class="status-text">{{ statusLabel }}</span>
        <span class="divider">|</span>
        <span class="sys-id">SYS-04</span>
      </div> -->
    </header>

    <div v-if="error && streamStatus !== 'playing'" class="tech-alert">
      <span>{{ error }}</span>
    </div>

    <div v-if="data || loading" class="content-grid">
      <section class="tech-panel video-panel">
        <div class="panel-frame">
          <div class="panel-header">
            <div class="header-left">
              <!-- <span class="header-icon" v-html="Icons.Video"></span> -->
              <span class="header-title">本地视频流预览（仅本地可见）</span>
            </div>
            <span class="header-tag">LIVE FEED</span>
          </div>
          <div class="panel-content">
            <StreamPlayer :streamUrl="streamUrl" @error="onStreamError" @connected="onStreamConnected"
              @disconnected="onStreamDisconnected" />
          </div>
        </div>
      </section>

      <section class="tech-panel desc-panel">
        <div class="panel-frame">
          <div class="panel-header">
            <div class="header-left">
              <!-- <span class="header-icon" v-html="Icons.Brain"></span> -->
              <span class="header-title">画面语义解析</span>
            </div>
            <span class="header-tag">AI INFERENCE</span>
          </div>
          <div class="panel-content desc-content" ref="historyContainer" @scroll="onScroll">
            <div v-if="descriptionHistory.length === 0" class="placeholder">等待模型推理...</div>
            <div v-else class="virtual-list" :style="{ height: totalHeight + 'px' }">
              <div class="virtual-list-inner" :style="{ transform: 'translateY(' + offsetY + 'px)' }">
                <div class="list-wrapper">
                  <div v-for="entry in visibleItems" :key="entry.id" class="history-entry" :class="{ latest: entry.id === descriptionHistory[0]?.id }">
                    <div class="entry-meta">
                      <span class="entry-location">{{ location }}</span>
                      <span class="entry-time">{{ entry.time }}</span>
                    </div>
                    <p class="desc-text">{{ entry.displayedDescription !== undefined ? entry.displayedDescription : entry.description }}</p>
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
              <!-- <span class="header-icon" v-html="Icons.Shield"></span> -->
              <span class="header-title">异常行为监测</span>
            </div>
            <span class="header-tag">VIOLATION DET</span>
          </div>
          <div class="panel-content detection-grid">
            <!-- <div v-for="item in detectionItems" :key="item.key" class="detection-card"
              :class="{ active: 1==1 }">
              <div class="card-top">
                <component :is="item.iconComponent" class="card-icon-svg" />
                <span class="card-label">{{ item.label }}</span>
              </div>
              <div class="card-bar">
                <div class="bar-track">
                  <div class="bar-fill" :class="{ active: 1==1 }"></div>
                </div>
                <span class="bar-label"
                  :style="{ color: 1==1 ? 'var(--alert)' : 'var(--safe)' }">
                  {{ 1==1 ? '⚠ 异常' : '✓ 正常' }}
                </span>
              </div>
            </div> -->
            <div v-for="item in detectionItems" :key="item.key" class="detection-card"
              :class="{ active: violationKeys.has(item.key) }">
              <div class="card-top">
                <component :is="item.iconComponent" class="card-icon-svg" />
                <span class="card-label">{{ item.label }}</span>
              </div>
              <div class="card-bar">
                <div class="bar-track">
                  <div class="bar-fill" :class="{ active: violationKeys.has(item.key) }"></div>
                </div>
                <span class="bar-label"
                  :style="{ color: violationKeys.has(item.key) ? 'var(--alert)' : 'var(--safe)' }">
                  {{ violationKeys.has(item.key) ? '⚠ 异常' : '✓ 正常' }}
                </span>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  </div>
</template>

<style scoped>
/* ✅ 核心修复：将变量定义在组件根类，避开 scoped 的 :root 失效问题 */
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

/* Background layer wrapper to contain all effects */
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

/* Animated grid line pulsing - brighter */
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
  0%, 100% { opacity: 0.4; }
  50% { opacity: 1; }
}

/* Dynamic floating particles - bigger & brighter */
.particle {
  position: absolute;
  border-radius: 50%;
  pointer-events: none;
  box-shadow: 0 0 6px currentColor;
}

.particle-1 {
  width: 5px; height: 5px;
  color: var(--accent-cyan);
  background: var(--accent-cyan);
  top: 15%; left: 10%;
  animation: particleFloat1 8s ease-in-out infinite;
}

.particle-2 {
  width: 4px; height: 4px;
  color: var(--accent-blue);
  background: var(--accent-blue);
  top: 45%; left: 5%;
  animation: particleFloat2 11s ease-in-out infinite;
}

.particle-3 {
  width: 6px; height: 6px;
  color: var(--accent-purple);
  background: var(--accent-purple);
  top: 70%; left: 20%;
  animation: particleFloat1 9s ease-in-out infinite 1s;
}

.particle-4 {
  width: 4px; height: 4px;
  color: var(--accent-cyan);
  background: var(--accent-cyan);
  top: 30%; right: 15%;
  animation: particleFloat2 10s ease-in-out infinite 2s;
}

.particle-5 {
  width: 5px; height: 5px;
  color: var(--accent-purple);
  background: var(--accent-purple);
  top: 60%; right: 8%;
  animation: particleFloat1 7s ease-in-out infinite 0.5s;
}

.particle-6 {
  width: 3px; height: 3px;
  color: var(--accent-cyan);
  background: var(--accent-cyan);
  top: 85%; right: 25%;
  animation: particleFloat2 12s ease-in-out infinite 3s;
}

.particle-7 {
  width: 5px; height: 5px;
  color: var(--accent-blue);
  background: var(--accent-blue);
  top: 10%; left: 50%;
  animation: particleFloat1 9s ease-in-out infinite 4s;
}

.particle-8 {
  width: 4px; height: 4px;
  color: var(--accent-purple);
  background: var(--accent-purple);
  top: 50%; left: 80%;
  animation: particleFloat2 8s ease-in-out infinite 1.5s;
}

.particle-9 {
  width: 5px; height: 5px;
  color: var(--accent-purple);
  background: var(--accent-purple);
  top: 8%; left: 25%;
  animation: particleFloat1 7s ease-in-out infinite 0.5s;
}

.particle-10 {
  width: 4px; height: 4px;
  color: var(--accent-cyan);
  background: var(--accent-cyan);
  top: 12%; right: 40%;
  animation: particleFloat2 9s ease-in-out infinite 2s;
}

.particle-11 {
  width: 6px; height: 6px;
  color: var(--accent-blue);
  background: var(--accent-blue);
  top: 5%; right: 15%;
  animation: particleFloat1 8s ease-in-out infinite 1.5s;
}

.particle-12 {
  width: 3px; height: 3px;
  color: var(--accent-cyan);
  background: var(--accent-cyan);
  top: 16%; left: 75%;
  animation: particleFloat2 10s ease-in-out infinite 3s;
}

@keyframes particleFloat1 {
  0%, 100% { opacity: 0; transform: translateY(0) translateX(0) scale(1); }
  20% { opacity: 1; }
  50% { opacity: 0.7; transform: translateY(-50px) translateX(25px) scale(1.8); box-shadow: 0 0 12px currentColor; }
  80% { opacity: 1; }
}

@keyframes particleFloat2 {
  0%, 100% { opacity: 0; transform: translateY(0) translateX(0) scale(1); }
  20% { opacity: 0.8; }
  50% { opacity: 0.5; transform: translateY(35px) translateX(-20px) scale(1.5); box-shadow: 0 0 10px currentColor; }
  80% { opacity: 0.8; }
}

/* Extra corner glows - more visible */
.corner-glow {
  position: absolute;
  border-radius: 50%;
  filter: blur(70px);
  pointer-events: none;
  opacity: 0.7;
}
.corner-glow-tl {
  width: 400px; height: 400px;
  background: radial-gradient(circle, rgba(99, 102, 241, 0.5), transparent 70%);
  top: -180px; left: -120px;
  animation: cornerPulse 3s ease-in-out infinite alternate;
}
.corner-glow-br {
  width: 450px; height: 450px;
  background: radial-gradient(circle, rgba(168, 85, 247, 0.45), transparent 70%);
  bottom: -180px; right: -120px;
  animation: cornerPulse 3.5s ease-in-out infinite alternate-reverse;
}
.corner-glow-tr {
  width: 350px; height: 350px;
  background: radial-gradient(circle, rgba(34, 211, 238, 0.35), transparent 70%);
  top: -100px; right: 10%;
  animation: cornerPulse 4s ease-in-out infinite alternate 1s;
}

@keyframes cornerPulse {
  0% { opacity: 0.3; transform: scale(1); }
  100% { opacity: 0.8; transform: scale(1.15); }
}

/* Dynamic light beam sweeping left to right */
.light-beam {
  position: absolute;
  top: 0;
  left: -20%;
  width: 25%;
  height: 100%;
  pointer-events: none;
  background: linear-gradient(90deg,
    transparent,
    rgba(99, 102, 241, 0.12) 30%,
    rgba(34, 211, 238, 0.15) 50%,
    rgba(168, 85, 247, 0.08) 70%,
    transparent
  );
  transform: skewX(-15deg);
  animation: beamSweep 6s ease-in-out infinite;
}

@keyframes beamSweep {
  0% { left: -25%; opacity: 0; }
  8% { opacity: 1; }
  45% { left: 110%; opacity: 1; }
  55% { opacity: 0; }
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
  /* letter-spacing: 0.2em; */
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
  width: 20px;
  height: 20px;
  stroke: var(--accent-cyan);
  transition: stroke 0.3s ease;
}

.detection-card.active .card-icon-svg {
  stroke: var(--alert);
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
}

.video-panel :deep(video),
.video-panel :deep(canvas) {
  width: 100%;
  height: 100%;
  object-fit: contain;
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
  padding: 0.1rem 0.5rem;
  font-size: 0.75rem;
  font-family: var(--font-tech);
  color: var(--alert);
  background: rgba(244, 63, 94, 0.12);
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

.desc-content::-webkit-scrollbar {
  width: 4px;
}

.desc-content::-webkit-scrollbar-track {
  background: transparent;
}

.desc-content::-webkit-scrollbar-thumb {
  background: var(--border-tech);
  border-radius: 2px;
}

.detection-grid {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.detection-card {
  background: rgba(5, 10, 40, 0.6);
  border: 1px solid rgba(99, 102, 241, 0.2);
  border-radius: 0.5rem;
  padding: 0.75rem;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.detection-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 0px;
  height: 100%;
  background: var(--accent-blue);
  opacity: 0.5;
  transition: all 0.3s ease;
}

.detection-card.active {
  border-color: rgba(244, 63, 94, 0.5);
  background: rgba(30, 10, 40, 0.4);
  box-shadow: 0 0 20px rgba(244, 63, 94, 0.1);
}

.detection-card.active::before {
  background: var(--alert);
  opacity: 1;
  box-shadow: 0 0 8px var(--alert);
}

.card-top {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}


.card-label {
  font-weight: 600;
  font-size: 1rem;
  color: #e2e8f0;
}

.card-bar {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.bar-track {
  flex: 1;
  height: 4px;
  background: rgba(255, 255, 255, 0.08);
  border-radius: 2px;
  overflow: hidden;
}

.bar-fill {
  height: 100%;
  width: 2%;
  background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
  border-radius: 2px;
  transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1), background 0.3s ease;
}

.bar-fill.active {
  width: 100%;
  background: linear-gradient(90deg, var(--alert), #fb7185);
  box-shadow: 0 0 10px rgba(244, 63, 94, 0.4);
}

.bar-label {
  font-family: var(--font-tech);
  font-size: 1rem;
  min-width: 45px;
  font-weight: 700;
  text-align: right;
  color: var(--text-secondary);
  transition: color 0.3s ease;
}

.detection-card.active .bar-label {
  color: #fda4af;
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
</style>