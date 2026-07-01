<script setup>
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { fetchShowClientData, parseInferenceResult } from '../api/index.js'
import StreamPlayer from './StreamPlayer.vue'

const streamUrl = import.meta.env.VITE_STREAM_URL
const combinedStreamUrl = import.meta.env.VITE_COMBINED_STREAM_URL
const is_Jetson = import.meta.env.VITE_IS_JETSON === 'true'
const loadDataInterval = parseInt(import.meta.env.VITE_LOAD_DATA_INTERVAL || '500', 10)
const typingDuration = parseInt(import.meta.env.VITE_TYPING_DURATION || '500', 10)
const freshInterval = parseInt(import.meta.env.VITE_FRESH_INTERVAL || '3000', 10)

const streamMode = ref('main')

function toggleStream() {
  streamMode.value = streamMode.value === 'main' ? 'combined' : 'main'
}

const data = ref(null)
const parsedResult = ref(null)
const error = ref(null)
const loading = ref(false)
const streamStatus = ref('idle')
const location = 'TeleAI-展厅'
const asset = (name) => `/images/${name}`

const descriptionHistory = ref([])
let entryId = 0
const MAX_HISTORY = 20

const detectionItems = [
  { key: '摔倒', label: '摔倒监测', iconNormal: asset('摔倒n1.jpg'), iconAlert: asset('摔倒n2.jpg') },
  { key: '挥手', label: '挥手监测', iconNormal: asset('挥手n1.jpg'), iconAlert: asset('挥手n2.jpg') },
  { key: '弯腰', label: '弯腰监测', iconNormal: asset('弯腰n1.jpg'), iconAlert: asset('弯腰n2.jpg') },
  { key: '打架', label: '打架监测', iconNormal: asset('打架n1.jpg'), iconAlert: asset('打架n2.jpg') },
]

// ===== TEMP MOCK: 已关闭 =====
const MOCK_TEST = false
const MOCK_VIOLATIONS = []
// ===========================

const violationKeys = computed(() => new Set(parsedResult.value?.violations ?? []))

let intervalId = null
let lastAddTime = 0
let lastHasPerson = null
let lastViolationsStr = ''

function onStreamError(msg) { error.value = msg; streamStatus.value = 'error' }
function onStreamConnected() { error.value = null; streamStatus.value = 'playing' }
function onStreamDisconnected() { streamStatus.value = 'idle' }

const ESTIMATED_ITEM_HEIGHT = 150
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

function getDetectionIcon(key) {
  const item = detectionItems.find(d => d.key === key)
  if (!item) return ''
  return violationKeys.value.has(key) ? item.iconAlert : item.iconNormal
}

async function loadData() {
  if (loading.value) return
  try {
    loading.value = true
    error.value = null

    // ===== TEMP MOCK: 测试挥手+弯腰异常，测完删 =====
    if (MOCK_TEST) {
      const mockResult = {
        server_response: {
          result: JSON.stringify({
            has_person: 1,
            violations: MOCK_VIOLATIONS,
            description: '画面中检测到有人物举手挥动，同时上半身前倾弯腰，疑似身体不适。'
          })
        }
      }
      data.value = mockResult
      parsedResult.value = parseInferenceResult(mockResult.server_response.result)
    } else {
    // =============================================
    const result = await fetchShowClientData()
    data.value = result
    parsedResult.value = parseInferenceResult(result.server_response.result)
    } // TEMP MOCK end

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
  <div class="tech-dashboard" :class="{ 'is-jetson': is_Jetson }">
    <!-- Full-screen background image -->
    <img class="bg-image" :src="asset('bg.png')" alt="" />

    <!-- Top decorative strip -->
    <img class="frame-top" :src="asset('frame-top.png')" alt="" />

    <!-- Left / Right decorative side images -->
    <img class="side-image-left" :src="asset('左侧.png')" alt="" />
    <img class="side-image-right" :src="asset('右侧.png')" alt="" />

    <!-- Header -->
    <header class="tech-header">
      <div class="header-inner">
        <img :src="asset('AIFlow.png')" alt="AI FLOW" class="header-logo" />
        <div class="brand">
          <h1 v-if="!is_Jetson">TeleAI 隐私相机演示系统</h1>
          <h1 v-else>TeleAI 隐私相机</h1>
        </div>
      </div>
    </header>

    <!-- Error alert -->
    <div v-if="error && streamStatus !== 'playing'" class="tech-alert">
      <span>{{ error }}</span>
    </div>

    <!-- Main content grid -->
    <div v-if="data || loading" class="content-grid">
      <!-- 1. Video panel -->
      <section class="tech-panel video-panel" v-if="!is_Jetson">
        <div class="panel-frame">
          <img class="panel-title-img title-video" :src="asset('监控视频标题.png')" alt="监控视频" />
          <div class="panel-content">
            <div class="video-top-bar">
              <span class="video-tip-text"><img class="warning-icon-img" :src="asset('frame-corner.png')" alt="警告" /> 原始视频仅用于展厅展示，云端无法获取原始视频。</span>
              <div class="toggle-group">
                <span class="toggle-label">{{ streamMode === 'main' ? '原始画面' : '隐私处理' }}</span>
                <button class="stream-toggle-btn" @click="toggleStream">
                  <span class="toggle-track">
                    <span class="toggle-dot" :class="{ active: streamMode === 'combined' }"></span>
                  </span>
                </button>
              </div>
            </div>
            <div class="stream-stage">
              <!-- 两路流保持常驻，只切换显示层，避免切换到隐私处理时重新建连导致黑屏等待 -->
              <StreamPlayer
                class="stream-layer"
                :cropRight="true"
                :class="{ visible: streamMode === 'main' }"
                :streamUrl="streamUrl"
                @error="onStreamError"
                @connected="onStreamConnected"
                @disconnected="onStreamDisconnected"
              />
              <StreamPlayer
                class="stream-layer"
                :class="{ visible: streamMode === 'combined' }"
                :privacy-mode="'True'"
                privacy-grid-cols="2"
                privacy-grid-rows="2"
                privacy-cell-width="2560"
                privacy-cell-height="1440"
                privacy-crop-width="2000"
                :streamUrl="combinedStreamUrl"
                @error="onStreamError"
                @connected="onStreamConnected"
                @disconnected="onStreamDisconnected"
              />
            </div>
          </div>
        </div>
      </section>

      <!-- 2. Semantic description panel -->
      <section class="tech-panel desc-panel" v-if="!is_Jetson">
        <div class="panel-frame">
          <img class="panel-title-img title-desc" :src="asset('画面语义标题.png')" alt="画面语义解析" />
          <div class="panel-content desc-content" ref="historyContainer" @scroll="onScroll">
            <div v-if="descriptionHistory.length === 0" class="placeholder">等待模型推理...</div>
            <div v-else class="virtual-list" :style="{ height: totalHeight + 'px' }">
              <div class="virtual-list-inner" :style="{ transform: 'translateY(' + offsetY + 'px)' }">
                <div class="list-wrapper">
                  <div
                    v-for="entry in visibleItems"
                    :key="entry.id"
                    class="history-entry-wrapper"
                    :class="{ 'has-violations': entry.violations.length > 0, latest: entry.id === descriptionHistory[0]?.id }"
                  >
                    <div class="history-entry">
                      <div class="entry-meta">
                        <span class="entry-location">{{ location }}</span>
                        <span class="entry-time">{{ entry.time }}</span>
                      </div>
                      <p class="desc-text">
                        {{ entry.displayedDescription !== undefined ? entry.displayedDescription : entry.description }}
                      </p>
                    </div>
                    <div v-if="entry.violations.length > 0" class="entry-violations-bar">
                      <span v-for="v in entry.violations" :key="v" class="violation-tag">{{ v }}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- 3. Anomaly detection panel -->
      <section class="tech-panel status-panel">
        <div class="panel-frame">
          <img class="panel-title-img title-status" :src="asset('异常行为标题.png')" alt="异常行为监测" />
          <div class="panel-content detection-grid">
            <div
              v-for="item in detectionItems"
              :key="item.key"
              class="detection-card"
              :class="{ active: violationKeys.has(item.key) }"
            >
              <img
                v-if="violationKeys.has(item.key)"
                :src="asset('告警背景.png')"
                class="alert-bg-img"
                alt=""
              />
              <div class="card-top">
                <img
                  class="card-icon-img"
                  :src="getDetectionIcon(item.key)"
                  :alt="item.label"
                />
                <span class="card-label">{{ item.label }}</span>
              </div>
              <div class="card-status-badge" :class="{ active: violationKeys.has(item.key) }">
                <div class="status-content" v-if="!violationKeys.has(item.key)">
                  <span class="status-dot"></span>
                  <span class="status-text">正常</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  </div>
</template>

<style scoped>
/* ========================================
   CSS Variables
   ======================================== */
.tech-dashboard {
  --bg-deep: #050a1a;
  --bg-panel: rgba(5, 18, 50, 0.85);
  --border-tech: rgba(60, 130, 220, 0.3);
  --accent-blue: #4a90d9;
  --accent-cyan: #22d3ee;
  --accent-purple: #a855f7;
  --safe: #34d399;
  --alert: #f43f5e;
  --text-primary: #e2e8f0;
  --text-secondary: #94a3b8;
  --font-tech: 'JetBrains Mono', 'Consolas', monospace;
  --font-ui: system-ui, -apple-system, 'Microsoft YaHei', sans-serif;

  position: relative;
  width: 100%;
  height: 100vh;
  min-height: 720px;
  background: var(--bg-deep);
  color: var(--text-primary);
  font-family: var(--font-ui);
  overflow: hidden;
  display: flex;
  flex-direction: column;
  box-sizing: border-box;
}

/* ========================================
   Background layer
   ======================================== */
.bg-image {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  pointer-events: none;
  opacity: 0.85;
  z-index: 0;
}

.frame-top {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: auto;
  pointer-events: none;
  z-index: 1;
  opacity: 1;
}

/* ========================================
   Side decorative images (1920x1080)
   ======================================== */
.side-image-left {
  position: absolute;
  left: calc(16 / 1920 * 100vw);
  top: calc(82 / 1080 * 100vh);
  width: calc(300 / 1920 * 100vw);
  height: calc(980 / 1080 * 100vh);
  pointer-events: none;
  z-index: 3;
  opacity: 1;
}

.side-image-right {
  position: absolute;
  left: calc(1604 / 1920 * 100vw);
  top: calc(82 / 1080 * 100vh);
  width: calc(300 / 1920 * 100vw);
  height: calc(980 / 1080 * 100vh);
  pointer-events: none;
  z-index: 3;
  opacity: 1;
}

/* ========================================
   Header
   ======================================== */
.tech-header {
  position: relative;
  z-index: 10;
  height: 78px;
  padding: 0;
  border-bottom: 0;
  background: transparent;
  pointer-events: none;
}

.header-inner {
  display: none;
}

.header-logo {
  width: 48px;
  height: auto;
}

.brand h1 {
  font-size: clamp(1.25rem, 2vw, 1.75rem);
  font-weight: 700;
  margin: 0;
  background: linear-gradient(90deg, #e0e7ff, #93c5fd, #60a5fa);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

/* ========================================
   Alert bar
   ======================================== */
.tech-alert {
  position: relative;
  z-index: 10;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem 1rem;
  margin: 0 1.5rem;
  background: rgba(244, 63, 94, 0.15);
  border: 1px solid rgba(244, 63, 94, 0.4);
  border-radius: 0.5rem;
  color: #fda4af;
  font-family: var(--font-tech);
}

/* ========================================
   Content grid
   ======================================== */
.content-grid {
  position: relative;
  z-index: 5;
  flex: 1;
  min-height: 0;
}

/* Panel absolute positioning — based on 1920×1080 design reference */
.content-grid .video-panel {
  position: absolute;
  left: calc(64 / 1920 * 100vw);
  top: calc(40 / 1080 * 100vh);
  width: calc(880 / 1920 * 100vw);
  bottom: calc(58 / 1080 * 100vh);
}

.content-grid .desc-panel {
  position: absolute;
  left: calc(960 / 1920 * 100vw);
  top: calc(40 / 1080 * 100vh);
  width: calc(466 / 1920 * 100vw);
  bottom: calc(58 / 1080 * 100vh);
}

.content-grid .status-panel {
  position: absolute;
  left: calc(1442 / 1920 * 100vw);
  top: calc(40 / 1080 * 100vh);
  width: calc(398.5 / 1920 * 100vw);
  bottom: calc(58 / 1080 * 100vh);
}

/* ========================================
   Tech panels
   ======================================== */
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
  backdrop-filter: blur(12px);
  display: flex;
  flex-direction: column;
}

.panel-frame::before {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: 0.75rem;
  padding: 1px;
  background: linear-gradient(135deg, transparent 35%, var(--accent-blue) 50%, var(--accent-cyan) 65%, transparent 75%);
  -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  -webkit-mask-composite: xor;
  mask-composite: exclude;
  pointer-events: none;
  opacity: 0.4;
}

.panel-title-img {
  display: block;
  width: 100%;
  height: 60px;
  margin: 0;
  object-fit: fill;
  object-position: left center;
  flex-shrink: 0;
}
.title-video,
.title-desc,
.title-status {
  width: 100%;
  height: 60px;
  object-fit: fill;
}

.panel-content {
  flex: 1;
  min-height: 0;
  padding: 0.75rem 1rem;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

/* ========================================
   Video panel
   ======================================== */
.video-panel .panel-content {
  padding: 0;
  background: rgba(0, 53, 104, 0.32);
  backdrop-filter: blur(16px);
  box-shadow: inset 0px 0px 16px 4px rgba(36, 179, 255, 0.3);
  border-radius: 8px;
  border: 1px solid #3A75D7;
  position: relative;
  overflow: hidden;
}

.stream-stage {
  position: absolute;
  left: 1rem;
  right: 1rem;
  top: 50%;
  transform: translateY(-50%);
  aspect-ratio: 16 / 9;
  overflow: hidden;
  background: #000;
}

.stream-layer {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  opacity: 0;
  visibility: hidden;
  pointer-events: none;
  transition: opacity 0.18s ease;
}

.stream-layer.visible {
  opacity: 1;
  visibility: visible;
  pointer-events: auto;
}

.stream-stage :deep(video),
.stream-stage :deep(canvas) {
  width: 100%;
  height: 100%;
  object-fit: contain;
  object-position: center;
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
  height: 52px;
  background: rgba(255, 136, 38, 0.2);
  box-shadow: inset 0px 0px 16px 0px rgba(255, 136, 38, 0.7);
  border-radius: 7px 7px 0px 0px;
  padding: 0 18px;
  pointer-events: none;
}

.video-tip-text {
  color: #ffd8a8;
  display: flex;
  align-items: center;
  font-size: clamp(0.85rem, 1.3vw, 1rem);
  font-weight: 500;
  letter-spacing: 0.5px;
  text-shadow: 0 1px 4px rgba(0,0,0,0.8);
}

.warning-icon-img {
  width: 18px;
  height: 18px;
  margin-right: 8px;
  object-fit: contain;
  flex: 0 0 auto;
}

.warning-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 18px;
  height: 18px;
  margin-right: 8px;
  border-radius: 50%;
  color: #fff;
  background: #ff8a1f;
  font-weight: 900;
}

.toggle-group {
  pointer-events: auto;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-shrink: 0;
}

.toggle-label {
  font-family: var(--font-tech);
  font-size: 0.9rem;
  color: var(--text-primary);
  white-space: nowrap;
}

.stream-toggle-btn {
  display: flex;
  align-items: center;
  padding: 0;
  background: none;
  border: none;
  cursor: pointer;
  flex-shrink: 0;
}

.toggle-track {
  display: flex;
  align-items: center;
  width: 46px;
  height: 26px;
  border-radius: 99px;
  background: linear-gradient(270deg, #03305B 0%, #0082FF 100%);
  border: 1px solid #288EF5;
  position: relative;
  transition: all 0.25s ease;
}

.toggle-dot {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: linear-gradient(270deg, #007BFF 0%, #B7E0FF 100%);
  box-shadow: 0px 2px 4px 0px rgba(0, 0, 0, 0.3);
  border: 1px solid #B8DAFF;
  position: absolute;
  left: 2px;
  transition: all 0.25s ease;
}

.toggle-dot.active {
  left: 22px;
}

/* ========================================
   Description panel
   ======================================== */
.desc-content {
  font-size: clamp(0.9rem, 1vw, 1.05rem);
  line-height: 1.7;
  color: var(--text-secondary);
  overflow-y: auto;
  flex: 1;
  min-height: 0;
  scrollbar-width: none;
}

.desc-content::-webkit-scrollbar {
  display: none;
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
  gap: 0.5rem;
}

/* Wrapper: each entry + optional violation bar */
.history-entry-wrapper {
  flex-shrink: 0;
}

.history-entry-wrapper.latest .history-entry {
  animation: slideIn 0.35s cubic-bezier(0.34, 1.56, 0.64, 1);
}

@keyframes slideIn {
  from { opacity: 0; transform: translateY(-12px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* Normal card */
.history-entry {
  background: rgba(0, 21, 42, 0.6);
  border: 1px solid #063560;
  border-radius: 8px;
  padding: 1rem 1.25rem;
}

/* Violation card — red gradient, no bottom radius */
.history-entry-wrapper.has-violations .history-entry {
  background: linear-gradient(180deg, rgba(255, 78, 78, 0.15) 0%, rgba(255, 78, 78, 0.3) 100%);
  border: none;
  border-radius: 8px 8px 0 0;
}
.history-entry-wrapper.has-violations .entry-location {
  color: var(--accent-cyan);
}
.history-entry-wrapper.has-violations .entry-time,
.history-entry-wrapper.has-violations .desc-text {
  color: #fff;
}
.history-entry-wrapper.has-violations .desc-text {
  border-left-color: rgba(255, 255, 255, 0.5);
}

/* Violation bar below */
.entry-violations-bar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
  padding: 0 1rem;
  height: 36px;
  background: rgba(255, 78, 78, 0.5);
  border-radius: 0 0 8px 8px;
}

.entry-meta {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 0.4rem;
  font-family: var(--font-tech);
  font-size: 0.9rem;
}

.entry-location {
  color: var(--accent-cyan);
  font-weight: 600;
}

.entry-time {
  color: var(--text-secondary);
  opacity: 0.8;
}

.desc-text {
  border-left: 2px solid var(--accent-blue);
  padding-left: 0.6rem;
  color: #cbd5e1;
  margin: 0;
  font-size: 0.95rem;
  line-height: 1.55;
}

.placeholder {
  color: var(--text-secondary);
  opacity: 0.5;
  font-style: italic;
}

/* Violation tags inside the red bar */
.entry-violations-bar .violation-tag {
  font-size: 0.9rem;
  font-weight: 600;
  color: #fff;
}

/* ========================================
   Detection panel
   ======================================== */
.detection-grid {
  display: flex;
  flex-direction: column;
  gap: 1.75rem;
  padding: 1rem 1.15rem;
}

.detection-card {
  position: relative;
  width: 100%;
  min-height: 100px;
  box-sizing: border-box;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  background: linear-gradient(360deg, rgba(84,255,204,0.3) 0%, rgba(84,255,204,0.05) 100%);
  border: 1px solid rgba(124, 255, 222, 0.55);
  border-radius: 8px;
  padding: 0 24px;
  box-shadow: 0px 4px 4px 0px rgba(0,0,0,0.25);
  transition: all 0.25s ease;
  overflow: hidden;
}

.detection-card::before {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: inherit;
  border: 1px solid transparent;
  background: linear-gradient(180deg, rgba(124, 255, 222, 0.2), rgba(124, 255, 222, 1)) border-box;
  -webkit-mask: linear-gradient(#fff 0 0) padding-box, linear-gradient(#fff 0 0);
  -webkit-mask-composite: xor;
  mask-composite: exclude;
  pointer-events: none;
}

.detection-card.active {
  background: rgba(255, 78, 78, 0.08);
  border-color: rgba(236, 69, 78, 0.4);
  box-shadow: 0 0 22px rgba(255, 78, 78, 0.12), 0px 4px 4px 0px rgba(0,0,0,0.25);
}

.detection-card.active::before {
  background: linear-gradient(180deg, rgba(236, 69, 78, 0.2), rgba(236, 69, 78, 1)) border-box;
}

.detection-card.active::after {
  content: '';
  position: absolute;
  left: 50%;
  bottom: -30px;
  width: 494px;
  height: 122px;
  transform: translateX(-50%);
  background: linear-gradient(180deg, rgba(236,69,78,0) 0%, #EC454E 51.49%, rgba(236,69,78,0) 100%);
  opacity: 0.27;
  pointer-events: none;
}

.card-top {
  position: relative;
  z-index: 1;
  display: flex;
  align-items: center;
  gap: 0.75rem;
  min-width: 0;
  margin: 0;
}

.card-icon-img {
  width: 52px;
  height: 52px;
  object-fit: contain;
  flex: 0 0 auto;
  transition: filter 0.3s ease, transform 0.3s ease;
}

.detection-card.active .card-icon-img {
  filter: drop-shadow(0 0 8px rgba(236, 69, 78, 0.75));
  transform: scale(1.05);
}

.card-label {
  font-weight: 700;
  font-size: 22px;
  color: #FFFFFF;
  white-space: nowrap;
}

.card-status-badge {
  position: relative;
  z-index: 1;
  display: inline-flex;
  align-items: center;
  justify-content: flex-end;
  padding: 0;
  background: transparent;
  border: none;
  transition: all 0.25s ease;
  overflow: visible;
}

.card-status-badge.active {
  background: transparent;
  border: none;
}

.alert-bg-img {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
  border-radius: 8px;
  pointer-events: none;
  z-index: 0;
}

.status-content {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  position: relative;
  z-index: 1;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #54FFCC;
  box-shadow: 0 0 8px rgba(84,255,204,0.95);
  transition: all 0.3s ease;
}

.card-status-badge.active .status-dot {
  background: #fff;
  box-shadow: 0 0 8px rgba(255,255,255,0.85);
  animation: subtlePulse 2s ease-in-out infinite;
}

.status-text {
  width: auto;
  min-width: 28px;
  height: 20px;
  font-family: 'PingFang SC', system-ui, -apple-system, 'Microsoft YaHei', sans-serif;
  font-weight: 400;
  font-size: 14px;
  line-height: 20px;
  color: #FFFFFF;
  text-align: left;
  font-style: normal;
  text-transform: none;
  white-space: nowrap;
  transition: color 0.3s ease;
}

@keyframes subtlePulse {
  0%, 100% { transform: scale(1); opacity: 1; }
  50%      { transform: scale(1.3); opacity: 0.7; }
}

/* ========================================
   Jetson 展厅大屏优化
   ======================================== */
.tech-dashboard.is-jetson .content-grid {
  position: relative;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  padding: 1rem 2rem;
}

.tech-dashboard.is-jetson .video-panel,
.tech-dashboard.is-jetson .desc-panel,
.tech-dashboard.is-jetson .status-panel {
  position: relative;
  left: auto;
  top: auto;
  width: auto;
  bottom: auto;
  flex: 1;
  min-height: 0;
}

.tech-dashboard.is-jetson .brand h1 {
  font-size: 2.25rem;
}

.tech-dashboard.is-jetson .status-panel .panel-title-img {
  height: 60px;
}

.tech-dashboard.is-jetson .panel-content {
  padding: 1.5rem 2rem;
}

.tech-dashboard.is-jetson .detection-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 2rem;
  height: 100%;
}

.tech-dashboard.is-jetson .detection-card {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
  padding: 2rem;
}

.tech-dashboard.is-jetson .card-top {
  flex-direction: column;
  align-items: center;
  gap: 1.5rem;
  margin-bottom: 2rem;
}

.tech-dashboard.is-jetson .card-icon-img {
  width: 80px;
  height: 80px;
}

.tech-dashboard.is-jetson .card-label {
  font-size: 1.75rem;
}

.tech-dashboard.is-jetson .card-status-badge {
  padding: 0.75rem 2rem;
  font-size: 1.25rem;
  min-width: 200px;
}

.tech-dashboard.is-jetson .status-dot {
  width: 14px;
  height: 14px;
}

/* Hide side images on Jetson */
.tech-dashboard.is-jetson .side-image-left,
.tech-dashboard.is-jetson .side-image-right {
  display: none;
}

/* ========================================
   Responsive
   ======================================== */
@media (max-width: 1024px) {
  .content-grid {
    position: relative;
    display: flex;
    flex-direction: column;
    gap: 1rem;
    padding: 1rem;
  }

  .content-grid .video-panel,
  .content-grid .desc-panel,
  .content-grid .status-panel {
    position: relative;
    left: auto;
    top: auto;
    width: auto;
    bottom: auto;
    flex: 1;
    min-height: 0;
  }

  .side-image-left,
  .side-image-right {
    display: none;
  }
}

@media (max-width: 640px) {
  .content-grid {
    position: relative;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    padding: 0.5rem;
  }

  .content-grid .video-panel,
  .content-grid .desc-panel,
  .content-grid .status-panel {
    position: relative;
    left: auto;
    top: auto;
    width: auto;
    bottom: auto;
    flex: 1;
    min-height: 0;
  }

  .tech-header {
    padding: 0.5rem 1rem;
  }
}
</style>

