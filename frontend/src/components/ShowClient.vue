<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { fetchShowClientData, parseInferenceResult, getWebSocketUrl } from '../api'
import type { ShowClientData, InferenceResult } from '../api/types'
import StreamPlayer from './StreamPlayer.vue'

const streamUrl = "http://192.168.153.50:8080/live/livestream.flv"

import {
  Cigarette,
  Swords as Fight, // Lucide 中没有直接的 Fight，通常用 Swords 或 Fists 代替
  Flame as Fire,
  PersonStanding as Fall
} from 'lucide-vue-next'

const data = ref<ShowClientData | null>(null)
const parsedResult = ref<InferenceResult | null>(null)
const error = ref<string | null>(null)
const loading = ref(false)
const streamStatus = ref<'connecting' | 'playing' | 'error' | 'idle'>('idle')


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
onUnmounted(() => { if (intervalId) clearInterval(intervalId) })
</script>

<template>
  <div class="tech-dashboard">
    <div class="dashboard-bg"></div>
    <div class="scan-line"></div>
    <div class="glow-orb glow-orb-1"></div>
    <div class="glow-orb glow-orb-2"></div>

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
          <div class="panel-content desc-content">
            <p v-if="parsedResult?.description" class="desc-text">{{ parsedResult.description }}</p>
            <p v-else class="placeholder">等待模型推理...</p>
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
  overflow: hidden;
  display: flex;
  flex-direction: column;
  box-sizing: border-box;
}

.dashboard-bg {
  position: absolute;
  inset: 0;
  z-index: -2;
  background:
    radial-gradient(ellipse at 20% 30%, rgba(99, 102, 241, 0.15) 0%, transparent 50%),
    radial-gradient(ellipse at 80% 70%, rgba(168, 85, 247, 0.1) 0%, transparent 40%),
    repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(99, 102, 241, 0.08) 40px),
    repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(99, 102, 241, 0.08) 40px);
}

.glow-orb {
  position: absolute;
  border-radius: 50%;
  filter: blur(90px);
  z-index: -1;
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

.scan-line {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
  opacity: 0.6;
  animation: scanMove 12s linear infinite;
  z-index: -1;
  pointer-events: none;
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
}

.panel-frame {
  position: relative;
  flex: 1;
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
}

.desc-text {
  animation: fadeIn 0.5s ease;
  border-left: 2px solid var(--accent-blue);
  padding-left: 0.75rem;
  color: #cbd5e1;
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