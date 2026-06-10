<script setup lang="ts">
import { ref, nextTick } from 'vue'
import { marked } from 'marked'

const apiHost = import.meta.env.VITE_API_HOST

interface QAStep {
  title: string
  type: 'thinking' | 'todo' | 'tool_call' | 'answer' | 'analysis'
  content: string
  status: 'pending' | 'active' | 'done'
}

interface AnalysisContent {
  thinking: string
  todos: { content: string; status: string }[]
}

interface QAMessage {
  id: number
  role: 'user' | 'assistant'
  content: string
  displayedContent: string
  steps?: QAStep[]
}

const qaInput = ref('')
const qaMessages = ref<QAMessage[]>([])
const qaProcessing = ref(false)
const qaActive = ref(false)
let qaMsgId = 0
const qaPanelRef = ref<HTMLElement | null>(null)

const timeRangeOptions = [
  { label: '15分钟', value: '15m' },
  { label: '1小时', value: '1h' },
  { label: '4小时', value: '4h' },
  { label: '8小时', value: '8h' },
  { label: '24小时', value: '24h' },
  { label: '长期', value: 'long' },
]
const qaTimeRange = ref('1h')
const qaTimeOpen = ref(false)

function toggleTimeDropdown() { qaTimeOpen.value = !qaTimeOpen.value }

function selectTime(val: string) {
  qaTimeRange.value = val
  qaTimeOpen.value = false
}

function formatDatetime(d: Date): string {
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
}

function buildTimeRangeText(): string {
  const end = new Date()
  const offsetMs: Record<string, number> = {
    '15m': 15 * 60 * 1000,
    '1h': 60 * 60 * 1000,
    '4h': 4 * 60 * 60 * 1000,
    '8h': 8 * 60 * 60 * 1000,
    '24h': 24 * 60 * 60 * 1000,
    'long': 30 * 24 * 60 * 60 * 1000,
  }
  const start = new Date(end.getTime() - (offsetMs[qaTimeRange.value] ?? 60 * 60 * 1000))
  return `查询时间范围：${formatDatetime(start)}~${formatDatetime(end)}`
}

function handlePresetClick(text: string) {
  qaInput.value = text
  handleQASend()
}

function renderMarkdown(text: string): string {
  return marked.parse(text, { async: false }) as string
}

function parseTodos(content: string): { content: string; status: string }[] {
  try { return JSON.parse(content) } catch { return [] }
}

function parseAnalysis(content: string): AnalysisContent {
  try { return JSON.parse(content) } catch { return { thinking: content, todos: [] } }
}


async function handleQASend() {
  const text = qaInput.value.trim()
  if (!text || qaProcessing.value) return
  qaInput.value = ''
  qaMessages.value = []
  qaActive.value = true

  qaMessages.value.push({ id: qaMsgId++, role: 'user', content: text, displayedContent: text })
  qaProcessing.value = true

  const questionWithTime = `${text}\n${buildTimeRangeText()}`

  const msg: QAMessage = { id: qaMsgId++, role: 'assistant', content: '', displayedContent: '', steps: [] }
  qaMessages.value.push(msg)

  nextTick(() => qaPanelRef.value?.scrollTo({ top: qaPanelRef.value.scrollHeight, behavior: 'smooth' }))

  if (import.meta.env.VITE_QA_DEV_MODE === 'true') {
    msg.steps!.push({
      title: '回答',
      type: 'answer',
      content: '本智能助手正在闭关，等我学成归来再为你解答~',
      status: 'done',
    })
    msg.displayedContent = '本智能助手正在闭关，等我学成归来再为你解答~'
    qaProcessing.value = false
    return
  }

  function findMsg() { return qaMessages.value.find(m => m.id === msg.id) }

  function applyEvent(ev: any) {
    const m = findMsg()
    if (!m) return
    if (ev.event === 'step_add') {
      const s = ev.step
      m.steps![s.idx] = { title: s.title, type: s.type, content: s.content ?? '', status: s.status ?? 'done' }
    } else if (ev.event === 'step_update') {
      const s = m.steps![ev.idx]
      if (s) { if (ev.content !== undefined) s.content = ev.content; if (ev.status) s.status = ev.status }
    } else if (ev.event === 'step_status') {
      const s = m.steps![ev.idx]
      if (s) s.status = ev.status
    } else if (ev.event === 'answer_chunk') {
      const s = m.steps![ev.idx]
      if (s) s.content += ev.text
      m.displayedContent += ev.text
    } else if (ev.event === 'error') {
      m.steps!.push({ title: '错误', type: 'thinking', content: ev.message, status: 'done' })
    }
    nextTick(() => qaPanelRef.value?.scrollTo({ top: qaPanelRef.value.scrollHeight, behavior: 'smooth' }))
  }

  try {
    const res = await fetch(`${apiHost}/qa/ask-stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: questionWithTime, history: [] }),
    })
    if (!res.body) throw new Error('No response body')

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buf = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      const lines = buf.split('\n')
      buf = lines.pop() ?? ''
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            applyEvent(JSON.parse(line.slice(6)))
          } catch { /* malformed line */ }
        }
      }
    }
  } catch (e: any) {
    const m = findMsg()
    if (m) m.steps!.push({ title: '错误', type: 'thinking', content: e.message || '请求失败', status: 'done' })
  } finally {
    qaProcessing.value = false
  }
}
</script>

<template>
  <div class="qa-zone" :class="{ 'is-active': qaActive }" @mouseleave="qaActive = false">
    <div class="qa-hover-bg"></div>
    <div class="qa-messages-area" ref="qaPanelRef" v-if="qaMessages.length > 0">
      <div v-for="msg in qaMessages" :key="msg.id" class="qa-msg" :class="msg.role">
        <div class="qa-msg-label">{{ msg.role === 'user' ? '你' : 'AI 助手' }}</div>
        <div v-if="msg.steps" class="qa-steps">
          <div v-for="(step, si) in msg.steps" :key="si" class="qa-step-wrap">
            <div v-if="step.type !== 'todo' && step.type !== 'analysis'" class="qa-step" :class="step.status">
              <span class="qa-step-icon">
                <span v-if="step.status === 'pending'" class="step-dot"></span>
                <span v-else-if="step.status === 'active'" class="step-spinner"></span>
                <svg v-else width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
              </span>
              <span class="qa-step-label">{{ step.title }}</span>
            </div>

            <template v-if="step.type === 'analysis'">
              <div class="qa-step" :class="step.status">
                <span class="qa-step-icon">
                  <span v-if="step.status === 'active'" class="step-spinner"></span>
                  <svg v-else width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                </span>
                <span class="qa-step-label">{{ step.title }}</span>
              </div>
              <template v-if="step.status === 'done' && step.content">
                <div v-if="parseAnalysis(step.content).thinking" class="qa-step-output is-thinking">{{ parseAnalysis(step.content).thinking }}</div>
                <div v-if="parseAnalysis(step.content).todos.length" class="qa-step-output is-todo">
                  <div v-for="(todo, ti) in parseAnalysis(step.content).todos" :key="ti" class="qa-todo-item">
                    <span class="qa-todo-icon"><span class="step-dot"></span></span>
                    <span class="qa-todo-text">{{ todo.content }}</span>
                  </div>
                </div>
              </template>
            </template>

            <div v-if="step.type === 'thinking' && step.content"
              class="qa-step-output is-thinking">{{ step.content }}</div>

            <div v-if="step.type === 'tool_call' && step.content"
              class="qa-step-output is-tool">{{ step.content }}</div>

            <div v-if="step.type === 'todo' && step.content" class="qa-step-output is-todo">
              <div v-for="(todo, ti) in parseTodos(step.content)" :key="ti" class="qa-todo-item" :class="todo.status">
                <span class="qa-todo-icon">
                  <span v-if="todo.status === 'pending'" class="step-dot"></span>
                  <span v-else-if="todo.status === 'in_progress'" class="step-spinner"></span>
                  <svg v-else width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                </span>
                <span class="qa-todo-text">{{ todo.content }}</span>
              </div>
            </div>

            <div v-if="step.type === 'answer' && step.content" class="qa-answer-content is-md"
              v-html="renderMarkdown(step.content)"></div>
          </div>
        </div>
        <div v-if="!msg.steps && msg.displayedContent" class="qa-msg-content">{{ msg.displayedContent }}</div>
      </div>
    </div>

    <div v-if="!qaProcessing" class="qa-presets">
      <span class="qa-preset" @click="handlePresetClick('展厅有什么异常吗？')">展厅有什么异常吗？</span>
      <span class="qa-preset" @click="handlePresetClick('有人在展厅吸烟过吗？')">有人在展厅吸烟过吗？</span>
      <span class="qa-preset" @click="handlePresetClick('监控系统运行状态如何？')">监控系统运行状态如何？</span>
    </div>

    <div class="qa-bar">
      <div class="qa-time-selector">
        <button class="qa-time-btn" @click="toggleTimeDropdown">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
          <span>{{ timeRangeOptions.find(o => o.value === qaTimeRange)?.label }}</span>
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
        </button>
        <div v-if="qaTimeOpen" class="qa-time-dropdown" @click.stop>
          <div v-for="opt in timeRangeOptions" :key="opt.value" class="qa-time-drop-item" :class="{ active: qaTimeRange === opt.value }" @click="selectTime(opt.value)">{{ opt.label }}</div>
        </div>
        <div v-if="qaTimeOpen" class="qa-time-backdrop" @click="qaTimeOpen = false"></div>
      </div>
      <input
        class="qa-input"
        type="text"
        v-model="qaInput"
        placeholder="输入您的问题..."
        @keyup.enter="handleQASend"
        :disabled="qaProcessing"
      />
      <button class="qa-send-btn" @click="handleQASend" :disabled="qaProcessing">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="22" y1="2" x2="11" y2="13"></line>
          <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
        </svg>
      </button>
    </div>
  </div>
</template>

<style scoped>
.qa-zone {
  --font-tech: 'JetBrains Mono', 'Consolas', monospace;
  --font-ui: system-ui, -apple-system, sans-serif;
  --accent-cyan: #22d3ee;
  --safe: #34d399;
  --alert: #f43f5e;

  position: fixed;
  bottom: 2rem;
  left: 50%;
  transform: translateX(-50%);
  z-index: 100;
  width: min(600px, calc(100% - 3rem));
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
  pointer-events: auto;
  transition: filter 0.3s ease;
}

.qa-messages-area {
  width: 100%;
  max-height: 320px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
  padding: 0.6rem 0.6rem 0.4rem;
  border-radius: 1rem;
  background: rgba(5, 5, 26, 0.4);
  backdrop-filter: blur(12px);
  scrollbar-width: none;
  transition: background 0.3s ease, opacity 0.3s ease;
  opacity: 0;
  pointer-events: none;
}

.qa-zone:hover .qa-messages-area,
.qa-zone:focus-within .qa-messages-area,
.qa-zone.is-active .qa-messages-area {
  opacity: 1;
  pointer-events: auto;
  background: rgba(5, 5, 26, 0.6);
}

.qa-messages-area::-webkit-scrollbar { display: none; }

.qa-msg {
  animation: qaFadeIn 0.3s ease;
  max-width: 85%;
}

.qa-msg.user {
  align-self: flex-end;
}

@keyframes qaFadeIn {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

.qa-msg-label {
  font-size: 0.65rem;
  font-weight: 600;
  color: rgba(148, 163, 184, 0.5);
  margin-bottom: 0.25rem;
  letter-spacing: 0.04em;
  font-family: var(--font-tech, 'JetBrains Mono', 'Consolas', monospace);
}

.qa-msg.user .qa-msg-label {
  text-align: right;
}

.qa-msg-content {
  font-size: 0.88rem;
  line-height: 1.65;
  color: #e2e8f0;
  padding: 0.5rem 0.85rem;
  background: rgba(99, 102, 241, 0.12);
  border: 1px solid rgba(148, 163, 255, 0.2);
  border-radius: 1rem;
  border-bottom-left-radius: 0.25rem;
  display: inline-block;
  text-align: left;
}

.qa-msg.user .qa-msg-content {
  background: rgba(99, 102, 241, 0.3);
  border-color: rgba(148, 163, 255, 0.35);
  border-bottom-left-radius: 1rem;
  border-bottom-right-radius: 0.25rem;
}

.qa-steps {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  padding: 0.4rem 0;
}

.qa-step {
  display: flex;
  align-items: center;
  gap: 0.45rem;
  font-size: 0.8rem;
  color: rgba(148, 163, 184, 0.5);
  transition: color 0.3s ease;
}

.qa-step.active { color: #22d3ee; }
.qa-step.done { color: #34d399; }

.qa-step-icon {
  width: 16px;
  height: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.step-dot { width: 5px; height: 5px; border-radius: 50%; background: currentColor; }

.step-spinner {
  width: 12px; height: 12px;
  border: 2px solid transparent;
  border-top-color: currentColor;
  border-radius: 50%;
  animation: stepSpin 0.7s linear infinite;
}

@keyframes stepSpin { to { transform: rotate(360deg); } }

.qa-step-label { font-family: system-ui, -apple-system, sans-serif; letter-spacing: 0.02em; }

.qa-step-output {
  font-size: 0.82rem;
  line-height: 1.6;
  color: rgba(180, 195, 220, 0.7);
  margin: 0.25rem 0 0.4rem 1.4rem;
  padding-left: 0.6rem;
  border-left: 2px solid rgba(130, 140, 255, 0.15);
  white-space: pre-wrap;
  word-break: break-word;
}

.qa-step-output.is-thinking {
  color: rgba(148, 163, 184, 0.5);
  font-style: italic;
  border-left-color: rgba(99, 102, 241, 0.2);
}

.qa-step-output.is-tool {
  color: rgba(148, 163, 184, 0.55);
  font-family: 'JetBrains Mono', 'Consolas', monospace;
  font-size: 0.76rem;
  border-left-color: rgba(34, 211, 238, 0.2);
}

.qa-step-output.is-todo {
  border-left-color: rgba(99, 102, 241, 0.2);
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  white-space: normal;
}

.qa-todo-item {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.8rem;
  color: rgba(148, 163, 184, 0.5);
  transition: color 0.2s ease;
}

.qa-todo-item.in_progress { color: #22d3ee; }
.qa-todo-item.completed { color: #34d399; }

.qa-todo-icon {
  width: 14px;
  height: 14px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.qa-todo-text {
  font-family: system-ui, -apple-system, sans-serif;
  letter-spacing: 0.01em;
}

.qa-step-output.is-md {
  white-space: normal;
}

.qa-step-output.is-md ul,
.qa-step-output.is-md ol {
  margin: 0.25rem 0;
  padding-left: 1.2rem;
}

.qa-step-output.is-md li {
  margin-bottom: 0.15rem;
  line-height: 1.5;
}

.qa-step-output.is-md p {
  margin: 0.25rem 0;
}

.qa-step-output.is-md code {
  background: rgba(130, 140, 255, 0.1);
  padding: 0.05rem 0.35rem;
  border-radius: 0.2rem;
  font-size: 0.78rem;
  font-family: 'JetBrains Mono', 'Consolas', monospace;
}

.qa-step-output.is-md strong {
  color: rgba(200, 215, 240, 0.85);
}

.qa-step-wrap:last-child .qa-step-output {
  border-left-color: rgba(99, 102, 241, 0.25);
}

.qa-answer-content {
  font-size: 0.9rem;
  line-height: 1.7;
  color: #e2e8f0;
  margin-top: 0.5rem;
  white-space: normal;
  word-break: break-word;
}

.qa-answer-content ul,
.qa-answer-content ol {
  margin: 0.3rem 0;
  padding-left: 1.2rem;
}

.qa-answer-content li { margin-bottom: 0.2rem; line-height: 1.6; }
.qa-answer-content p { margin: 0.3rem 0; }

.qa-answer-content code {
  background: rgba(130, 140, 255, 0.12);
  padding: 0.05rem 0.35rem;
  border-radius: 0.2rem;
  font-size: 0.8rem;
  font-family: 'JetBrains Mono', 'Consolas', monospace;
  color: #c4b5fd;
}

.qa-answer-content strong { color: #f1f5f9; }
.qa-answer-content em { color: #cbd5e1; }

.qa-presets {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
  justify-content: center;
  width: 100%;
  opacity: 0;
  transform: translateY(6px);
  transition: opacity 0.3s ease, transform 0.3s ease;
  pointer-events: none;
}

.qa-zone:hover .qa-presets,
.qa-zone:focus-within .qa-presets,
.qa-zone.is-active .qa-presets {
  opacity: 1;
  transform: translateY(0);
  pointer-events: auto;
}

.qa-preset {
  padding: 0.3rem 0.9rem;
  font-size: 0.78rem;
  color: rgba(180, 190, 210, 0.75);
  background: rgba(5, 5, 26, 0.5);
  border: 1px solid rgba(130, 140, 255, 0.2);
  border-radius: 2rem;
  cursor: pointer;
  transition: all 0.25s ease;
  white-space: nowrap;
  user-select: none;
  backdrop-filter: blur(8px);
}

.qa-preset:hover {
  color: #eef2ff !important;
  background: rgba(99, 102, 241, 0.25) !important;
  border-color: rgba(148, 163, 255, 0.5) !important;
}

.qa-hover-bg {
  position: absolute;
  inset: -20px -24px;
  border-radius: 28px;
  background: rgba(15, 15, 60, 0.75);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid rgba(130, 140, 255, 0.15);
  box-shadow: 0 0 60px rgba(99, 102, 241, 0.15);
  opacity: 0;
  transition: opacity 0.35s ease;
  pointer-events: none;
}

.qa-zone:hover .qa-hover-bg,
.qa-zone:focus-within .qa-hover-bg,
.qa-zone.is-active .qa-hover-bg {
  opacity: 1;
}

.qa-bar {
  width: 100%;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.6rem 0.6rem 0.6rem 1.1rem;
  background: rgba(10, 10, 35, 0.7);
  border: 1px solid rgba(130, 140, 255, 0.25);
  border-radius: 3rem;
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
  transition: all 0.3s ease;
}

.qa-bar:focus-within {
  border-color: rgba(148, 163, 255, 0.5);
  background: rgba(12, 12, 40, 0.8);
  box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5),
              0 0 50px rgba(99, 102, 241, 0.12);
}

.qa-zone:hover .qa-bar {
  background: rgba(12, 12, 40, 0.78);
  border-color: rgba(148, 163, 255, 0.35);
  box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5),
              0 0 40px rgba(99, 102, 241, 0.1);
}

.qa-time-selector {
  position: relative;
  flex-shrink: 0;
}

.qa-time-btn {
  display: flex;
  align-items: center;
  gap: 0.2rem;
  padding: 0.15rem 0.6rem 0.15rem 0;
  background: none;
  border: none;
  border-right: 1px solid rgba(130, 140, 255, 0.2);
  color: rgba(160, 175, 210, 0.6);
  font-size: 0.72rem;
  font-family: system-ui, -apple-system, sans-serif;
  cursor: pointer;
  transition: color 0.2s ease;
  user-select: none;
  white-space: nowrap;
  line-height: 1;
}

.qa-time-btn:hover { color: #b0c0e0; }

.qa-time-dropdown {
  position: absolute;
  bottom: calc(100% + 8px);
  left: 0;
  min-width: 130px;
  background: rgba(10, 10, 35, 0.96);
  border: 1px solid rgba(130, 140, 255, 0.25);
  border-radius: 0.75rem;
  backdrop-filter: blur(20px);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
  overflow: hidden;
  z-index: 200;
  animation: qaDropIn 0.15s ease;
}

@keyframes qaDropIn {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}

.qa-time-drop-item {
  position: relative;
  padding: 0.5rem 0.9rem;
  font-size: 0.8rem;
  color: rgba(160, 175, 210, 0.7);
  cursor: pointer;
  transition: all 0.15s ease;
}

.qa-time-drop-item:hover { background: rgba(99, 102, 241, 0.12); color: #d0d8f0; }

.qa-time-drop-item.active {
  color: #eef2ff;
  background: rgba(99, 102, 241, 0.18);
}

.qa-time-drop-item.active::before {
  content: '';
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 2px;
  background: linear-gradient(180deg, #818cf8, #a78bfa);
}

.qa-time-backdrop {
  position: fixed; inset: 0; z-index: 199;
}

.qa-input {
  flex: 1;
  border: none;
  outline: none;
  background: transparent;
  color: #e8edf5;
  font-family: system-ui, -apple-system, sans-serif;
  font-size: 0.9rem;
  line-height: 1.5;
  padding: 0.2rem 0;
}

.qa-input::placeholder { color: rgba(148, 163, 184, 0.3); }
.qa-input:disabled { opacity: 0.4; }

.qa-send-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 2.2rem;
  height: 2.2rem;
  border: none;
  border-radius: 50%;
  background: linear-gradient(135deg, #818cf8, #a78bfa);
  color: #fff;
  cursor: pointer;
  transition: all 0.25s ease;
  flex-shrink: 0;
  box-shadow: 0 0 12px rgba(99, 102, 241, 0.2);
}

.qa-send-btn:hover:not(:disabled) {
  transform: scale(1.08);
  box-shadow: 0 0 24px rgba(99, 102, 241, 0.5);
}

.qa-send-btn:active:not(:disabled) { transform: scale(0.93); }
.qa-send-btn:disabled { opacity: 0.25; cursor: not-allowed; }
</style>
