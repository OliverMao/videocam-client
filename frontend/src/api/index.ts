import axios from 'axios'
import type { ShowClientData, InferenceResult, ViolationInfo } from './types'

const api = axios.create({
  baseURL: 'http://192.168.151.96:28001/api',
  timeout: 30000
})

const BASE_URL = 'http://192.168.151.96:28001/api'

const violationConfig: Record<string, ViolationInfo> = {
  '吸烟': { label: '吸烟', icon: '🚬', color: '#f97316', bgColor: '#7c2d12', severity: 'medium' },
  '抽烟': { label: '抽烟', icon: '🚬', color: '#f97316', bgColor: '#7c2d12', severity: 'medium' },
  '打架': { label: '打架', icon: '⚔', color: '#ef4444', bgColor: '#7f1d1d', severity: 'high' },
  '着火': { label: '着火', icon: '🔥', color: '#ef4444', bgColor: '#7f1d1d', severity: 'critical' },
  '火灾': { label: '火灾', icon: '🔥', color: '#ef4444', bgColor: '#7f1d1d', severity: 'critical' },
  '摔倒': { label: '摔倒', icon: '🚑', color: '#eab308', bgColor: '#713f12', severity: 'high' },
  '跌倒': { label: '跌倒', icon: '🚑', color: '#eab308', bgColor: '#713f12', severity: 'high' },
  '赌博': { label: '赌博', icon: '🎲', color: '#a855f7', bgColor: '#581c87', severity: 'medium' },
}

export function getViolationInfo(violation: string): ViolationInfo {
  return violationConfig[violation] || {
    label: violation,
    icon: '⚠',
    color: '#f97316',
    bgColor: '#7c2d12',
    severity: 'medium',
  }
}

export function fetchShowClientData(): Promise<ShowClientData> {
  return api.get('/show-client').then(res => res.data)
}

export function parseInferenceResult(resultStr: string): InferenceResult {
  if (!resultStr) return { description: '', violations: [] }
  try {
    const jsonMatch = resultStr.match(/```json\n([\s\S]*?)\n```/)
    const json = jsonMatch ? jsonMatch[1] : resultStr
    return JSON.parse(json)
  } catch {
    return { description: resultStr, violations: [] }
  }
}

export function getStreamUrl(): string {
  return `${BASE_URL}/stream-sse`
}

export function getWebSocketUrl(): string {
  const url = new URL(BASE_URL)
  const wsProtocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${wsProtocol}//${url.host}/api/ws/stream`
}