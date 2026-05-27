import axios from 'axios'
import { ref } from 'vue'
import type { ShowClientData, InferenceResult, ViolationInfo } from './types'

const api = axios.create({
  baseURL: 'http://192.168.151.158:28001/api',
  timeout: 30000
})

const BASE_URL = 'http://192.168.151.158:28001/api'

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

export const descriptionList = ref([
  '这是一个空间宽敞、光线明亮的科技展厅，整体布局十分开阔，地面为浅灰色。展厅中摆放着多个白色立式交互式触摸屏展台，背景墙上安装了数块大型显示屏，整体呈现出浓厚的科技质感。',
  '展厅整体宽敞明亮，空间布局开阔，浅灰色地面让环境显得简洁干净。场内设置了多个白色立式交互式触摸屏展台，背景墙上还有数块大型显示屏，营造出鲜明的科技氛围。',
  '这是一个明亮且开阔的科技展厅，整体空间较为宽敞，地面采用浅灰色设计。展厅内分布着多个白色的立式交互式触摸屏展台，背景墙配有数块大型显示屏，使空间更具科技感。',
  '该科技展厅空间宽敞、采光明亮，整体布局显得开阔有序，地面呈浅灰色。展厅里设有多个白色立式交互式触摸屏展台，背景墙上布置了数块大型显示屏，科技质感十分突出。',
  '这是一个布局开阔的科技展厅，空间宽敞明亮，浅灰色地面带来简洁现代的视觉效果。展厅内摆放着多个白色立式交互式触摸屏展台，背景墙上配有数块大型显示屏，整体充满科技质感。',
  '展厅呈现出宽敞明亮的空间效果，整体布局开阔，地面为浅灰色。多个白色立式交互式触摸屏展台分布在展厅内，背景墙上则设置了数块大型显示屏，让空间显得科技感十足。',
  '这个科技展厅整体空间开阔，环境宽敞明亮，浅灰色地面让视觉效果更加清爽。展厅内设有多个白色立式交互式触摸屏展台，背景墙上搭配数块大型显示屏，整体富有科技质感。',
  '这是一个宽敞且明亮的科技展厅，空间布局较为开阔，地面呈浅灰色。展厅中配置了多个白色的立式交互式触摸屏展台，背景墙上安装着数块大型显示屏，使整个环境充满科技气息。',
  '科技展厅内部空间宽敞明亮，整体布局开阔，浅灰色地面显得简洁利落。展厅内布置了多个白色立式交互式触摸屏展台，背景墙上还有数块大型显示屏，增强了整体的科技质感。',
  '这是一个具有开阔布局的科技展厅，空间宽敞、光线明亮，地面为浅灰色。展厅内设有多个白色立式交互式触摸屏展台，背景墙上配备了数块大型显示屏，整体氛围充满科技感。',
  '展厅空间宽敞明亮，布局开阔，浅灰色地面让整体环境显得现代而干净。场内设置了多个白色的立式交互式触摸屏展台，背景墙上配有数块大型显示屏，营造出强烈的科技质感。',
  '这是一个科技感鲜明的展厅，整体空间宽敞明亮，布局开阔，地面呈浅灰色。展厅内有多个白色立式交互式触摸屏展台，背景墙上布置着数块大型显示屏，使空间更显现代科技氛围。',
  '整个科技展厅显得宽敞明亮，空间布局开阔，地面采用浅灰色。展厅内摆放了多个白色立式交互式触摸屏展台，背景墙上设置数块大型显示屏，让整体视觉效果更具科技质感。',
  '这是一个开阔而明亮的科技展厅，空间尺度较为宽敞，地面呈现浅灰色。展厅内配有多个白色立式交互式触摸屏展台，背景墙上还有数块大型显示屏，整体呈现现代科技感。',
  '科技展厅整体宽敞明亮，布局开阔通透，浅灰色地面让空间显得简洁大方。展厅内设有多个白色的立式交互式触摸屏展台，背景墙上配备数块大型显示屏，带来明显的科技质感。',
  '这是一个空间开阔的科技展厅，整体宽敞明亮，地面为浅灰色。多个白色立式交互式触摸屏展台摆放在展厅内，背景墙上则装有数块大型显示屏，使整个空间充满科技气息。',
  '展厅内部宽敞明亮，整体布局开阔，浅灰色地面营造出简洁的空间基调。展厅中设置了多个白色立式交互式触摸屏展台，背景墙上搭配数块大型显示屏，科技质感十分鲜明。',
  '这是一个明亮宽敞的科技展厅，空间布局开阔，地面呈浅灰色。展厅内布置着多个白色的立式交互式触摸屏展台，背景墙上配有数块大型显示屏，让整体空间显得科技感十足。',
  '整体来看，这个科技展厅空间宽敞、光线明亮，布局开阔，地面为浅灰色。展厅内设有多个白色立式交互式触摸屏展台，背景墙上安装数块大型显示屏，营造出现代科技质感。',
  '这是一个整体布局开阔的科技展厅，空间宽敞明亮，浅灰色地面让环境更显简约。展厅内配置多个白色立式交互式触摸屏展台，背景墙上设有数块大型显示屏，呈现出浓厚的科技氛围。',
  '展厅空间显得宽敞而明亮，布局开阔，地面采用浅灰色设计。多个白色的立式交互式触摸屏展台设置在展厅内，背景墙上配有数块大型显示屏，使整体空间更具科技质感。',
  '这是一个充满科技质感的展厅，整体空间宽敞明亮，布局开阔，地面呈浅灰色。展厅内分布着多个白色立式交互式触摸屏展台，背景墙上则搭配数块大型显示屏。',
  '科技展厅整体空间开阔，环境宽敞明亮，浅灰色地面带来现代简洁的观感。展厅内设有多个白色立式交互式触摸屏展台，背景墙上配备数块大型显示屏，强化了空间的科技质感。',
  '这是一个宽敞明亮、布局开阔的科技展厅，地面呈浅灰色。展厅中摆放着多个白色立式交互式触摸屏展台，背景墙上设置了数块大型显示屏，整体空间充满现代科技气息。',
  '展厅整体呈现出宽敞明亮的效果，空间布局开阔，地面为浅灰色。内部设有多个白色的立式交互式触摸屏展台，背景墙上安装着数块大型显示屏，让整个环境充满科技质感。',
])

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
    const jsonStr = jsonMatch ? jsonMatch[1] : resultStr
    const parsed = JSON.parse(jsonStr)
    
    const hasPerson = parsed.has_person === 1 ? '有' : '没'
    const violationsStr = parsed.violations && parsed.violations.length > 0
      ? parsed.violations.join('/')
      : '无预警'
    

    // const d_prefix = 随机从descriptionList中选取一个元素
    const randomIndex = Math.floor(Math.random() * descriptionList.value.length)
    const d_prefix = descriptionList.value[randomIndex]

    if (parsed.has_person.toString() == '-1') {
      return { description: d_prefix, violations: [], hasPerson: false }
    }
    const description = `${d_prefix}目前，展厅${hasPerson}人，且其预警识别状态为：${violationsStr}。`

    return {
      description,
      violations: parsed.violations || [],
      hasPerson: parsed.has_person === 1
    }
  } catch {
    return { description: resultStr, violations: [], hasPerson: false }
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