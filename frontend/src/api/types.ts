export interface ServerResponse {
  result: string
  api_time_ms: number
  inference_time_ms: number
}

export interface InferenceResult {
  description: string
  violations: string[]
}

export interface ViolationInfo {
  label: string
  icon: string
  color: string
  bgColor: string
  severity: 'low' | 'medium' | 'high' | 'critical'
}

export interface ShowClientData {
  vision_split_time_ms: number
  total_api_time_ms: number
  server_response: ServerResponse
  image_processing_time_ms: number
}