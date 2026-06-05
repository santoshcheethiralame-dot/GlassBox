import type { ForwardResponse, HealthResponse, PatchResponse } from './types'

const BASE = '/api'

async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const detail = await res
      .json()
      .then((j) => j.detail ?? res.statusText)
      .catch(() => res.statusText)
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export function runForward(prompt: string, topK = 10): Promise<ForwardResponse> {
  return postJSON<ForwardResponse>('/forward', { prompt, top_k: topK })
}

export interface PatchInput {
  clean_prompt: string
  corrupted_prompt: string
  answer: string
  corrupted_answer: string
}

export function runPatching(input: PatchInput): Promise<PatchResponse> {
  return postJSON<PatchResponse>('/patch', input)
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE}/health`)
  if (!res.ok) throw new Error('backend not reachable')
  return res.json() as Promise<HealthResponse>
}
