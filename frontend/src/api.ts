import type {
  ConceptInfo,
  ForwardResponse,
  HealthResponse,
  NeuronDetail,
  NeuronScanResponse,
  PatchResponse,
  ProbeResponse,
  TrajectoryResponse,
} from './types'

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

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
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

export function getConcepts(): Promise<ConceptInfo[]> {
  return getJSON<ConceptInfo[]>('/probe/concepts')
}

export function runProbes(concepts: string[]): Promise<ProbeResponse> {
  return postJSON<ProbeResponse>('/probe', { concepts })
}

export function runTrajectory(prompt: string): Promise<TrajectoryResponse> {
  return postJSON<TrajectoryResponse>('/trajectory', { prompt })
}

export function scanLayer(layer: number): Promise<NeuronScanResponse> {
  return getJSON<NeuronScanResponse>(`/neuron/scan?layer=${layer}`)
}

export function neuronDetail(layer: number, index: number): Promise<NeuronDetail> {
  return getJSON<NeuronDetail>(`/neuron?layer=${layer}&index=${index}`)
}

export function getHealth(): Promise<HealthResponse> {
  return getJSON<HealthResponse>('/health')
}
