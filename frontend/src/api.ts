import type {
  ConceptInfo,
  ForwardResponse,
  HealthResponse,
  ModelInfo,
  NeuronDetail,
  NeuronScanResponse,
  PatchResponse,
  ProbeResponse,
  SaeFeaturesResponse,
  SaeInfo,
  TrajectoryResponse,
} from './types'

const BASE = '/api'

async function rfetch(url: string, opts?: RequestInit): Promise<Response> {
  let last: unknown
  for (let a = 0; a < 4; a++) {
    try {
      return await fetch(url, opts)
    } catch (e) {
      last = e
      await new Promise((r) => setTimeout(r, 350))
    }
  }
  throw last instanceof Error ? last : new Error('network error')
}

async function detailOf(res: Response): Promise<string> {
  return res
    .json()
    .then((j) => j.detail ?? res.statusText)
    .catch(() => res.statusText)
}

async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const res = await rfetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(await detailOf(res))
  return res.json() as Promise<T>
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await rfetch(`${BASE}${path}`)
  if (!res.ok) throw new Error(await detailOf(res))
  return res.json() as Promise<T>
}

export function runForward(prompt: string, topK = 10): Promise<ForwardResponse> {
  return postJSON<ForwardResponse>('/forward', { prompt, top_k: topK })
}

export function runTrajectory(prompt: string): Promise<TrajectoryResponse> {
  return postJSON<TrajectoryResponse>('/trajectory', { prompt })
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

export function scanLayer(layer: number): Promise<NeuronScanResponse> {
  return getJSON<NeuronScanResponse>(`/neuron/scan?layer=${layer}`)
}

export function neuronDetail(layer: number, index: number): Promise<NeuronDetail> {
  return getJSON<NeuronDetail>(`/neuron?layer=${layer}&index=${index}`)
}

export function getHealth(): Promise<HealthResponse> {
  return getJSON<HealthResponse>('/health')
}

export function getModels(): Promise<ModelInfo[]> {
  return getJSON<ModelInfo[]>('/models')
}

export function getSaeInfo(model: string): Promise<SaeInfo> {
  return getJSON<SaeInfo>(`/sae/info?model_key=${encodeURIComponent(model)}`)
}

export function runSaeFeatures(
  prompt: string,
  layer: number,
  model: string,
  topK = 12,
): Promise<SaeFeaturesResponse> {
  return postJSON<SaeFeaturesResponse>('/sae/features', {
    prompt,
    layer,
    top_k: topK,
    model_key: model,
  })
}

export function getSaeLabels(
  model: string,
  layer: number,
  indices: number[],
): Promise<Record<string, string | null>> {
  return postJSON<Record<string, string | null>>('/sae/labels', {
    model_key: model,
    layer,
    indices,
  })
}
