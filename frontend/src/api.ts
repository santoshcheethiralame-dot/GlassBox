import type {
  AttributeResponse,
  ConceptInfo,
  ExperimentResponse,
  ForwardResponse,
  HealthResponse,
  InterveneResponse,
  ModelInfo,
  NeuronDetail,
  NeuronScanResponse,
  PatchResponse,
  ProbeResponse,
  SaeFeaturesResponse,
  SaeInfo,
  SaeTrackResponse,
  TrajectoryResponse,
} from './types'

const BASE = '/api'

// Only idempotent GETs are retried. POSTs run real compute on the server, so a
// network throw could mean the request already landed — retrying would double-run it.
async function rfetch(url: string, opts: RequestInit | undefined, attempts: number): Promise<Response> {
  let last: unknown
  for (let a = 0; a < attempts; a++) {
    try {
      return await fetch(url, opts)
    } catch (e) {
      last = e
      if (a < attempts - 1) await new Promise((r) => setTimeout(r, 350))
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
  const res = await rfetch(
    `${BASE}${path}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    },
    1,
  )
  if (!res.ok) throw new Error(await detailOf(res))
  return res.json() as Promise<T>
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await rfetch(`${BASE}${path}`, undefined, 4)
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

export function runSaeTrack(
  prompt: string,
  layer: number,
  feature: number,
  model: string,
): Promise<SaeTrackResponse> {
  return postJSON<SaeTrackResponse>('/sae/track', { prompt, layer, feature, model_key: model })
}

export interface InterveneInput {
  prompt: string
  layer: number
  feature: number
  mode: 'ablate' | 'steer'
  coeff: number
  model: string
}

export function runIntervene(input: InterveneInput): Promise<InterveneResponse> {
  return postJSON<InterveneResponse>('/sae/intervene', {
    prompt: input.prompt,
    layer: input.layer,
    feature: input.feature,
    mode: input.mode,
    coeff: input.coeff,
    model_key: input.model,
  })
}

export interface AttributeInput {
  clean_prompt: string
  corrupted_prompt: string
  answer: string
  corrupted_answer: string
  method: 'attribution' | 'activation'
  model: string
}

export function runAttribute(input: AttributeInput): Promise<AttributeResponse> {
  return postJSON<AttributeResponse>('/attribute', {
    clean_prompt: input.clean_prompt,
    corrupted_prompt: input.corrupted_prompt,
    answer: input.answer,
    corrupted_answer: input.corrupted_answer,
    method: input.method,
    model_key: input.model,
  })
}

export function runExperiment(model: string, layer = 7): Promise<ExperimentResponse> {
  return postJSON<ExperimentResponse>('/experiment', { model_key: model, layer })
}
