export interface TokenPrediction {
  token: string
  token_id: number
  logit: number
  prob: number
}

export interface ForwardResponse {
  tokens: string[]
  token_ids: number[]
  n_layers: number
  n_heads: number
  top_predictions: TokenPrediction[]
  attention: number[][][][]
}

export interface HealthResponse {
  status: string
  model: string
  n_layers: number
  n_heads: number
  d_model: number
}

export interface PatchResponse {
  tokens: string[]
  corrupted_tokens: string[]
  n_layers: number
  seq: number
  answer: string
  corrupted_answer: string
  logit_diff_clean: number
  logit_diff_corrupted: number
  scores: number[][]
}

export interface ConceptInfo {
  key: string
  label: string
  kind: string
  positive: string
  negative: string
}

export interface ProbeConceptResult {
  key: string
  label: string
  kind: string
  positive: string
  negative: string
  n_examples: number
  n_train: number
  n_test: number
  baseline: number
  train_acc: number[]
  test_acc: number[]
}

export interface ProbeResponse {
  n_layers: number
  results: ProbeConceptResult[]
}

export interface TrajectoryResponse {
  tokens: string[]
  layer_labels: string[]
  explained_variance: number[]
  trajectories: number[][][]
}

export interface NeuronContext {
  tokens: string[]
  acts: number[]
  max_pos: number
  max_act: number
}

export interface NeuronSummary {
  index: number
  max_act: number
  density: number
  contexts: NeuronContext[]
}

export interface NeuronScanResponse {
  layer: number
  d_mlp: number
  n_sentences: number
  neurons: NeuronSummary[]
}

export interface NeuronDetail {
  layer: number
  index: number
  max_act: number
  density: number
  contexts: NeuronContext[]
}

export interface ModelInfo {
  key: string
  name: string
  device: string
}

export interface SaeInfo {
  available: boolean
  model?: string
  n_layers?: number
  pt_it_caveat?: boolean
  release?: string
}

export interface SaeFeature {
  index: number
  act: number
}

export interface SaeFeaturesResponse {
  model: string
  layer: number
  hook: string
  d_sae: number
  pt_it_caveat: boolean
  tokens: string[]
  features: SaeFeature[][]
}

export interface Prediction {
  token: string
  logit: number
  prob: number
}

export interface FeatureDelta {
  token: string
  base: number
  delta: number
}

export interface InterveneResponse {
  model: string
  layer: number
  feature: number
  mode: string
  coeff: number
  tokens: string[]
  feature_acts: number[]
  baseline: Prediction[]
  intervened: Prediction[]
  deltas: FeatureDelta[]
}

export interface AttributeResponse {
  model: string
  method: string
  tokens: string[]
  corrupted_tokens: string[]
  answer: string
  corrupted_answer: string
  logit_diff_clean: number
  logit_diff_corrupted: number
  attribution: number[]
}

export interface ExperimentRow {
  subject: string
  grounded: string
  parametric: string
  predicted: string
  label: string
  logit_diff: number
  attr_peak: string | null
  attr_peak_value: number | null
  attr_on_context: boolean | null
}

export interface ExperimentFlip {
  subject: string
  grounded: string
  parametric: string
  feature: number
  feature_label: string | null
  ld_before: number
  ld_after: number
  shift: number
}

export interface ExperimentResponse {
  model: string
  layer: number
  n_total: number
  n_grounded: number
  n_confabulated: number
  rows: ExperimentRow[]
  flip: ExperimentFlip | null
}
