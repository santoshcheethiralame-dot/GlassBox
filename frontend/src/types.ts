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
