# GlassBox

Look inside **GPT-2 small** — attention patterns, activation patching (causal
tracing), linear probes, and neuron analysis, visualised in the browser.

Built on [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
for model internals, FastAPI for the backend, and React + D3 for the
visualisation layer.

## Stack
- **Backend:** Python 3.12, FastAPI, TransformerLens, PyTorch (CPU), scikit-learn
- **Frontend:** Vite + React + TypeScript, d3-scale for colour mapping
- **Model:** GPT-2 small (124M) — 12 layers x 12 heads, d_model 768

## Running locally

### Backend
```powershell
cd backend
.venv\Scripts\Activate.ps1
uvicorn main:app --reload --port 8000
```
First run downloads GPT-2 weights (~500 MB) to the HuggingFace cache.

### Frontend
```powershell
cd frontend
npm install
npm run dev
```
Open http://localhost:5173 (the dev server proxies API calls to :8000).

## Features
- [x] Forward pass: tokenization + top-k next-token predictions
- [x] Attention heatmaps: per layer/head attention patterns (12 × 12), small-multiples + detail view
- [x] Activation patching: residual-stream causal-tracing heatmap (clean vs corrupted)
- [ ] Linear probing: per-layer concept decodability
- [ ] Neuron analysis: maximally-activating examples

The first three are built and verified end-to-end; linear probing and neuron
analysis are next.

## API
- `GET  /health` — model dimensions
- `POST /forward` — `{prompt, top_k}` → tokens, top-k predictions, attention `[layer][head][q][k]`
- `POST /patch` — `{clean_prompt, corrupted_prompt, answer, corrupted_answer}` → restoration `scores[layer][pos]`
