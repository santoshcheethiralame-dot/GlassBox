# GlassBox

A browser-based microscope for **GPT-2 small**. Type a prompt and watch the model
think: attention patterns, causal tracing through activation patching, what each
layer linearly encodes, how the residual stream travels through layer space, and
what individual neurons fire on.

Built on [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
for model internals, FastAPI for the service layer, and React for the dashboard.

## What you can see

- **Forward pass** — tokenization and the top-k next-token distribution.
- **Attention** — every 12 × 12 head pattern as small multiples, plus a labelled
  detail view with the causal mask drawn explicitly.
- **Activation patching** — copy clean residual-stream activations into a
  corrupted run, one (layer, position) at a time, and measure how much of the
  answer comes back. The heatmap localises *where* the model holds the fact.
- **Linear probes** — per-layer logistic-regression probes for several concepts,
  contrasting lexical features (readable from layer 0) against a contextual one
  the model has to compute.
- **Residual trajectory** — each token's path through layer space, projected to
  2D with PCA over L2-normalised activations.
- **Neurons** — the most selective MLP neurons in a layer and their
  maximally-activating contexts.

## A few things it surfaces

- For `The capital of France is`, patching pins the answer to the **` France`
  token at layer 0**, which then routes to the final position in the upper
  layers — the logit difference swings from +1.89 (clean) to −2.81 (corrupted).
- A **subject-number** probe — the number of a noun read past a distractor —
  starts *below* chance, because early layers latch onto the nearer distractor,
  then resolves to ~1.0 by layer 3. Plain lexical number is linearly present from
  the start.
- Layer 0, neuron **1133** is a clean *they / their* detector.

## Stack

- **Backend** — Python 3.12, FastAPI, TransformerLens, PyTorch (CPU), scikit-learn.
- **Frontend** — Vite + React + TypeScript. The charts are hand-rolled inline SVG,
  no charting library.
- **Model** — GPT-2 small (124M): 12 layers × 12 heads, d_model 768.

## Layout

```
backend/    FastAPI service over a TransformerLens GPT-2
  model.py        shared model singleton, guarded by a lock (see Notes)
  interp.py       forward pass, attention, activation patching
  probing.py      per-layer linear probes
  trajectory.py   residual-stream PCA
  neurons.py      neuron scan + maximally-activating contexts
  corpus.py       sentences for neuron analysis
frontend/   React dashboard, one panel per feature
  src/components/
```

## Running locally

Two processes: the API on `:8000` and the Vite dev server on `:5173`, which
proxies `/api` to the backend.

### Backend

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The first run downloads GPT-2 weights (~500 MB) into the HuggingFace cache.

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

Then open <http://localhost:5173>.

## API

| Method | Route | Body / query | Returns |
| --- | --- | --- | --- |
| `GET`  | `/health` | — | model dimensions |
| `POST` | `/forward` | `{prompt, top_k}` | tokens, top-k predictions, attention `[layer][head][query][key]` |
| `POST` | `/patch` | `{clean_prompt, corrupted_prompt, answer, corrupted_answer}` | restoration `scores[layer][position]` |
| `GET`  | `/probe/concepts` | — | available probe concepts |
| `POST` | `/probe` | `{concepts}` | per-layer train/test accuracy for each concept |
| `POST` | `/trajectory` | `{prompt}` | per-token 2D PCA coordinates at each layer |
| `GET`  | `/neuron/scan` | `?layer=L` | the most selective neurons in a layer, with top contexts |
| `GET`  | `/neuron` | `?layer=L&index=N` | one neuron's maximally-activating contexts |

## Notes

A few decisions that aren't obvious from the code:

- **One shared model, behind a lock.** TransformerLens caches activations by
  mutating hook state, and the dashboard fires five requests at once on load.
  Without serialising the forward passes those caches clobber each other —
  mismatched batch sizes, missing hook keys. A single process-wide lock trades
  concurrency for correctness, which is the right call for a single-user tool.
- **PCA runs on L2-normalised residuals.** Residual-stream norm grows with depth,
  so on raw vectors PC1 is ~97% "how deep are we" and the trajectory is a
  straight slide. Normalising first exposes the actual directional structure.
- **Probes split by source word, not by example.** A `StratifiedGroupKFold`
  grouped on the word keeps the same word out of both train and test; otherwise
  the probe just memorises token identity and every layer scores ~1.0.
