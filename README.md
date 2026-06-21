---
title: GlassBox
emoji: 🔬
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: "GPT-2 internals: SAE features, interventions, attribution"
---

# GlassBox

**[▶ Try it live — GPT-2 in your browser, no signup](https://numero00-glassbox-interp.hf.space)**

<!-- hero shot goes here: ![GlassBox](docs/glassbox.png) -->

A browser-based microscope for transformer language models. Type a prompt and
watch the model think — attention, the residual stream, what each layer encodes —
then *intervene*: decode the residual stream into interpretable features, knock
those features out, and trace which input tokens actually caused an answer.

Runs on **GPT-2 small** out of the box — CPU, no setup. The same dashboard drives
**Gemma-2-2B-it** with Gemma Scope SAEs on a GPU, behind one public URL (see
[Gemma](#gemma-gpu)).

Built on [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
and [SAELens](https://github.com/jbloomAus/SAELens) for model internals,
[Neuronpedia](https://www.neuronpedia.org) for feature labels, FastAPI for the
service layer, and React for the dashboard. The charts are hand-rolled inline
SVG, no charting library.

## What you can see

The panels run from passive observation, to active intervention, to a causal
experiment that ties the rest together.

### Observe

- **Forward pass** — tokenization and the top-k next-token distribution.
- **Attention** — every head pattern as small multiples, plus a labelled detail
  view with the causal mask drawn explicitly.

### Decode

- **SAE features** — a sparse autoencoder reads the residual stream at a chosen
  layer and decomposes it into interpretable features, each labelled from
  Neuronpedia. GPT-2 uses the `res-jb` SAEs; Gemma uses Gemma Scope. These
  features are the vocabulary everything downstream intervenes on.

### Intervene

- **Ablation & steering** — remove a feature from the residual stream, or amplify
  it along its decoder direction with a coefficient slider, and watch the
  next-token distribution shift. The effect is causal and live.

### Trace

- **Activation patching** — copy clean residual activations into a corrupted run,
  one (layer, position) at a time, and measure how much of the answer comes back.
  The heatmap localises *where* the model holds a fact.
- **Context attribution** — for a counterfactual clean / corrupted pair, score
  every input token by how much it causes the answer. A **fast** path (attribution
  patching — one backward pass) and a **verified** path (activation patching — one
  forward pass per position) agree to within rounding.

### Experiment

- **Hallucination lab** — a batch of knowledge-conflict prompts, each asserting a
  *false* fact. Does the model follow the context (**grounded**) or fall back on
  what it memorised (**confabulated**)? Then the payoff: find the single SAE
  feature whose ablation most pushes a confabulation back toward the context, and
  flip it.

### Plus

- **Linear probes** — per-layer logistic-regression probes for several concepts,
  contrasting lexical features (readable from layer 0) against a contextual one
  the model has to compute.
- **Residual trajectory** — each token's path through layer space, projected to
  2D with PCA over L2-normalised activations.
- **Neurons** — the most selective MLP neurons in a layer and their
  maximally-activating contexts.

## A few things it surfaces

- On **Gemma-2-2B** the hallucination batch leans hard on memory — it follows
  only one of six false contexts, confabulating the rest. For `The Brandenburg
  Gate is in [false city]`, ablating a single SAE feature ("names of political
  parties & significant events") moves the grounded−memorised logit-diff from
  **−9.2 to −4.4 (+4.8)** — roughly halving the model's pull toward the memorised
  answer. One interpretable unit, one measurable lever on a confabulation. (GPT-2,
  by contrast, *follows* five of six of the same false contexts.)
- For `The capital of France is`, patching pins the answer to the **` France`
  token at layer 0**, which then routes to the final position in the upper
  layers — the logit difference swings from +1.89 (clean) to −2.81 (corrupted).
- In the hallucination lab on GPT-2, five of six false contexts are followed; the
  **Kremlin** case confabulates ` Moscow` over the context's ` Vienna`
  (logit-diff −0.74). Ablating one SAE feature shifts it **+0.54 back toward the
  context** — a single interpretable unit, measurably suppressing a confabulation.
- A **subject-number** probe — the number of a noun read past a distractor —
  starts *below* chance, because early layers latch onto the nearer distractor,
  then resolves to ~1.0 by layer 3. Plain lexical number is linearly present from
  the start.
- Layer 0, neuron **1133** is a clean *they / their* detector.

## Models

- **GPT-2 small** (124M) — 12 layers × 12 heads, d_model 768. Runs on CPU; the
  default, and the only model the public Space serves.
- **Gemma-2-2B-it** (2.6B) — 26 layers, d_model 2304. Needs a GPU; the model
  selector only offers it when one is present, so the CPU Space never tries to
  load it.

SAEs: GPT-2 `gpt2-small-res-jb` (`resid_pre`); Gemma `gemma-scope-2b-pt-res`,
16k-width canonical (`resid_post`). Gemma Scope is trained on the *base* model and
applied here to the *instruct* model — a deliberate, labelled approximation that
the SAE panel flags.

## Stack

- **Backend** — Python 3.12, FastAPI, TransformerLens, SAELens, PyTorch,
  scikit-learn.
- **Frontend** — Vite + React + TypeScript, inline SVG charts.
- **Serving** — in production FastAPI serves the built React bundle as static
  files, so the whole app is one origin on one port.

## Layout

```
backend/    FastAPI service over TransformerLens
  model.py        per-model singletons behind reentrant locks (see Notes)
  interp.py       forward pass, attention, activation patching
  saes.py         SAE features, Neuronpedia labels, ablation / steering
  attribution.py  counterfactual context attribution (fast + verified)
  experiment.py   the knowledge-conflict hallucination batch
  probing.py      per-layer linear probes
  trajectory.py   residual-stream PCA
  neurons.py      neuron scan + maximally-activating contexts
frontend/   React dashboard, one panel per feature
  src/components/
colab/      turnkey notebooks that build the app and serve Gemma over a tunnel
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

The first run downloads GPT-2 weights and the SAEs into the HuggingFace cache.

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

Then open <http://localhost:5173>. The model selector defaults to GPT-2.

## Gemma (GPU)

The free Space is CPU-only, so it serves GPT-2. To drive Gemma, run a notebook —
it builds this same app and serves it behind a public URL:

- **Colab** (T4) — `colab/glassbox.ipynb` → Runtime → T4 GPU → Run all.
- **Kaggle** (T4 ×2, ~29 GB RAM, loads Gemma comfortably) —
  `colab/glassbox_kaggle.ipynb`.

Accept the [gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it) license
with the same account as your HF token; Gemma Scope itself is public.

## API

All routes are under `/api`.

| Method | Route | Body / query | Returns |
| --- | --- | --- | --- |
| `GET`  | `/health` | — | model dimensions |
| `GET`  | `/models` | — | available models (GPU-gated) |
| `POST` | `/forward` | `{prompt, top_k, model_key}` | tokens, top-k predictions, attention |
| `POST` | `/patch` | `{clean_prompt, corrupted_prompt, answer, corrupted_answer}` | restoration `scores[layer][position]` |
| `GET`  | `/sae/info` | `?model_key` | SAE release, layers, hook point |
| `POST` | `/sae/features` | `{prompt, layer, top_k, model_key}` | top features per token, with labels |
| `POST` | `/sae/intervene` | `{prompt, layer, feature, mode, coeff, model_key}` | baseline vs intervened top-k + logit deltas |
| `POST` | `/attribute` | `{clean, corrupted, answer, corrupted_answer, method, model_key}` | per-token causal scores |
| `POST` | `/experiment` | `{model_key, layer}` | grounded/confabulated rows + the ablation flip |
| `GET`  | `/probe/concepts` | — | available probe concepts |
| `POST` | `/probe` | `{concepts}` | per-layer train/test accuracy |
| `POST` | `/trajectory` | `{prompt}` | per-token 2D PCA coordinates per layer |
| `GET`  | `/neuron/scan` | `?layer=L` | the most selective neurons, with top contexts |
| `GET`  | `/neuron` | `?layer=L&index=N` | one neuron's maximally-activating contexts |

## Notes

A few decisions that aren't obvious from the code:

- **One model per key, behind a reentrant lock, under `no_grad` by default.**
  TransformerLens caches activations by mutating hook state, and the dashboard
  fires several requests at once on load; serialising the forward passes keeps
  those caches from clobbering each other. PyTorch's grad-enabled flag is
  *thread-local*, so the FastAPI worker threads run with gradients off — otherwise
  every forward pass would build an autograd graph and exhaust memory. Attribution
  re-enables grad locally, inside its own lock, which is why the lock is reentrant.
- **Attribution uses a counterfactual pair, not one prompt.** A single prompt's
  mean-ablation is dominated by the BOS attention sink, and leave-one-out and
  gradients end up measuring different things. A clean / corrupted pair with a
  logit-difference metric makes the fast (gradient) and verified (patching) paths
  line up almost exactly — a cheap correctness check on the fast path.
- **PCA runs on L2-normalised residuals.** Residual-stream norm grows with depth,
  so on raw vectors PC1 is ~97% "how deep are we" and the trajectory is a straight
  slide. Normalising first exposes the actual directional structure.
- **Probes split by source word, not by example.** A `StratifiedGroupKFold`
  grouped on the word keeps the same word out of both train and test; otherwise
  the probe just memorises token identity and every layer scores ~1.0.
