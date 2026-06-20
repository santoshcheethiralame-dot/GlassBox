from __future__ import annotations

import numpy as np
import torch
from fastapi import HTTPException

from corpus import CORPUS
from model import DEFAULT_MODEL, get_model, synchronized

# tokenized corpus is per-model (different tokenizers); the activation cache is a
# single slot keyed by (model_key, layer) so a model switch never serves stale acts.
_tokens: dict[str, tuple] = {}
_cache_key = None
_cache = None


def _ensure_tokens(model_key: str = DEFAULT_MODEL):
    if model_key not in _tokens:
        model = get_model(model_key)
        str_tokens = [model.to_str_tokens(s) for s in CORPUS]
        rows = [model.to_tokens(s)[0] for s in CORPUS]
        lengths = [int(r.shape[0]) for r in rows]
        max_len = max(lengths)
        pad = model.tokenizer.pad_token_id
        if pad is None:
            pad = model.tokenizer.eos_token_id
        batch = torch.full((len(rows), max_len), pad, dtype=torch.long)
        for i, r in enumerate(rows):
            batch[i, : r.shape[0]] = r
        _tokens[model_key] = (str_tokens, batch, lengths)
    return _tokens[model_key]


@synchronized
def _layer_acts(layer, model_key: str = DEFAULT_MODEL):
    global _cache_key, _cache
    if _cache_key == (model_key, layer):
        return _cache
    model = get_model(model_key)
    str_tokens, batch, lengths = _ensure_tokens(model_key)
    name = f"blocks.{layer}.mlp.hook_post"
    _, cache = model.run_with_cache(batch, names_filter=lambda n: n == name, return_type=None)
    _cache_key = (model_key, layer)
    _cache = {"acts": cache[name].float().cpu().numpy(), "lengths": lengths, "str_tokens": str_tokens}
    return _cache


def _window(toks, acts_row, pos, before=6, after=2):
    lo = max(1, pos - before)
    hi = min(len(toks), pos + after + 1)
    return {
        "tokens": toks[lo:hi],
        "acts": [round(float(acts_row[p]), 3) for p in range(lo, hi)],
        "max_pos": pos - lo,
        "max_act": round(float(acts_row[pos]), 3),
    }


def _top_contexts(c, index, top_k, distinct=True):
    acts = c["acts"]
    lengths = c["lengths"]
    str_tokens = c["str_tokens"]
    triples = []
    for i, length in enumerate(lengths):
        col = acts[i, :, index]
        for pos in range(1, length):
            triples.append((float(col[pos]), i, pos))
    triples.sort(key=lambda x: -x[0])
    res = []
    used = set()
    for _, i, pos in triples:
        if distinct and i in used:
            continue
        used.add(i)
        res.append(_window(str_tokens[i], acts[i, :, index], pos))
        if len(res) >= top_k:
            break
    return res


def scan_layer(layer, n_neurons=12, model_key: str = DEFAULT_MODEL):
    model = get_model(model_key)
    if layer < 0 or layer >= model.cfg.n_layers:
        raise HTTPException(status_code=400, detail=f"layer must be 0..{model.cfg.n_layers - 1}")
    c = _layer_acts(layer, model_key=model_key)
    acts = c["acts"]
    lengths = c["lengths"]
    n_seq, max_len, d_mlp = acts.shape
    mask = np.zeros((n_seq, max_len), dtype=bool)
    for i, length in enumerate(lengths):
        mask[i, 1:length] = True
    real = acts[mask]
    max_per = real.max(axis=0)
    thresh = np.maximum(0.5 * max_per, 1e-6)
    density = (real > thresh[None, :]).mean(axis=0)
    score = max_per * (1.0 - density)
    order = np.argsort(-score)[:n_neurons]
    neurons = [
        {
            "index": int(ni),
            "max_act": round(float(max_per[ni]), 3),
            "density": round(float(density[ni]), 4),
            "contexts": _top_contexts(c, int(ni), 3),
        }
        for ni in order
    ]
    return {"layer": layer, "d_mlp": int(d_mlp), "n_sentences": n_seq, "neurons": neurons}


def neuron_detail(layer, index, model_key: str = DEFAULT_MODEL):
    model = get_model(model_key)
    if layer < 0 or layer >= model.cfg.n_layers:
        raise HTTPException(status_code=400, detail=f"layer must be 0..{model.cfg.n_layers - 1}")
    if index < 0 or index >= model.cfg.d_mlp:
        raise HTTPException(status_code=400, detail=f"neuron index must be 0..{model.cfg.d_mlp - 1}")
    c = _layer_acts(layer, model_key=model_key)
    acts = c["acts"]
    lengths = c["lengths"]
    allv = np.concatenate([acts[i, 1:length, index] for i, length in enumerate(lengths)])
    max_act = float(allv.max())
    density = float((allv > max(0.5 * max_act, 1e-6)).mean())
    return {
        "layer": layer,
        "index": index,
        "max_act": round(max_act, 3),
        "density": round(density, 4),
        "contexts": _top_contexts(c, index, 10),
    }
