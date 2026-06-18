from __future__ import annotations

import threading

import requests
from fastapi import HTTPException

from model import get_model, synchronized

SAE_SPECS = {
    "gpt2": {
        "release": "gpt2-small-res-jb",
        "sae_id": "blocks.{layer}.hook_resid_pre",
        "np_model": "gpt2-small",
        "np_source": "{layer}-res-jb",
        "n_layers": 12,
        "pt_it_caveat": False,
    },
    "gemma-2-2b-it": {
        "release": "gemma-scope-2b-pt-res-canonical",
        "sae_id": "layer_{layer}/width_16k/canonical",
        "np_model": "gemma-2-2b",
        "np_source": "{layer}-gemmascope-res-16k",
        "n_layers": 26,
        "pt_it_caveat": True,
    },
}

_saes: dict = {}
_sae_load_lock = threading.Lock()
_labels: dict = {}


def has_sae(model_key: str) -> bool:
    return model_key in SAE_SPECS


def sae_info(model_key: str) -> dict:
    spec = SAE_SPECS[model_key]
    return {
        "model": model_key,
        "n_layers": spec["n_layers"],
        "pt_it_caveat": spec["pt_it_caveat"],
        "release": spec["release"],
    }


def _load_sae(model_key: str, layer: int):
    from sae_lens import SAE

    spec = SAE_SPECS[model_key]
    device = str(next(get_model(model_key).parameters()).device)
    out = SAE.from_pretrained(
        release=spec["release"],
        sae_id=spec["sae_id"].format(layer=layer),
        device=device,
    )
    return out[0] if isinstance(out, tuple) else out


def get_sae(model_key: str, layer: int):
    cache_key = (model_key, layer)
    if cache_key not in _saes:
        with _sae_load_lock:
            if cache_key not in _saes:
                _saes[cache_key] = _load_sae(model_key, layer)
    return _saes[cache_key]


def _hook_name(sae) -> str:
    meta = getattr(sae.cfg, "metadata", None)
    if meta is not None and getattr(meta, "hook_name", None):
        return meta.hook_name
    return sae.cfg.hook_name


def label(model_key: str, layer: int, index: int):
    spec = SAE_SPECS[model_key]
    source = spec["np_source"].format(layer=layer)
    cache_key = (spec["np_model"], source, int(index))
    if cache_key in _labels:
        return _labels[cache_key]
    url = f"https://www.neuronpedia.org/api/feature/{spec['np_model']}/{source}/{index}"
    text = None
    try:
        r = requests.get(url, timeout=15)
        if r.ok:
            exps = r.json().get("explanations") or []
            if exps:
                text = exps[0].get("description")
    except requests.RequestException:
        text = None
    _labels[cache_key] = text
    return text


def labels_for(model_key: str, layer: int, indices: list[int]) -> dict:
    return {str(int(i)): label(model_key, layer, int(i)) for i in indices}


@synchronized
def sae_features(prompt: str, layer: int, top_k: int = 12, model_key: str = "gpt2") -> dict:
    model = get_model(model_key)
    sae = get_sae(model_key, layer)
    hook = _hook_name(sae)
    sdtype = next(sae.parameters()).dtype
    tokens = model.to_tokens(prompt)
    str_tokens = model.to_str_tokens(prompt)
    _, cache = model.run_with_cache(tokens, names_filter=hook)
    acts = sae.encode(cache[hook].to(sdtype))[0]
    per_token = []
    for pos in range(acts.shape[0]):
        vals, idxs = acts[pos].topk(top_k)
        per_token.append(
            [
                {"index": int(i), "act": round(float(v), 3)}
                for v, i in zip(vals.tolist(), idxs.tolist())
                if float(v) > 0
            ]
        )
    return {
        "model": model_key,
        "layer": layer,
        "hook": hook,
        "d_sae": int(acts.shape[1]),
        "pt_it_caveat": SAE_SPECS[model_key]["pt_it_caveat"],
        "tokens": str_tokens,
        "features": per_token,
    }


@synchronized
def intervene(prompt, layer, feature, mode="ablate", coeff=8.0, top_k=10, model_key="gpt2"):
    model = get_model(model_key)
    sae = get_sae(model_key, layer)
    if feature < 0 or feature >= sae.cfg.d_sae:
        raise HTTPException(status_code=400, detail=f"feature must be 0..{sae.cfg.d_sae - 1}")
    hook = _hook_name(sae)
    sdtype = next(sae.parameters()).dtype
    w_dec = sae.W_dec[feature]
    tokens = model.to_tokens(prompt)
    str_tokens = model.to_str_tokens(prompt)

    base_logits_full, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook)
    base_logits = base_logits_full[0, -1]
    feat_acts = sae.encode(cache[hook].to(sdtype))[0, :, feature]

    def edit(resid, hook):
        if mode == "steer":
            return resid + (coeff * w_dec).to(resid.dtype)
        acts = sae.encode(resid.to(sdtype))
        return resid - (acts[..., feature : feature + 1] * w_dec).to(resid.dtype)

    int_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook, edit)])[0, -1]

    base_probs = base_logits.softmax(dim=-1)
    int_probs = int_logits.softmax(dim=-1)

    def top(logits, probs):
        return [
            {"token": model.tokenizer.decode([i]), "logit": round(float(logits[i]), 3), "prob": round(float(probs[i]), 5)}
            for i in logits.topk(top_k).indices.tolist()
        ]

    union = list(dict.fromkeys(base_logits.topk(top_k).indices.tolist() + int_logits.topk(top_k).indices.tolist()))
    deltas = [
        {
            "token": model.tokenizer.decode([i]),
            "base": round(float(base_logits[i]), 3),
            "delta": round(float(int_logits[i] - base_logits[i]), 3),
        }
        for i in union
    ]
    deltas.sort(key=lambda d: -abs(d["delta"]))

    return {
        "model": model_key,
        "layer": layer,
        "feature": feature,
        "mode": mode,
        "coeff": coeff,
        "tokens": str_tokens,
        "feature_acts": [round(float(x), 3) for x in feat_acts.tolist()],
        "baseline": top(base_logits, base_probs),
        "intervened": top(int_logits, int_probs),
        "deltas": deltas[:12],
    }
