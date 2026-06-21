from __future__ import annotations

import threading

import torch
from fastapi import HTTPException
from transformer_lens import HookedTransformer

MODELS = {
    "gpt2": {
        "tl_name": "gpt2",
        "device": "cpu",
        "dtype": None,
        "kwargs": {},
    },
    "gemma-2-2b-it": {
        "tl_name": "google/gemma-2-2b-it",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dtype": torch.float16,
        "kwargs": {
            "fold_ln": True,
            "center_writing_weights": False,
            "center_unembed": False,
        },
        "low_mem": True,
        "no_processing": True,
        "requires_gpu": True,
    },
}

DEFAULT_MODEL = "gpt2"
MODEL_NAME = "gpt2"

_models: dict[str, HookedTransformer] = {}
_locks: dict[str, threading.RLock] = {key: threading.RLock() for key in MODELS}
_load_lock = threading.Lock()
LOCK = _locks[DEFAULT_MODEL]


def list_models() -> list[dict]:
    cuda = torch.cuda.is_available()
    return [
        {"key": key, "name": spec["tl_name"], "device": spec["device"]}
        for key, spec in MODELS.items()
        if cuda or not spec.get("requires_gpu")
    ]


def get_model(key: str = DEFAULT_MODEL) -> HookedTransformer:
    if key not in MODELS:
        raise HTTPException(status_code=400, detail=f"unknown model {key!r}; options are {list(MODELS)}")
    if MODELS[key].get("requires_gpu") and not torch.cuda.is_available():
        raise HTTPException(
            status_code=503,
            detail=f"{key} needs a GPU; this host is CPU-only. Run the Colab/Kaggle notebook for Gemma.",
        )
    if key not in _models:
        with _load_lock:
            if key not in _models:
                torch.set_grad_enabled(False)
                spec = MODELS[key]
                opts = dict(spec["kwargs"])
                if spec["dtype"] is not None:
                    opts["dtype"] = spec["dtype"]
                if spec.get("no_processing"):
                    # Let TransformerLens load + place the weights itself. It streams from
                    # safetensors (memory-mapped, no second full host copy) and frees its
                    # internal HF model before allocating params — so host-RAM peak stays
                    # near one copy instead of the ~3 copies (hf + state dict + params) that
                    # OOM-kill a free 12 GB Colab T4 when an external hf_model is passed.
                    # no_processing skips weight folding/centering, which is what spikes RAM;
                    # fold_ln is mathematically equivalent, so the residual stream (and thus
                    # SAE features / the experiment) is unchanged.
                    model = HookedTransformer.from_pretrained_no_processing(
                        spec["tl_name"], device=spec["device"], dtype=spec["dtype"]
                    )
                else:
                    model = HookedTransformer.from_pretrained(
                        spec["tl_name"], device=spec["device"], **opts
                    )
                model.eval()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                _models[key] = model
    return _models[key]


def lock_for(key: str = DEFAULT_MODEL) -> threading.Lock:
    return _locks.get(key, LOCK)


def synchronized(fn):
    def wrapper(*args, **kwargs):
        with lock_for(kwargs.get("model_key", DEFAULT_MODEL)):
            with torch.no_grad():
                return fn(*args, **kwargs)

    return wrapper
