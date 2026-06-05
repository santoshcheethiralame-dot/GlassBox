from __future__ import annotations

import threading

import torch
from transformer_lens import HookedTransformer

MODEL_NAME = "gpt2"
_model: HookedTransformer | None = None
LOCK = threading.Lock()


def get_model() -> HookedTransformer:
    global _model
    if _model is None:
        torch.set_grad_enabled(False)
        _model = HookedTransformer.from_pretrained(MODEL_NAME, device="cpu")
        _model.eval()
    return _model


def synchronized(fn):
    def wrapper(*args, **kwargs):
        with LOCK:
            return fn(*args, **kwargs)

    return wrapper
