from __future__ import annotations

import numpy as np
import torch
from fastapi import HTTPException
from sklearn.decomposition import PCA

from model import get_model
from schemas import MAX_TOKENS


def run_trajectory(prompt: str) -> dict:
    model = get_model()
    tokens = model.to_tokens(prompt)
    seq = tokens.shape[1]
    if seq > MAX_TOKENS:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt is {seq} tokens; max is {MAX_TOKENS}. Use a shorter prompt.",
        )
    str_tokens = model.to_str_tokens(prompt)
    n_layers = model.cfg.n_layers

    keep = {"blocks.0.hook_resid_pre"} | {f"blocks.{l}.hook_resid_post" for l in range(n_layers)}
    _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in keep, return_type=None)

    points = [cache["blocks.0.hook_resid_pre"][0]]
    for layer in range(n_layers):
        points.append(cache["resid_post", layer][0])
    stacked = torch.stack(points, dim=0).float().cpu().numpy()
    n_points = stacked.shape[0]

    flat = stacked.reshape(n_points * seq, -1)
    flat = flat / np.clip(np.linalg.norm(flat, axis=1, keepdims=True), 1e-8, None)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(flat).reshape(n_points, seq, 2)

    labels = ["emb"] + [f"L{l}" for l in range(n_layers)]
    trajectories = [
        [[round(float(coords[p, t, 0]), 4), round(float(coords[p, t, 1]), 4)] for p in range(n_points)]
        for t in range(seq)
    ]
    return {
        "tokens": str_tokens,
        "layer_labels": labels,
        "explained_variance": [
            round(float(pca.explained_variance_ratio_[0]), 4),
            round(float(pca.explained_variance_ratio_[1]), 4),
        ],
        "trajectories": trajectories,
    }
