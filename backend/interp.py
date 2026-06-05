from __future__ import annotations

import numpy as np
import torch
from fastapi import HTTPException

from model import get_model, synchronized
from schemas import MAX_PATCH_TOKENS, MAX_TOKENS


def _check_length(tokens: torch.Tensor, limit: int = MAX_TOKENS) -> None:
    seq = tokens.shape[1]
    if seq > limit:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt is {seq} tokens; max is {limit}. Use a shorter prompt.",
        )


@synchronized
def run_forward(prompt: str, top_k: int = 10) -> dict:
    model = get_model()
    tokens = model.to_tokens(prompt)
    _check_length(tokens)
    str_tokens = model.to_str_tokens(prompt)

    logits, cache = model.run_with_cache(tokens, return_type="logits")
    final_logits = logits[0, -1]
    probs = final_logits.softmax(dim=-1)

    top = torch.topk(probs, top_k)
    top_predictions = [
        {
            "token": model.tokenizer.decode([idx]),
            "token_id": int(idx),
            "logit": round(float(final_logits[idx]), 4),
            "prob": round(float(p), 6),
        }
        for p, idx in zip(top.values.tolist(), top.indices.tolist())
    ]

    attention = [
        np.round(cache["pattern", layer][0].cpu().numpy(), 4).tolist()
        for layer in range(model.cfg.n_layers)
    ]

    return {
        "tokens": str_tokens,
        "token_ids": tokens[0].tolist(),
        "n_layers": model.cfg.n_layers,
        "n_heads": model.cfg.n_heads,
        "top_predictions": top_predictions,
        "attention": attention,
    }


def _single_token_id(model, text: str) -> int:
    ids = model.to_tokens(text, prepend_bos=False)[0]
    if ids.shape[0] != 1:
        raise HTTPException(
            status_code=400,
            detail=f"{text!r} is not a single token (got {ids.shape[0]}). Try a leading space, e.g. ' Paris'.",
        )
    return int(ids[0])


@synchronized
def run_patching(
    clean_prompt: str, corrupted_prompt: str, answer: str, corrupted_answer: str
) -> dict:
    model = get_model()
    clean_tokens = model.to_tokens(clean_prompt)
    corrupted_tokens = model.to_tokens(corrupted_prompt)

    if clean_tokens.shape[1] != corrupted_tokens.shape[1]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Clean ({clean_tokens.shape[1]} tok) and corrupted "
                f"({corrupted_tokens.shape[1]} tok) prompts must tokenize to the same length."
            ),
        )
    _check_length(clean_tokens, MAX_PATCH_TOKENS)
    seq = clean_tokens.shape[1]

    answer_id = _single_token_id(model, answer)
    corr_id = _single_token_id(model, corrupted_answer)

    def logit_diff(final_logits: torch.Tensor) -> torch.Tensor:
        return final_logits[..., answer_id] - final_logits[..., corr_id]

    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    ld_clean = float(logit_diff(clean_logits[0, -1]))
    ld_corrupted = float(logit_diff(model(corrupted_tokens)[0, -1]))
    denom = ld_clean - ld_corrupted
    if abs(denom) < 1e-6:
        denom = 1e-6 if denom >= 0 else -1e-6

    pos = torch.arange(seq)
    batch = corrupted_tokens.repeat(seq, 1)

    scores: list[list[float]] = []
    for layer in range(model.cfg.n_layers):
        hook_name = f"blocks.{layer}.hook_resid_pre"
        clean_resid = clean_cache[hook_name]

        def patch(resid: torch.Tensor, hook, _clean=clean_resid) -> torch.Tensor:
            resid[pos, pos, :] = _clean[0, pos, :]
            return resid

        patched_logits = model.run_with_hooks(batch, fwd_hooks=[(hook_name, patch)])
        ld = logit_diff(patched_logits[:, -1, :])
        row = (ld - ld_corrupted) / denom
        scores.append([round(float(x), 4) for x in row])

    return {
        "tokens": model.to_str_tokens(clean_prompt),
        "corrupted_tokens": model.to_str_tokens(corrupted_prompt),
        "n_layers": model.cfg.n_layers,
        "seq": seq,
        "answer": answer,
        "corrupted_answer": corrupted_answer,
        "logit_diff_clean": round(ld_clean, 4),
        "logit_diff_corrupted": round(ld_corrupted, 4),
        "scores": scores,
    }
