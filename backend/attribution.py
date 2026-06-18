from __future__ import annotations

import torch
from fastapi import HTTPException

from model import get_model, synchronized

HOOK = "blocks.0.hook_resid_pre"


def _single_token(model, text):
    ids = model.to_tokens(text, prepend_bos=False)[0]
    if ids.shape[0] != 1:
        raise HTTPException(
            status_code=400,
            detail=f"{text!r} is not a single token (got {ids.shape[0]}); try a leading space, e.g. ' Paris'.",
        )
    return int(ids[0])


@synchronized
def attribute(
    clean_prompt, corrupted_prompt, answer, corrupted_answer, method="attribution", model_key="gpt2"
):
    model = get_model(model_key)
    clean_tokens = model.to_tokens(clean_prompt)
    corr_tokens = model.to_tokens(corrupted_prompt)
    if clean_tokens.shape[1] != corr_tokens.shape[1]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"clean ({clean_tokens.shape[1]} tok) and corrupted ({corr_tokens.shape[1]} tok) "
                "must tokenize to the same length."
            ),
        )
    seq = clean_tokens.shape[1]
    str_tokens = model.to_str_tokens(clean_prompt)
    corr_str_tokens = model.to_str_tokens(corrupted_prompt)
    ans_id = _single_token(model, answer)
    corr_id = _single_token(model, corrupted_answer)

    def ld(final):
        return final[ans_id] - final[corr_id]

    clean_logits, clean_cache = model.run_with_cache(clean_tokens, names_filter=lambda n: n == HOOK)
    clean_resid = clean_cache[HOOK][0]
    ld_clean = float(ld(clean_logits[0, -1]))

    if method == "activation":
        ld_corr = float(ld(model(corr_tokens)[0, -1]))
        attribution = []
        for pos in range(seq):

            def patch(resid, hook, p=pos):
                resid = resid.clone()
                resid[0, p] = clean_resid[p]
                return resid

            patched = model.run_with_hooks(corr_tokens, fwd_hooks=[(HOOK, patch)])
            attribution.append(float(ld(patched[0, -1])) - ld_corr)
    else:
        store = {}
        with torch.enable_grad():

            def grab(resid, hook):
                r = resid.detach().requires_grad_(True)
                store["r"] = r
                return r

            corr_logits = model.run_with_hooks(corr_tokens, fwd_hooks=[(HOOK, grab)])
            metric = ld(corr_logits[0, -1])
            grad = torch.autograd.grad(metric, store["r"])[0][0]
        ld_corr = float(metric.detach())
        corr_resid = store["r"].detach()[0]
        attribution = ((clean_resid - corr_resid) * grad).sum(-1).tolist()

    attribution = [round(float(x), 4) for x in attribution]

    return {
        "model": model_key,
        "method": method,
        "tokens": str_tokens,
        "corrupted_tokens": corr_str_tokens,
        "answer": answer,
        "corrupted_answer": corrupted_answer,
        "logit_diff_clean": round(ld_clean, 4),
        "logit_diff_corrupted": round(ld_corr, 4),
        "attribution": attribution,
    }
