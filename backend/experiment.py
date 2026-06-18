from __future__ import annotations

from attribution import attribute
from model import get_model, synchronized
from saes import _hook_name, get_sae, label, sae_features

CONFLICTS = [
    {"subj": "The Eiffel Tower is in", "grounded": " Rome", "parametric": " Paris"},
    {"subj": "The Colosseum is in", "grounded": " London", "parametric": " Rome"},
    {"subj": "Big Ben is in", "grounded": " Paris", "parametric": " London"},
    {"subj": "The Brandenburg Gate is in", "grounded": " Madrid", "parametric": " Berlin"},
    {"subj": "The Kremlin is in", "grounded": " Vienna", "parametric": " Moscow"},
    {"subj": "The Sphinx is in", "grounded": " Berlin", "parametric": " Cairo"},
]


def _tok_id(model, text):
    return int(model.to_tokens(text, prepend_bos=False)[0][0])


def _prompt(subj, fact):
    return f"{subj}{fact}. {subj}"


@synchronized
def run_experiment(model_key="gpt2", layer=7, top_k=6):
    model = get_model(model_key)
    rows = []
    for c in CONFLICTS:
        clean = _prompt(c["subj"], c["grounded"])
        corrupt = _prompt(c["subj"], c["parametric"])
        logits = model(model.to_tokens(clean))[0, -1]
        g_id = _tok_id(model, c["grounded"])
        p_id = _tok_id(model, c["parametric"])
        top = int(logits.argmax())
        lab = "grounded" if top == g_id else "confabulated" if top == p_id else "other"
        ld = float(logits[g_id] - logits[p_id])

        try:
            attr = attribute(clean, corrupt, c["grounded"], c["parametric"], "attribution", model_key)
            a = attr["attribution"]
            peak = max(range(len(a)), key=lambda i: a[i])
            attr_peak = attr["tokens"][peak]
            attr_peak_value = round(a[peak], 3)
            attr_on_context = attr_peak.strip() == c["grounded"].strip()
        except Exception:
            attr_peak, attr_peak_value, attr_on_context = None, None, None

        rows.append(
            {
                "subject": c["subj"],
                "grounded": c["grounded"],
                "parametric": c["parametric"],
                "predicted": model.tokenizer.decode([top]),
                "label": lab,
                "logit_diff": round(ld, 3),
                "attr_peak": attr_peak,
                "attr_peak_value": attr_peak_value,
                "attr_on_context": attr_on_context,
            }
        )

    flip = _ablation_flip(model, model_key, layer, top_k, rows)
    return {
        "model": model_key,
        "layer": layer,
        "n_total": len(rows),
        "n_grounded": sum(r["label"] == "grounded" for r in rows),
        "n_confabulated": sum(r["label"] == "confabulated" for r in rows),
        "rows": rows,
        "flip": flip,
    }


def _ablation_flip(model, model_key, layer, top_k, rows):
    target = min(rows, key=lambda r: r["logit_diff"])
    prompt = _prompt(target["subject"], target["grounded"])
    tokens = model.to_tokens(prompt)
    g_id = _tok_id(model, target["grounded"])
    p_id = _tok_id(model, target["parametric"])
    sae = get_sae(model_key, layer)
    hook = _hook_name(sae)
    sdtype = next(sae.parameters()).dtype

    base = model(tokens)[0, -1]
    base_ld = float(base[g_id] - base[p_id])

    _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook)
    acts = sae.encode(cache[hook].to(sdtype))[0, -1]
    top_idx = acts.topk(top_k).indices.tolist()

    best = None
    for fi in top_idx:
        w = sae.W_dec[fi]

        def ab(resid, hook, _fi=fi, _w=w):
            a = sae.encode(resid.to(sdtype))
            return resid - (a[..., _fi : _fi + 1] * _w).to(resid.dtype)

        new = model.run_with_hooks(tokens, fwd_hooks=[(hook, ab)])[0, -1]
        new_ld = float(new[g_id] - new[p_id])
        if best is None or new_ld > best["ld_after"]:
            best = {
                "feature": int(fi),
                "ld_after": round(new_ld, 3),
                "feature_label": label(model_key, layer, int(fi)),
            }

    return {
        "subject": target["subject"],
        "grounded": target["grounded"],
        "parametric": target["parametric"],
        "feature": best["feature"],
        "feature_label": best["feature_label"],
        "ld_before": round(base_ld, 3),
        "ld_after": best["ld_after"],
        "shift": round(best["ld_after"] - base_ld, 3),
    }
