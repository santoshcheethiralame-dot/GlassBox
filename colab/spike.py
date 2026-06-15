import os
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
MODEL = "google/gemma-2-2b-it"
LAYER = int(os.environ.get("GLASSBOX_LAYER", "20"))
PROMPT = os.environ.get("GLASSBOX_PROMPT", "The Eiffel Tower stands in the city of")
TOPK = 15


def banner(s):
    print("\n" + "=" * 10 + " " + s + " " + "=" * 10)


def load_tl():
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained(
        MODEL,
        device=DEVICE,
        dtype=DTYPE,
        fold_ln=True,
        center_writing_weights=False,
        center_unembed=False,
    )


def fidelity(model):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(DEVICE)
    tl_logits = model(ids, return_type="logits")[0, -1].float()
    hf = None
    try:
        hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE).to(DEVICE)
        hf.eval()
        with torch.no_grad():
            hf_logits = hf(ids).logits[0, -1].float()
        hf_top = int(hf_logits.argmax())
        tl_top = int(tl_logits.argmax())
        maxdiff = (hf_logits - tl_logits).abs().max().item()
        print("HF top1:", repr(tok.decode([hf_top])), "| TL top1:", repr(tok.decode([tl_top])))
        print("top-1 agree:", hf_top == tl_top, "| max|delta logit|:", round(maxdiff, 3))
        if hf_top != tl_top or maxdiff > 1.0:
            print("WARNING: TL and HF diverge — consider the nnsight fallback path.")
    except RuntimeError as e:
        print("fidelity check skipped:", e)
    finally:
        del hf
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    return tok


def load_sae(layer):
    from sae_lens import SAE

    out = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res-canonical",
        sae_id=f"layer_{layer}/width_16k/canonical",
        device=DEVICE,
    )
    return out[0] if isinstance(out, tuple) else out


def label(layer, index, dump=False):
    import json
    import requests

    source = f"{layer}-gemmascope-res-16k"
    url = f"https://www.neuronpedia.org/api/feature/gemma-2-2b/{source}/{index}"
    headers = {}
    key = os.environ.get("NEURONPEDIA_API_KEY")
    if key:
        headers["x-api-key"] = key
    try:
        r = requests.get(url, headers=headers, timeout=20)
    except Exception as e:
        return f"(request failed: {e})"
    if not r.ok:
        return f"(http {r.status_code} @ {url})"
    j = r.json()
    if dump:
        print("  raw keys:", list(j.keys()))
        print("  sample:", json.dumps(j, indent=2)[:600])
    exps = j.get("explanations") or []
    if exps:
        return exps[0].get("description") or "(empty description)"
    return "(no explanation stored)"


def main():
    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    banner("device")
    print("device:", DEVICE, "| dtype:", DTYPE)
    if DEVICE != "cuda":
        print("WARNING: no CUDA detected — Gemma-2-2B is unusably slow on CPU. Use a GPU runtime.")

    banner("load gemma-2-2b-it (transformer_lens)")
    model = load_tl()
    print("n_layers:", model.cfg.n_layers, "| d_model:", model.cfg.d_model)

    banner("fidelity vs raw HF logits")
    tok = fidelity(model)

    banner(f"load gemma scope residual SAE (layer {LAYER})")
    sae = load_sae(LAYER)
    sdtype = next(sae.parameters()).dtype
    print(
        "d_in:", getattr(sae.cfg, "d_in", "?"),
        "| d_sae:", getattr(sae.cfg, "d_sae", "?"),
        "| hook:", getattr(sae.cfg, "hook_name", "?"),
        "| dtype:", sdtype,
    )

    banner("forward + SAE encode at hook_resid_post")
    ids = model.to_tokens(PROMPT)
    hook = f"blocks.{LAYER}.hook_resid_post"
    _, cache = model.run_with_cache(ids, names_filter=hook)
    feats = sae.encode(cache[hook].to(sdtype))
    last = feats[0, -1]
    print("prompt:", repr(PROMPT))
    print("tokens:", model.to_str_tokens(PROMPT))
    print("active features on last token:", int((last > 0).sum()), "/", last.numel())

    banner(f"top-{TOPK} features + neuronpedia labels")
    print("neuronpedia source:", f"{LAYER}-gemmascope-res-16k")
    vals, idxs = last.topk(TOPK)
    for n, (v, i) in enumerate(zip(vals.tolist(), idxs.tolist())):
        print(f"  f/{i:<6} act={v:6.2f}  {label(LAYER, i, dump=(n == 0))}")

    banner("done — foundation OK if labels look coherent")


if __name__ == "__main__":
    main()
