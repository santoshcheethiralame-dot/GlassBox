from __future__ import annotations

import numpy as np
import torch
from fastapi import HTTPException
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from model import get_model, synchronized

_POS = [
    "wonderful", "fantastic", "excellent", "delightful", "amazing", "superb",
    "brilliant", "lovely", "gorgeous", "charming", "pleasant", "magnificent",
    "marvellous", "terrific", "splendid",
]
_NEG = [
    "terrible", "awful", "horrible", "dreadful", "disgusting", "miserable",
    "lousy", "hideous", "appalling", "unpleasant", "atrocious", "abysmal",
    "repulsive", "gloomy", "dismal",
]
_SENT_TEMPLATES = [
    "The movie was {}",
    "Overall the experience felt {}",
    "I thought the meal was absolutely {}",
    "Her performance was truly {}",
    "The little town seemed rather {}",
]

_SINGULAR = [
    "cat", "dog", "book", "car", "tree", "house", "river", "mountain",
    "child", "song", "city", "road", "garden", "clock", "letter",
]
_PLURAL = [
    "cats", "dogs", "books", "cars", "trees", "houses", "rivers", "mountains",
    "children", "songs", "cities", "roads", "gardens", "clocks", "letters",
]
_NUM_TEMPLATES = [
    "She was looking at the {}",
    "He wrote a story about the {}",
    "We talked for hours about the {}",
    "They walked slowly past the {}",
    "I keep thinking about the {}",
]

_LIVING = [
    "dog", "horse", "rabbit", "sparrow", "farmer", "child", "dolphin", "tiger",
    "spider", "grandmother", "salmon", "beetle", "monkey", "nurse", "goat",
]
_NONLIVING = [
    "rock", "table", "hammer", "cloud", "engine", "mirror", "kettle", "pebble",
    "curtain", "tractor", "balloon", "ladder", "bucket", "candle", "helmet",
]
_LIVING_TEMPLATES = [
    "In the distance she noticed a {}",
    "He slowly walked toward the {}",
    "On the muddy path there was a {}",
    "The old painting showed a {}",
    "Near the window sat a {}",
]

_SN_PAIRS = [
    ("cat", "cats"), ("dog", "dogs"), ("bird", "birds"), ("book", "books"),
    ("key", "keys"), ("car", "cars"), ("lamp", "lamps"), ("tree", "trees"),
    ("cup", "cups"), ("friend", "friends"), ("window", "windows"), ("painting", "paintings"),
]
_SN_TEMPLATES = [
    "The {s} behind the {d} over there",
    "The {s} beside the {d} sitting there",
    "The {s} under the {d} resting there",
    "The {s} above the {d} placed there",
]


def _build(pos_words, neg_words, templates):
    prompts, labels, groups = [], [], []
    for word in pos_words:
        for t in templates:
            prompts.append(t.format(word))
            labels.append(1)
            groups.append(word)
    for word in neg_words:
        for t in templates:
            prompts.append(t.format(word))
            labels.append(0)
            groups.append(word)
    return prompts, labels, groups


def _build_subject_number(pairs, templates):
    prompts, labels, groups = [], [], []
    n = len(pairs)
    for i, (sing, plur) in enumerate(pairs):
        d_sing, d_plur = pairs[(i + 1) % n]
        for t in templates:
            prompts.append(t.format(s=sing, d=d_plur))
            labels.append(0)
            groups.append(sing)
            prompts.append(t.format(s=plur, d=d_sing))
            labels.append(1)
            groups.append(plur)
    return prompts, labels, groups


CONCEPTS = {
    "number": {
        "label": "Grammatical number (singular vs plural)",
        "kind": "syntactic",
        "positive": "plural",
        "negative": "singular",
        "data": lambda: _build(_PLURAL, _SINGULAR, _NUM_TEMPLATES),
    },
    "sentiment": {
        "label": "Sentiment (positive vs negative)",
        "kind": "semantic",
        "positive": "positive",
        "negative": "negative",
        "data": lambda: _build(_POS, _NEG, _SENT_TEMPLATES),
    },
    "animacy": {
        "label": "Animacy (living vs non-living)",
        "kind": "semantic",
        "positive": "living",
        "negative": "non-living",
        "data": lambda: _build(_LIVING, _NONLIVING, _LIVING_TEMPLATES),
    },
    "subject_number": {
        "label": "Subject number past a distractor (contextual)",
        "kind": "contextual",
        "positive": "plural subject",
        "negative": "singular subject",
        "data": lambda: _build_subject_number(_SN_PAIRS, _SN_TEMPLATES),
    },
}


def list_concepts():
    return [
        {
            "key": k,
            "label": v["label"],
            "kind": v["kind"],
            "positive": v["positive"],
            "negative": v["negative"],
        }
        for k, v in CONCEPTS.items()
    ]


@synchronized
def _last_token_resids(prompts):
    model = get_model()
    rows = [model.to_tokens(p)[0] for p in prompts]
    lengths = torch.tensor([r.shape[0] for r in rows])
    max_len = int(lengths.max())
    pad_id = model.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = model.tokenizer.eos_token_id
    batch = torch.full((len(rows), max_len), pad_id, dtype=torch.long)
    for i, r in enumerate(rows):
        batch[i, : r.shape[0]] = r
    _, cache = model.run_with_cache(
        batch, names_filter=lambda n: n.endswith("hook_resid_post"), return_type=None
    )
    sel = torch.arange(len(rows))
    last = lengths - 1
    return [
        cache["resid_post", layer][sel, last, :].float().cpu().numpy()
        for layer in range(model.cfg.n_layers)
    ]


def _probe_layers(resids, labels, groups):
    y = np.array(labels)
    g = np.array(groups)
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=0)
    train_idx, test_idx = next(splitter.split(resids[0], y, g))
    train_acc, test_acc = [], []
    for X in resids:
        scaler = StandardScaler().fit(X[train_idx])
        xs = scaler.transform(X)
        clf = LogisticRegression(max_iter=1000, C=0.4)
        clf.fit(xs[train_idx], y[train_idx])
        train_acc.append(round(float(clf.score(xs[train_idx], y[train_idx])), 4))
        test_acc.append(round(float(clf.score(xs[test_idx], y[test_idx])), 4))
    return train_acc, test_acc, len(train_idx), len(test_idx)


def run_probes(keys):
    model = get_model()
    if not keys:
        keys = list(CONCEPTS.keys())
    unknown = [k for k in keys if k not in CONCEPTS]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown concept(s) {unknown}; options are {list(CONCEPTS.keys())}.",
        )

    results = []
    for key in keys:
        spec = CONCEPTS[key]
        prompts, labels, groups = spec["data"]()
        resids = _last_token_resids(prompts)
        train_acc, test_acc, n_train, n_test = _probe_layers(resids, labels, groups)
        baseline = float(max(np.mean(labels), 1 - np.mean(labels)))
        results.append(
            {
                "key": key,
                "label": spec["label"],
                "kind": spec["kind"],
                "positive": spec["positive"],
                "negative": spec["negative"],
                "n_examples": len(prompts),
                "n_train": n_train,
                "n_test": n_test,
                "baseline": round(baseline, 4),
                "train_acc": train_acc,
                "test_acc": test_acc,
            }
        )
    return {"n_layers": model.cfg.n_layers, "results": results}
