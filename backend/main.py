from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from attribution import attribute
from interp import run_forward, run_patching
from model import MODEL_NAME, get_model, list_models
from neurons import neuron_detail, scan_layer
from probing import list_concepts, run_probes
from saes import has_sae, intervene, labels_for, sae_features, sae_info
from schemas import (
    AttributeRequest,
    ForwardRequest,
    InterveneRequest,
    PatchRequest,
    ProbeRequest,
    SaeFeaturesRequest,
    SaeLabelsRequest,
    TrajectoryRequest,
)
from trajectory import run_trajectory


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()
    yield


app = FastAPI(title="GlassBox", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    model = get_model()
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "n_layers": model.cfg.n_layers,
        "n_heads": model.cfg.n_heads,
        "d_model": model.cfg.d_model,
    }


@app.post("/api/forward")
def forward(req: ForwardRequest) -> dict:
    return run_forward(req.prompt, req.top_k)


@app.post("/api/patch")
def patch(req: PatchRequest) -> dict:
    return run_patching(req.clean_prompt, req.corrupted_prompt, req.answer, req.corrupted_answer)


@app.get("/api/probe/concepts")
def probe_concepts() -> list:
    return list_concepts()


@app.post("/api/probe")
def probe(req: ProbeRequest) -> dict:
    return run_probes(req.concepts)


@app.post("/api/trajectory")
def trajectory(req: TrajectoryRequest) -> dict:
    return run_trajectory(req.prompt)


@app.get("/api/neuron/scan")
def neuron_scan(layer: int = Query(0, ge=0, le=11)) -> dict:
    return scan_layer(layer)


@app.get("/api/neuron")
def neuron(layer: int = Query(..., ge=0, le=11), index: int = Query(..., ge=0, le=3071)) -> dict:
    return neuron_detail(layer, index)


@app.get("/api/models")
def models() -> list:
    return list_models()


@app.get("/api/sae/info")
def sae_info_route(model_key: str = Query("gpt2")) -> dict:
    if not has_sae(model_key):
        return {"model": model_key, "available": False}
    return {"available": True, **sae_info(model_key)}


@app.post("/api/sae/features")
def sae_features_route(req: SaeFeaturesRequest) -> dict:
    return sae_features(req.prompt, req.layer, req.top_k, model_key=req.model_key)


@app.post("/api/sae/labels")
def sae_labels_route(req: SaeLabelsRequest) -> dict:
    return labels_for(req.model_key, req.layer, req.indices)


@app.post("/api/sae/intervene")
def sae_intervene_route(req: InterveneRequest) -> dict:
    return intervene(
        req.prompt, req.layer, req.feature, req.mode, req.coeff, req.top_k, model_key=req.model_key
    )


@app.post("/api/attribute")
def attribute_route(req: AttributeRequest) -> dict:
    return attribute(
        req.clean_prompt,
        req.corrupted_prompt,
        req.answer,
        req.corrupted_answer,
        req.method,
        model_key=req.model_key,
    )


_static = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static):
    app.mount("/", StaticFiles(directory=_static, html=True), name="static")
