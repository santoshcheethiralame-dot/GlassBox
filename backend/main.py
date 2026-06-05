from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from interp import run_forward, run_patching
from model import MODEL_NAME, get_model
from neurons import neuron_detail, scan_layer
from probing import list_concepts, run_probes
from schemas import ForwardRequest, PatchRequest, ProbeRequest, TrajectoryRequest
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


@app.get("/health")
def health() -> dict:
    model = get_model()
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "n_layers": model.cfg.n_layers,
        "n_heads": model.cfg.n_heads,
        "d_model": model.cfg.d_model,
    }


@app.post("/forward")
def forward(req: ForwardRequest) -> dict:
    return run_forward(req.prompt, req.top_k)


@app.post("/patch")
def patch(req: PatchRequest) -> dict:
    return run_patching(req.clean_prompt, req.corrupted_prompt, req.answer, req.corrupted_answer)


@app.get("/probe/concepts")
def probe_concepts() -> list:
    return list_concepts()


@app.post("/probe")
def probe(req: ProbeRequest) -> dict:
    return run_probes(req.concepts)


@app.post("/trajectory")
def trajectory(req: TrajectoryRequest) -> dict:
    return run_trajectory(req.prompt)


@app.get("/neuron/scan")
def neuron_scan(layer: int = Query(0, ge=0, le=11)) -> dict:
    return scan_layer(layer)


@app.get("/neuron")
def neuron(layer: int = Query(..., ge=0, le=11), index: int = Query(..., ge=0, le=3071)) -> dict:
    return neuron_detail(layer, index)
