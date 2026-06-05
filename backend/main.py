from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from interp import run_forward, run_patching
from model import MODEL_NAME, get_model
from schemas import ForwardRequest, PatchRequest


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
