from __future__ import annotations

from pydantic import BaseModel, Field

MAX_TOKENS = 128
MAX_PATCH_TOKENS = 32


class ForwardRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=50)


class PatchRequest(BaseModel):
    clean_prompt: str = Field(..., min_length=1)
    corrupted_prompt: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    corrupted_answer: str = Field(..., min_length=1)


class ProbeRequest(BaseModel):
    concepts: list[str] = Field(default_factory=list)


class TrajectoryRequest(BaseModel):
    prompt: str = Field(..., min_length=1)


class SaeFeaturesRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    layer: int = Field(0, ge=0)
    top_k: int = Field(12, ge=1, le=50)
    model_key: str = Field("gpt2")


class SaeLabelsRequest(BaseModel):
    layer: int = Field(0, ge=0)
    indices: list[int] = Field(default_factory=list)
    model_key: str = Field("gpt2")


class InterveneRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    layer: int = Field(0, ge=0)
    feature: int = Field(..., ge=0)
    mode: str = Field("ablate")
    coeff: float = Field(8.0)
    top_k: int = Field(10, ge=1, le=30)
    model_key: str = Field("gpt2")
