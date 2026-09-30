"""HTTP API. Run with: uvicorn dgalens.api:app"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from dgalens import __version__
from dgalens.domains import normalize
from dgalens.model import MarkovModel

MAX_BATCH = 1000


class ScoreRequest(BaseModel):
    domain: str = Field(min_length=1, max_length=253, examples=["xjkqpzwvbt.com"])


class BatchRequest(BaseModel):
    domains: list[str] = Field(min_length=1, max_length=MAX_BATCH)


class ScoreResult(BaseModel):
    domain: str
    label: str
    score: float = Field(description="Mean log probability per character; lower looks more generated.")
    suspicious: bool


class BatchResult(BaseModel):
    results: list[ScoreResult]


def create_app(model: MarkovModel | None = None) -> FastAPI:
    """Build the app. Pass ``model`` directly in tests; otherwise it is loaded
    from ``$DGALENS_MODEL`` at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model = model or MarkovModel.load(os.environ.get("DGALENS_MODEL", "models/dgalens.json"))
        if app.state.model.threshold is None:
            raise RuntimeError("model has no threshold; train it with `dgalens train`")
        yield

    api = FastAPI(title="dgalens", version=__version__, lifespan=lifespan)

    def score_one(request: Request, raw: str) -> ScoreResult:
        label = normalize(raw)
        if not label:
            raise HTTPException(status_code=422, detail=f"no scoreable label in {raw!r}")
        m: MarkovModel = request.app.state.model
        return ScoreResult(domain=raw, label=label, score=m.score(label), suspicious=m.is_suspicious(label))

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.post("/score")
    def score(body: ScoreRequest, request: Request) -> ScoreResult:
        return score_one(request, body.domain)

    @api.post("/score/batch")
    def score_batch(body: BatchRequest, request: Request) -> BatchResult:
        return BatchResult(results=[score_one(request, d) for d in body.domains])

    return api


app = create_app()
