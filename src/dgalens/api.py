"""HTTP API. Run with: uvicorn dgalens.api:app"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from dgalens import __version__
from dgalens.domains import normalize
from dgalens.model import MarkovModel

MAX_BATCH = 1000
MAX_DOMAIN_LENGTH = 253  # the DNS limit for a full name


class ScoreRequest(BaseModel):
    domain: str = Field(min_length=1, max_length=MAX_DOMAIN_LENGTH, examples=["xjkqpzwvbt.com"])


class BatchRequest(BaseModel):
    domains: list[Annotated[str, Field(max_length=MAX_DOMAIN_LENGTH)]] = Field(
        min_length=1, max_length=MAX_BATCH
    )


class ScoreResult(BaseModel):
    domain: str
    label: str
    score: float = Field(description="Mean log probability per character; lower looks more generated.")
    suspicious: bool


class ScoreError(BaseModel):
    domain: str
    error: str


class BatchResult(BaseModel):
    results: list[ScoreResult | ScoreError] = Field(
        description="One entry per input domain, in order. Inputs with no scoreable label "
        "(a bare suffix, an IP address) get an error entry instead of failing the batch."
    )


def _score(model: MarkovModel, raw: str) -> ScoreResult | ScoreError:
    label = normalize(raw)
    if not label:
        return ScoreError(domain=raw, error=f"no scoreable label in {raw!r}")
    return ScoreResult(
        domain=raw, label=label, score=model.score(label), suspicious=model.is_suspicious(label)
    )


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

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.post("/score")
    def score(body: ScoreRequest, request: Request) -> ScoreResult:
        result = _score(request.app.state.model, body.domain)
        if isinstance(result, ScoreError):
            raise HTTPException(status_code=422, detail=result.error)
        return result

    @api.post("/score/batch")
    def score_batch(body: BatchRequest, request: Request) -> BatchResult:
        model: MarkovModel = request.app.state.model
        return BatchResult(results=[_score(model, domain) for domain in body.domains])

    return api


app = create_app()
