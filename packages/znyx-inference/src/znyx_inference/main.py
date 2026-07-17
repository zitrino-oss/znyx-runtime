"""ZNYX Inference Service — optional sidecar FastAPI app.

Endpoints:
  POST /v1/infer/{task}  → the confidence contract (cached + batched)
  GET  /healthz          → liveness
  GET  /v1/models        → registered models + availability (feeds the model registry)
  GET  /v1/stats         → cache + batcher metrics (observability for cache-hit/batching)

Boots on the dependency-free StubRunner with no ML stack installed.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from znyx_inference.batching import Saturated
from znyx_inference.cache import ContentHashCache, content_key
from znyx_inference.config import InferenceConfig
from znyx_inference.contract import (
    BatchInferResponse,
    InferRequest,
    InferResponse,
    InferResult,
)
from znyx_inference.install import InstallManager
from znyx_inference.registry import RunnerRegistry

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = InferenceConfig.from_env()
    registry = RunnerRegistry(config)
    await registry.start_all()
    app.state.config = config
    app.state.registry = registry
    app.state.cache = ContentHashCache(maxsize=config.cache_maxsize)
    app.state.install_manager = InstallManager()
    available = [m.task for m in registry.list_models() if m.available]
    logger.info("ZNYX Inference ready — available tasks: %s", available)
    try:
        yield
    finally:
        await registry.stop_all()


app = FastAPI(title="ZNYX Inference Service", version="1.0.0", lifespan=lifespan)


def _result(out, model_version: str) -> InferResult:
    return InferResult(
        decision=out.decision, risk_score=out.risk_score, confidence=out.confidence,
        label_scores=out.label_scores, calibrated_score=out.calibrated_score,
        threshold=out.threshold, model_version=model_version,
    )


def _assert_pin(req: InferRequest, model_version: str) -> None:
    """If the caller pinned model_id/revision, the loaded model MUST match (model
    pinning) — otherwise the caller could be silently scored by the wrong model. The
    loaded identity is ``model_id@revision``; compare each pinned component against it."""
    if not req.model_id:
        return
    loaded_id, _, loaded_rev = model_version.partition("@")
    if req.model_id != loaded_id or (req.revision is not None and req.revision != loaded_rev):
        want = req.model_id + (f"@{req.revision}" if req.revision else "")
        raise HTTPException(status_code=409, detail=(
            f"model pin mismatch: requested {want}, loaded {model_version}"))


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models(request: Request):
    return {"models": [m.model_dump() for m in request.app.state.registry.list_models()]}


@app.get("/v1/stats")
async def stats(request: Request):
    return {
        "cache": request.app.state.cache.stats(),
        "batchers": request.app.state.registry.batcher_stats(),
    }


@app.post("/v1/infer/{task}")
async def infer(task: str, req: InferRequest, request: Request):
    registry: RunnerRegistry = request.app.state.registry
    cache: ContentHashCache = request.app.state.cache
    batcher = registry.get(task)
    if batcher is None:
        raise HTTPException(status_code=503, detail=f"task '{task}' unavailable")
    model_version = registry.model_version(task) or "unknown"
    _assert_pin(req, model_version)

    texts = req.items()
    t0 = time.perf_counter()

    # Explicit multi-item batch → no per-item cache, return BatchInferResponse.
    if req.texts is not None:
        try:
            outs = [await batcher.submit(t) for t in texts]
        except Saturated as exc:
            raise HTTPException(status_code=429, detail=str(exc))
        return BatchInferResponse(
            results=[_result(o, model_version) for o in outs],
            latency_ms=int((time.perf_counter() - t0) * 1000),
            model_version=model_version,
        )

    # Single item → content-hash cache then batch.
    text = texts[0]
    key = content_key(model_version, text)
    cached = cache.get(key)
    if cached is not None:
        resp = _result(cached, model_version).model_dump()
        return InferResponse(**resp, latency_ms=int((time.perf_counter() - t0) * 1000), cached=True)
    try:
        out = await batcher.submit(text)
    except Saturated as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    cache.put(key, out)
    resp = _result(out, model_version).model_dump()
    return InferResponse(**resp, latency_ms=int((time.perf_counter() - t0) * 1000), cached=False)


# ---------------------------------------------------------------------------
# Model install + reload (operator-triggered via the console UI)
# ---------------------------------------------------------------------------

from pydantic import BaseModel
from fastapi.responses import JSONResponse


class InstallRequest(BaseModel):
    task: str
    model_id: str | None = None
    revision: str | None = None


class ReloadRequest(BaseModel):
    task: str
    spec: dict


@app.post("/v1/models/install")
async def install_model(req: InstallRequest, request: Request):
    """Start a background model install job. Returns 202 with the job_id."""
    manager: InstallManager = request.app.state.install_manager
    try:
        job = manager.start_install(req.task, model_id=req.model_id, revision=req.revision)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(status_code=202, content=job.to_dict())


@app.get("/v1/models/install/{job_id}")
async def get_install_status(job_id: str, request: Request):
    """Poll the status of an install job."""
    manager: InstallManager = request.app.state.install_manager
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job '{job_id}'")
    return job.to_dict()


@app.post("/v1/models/reload")
async def reload_model(req: ReloadRequest, request: Request):
    """Hot-reload a task's runner after a model install."""
    registry: RunnerRegistry = request.app.state.registry
    info = await registry.reload_task(req.task, req.spec)
    return info.model_dump()
