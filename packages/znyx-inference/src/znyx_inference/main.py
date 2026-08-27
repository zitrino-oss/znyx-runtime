"""ZNYX Inference Service — optional sidecar FastAPI app.

Endpoints:
  POST /v1/infer/{task}    → the confidence contract (cached + batched)
  GET  /healthz            → liveness
  GET  /v1/models          → registered models + availability (feeds the model registry)
  GET  /v1/stats           → cache + batcher metrics (observability for cache-hit/batching)
  POST /v1/models/desired  → reconcile loaded models against a desired pin set
  POST /v1/models/install  → operator-triggered download (active slot)
  POST /v1/models/reload   → operator-triggered hot-reload (active slot)

Boots on the dependency-free StubRunner with no ML stack installed.

This service NEVER calls the control plane. The runtime owns that channel and pushes
desired model pins here via POST /v1/models/desired on each of its bundle cycles; see
``reconciler.py`` for why desired state flows in that direction. Consequently the sidecar
needs no API key and no outbound internet access beyond fetching model weights.
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

# Nothing in this package configured a level/handler before, so every INFO-level log
# (pin-sync's fetch/load progress, registry's successful-load line) was silently dropped —
# Python's root logger defaults to WARNING with no handler, and uvicorn's own dictConfig
# only touches its own "uvicorn"/"uvicorn.access"/"uvicorn.error" loggers, not this app's.
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

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
    # Desired state is PUSHED here by the runtime (POST /v1/models/desired), not polled.
    # Nothing to start or stop: the reconciler holds only the last pin set, so it needs no
    # background task and no credential.
    from znyx_inference.reconciler import ModelReconciler
    app.state.reconciler = ModelReconciler(registry)
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


def _resolve_batcher(registry: RunnerRegistry, task: str, req: InferRequest):
    """Route the request to the batcher serving the model it pinned (multi-model).

    Unpinned → the task's active slot. Pinned → the active slot when it matches, else a
    loaded variant of that exact model. A pin no loaded model satisfies is still a 409 —
    the caller must never be silently scored by the wrong model."""
    if not req.model_id:
        batcher = registry.get(task)
        if batcher is None:
            raise HTTPException(status_code=503, detail=f"task '{task}' unavailable")
        return batcher, registry.model_version(task) or "unknown"

    resolved = registry.get_for(task, req.model_id, req.revision)
    if resolved is not None:
        return resolved
    if registry.get(task) is None and not registry.serves(task, req.model_id, req.revision):
        raise HTTPException(status_code=503, detail=f"task '{task}' unavailable")
    want = req.model_id + (f"@{req.revision}" if req.revision else "")
    loaded = registry.model_version(task) or "unknown"
    raise HTTPException(status_code=409, detail=(
        f"model pin mismatch: requested {want}, loaded {loaded}"))


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
    batcher, model_version = _resolve_batcher(registry, task, req)

    texts = req.items()
    t0 = time.perf_counter()

    def _key(text: str) -> str:
        # The full serving identity: exact text + task + model + runner/config scope +
        # request params. Anything that can change the decision is in the key.
        return content_key(model_version, text, task=task,
                           scope=batcher.cache_scope, params=req.params)

    # Per-request params (e.g. allowed_languages for the language runner) can't be
    # coalesced with other callers' texts, so they run on the batcher's bounded direct
    # path - same in-flight cap and latency budget, same 429 on saturation - instead of
    # an unbounded bypass. Single items are cached like the batched path (params are
    # part of the key).
    if req.params:
        if len(texts) == 1:
            key = _key(texts[0])
            cached = cache.get(key)
            if cached is not None:
                resp = _result(cached, model_version).model_dump()
                return InferResponse(**resp, latency_ms=int((time.perf_counter() - t0) * 1000), cached=True)
        try:
            outs = await batcher.run_direct(texts, req.params)
        except Saturated as exc:
            raise HTTPException(status_code=429, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        if len(texts) == 1:
            cache.put(key, outs[0])
            resp = _result(outs[0], model_version).model_dump()
            return InferResponse(**resp, latency_ms=int((time.perf_counter() - t0) * 1000), cached=False)
        return BatchInferResponse(
            results=[_result(o, model_version) for o in outs],
            latency_ms=int((time.perf_counter() - t0) * 1000),
            model_version=model_version,
        )

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
    key = _key(text)
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

from pydantic import BaseModel, Field
from typing import Any, Dict
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
    """Hot-reload a task's runner after a model install.

    Also drops the task's cached decisions: a reload can swap weights without changing
    the pin (a re-install over the same model_id@revision), and a spec change alone is
    already covered by the runner/config scope inside every cache key."""
    registry: RunnerRegistry = request.app.state.registry
    info = await registry.reload_task(req.task, req.spec)
    request.app.state.cache.invalidate_task(req.task)
    return info.model_dump()


# ---------------------------------------------------------------------------
# Desired-state reconcile (pushed by the runtime on every bundle cycle)
# ---------------------------------------------------------------------------

# There are 7 inference tasks in the catalog. A desired set can only ever name a task the
# catalog knows, so anything beyond a small multiple of that is a malformed or hostile payload
# rather than a real configuration. Bounded so a junk body cannot make the reconciler churn
# through thousands of failed shortlist lookups, each with its own log line.
_MAX_DESIRED_PINS = 64


class DesiredModelsRequest(BaseModel):
    # {task: {model_id, revision, threshold, runner, sha256}} — the full desired set for
    # this deployment. It must be complete rather than a delta: a task absent from the map
    # is what tells the reconciler that its pin was removed and the variant can be evicted.
    pins: Dict[str, Any] = Field(default_factory=dict)


@app.post("/v1/models/desired")
async def set_desired_models(req: DesiredModelsRequest, request: Request):
    """Converge the loaded models onto ``pins`` and report what is now loaded.

    Called by the runtime, not by the control plane, which cannot reach this process. The
    runtime re-pushes the same set every cycle, so this is idempotent and cheap when
    nothing has changed, and a pin whose download failed last time is retried here.

    Auth matches the sibling install/reload endpoints: none, by deployment contract. This
    service is a same-pod or same-network component and MUST NOT be published beyond its own
    deployment: the endpoints here can evict a loaded model (degrading a detector to its
    deterministic fallback) and can trigger multi-gigabyte downloads. The shipped compose binds
    it to loopback for exactly that reason. If you run the sidecar as its own Service, restrict
    it with a NetworkPolicy.
    """
    reconciler = getattr(request.app.state, "reconciler", None)
    if reconciler is None:      # pragma: no cover - lifespan always sets this
        raise HTTPException(status_code=503, detail="reconciler unavailable")
    if len(req.pins) > _MAX_DESIRED_PINS:
        raise HTTPException(
            status_code=422,
            detail=f"too many pins: {len(req.pins)} (max {_MAX_DESIRED_PINS})")
    return await reconciler.reconcile(req.pins)
