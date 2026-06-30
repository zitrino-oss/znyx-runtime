"""
Runtime API routes - evaluation endpoints only.
No DB dependency, no admin routes.
"""
import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Header

from app.shared.core.models import (
    EvaluationRequest, EvaluationResponse, HealthResponse, ToolEvaluationRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _is_prod() -> bool:
    env = (os.getenv("ZNYX_ENV") or os.getenv("ENVIRONMENT") or "").strip().lower()
    return env in ("prod", "production")


def _runtime_auth_required() -> bool:
    """Default: on. Only explicitly disabled when RUNTIME_REQUIRE_AUTH=false,
    and only honoured outside production — production always requires auth."""
    raw = os.getenv("RUNTIME_REQUIRE_AUTH", "true").strip().lower()
    if _is_prod():
        return True
    return raw not in ("0", "false", "no", "off")


async def _optional_runtime_auth(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None),
) -> None:
    """Enforce runtime auth. Default-on; only disablable in non-production."""
    if not _runtime_auth_required():
        return

    expected = os.getenv("RUNTIME_API_KEY", "").strip()
    if not expected:
        raise HTTPException(
            status_code=500,
            detail="Server misconfiguration: RUNTIME_REQUIRE_AUTH is enabled but RUNTIME_API_KEY is not set.",
        )

    # Accept either X-API-Key or Authorization: Bearer
    token = x_api_key
    if not token and authorization and authorization.startswith("Bearer "):
        token = authorization[7:]

    if not token:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Set X-API-Key or Authorization: Bearer header.",
        )

    # Constant-time compare to prevent timing attacks.
    import hmac as _hmac

    if not _hmac.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="Invalid authentication token")


def _get_evaluator():
    from app.runtime.main import evaluator
    return evaluator


def _get_bundle_manager():
    from app.runtime.main import bundle_manager
    return bundle_manager


# Public aliases consumed by stream_routes and any other runtime-scoped code.
# These intentionally shadow the private underscore names so callers can do:
#   from app.runtime.api.routes import verify_runtime_auth, get_evaluator
verify_runtime_auth = _optional_runtime_auth
get_evaluator = _get_evaluator


@router.post("/v1/evaluate/input", response_model=EvaluationResponse, dependencies=[Depends(_optional_runtime_auth)])
async def evaluate_input(request: EvaluationRequest) -> EvaluationResponse:
    """Evaluate input text before sending to LLM."""
    try:
        bm = _get_bundle_manager()
        if bm.effective_env:
            request = request.model_copy(update={"env": bm.effective_env})
        policy = bm.get_policy(
            tenant_id=request.tenant_id,
            app_id=request.app_id,
            agent_id=request.agent_id,
            env=request.env,
        )
        ev = _get_evaluator()
        return await ev.evaluate(request, context="input", policy=policy)
    except RuntimeError as e:
        logger.error(f"Policy unavailable: {e}")
        raise HTTPException(status_code=503, detail="Policy unavailable (fail-closed)")
    except Exception as e:
        logger.exception("Evaluation failed")
        raise HTTPException(status_code=500, detail="Evaluation failed")


@router.post("/v1/evaluate/output", response_model=EvaluationResponse, dependencies=[Depends(_optional_runtime_auth)])
async def evaluate_output(request: EvaluationRequest) -> EvaluationResponse:
    """Evaluate output text from LLM before returning to user."""
    try:
        bm = _get_bundle_manager()
        if bm.effective_env:
            request = request.model_copy(update={"env": bm.effective_env})
        policy = bm.get_policy(
            tenant_id=request.tenant_id,
            app_id=request.app_id,
            agent_id=request.agent_id,
            env=request.env,
        )
        ev = _get_evaluator()
        return await ev.evaluate(request, context="output", policy=policy)
    except RuntimeError as e:
        logger.error(f"Policy unavailable: {e}")
        raise HTTPException(status_code=503, detail="Policy unavailable (fail-closed)")
    except Exception as e:
        logger.exception("Evaluation failed")
        raise HTTPException(status_code=500, detail="Evaluation failed")


@router.post("/v1/evaluate/tool", response_model=EvaluationResponse, dependencies=[Depends(_optional_runtime_auth)])
async def evaluate_tool(request: ToolEvaluationRequest) -> EvaluationResponse:
    """Evaluate tool invocation against governance policies."""
    try:
        bm = _get_bundle_manager()
        if bm.effective_env:
            request = request.model_copy(update={"env": bm.effective_env})
        policy = bm.get_policy(
            tenant_id=request.tenant_id,
            app_id=request.app_id,
            agent_id=request.agent_id,
            env=request.env,
        )
        ev = _get_evaluator()
        return await ev.evaluate_tool(request, policy=policy)
    except RuntimeError as e:
        logger.error(f"Policy unavailable: {e}")
        raise HTTPException(status_code=503, detail="Policy unavailable (fail-closed)")
    except Exception as e:
        logger.exception("Tool evaluation failed")
        raise HTTPException(status_code=500, detail="Tool evaluation failed")


@router.get("/healthz", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check - always returns 200."""
    return HealthResponse(status="ok")


@router.get("/readyz", response_model=HealthResponse)
async def readiness_check() -> HealthResponse:
    """Readiness check - returns 200 only if a policy is loaded."""
    bm = _get_bundle_manager()
    if bm.is_ready:
        return HealthResponse(status="ready")
    raise HTTPException(status_code=503, detail="Not ready - no policy loaded")


@router.get("/v1/bundle/status")
async def bundle_status():
    """Return metadata about the currently loaded policy bundle."""
    bm = _get_bundle_manager()
    return {
        "ready": bm.is_ready,
        **bm.bundle_info,
    }
