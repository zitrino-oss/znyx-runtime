"""
Runtime API routes - evaluation endpoints only.
No DB dependency, no admin routes.
"""
import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Header

from znyx_core.core.models import (
    EvaluationRequest, EvaluationResponse, HealthResponse, ToolEvaluationRequest,
    RetrievalEvaluationRequest, AgentPlanEvaluationRequest, AgentStepEvaluationRequest,
    MemoryWriteEvaluationRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _is_prod() -> bool:
    from znyx_core.utils.env import is_production
    return is_production()


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
    from znyx_runtime.main import evaluator
    return evaluator


def _get_bundle_manager():
    from znyx_runtime.main import bundle_manager
    return bundle_manager


def _get_heartbeat():
    from znyx_runtime.main import heartbeat
    return heartbeat


def _get_runtime_judge():
    from znyx_runtime.main import runtime_judge
    return runtime_judge


def _runtime_judge_ctx(policy, scoped):
    """Build the per-request judge context for a runtime evaluation, or None
    when judges aren't used / judge audit is off. Stamps the request's tenant as org_scope
    so the CP drain can attribute the spooled rows."""
    rj = _get_runtime_judge()
    if rj is None:
        return None
    return rj.context_for(policy, env=getattr(scoped, "env", None) or "prod",
                          org_scope=getattr(scoped, "tenant_id", None))


# Public aliases consumed by stream_routes and any other runtime-scoped code.
# These intentionally shadow the private underscore names so callers can do:
#   from znyx_runtime.api.routes import verify_runtime_auth, get_evaluator
verify_runtime_auth = _optional_runtime_auth
get_evaluator = _get_evaluator
get_heartbeat = _get_heartbeat


# OpenAPI metadata (tags / operation_id / summary) on the evaluate routes.
#
# These are the runtime's public contract, so a self-hosted deployment's own
# /openapi.json should describe them properly rather than emitting anonymous
# "evaluate_input__post" operation ids under no tag. Generated clients and doc
# tooling both key off these.
#
# The values intentionally mirror the ones the ZNYX control plane uses for the
# same paths, so tooling that has seen either spec resolves the same operation.


@router.post(
    "/v1/evaluate/input",
    response_model=EvaluationResponse,
    dependencies=[Depends(_optional_runtime_auth)],
    tags=["Runtime API"],
    operation_id="runtime.evaluateInput",
    summary="Evaluate user input before calling the LLM",
)
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
        result = await ev.evaluate(request, context="input", policy=policy,
                                   judge_ctx=_runtime_judge_ctx(policy, request))
        hb = _get_heartbeat()
        if hb:
            hb.increment_eval_count()
        return result
    except RuntimeError as e:
        logger.error(f"Policy unavailable: {e}")
        raise HTTPException(status_code=503, detail="Policy unavailable (fail-closed)")
    except Exception:
        logger.exception("Evaluation failed")
        raise HTTPException(status_code=500, detail="Evaluation failed")


@router.post(
    "/v1/evaluate/output",
    response_model=EvaluationResponse,
    dependencies=[Depends(_optional_runtime_auth)],
    tags=["Runtime API"],
    operation_id="runtime.evaluateOutput",
    summary="Evaluate an LLM response before returning it to the user",
)
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
        result = await ev.evaluate(request, context="output", policy=policy,
                                   judge_ctx=_runtime_judge_ctx(policy, request))
        hb = _get_heartbeat()
        if hb:
            hb.increment_eval_count()
        return result
    except RuntimeError as e:
        logger.error(f"Policy unavailable: {e}")
        raise HTTPException(status_code=503, detail="Policy unavailable (fail-closed)")
    except Exception:
        logger.exception("Evaluation failed")
        raise HTTPException(status_code=500, detail="Evaluation failed")


@router.post(
    "/v1/evaluate/tool",
    response_model=EvaluationResponse,
    dependencies=[Depends(_optional_runtime_auth)],
    tags=["Runtime API"],
    operation_id="runtime.evaluateTool",
    summary="Evaluate a tool / function invocation",
)
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
        result = await ev.evaluate_tool(request, policy=policy,
                                        judge_ctx=_runtime_judge_ctx(policy, request))
        hb = _get_heartbeat()
        if hb:
            hb.increment_eval_count()
        return result
    except RuntimeError as e:
        logger.error(f"Policy unavailable: {e}")
        raise HTTPException(status_code=503, detail="Policy unavailable (fail-closed)")
    except Exception:
        logger.exception("Tool evaluation failed")
        raise HTTPException(status_code=500, detail="Tool evaluation failed")


# ── New-stage evaluate endpoints (generalized stage dispatch) ──────

async def _evaluate_stage_runtime(scoped, stage: str) -> EvaluationResponse:
    """Shared runtime handler for the per-stage endpoints: resolve the bundle
    policy for the request scope and dispatch through the generalized stage pipeline."""
    bm = _get_bundle_manager()
    if bm.effective_env:
        scoped = scoped.model_copy(update={"env": bm.effective_env})
    policy = bm.get_policy(
        tenant_id=scoped.tenant_id,
        app_id=scoped.app_id,
        agent_id=scoped.agent_id,
        env=scoped.env,
    )
    ev = _get_evaluator()
    result = await ev.evaluate_stage(scoped, stage, policy=policy,
                                     judge_ctx=_runtime_judge_ctx(policy, scoped))
    hb = _get_heartbeat()
    if hb:
        hb.increment_eval_count()
    return result


@router.post(
    "/v1/evaluate/retrieval",
    response_model=EvaluationResponse,
    dependencies=[Depends(_optional_runtime_auth)],
    tags=["Runtime API"],
    operation_id="runtime.evaluateRetrieval",
    summary="Evaluate retrieved RAG chunks for indirect prompt injection (LLM01)",
)
async def evaluate_retrieval(request: RetrievalEvaluationRequest) -> EvaluationResponse:
    """Evaluate retrieved RAG chunks for indirect prompt injection before they enter context (LLM01)."""
    try:
        return await _evaluate_stage_runtime(request, "retrieval")
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Policy unavailable (fail-closed)")
    except Exception:
        logger.exception("Retrieval evaluation failed")
        raise HTTPException(status_code=500, detail="Evaluation failed")


@router.post(
    "/v1/evaluate/agent-plan",
    response_model=EvaluationResponse,
    dependencies=[Depends(_optional_runtime_auth)],
    tags=["Runtime API"],
    operation_id="runtime.evaluateAgentPlan",
    summary="Evaluate a proposed agent plan for excessive agency (LLM03)",
)
async def evaluate_agent_plan(request: AgentPlanEvaluationRequest) -> EvaluationResponse:
    """Evaluate a proposed multi-step agent plan for excessive agency (LLM03)."""
    try:
        return await _evaluate_stage_runtime(request, "agent_plan")
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Policy unavailable (fail-closed)")
    except Exception:
        logger.exception("Agent-plan evaluation failed")
        raise HTTPException(status_code=500, detail="Evaluation failed")


@router.post(
    "/v1/evaluate/agent-step",
    response_model=EvaluationResponse,
    dependencies=[Depends(_optional_runtime_auth)],
    tags=["Runtime API"],
    operation_id="runtime.evaluateAgentStep",
    summary="Evaluate a single agent-loop iteration for budget/depth caps (LLM06)",
)
async def evaluate_agent_step(request: AgentStepEvaluationRequest) -> EvaluationResponse:
    """Evaluate a single agent-loop iteration for budget/depth caps (LLM06)."""
    try:
        return await _evaluate_stage_runtime(request, "agent_loop")
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Policy unavailable (fail-closed)")
    except Exception:
        logger.exception("Agent-step evaluation failed")
        raise HTTPException(status_code=500, detail="Evaluation failed")


@router.post(
    "/v1/evaluate/memory-write",
    response_model=EvaluationResponse,
    dependencies=[Depends(_optional_runtime_auth)],
    tags=["Runtime API"],
    operation_id="runtime.evaluateMemoryWrite",
    summary="Evaluate text written to agent memory for persistent injection (LLM01)",
)
async def evaluate_memory_write(request: MemoryWriteEvaluationRequest) -> EvaluationResponse:
    """Evaluate text being written to agent memory for persistent injection (LLM01)."""
    try:
        return await _evaluate_stage_runtime(request, "memory_write")
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Policy unavailable (fail-closed)")
    except Exception:
        logger.exception("Memory-write evaluation failed")
        raise HTTPException(status_code=500, detail="Evaluation failed")


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
