"""
Runtime API routes - evaluation endpoints only.
No DB dependency, no admin routes.
"""
import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Header, Request

from znyx_core.core.models import (
    EvaluationRequest, EvaluationResponse, HealthResponse, ToolEvaluationRequest,
    RetrievalEvaluationRequest, AgentPlanEvaluationRequest, AgentStepEvaluationRequest,
    MemoryWriteEvaluationRequest,
)
from znyx_core.engine.evaluator import EvaluatorOverloadedError
from znyx_runtime.sdk_registry import get_sdk_registry

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


async def _caller_key(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None),
):
    """Who is calling an evaluate route: an app or agent gateway key (its scope is
    then authoritative, see ``_apply_key_scope``), else the instance key or, with
    auth off outside production, anyone (``None``: the body names the scope, as
    before)."""
    token = x_api_key
    if not token and authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
    gateway = getattr(request.app.state, "gateway", None)
    if token and gateway is not None:
        from znyx_runtime.gateway.config import key_digest
        key = gateway.config.keys_by_digest.get(key_digest(token))
        if key is not None:
            if key.disabled or not gateway.key_usable(key):
                raise HTTPException(status_code=401, detail="Invalid authentication token")
            return key
    await _optional_runtime_auth(x_api_key=x_api_key, authorization=authorization)
    return None


def _apply_key_scope(body, key):
    """Validate the body against the key: an app or agent key evaluates only for its
    own tenant, app and agent. A body naming another scope is refused rather than
    silently rewritten (it would otherwise pick that scope's, possibly weaker,
    policy); a body that leaves them at the defaults gets the key's."""
    if key is None or key.instance_key:
        return body
    updates = {}
    for field, value in (("tenant_id", key.tenant_id), ("app_id", key.app_id),
                         ("agent_id", key.agent_id)):
        current = getattr(body, field, None)
        if current not in (None, "", "default", value):
            raise HTTPException(status_code=403,
                                detail=f"This key is scoped to {field[:-3]} {value!r}; the request "
                                       f"names {current!r}")
        updates[field] = value
    if hasattr(body, "env"):
        # The environment the gateway uses for this key: a scoped key does not get to
        # pick another environment's (possibly weaker) policy.
        try:
            effective = _get_bundle_manager().effective_env
        except Exception:  # noqa: BLE001 - no bundle manager: the key's own setting
            effective = None
        updates["env"] = effective or key.environment or "prod"
    return body.model_copy(update=updates)


from znyx_runtime.bundle_manager import ScopeNotServed  # noqa: E402

# Checks of what an agent does (not what it says): refused for a quarantined caller.
_ACTING_STAGES = frozenset({"tool", "agent_plan", "agent_step", "memory_write"})


def _get_gateway():
    from znyx_runtime.main import app
    return getattr(app.state, "gateway", None)


def _stop_decision(body, key, stage: str) -> Optional[EvaluationResponse]:
    """The kill switch on the evaluate API: a stopped caller (or a quarantined one checking
    an action, or a stopped tool) gets a BLOCK decision rather than an error, so an SDK set
    to fail open still stops."""
    gateway = _get_gateway()
    stops = gateway.stop_set() if gateway is not None else None
    if not stops:
        return None
    from znyx_core.governance import RULE_QUARANTINED, RULE_STOPPED, RULE_TOOL_STOPPED
    try:
        member = _get_bundle_manager().member_for(getattr(key, "member", "") or None,
                                                  getattr(body, "app_id", None), getattr(body, "env", None))
    except ScopeNotServed:
        return None                          # the evaluation itself refuses the call
    call = stops.for_caller(project_id=member.project_id if member is not None else "",
                            environment_id=member.environment_id if member is not None else "",
                            app_id=getattr(body, "app_id", None), agent_id=getattr(body, "agent_id", None),
                            key_id=key.id if key is not None and not key.instance_key else None)
    hit, rule = call.stopped, RULE_STOPPED
    if hit is None and stage in _ACTING_STAGES:
        hit, rule = call.quarantined, RULE_QUARANTINED
    if hit is None and stage == "tool":
        hit, rule = call.tool_stop(getattr(body, "tool_name", None)), RULE_TOOL_STOPPED
    if hit is None:
        return None
    from znyx_core.core.models import Decision, RuleHit, Severity
    logger.info("Evaluate %s refused by stop %s (%s %s)", stage, hit.id or "-", hit.target, hit.match or "")
    return EvaluationResponse(
        request_id=str(getattr(body, "request_id", "") or ""), decision=Decision.BLOCK, risk_score=100,
        policy_version="kill_switch", rule_hits=[RuleHit(rule_id=rule, severity=Severity.HIGH, message=hit.message())],
        user_message=hit.message(), developer_message=f"Refused by stop {hit.id}" if hit.id else "Refused by a stop",
        trace_id=getattr(body, "trace_id", None), session_id=getattr(body, "session_id", None),
        span_id=getattr(body, "span_id", None))


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
get_bundle_manager = _get_bundle_manager
get_heartbeat = _get_heartbeat
runtime_judge_ctx = _runtime_judge_ctx
caller_key = _caller_key
apply_key_scope = _apply_key_scope


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
    tags=["Runtime API"],
    operation_id="runtime.evaluateInput",
    summary="Evaluate user input before calling the LLM",
)
async def evaluate_input(request: EvaluationRequest, key=Depends(_caller_key)) -> EvaluationResponse:
    """Evaluate input text before sending to LLM."""
    request = _apply_key_scope(request, key)
    stopped = _stop_decision(request, key, "input")
    if stopped is not None:
        return stopped
    try:
        bm = _get_bundle_manager()
        request, policy = bm.resolve(request, key)
        ev = _get_evaluator()
        result = await ev.evaluate(request, context="input", policy=policy,
                                   judge_ctx=_runtime_judge_ctx(policy, request))
        hb = _get_heartbeat()
        if hb:
            hb.increment_eval_count()
        return result
    except EvaluatorOverloadedError:
        raise HTTPException(status_code=503, detail="Evaluator saturated, retry shortly",
                            headers={"Retry-After": "1"})
    except ScopeNotServed as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Policy unavailable: {e}")
        raise HTTPException(status_code=503, detail="Policy unavailable (fail-closed)")
    except Exception:
        logger.exception("Evaluation failed")
        raise HTTPException(status_code=500, detail="Evaluation failed")


@router.post(
    "/v1/evaluate/output",
    response_model=EvaluationResponse,
    tags=["Runtime API"],
    operation_id="runtime.evaluateOutput",
    summary="Evaluate an LLM response before returning it to the user",
)
async def evaluate_output(request: EvaluationRequest, key=Depends(_caller_key)) -> EvaluationResponse:
    """Evaluate output text from LLM before returning to user."""
    request = _apply_key_scope(request, key)
    stopped = _stop_decision(request, key, "output")
    if stopped is not None:
        return stopped
    try:
        bm = _get_bundle_manager()
        request, policy = bm.resolve(request, key)
        ev = _get_evaluator()
        result = await ev.evaluate(request, context="output", policy=policy,
                                   judge_ctx=_runtime_judge_ctx(policy, request))
        hb = _get_heartbeat()
        if hb:
            hb.increment_eval_count()
        return result
    except EvaluatorOverloadedError:
        raise HTTPException(status_code=503, detail="Evaluator saturated, retry shortly",
                            headers={"Retry-After": "1"})
    except ScopeNotServed as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Policy unavailable: {e}")
        raise HTTPException(status_code=503, detail="Policy unavailable (fail-closed)")
    except Exception:
        logger.exception("Evaluation failed")
        raise HTTPException(status_code=500, detail="Evaluation failed")


@router.post(
    "/v1/evaluate/tool",
    response_model=EvaluationResponse,
    tags=["Runtime API"],
    operation_id="runtime.evaluateTool",
    summary="Evaluate a tool / function invocation",
)
async def evaluate_tool(request: ToolEvaluationRequest, key=Depends(_caller_key)) -> EvaluationResponse:
    """Evaluate tool invocation against governance policies."""
    request = _apply_key_scope(request, key)
    stopped = _stop_decision(request, key, "tool")
    if stopped is not None:
        return stopped
    try:
        bm = _get_bundle_manager()
        request, policy = bm.resolve(request, key)
        ev = _get_evaluator()
        result = await ev.evaluate_tool(request, policy=policy,
                                        judge_ctx=_runtime_judge_ctx(policy, request))
        hb = _get_heartbeat()
        if hb:
            hb.increment_eval_count()
        return result
    except EvaluatorOverloadedError:
        raise HTTPException(status_code=503, detail="Evaluator saturated, retry shortly",
                            headers={"Retry-After": "1"})
    except ScopeNotServed as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Policy unavailable: {e}")
        raise HTTPException(status_code=503, detail="Policy unavailable (fail-closed)")
    except Exception:
        logger.exception("Tool evaluation failed")
        raise HTTPException(status_code=500, detail="Tool evaluation failed")


# ── New-stage evaluate endpoints (generalized stage dispatch) ──────

async def _evaluate_stage_runtime(scoped, stage: str, key=None) -> EvaluationResponse:
    """Shared runtime handler for the per-stage endpoints: resolve the bundle
    policy for the request scope and dispatch through the generalized stage pipeline."""
    bm = _get_bundle_manager()
    scoped, policy = bm.resolve(scoped, key)
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
    tags=["Runtime API"],
    operation_id="runtime.evaluateRetrieval",
    summary="Evaluate retrieved RAG chunks for indirect prompt injection (LLM01)",
)
async def evaluate_retrieval(request: RetrievalEvaluationRequest, key=Depends(_caller_key)) -> EvaluationResponse:
    """Evaluate retrieved RAG chunks for indirect prompt injection before they enter context (LLM01)."""
    request = _apply_key_scope(request, key)
    stopped = _stop_decision(request, key, "retrieval")
    if stopped is not None:
        return stopped
    try:
        return await _evaluate_stage_runtime(request, "retrieval", key)
    except EvaluatorOverloadedError:
        raise HTTPException(status_code=503, detail="Evaluator saturated, retry shortly",
                            headers={"Retry-After": "1"})
    except ScopeNotServed as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Policy unavailable (fail-closed)")
    except Exception:
        logger.exception("Retrieval evaluation failed")
        raise HTTPException(status_code=500, detail="Evaluation failed")


@router.post(
    "/v1/evaluate/agent-plan",
    response_model=EvaluationResponse,
    tags=["Runtime API"],
    operation_id="runtime.evaluateAgentPlan",
    summary="Evaluate a proposed agent plan for excessive agency (LLM03)",
)
async def evaluate_agent_plan(request: AgentPlanEvaluationRequest, key=Depends(_caller_key)) -> EvaluationResponse:
    """Evaluate a proposed multi-step agent plan for excessive agency (LLM03)."""
    request = _apply_key_scope(request, key)
    stopped = _stop_decision(request, key, "agent_plan")
    if stopped is not None:
        return stopped
    try:
        return await _evaluate_stage_runtime(request, "agent_plan", key)
    except EvaluatorOverloadedError:
        raise HTTPException(status_code=503, detail="Evaluator saturated, retry shortly",
                            headers={"Retry-After": "1"})
    except ScopeNotServed as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Policy unavailable (fail-closed)")
    except Exception:
        logger.exception("Agent-plan evaluation failed")
        raise HTTPException(status_code=500, detail="Evaluation failed")


@router.post(
    "/v1/evaluate/agent-step",
    response_model=EvaluationResponse,
    tags=["Runtime API"],
    operation_id="runtime.evaluateAgentStep",
    summary="Evaluate a single agent-loop iteration for budget/depth caps (LLM06)",
)
async def evaluate_agent_step(request: AgentStepEvaluationRequest, key=Depends(_caller_key)) -> EvaluationResponse:
    """Evaluate a single agent-loop iteration for budget/depth caps (LLM06)."""
    request = _apply_key_scope(request, key)
    stopped = _stop_decision(request, key, "agent_step")
    if stopped is not None:
        return stopped
    try:
        return await _evaluate_stage_runtime(request, "agent_loop", key)
    except EvaluatorOverloadedError:
        raise HTTPException(status_code=503, detail="Evaluator saturated, retry shortly",
                            headers={"Retry-After": "1"})
    except ScopeNotServed as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Policy unavailable (fail-closed)")
    except Exception:
        logger.exception("Agent-step evaluation failed")
        raise HTTPException(status_code=500, detail="Evaluation failed")


@router.post(
    "/v1/evaluate/memory-write",
    response_model=EvaluationResponse,
    tags=["Runtime API"],
    operation_id="runtime.evaluateMemoryWrite",
    summary="Evaluate text written to agent memory for persistent injection (LLM01)",
)
async def evaluate_memory_write(request: MemoryWriteEvaluationRequest, key=Depends(_caller_key)) -> EvaluationResponse:
    """Evaluate text being written to agent memory for persistent injection (LLM01)."""
    request = _apply_key_scope(request, key)
    stopped = _stop_decision(request, key, "memory_write")
    if stopped is not None:
        return stopped
    try:
        return await _evaluate_stage_runtime(request, "memory_write", key)
    except EvaluatorOverloadedError:
        raise HTTPException(status_code=503, detail="Evaluator saturated, retry shortly",
                            headers={"Retry-After": "1"})
    except ScopeNotServed as e:
        raise HTTPException(status_code=400, detail=str(e))
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


@router.get("/v1/sdk-clients", include_in_schema=False)
async def sdk_clients():
    """Which SDKs have called this runtime since it started.

    The same list the runtime sends to the control plane on its heartbeat, exposed
    directly so an operator can confirm SDK identification is working without
    standing up a control plane and waiting out a heartbeat cycle.

    In-memory and per-process: it resets on restart, and a multi-replica deployment
    answers only for the replica that served this request. Carries no secrets - SDK
    name and version are the same metadata the caller volunteered in its headers -
    and it is kept out of the OpenAPI schema as a diagnostic, not a public contract.
    """
    return {"sdk_clients": get_sdk_registry().snapshot()}


@router.get("/v1/bundle/status")
async def bundle_status(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None),
):
    """Return metadata about the currently loaded policy bundles. Readiness is public
    (health checks); which projects and environments the runtime serves, and their
    bundle ids, only for the instance's own key (or with auth off)."""
    bm = _get_bundle_manager()
    try:
        await _optional_runtime_auth(x_api_key=x_api_key, authorization=authorization)
    except HTTPException:
        return {"ready": bm.is_ready, "environments": len(bm.members) if bm.members else None}
    return {
        "ready": bm.is_ready,
        **bm.bundle_info,
    }
