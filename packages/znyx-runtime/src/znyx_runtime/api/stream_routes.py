"""Streaming guardrails SSE endpoint - evaluate LLM output in real-time."""
import json
import logging
from typing import List, Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from znyx_runtime.api.routes import (
    apply_key_scope,
    caller_key,
    get_bundle_manager,
    get_evaluator,
    get_heartbeat,
    runtime_judge_ctx,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Runtime API"])


class StreamEvaluateRequest(BaseModel):
    """Request body for streaming evaluation.

    Deliberately has no inline ``policy`` field. The policy is resolved from the
    active bundle exactly as on ``/v1/evaluate/input`` and ``/v1/evaluate/output``,
    so a caller cannot hand this endpoint a weaker rule set than the one the
    control plane published for its scope.
    """
    request_id: str = "stream-0"
    tenant_id: str = "default"
    app_id: str = "default"
    agent_id: str = "default"
    env: str = "prod"
    context: Literal["input", "output"] = Field(
        default="output",
        description="Which detector set to run: input or output",
    )
    chunks: List[str] = Field(..., description="Text chunks to evaluate in order")
    window_size: int = Field(default=400, ge=50, le=4000)
    overlap: int = Field(
        default=160, ge=0, le=2000,
        description="Characters held back after each release point. Values below the "
                    "safe minimum (160) are raised to it; the done event reports the "
                    "effective window size and overlap.",
    )
    mode: Literal["incremental", "buffered"] = Field(
        default="incremental",
        description="incremental releases text as windows clear it, then runs a "
                    "full-response check; buffered evaluates the whole response before "
                    "releasing any of it (identical to /v1/evaluate/output). A policy "
                    "with streaming.mode = buffered forces buffered.",
    )


@router.post(
    "/v1/evaluate/stream",
    operation_id="runtime.evaluateStream",
    summary="Evaluate streaming LLM output as it arrives",
)
async def evaluate_stream(
    body: StreamEvaluateRequest,
    key=Depends(caller_key),
):
    """Evaluate text chunks via Server-Sent Events.

    Accepts a list of text chunks and streams back SSE events as each window
    is evaluated. Events:
      - ``guardrail``: window evaluation result (non-blocking)
      - ``chunk``: text a window has evaluated and allowed
      - ``block``: window triggered a BLOCK decision
      - ``done``: final summary with aggregate metrics

    A ``chunk`` is only emitted once the window covering it has been evaluated,
    and it carries the window's redactions, so nothing this endpoint forwards
    has skipped the detectors.

    Windows run on the shared detector pool with the same admission control,
    deadline, egress audit sink, custom plugins and judge context as
    ``/v1/evaluate/output``. A saturated pool or a missed deadline stops the
    stream fail-closed with a ``block`` event. The whole stream is recorded as
    one evaluation in telemetry and metrics.
    """
    import time

    from znyx_core.engine.evaluator import DetectorDeadlineExceeded, EvaluatorOverloadedError
    from znyx_core.engine import streaming as streaming_mod
    from znyx_core.engine.streaming import StreamingEvaluator
    from znyx_core.core.models import EvaluationRequest

    body = apply_key_scope(body, key)

    # Bound the work one request can ask for. An unbroken token is evaluated whole
    # however long it is, so the request itself is what must be capped.
    total_chars = sum(len(c) for c in body.chunks)
    if total_chars > streaming_mod.MAX_RUN:
        raise HTTPException(
            status_code=413,
            detail=f"Stream too large: {total_chars} characters (limit {streaming_mod.MAX_RUN})",
        )

    # Resolve the policy from the active bundle, the same way the non-streaming
    # evaluate routes do. Done before the response starts so a fail-closed
    # runtime with no bundle can still answer with a status code rather than
    # opening a stream it cannot police.
    from znyx_runtime.bundle_manager import ScopeNotServed
    bm = get_bundle_manager()
    try:
        body, policy = bm.resolve(body, key)
        env = body.env
    except ScopeNotServed as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Policy unavailable: {e}")
        raise HTTPException(status_code=503, detail="Policy unavailable (fail-closed)")

    request = EvaluationRequest(
        request_id=body.request_id,
        tenant_id=body.tenant_id,
        app_id=body.app_id,
        agent_id=body.agent_id,
        env=env,
        text="",  # text comes in chunks
    )

    ev = get_evaluator()
    stream_eval = StreamingEvaluator(
        policy=policy,
        context=body.context,
        window_size=body.window_size,
        overlap=body.overlap,
        request=request,
        orchestrator=ev.orchestrator,
        judge_ctx=runtime_judge_ctx(policy, request),
        remediation=ev.remediation,
        mode=body.mode,
    )

    def _sse(event) -> str:
        return f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"

    async def event_generator():
        t0 = time.perf_counter()
        try:
            try:
                for chunk in body.chunks:
                    # Detector work runs off the event loop, admission-controlled and
                    # under the per-call deadline, like a non-streaming evaluation.
                    for event in await ev.run_bounded(stream_eval.push, chunk):
                        yield _sse(event)
                    if stream_eval.is_blocked:
                        break

                # flush() returns the trailing events (final verdict, any released
                # tail) followed by `done`. Iterated in full: the final window's block
                # event used to be dropped here, which is what let a short stream through.
                for event in await ev.run_bounded(stream_eval.flush):
                    yield _sse(event)
            except EvaluatorOverloadedError:
                for event in stream_eval.fail_closed("Evaluator saturated, retry shortly"):
                    yield _sse(event)
            except DetectorDeadlineExceeded:
                for event in stream_eval.fail_closed("Evaluation deadline exceeded (fail-closed)"):
                    yield _sse(event)
        finally:
            # Runs even when the client disconnects mid-stream, so every stream is
            # recorded as one evaluation.
            try:
                ev.record_stream(
                    request,
                    decision=stream_eval.final_decision,
                    risk_score=stream_eval.max_risk,
                    rule_hits=stream_eval.rule_hits,
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                    context=body.context,
                    policy_version=policy.get("policy_version"),
                )
            except Exception:
                logger.debug("Stream telemetry failed", exc_info=True)

            hb = get_heartbeat()
            if hb:
                hb.increment_eval_count()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
