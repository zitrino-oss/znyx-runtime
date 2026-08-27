"""Streaming guardrails SSE endpoint - evaluate LLM output in real-time."""
import json
import logging
from typing import List, Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from znyx_runtime.api.routes import (
    verify_runtime_auth,
    get_bundle_manager,
    get_heartbeat,
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
    window_size: int = Field(default=200, ge=50, le=2000)
    overlap: int = Field(default=40, ge=0, le=500)


@router.post(
    "/v1/evaluate/stream",
    operation_id="runtime.evaluateStream",
    summary="Evaluate streaming LLM output as it arrives",
)
async def evaluate_stream(
    body: StreamEvaluateRequest,
    _auth=Depends(verify_runtime_auth),
):
    """Evaluate text chunks via Server-Sent Events.

    Accepts a list of text chunks and streams back SSE events as each window
    is evaluated. Events:
      - ``guardrail``: window evaluation result (non-blocking)
      - ``chunk``: text a window has evaluated and allowed
      - ``block``: window triggered a BLOCK decision
      - ``done``: final summary with aggregate metrics

    A ``chunk`` is only emitted once the window covering it has been evaluated
    and allowed, so nothing this endpoint forwards has skipped the detectors.
    """
    from znyx_core.engine.streaming import StreamingEvaluator
    from znyx_core.core.models import EvaluationRequest

    # Resolve the policy from the active bundle, the same way the non-streaming
    # evaluate routes do. Done before the response starts so a fail-closed
    # runtime with no bundle can still answer with a status code rather than
    # opening a stream it cannot police.
    bm = get_bundle_manager()
    env = bm.effective_env or body.env
    try:
        policy = bm.get_policy(
            tenant_id=body.tenant_id,
            app_id=body.app_id,
            agent_id=body.agent_id,
            env=env,
        )
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

    stream_eval = StreamingEvaluator(
        policy=policy,
        context=body.context,
        window_size=body.window_size,
        overlap=body.overlap,
        request=request,
    )

    def _sse(event) -> str:
        return f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"

    async def event_generator():
        for chunk in body.chunks:
            for event in stream_eval.push(chunk):
                yield _sse(event)
            if stream_eval.is_blocked:
                break

        # flush() returns the trailing events (final verdict, any released tail)
        # followed by `done`. Iterated in full: the final window's block event
        # used to be dropped here, which is what let a short stream through.
        for event in stream_eval.flush():
            yield _sse(event)

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
