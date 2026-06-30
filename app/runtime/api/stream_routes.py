"""Streaming guardrails SSE endpoint - evaluate LLM output in real-time."""
import json
import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.runtime.api.routes import verify_runtime_auth, get_evaluator

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Runtime API"])


class StreamEvaluateRequest(BaseModel):
    """Request body for streaming evaluation."""
    request_id: str = "stream-0"
    tenant_id: str = "default"
    app_id: str = "default"
    context: str = Field(default="output", description="input or output")
    chunks: List[str] = Field(..., description="Text chunks to evaluate in order")
    policy: Optional[Dict[str, Any]] = Field(default=None, description="Inline policy (optional)")
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
    evaluator=Depends(get_evaluator),
):
    """Evaluate text chunks via Server-Sent Events.

    Accepts a list of text chunks and streams back SSE events as each window
    is evaluated. Events:
      - ``chunk``: forwarded text chunk
      - ``guardrail``: window evaluation result (non-blocking)
      - ``block``: window triggered a BLOCK decision
      - ``done``: final summary with aggregate metrics
    """
    from app.shared.engine.streaming import StreamingEvaluator
    from app.shared.core.models import EvaluationRequest

    # Resolve policy
    policy = body.policy
    if policy is None and evaluator.policy_resolver:
        policy = evaluator.policy_resolver.resolve(
            tenant_id=body.tenant_id,
            app_id=body.app_id,
            agent_id="default",
            env="prod",
        )
    if policy is None:
        policy = {}

    request = EvaluationRequest(
        request_id=body.request_id,
        tenant_id=body.tenant_id,
        app_id=body.app_id,
        text="",  # text comes in chunks
    )

    stream_eval = StreamingEvaluator(
        policy=policy,
        context=body.context,
        window_size=body.window_size,
        overlap=body.overlap,
        request=request,
    )

    async def event_generator():
        for chunk in body.chunks:
            events = stream_eval.push(chunk)
            for event in events:
                yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"
                if event["event"] == "block":
                    # Still send done after block
                    break
            if stream_eval.is_blocked:
                break

        summary = stream_eval.flush()
        yield f"event: {summary['event']}\ndata: {json.dumps(summary['data'])}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
