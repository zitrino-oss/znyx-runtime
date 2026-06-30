"""Orchestrator for the Playground "run with LLM" flow.

Given a draft policy, an input prompt, and an LLM provider + key, this
module drives the full round-trip:

    Input  →  GL(input)  →  LLM  →  GL(output)  →  Output

Each stage emits a :class:`StageResult` so the client can render the
traceability timeline without extra round-trips. If any guardrail stage
blocks, downstream stages are **skipped** and the trace surfaces the
short-circuit — same behaviour as the production pipeline.

The trace is deliberately **single-shot + ephemeral**:

- We never persist the transcript or the API key.
- Telemetry is disabled on the internal evaluator so the round-trip
  doesn't inflate the user's usage meter.
- Evaluation results carry the same shape the regular Playground emits
  so the UI can reuse components.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from app.shared.core.models import EvaluationRequest
from app.shared.engine.evaluator import GuardrailsEvaluator
from app.shared.llm.providers import (
    CompletionRequest,
    CompletionResult,
    LLMCallError,
    get_provider,
)

logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    """One stage of the round-trip trace.

    ``kind`` identifies the stage:

    - ``input_guardrail``  — GL on the raw user input
    - ``llm_call``         — forwarded to the customer's LLM
    - ``output_guardrail`` — GL on the LLM response

    Every stage carries its own latency and, where applicable, decision
    + detector breakdown. The ``blocked`` flag is true when **this** stage
    short-circuited the pipeline; ``skipped=True`` means a prior stage
    blocked so this one never ran.
    """

    kind: str
    started_at: float
    latency_ms: int
    skipped: bool = False
    blocked: bool = False

    # Populated for guardrail stages
    decision: Optional[str] = None
    risk_score: Optional[int] = None
    rule_hits: List[Dict[str, Any]] = field(default_factory=list)
    detector_results: List[Dict[str, Any]] = field(default_factory=list)
    user_message: Optional[str] = None
    developer_message: Optional[str] = None
    sanitized_text: Optional[str] = None

    # Populated for LLM stage
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    llm_error: Optional[Dict[str, Any]] = None  # {kind, message}

    # Payload snapshots
    input_text: Optional[str] = None
    output_text: Optional[str] = None


@dataclass
class RoundTripTrace:
    request_id: str
    final_decision: str  # ALLOW | BLOCK | TRANSFORM
    final_output: Optional[str]  # text shown to end user (None if blocked pre-LLM)
    total_latency_ms: int
    stages: List[StageResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "final_decision": self.final_decision,
            "final_output": self.final_output,
            "total_latency_ms": self.total_latency_ms,
            "stages": [asdict(s) for s in self.stages],
        }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


async def run_round_trip(
    *,
    org_id: str,
    input_text: str,
    policy: Dict[str, Any],
    provider_name: str,
    api_key: str,
    model: str,
    system_prompt: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> RoundTripTrace:
    """Drive the Input → GL → LLM → GL → Output pipeline and emit a trace.

    The evaluator is constructed fresh per call with telemetry disabled so
    the user's usage meter isn't inflated by playground testing. If the
    input guardrail blocks, we short-circuit before calling the LLM —
    surfaces to the UI as a ``blocked`` input stage + two ``skipped``
    downstream stages.
    """
    import uuid

    request_id = f"playground-llm-{uuid.uuid4().hex[:12]}"
    stages: List[StageResult] = []
    run_start = time.perf_counter()

    # Fresh evaluator per call. telemetry is skipped because we pass no
    # org_id into the evaluator; it operates purely on the provided policy.
    evaluator = GuardrailsEvaluator(policy_resolver=None, log_redacted_text=False)

    # --- Stage 1: input guardrail -----------------------------------------
    stage_in = await _run_guardrail_stage(
        evaluator=evaluator,
        kind="input_guardrail",
        text=input_text,
        policy=policy,
        context="input",
        request_id=request_id,
        org_id=org_id,
    )
    stages.append(stage_in)

    if stage_in.blocked:
        # Input blocked — skip LLM + output guardrail for an honest trace.
        stages.append(_skipped_stage("llm_call", provider=provider_name, model=model))
        stages.append(_skipped_stage("output_guardrail"))
        return RoundTripTrace(
            request_id=request_id,
            final_decision="BLOCK",
            final_output=None,
            total_latency_ms=int((time.perf_counter() - run_start) * 1000),
            stages=stages,
        )

    # If the input stage transformed the text (e.g. PII redaction), the
    # forwarded prompt is the sanitized version. Most deployments run this
    # way because sending the raw PII to the LLM would defeat the point.
    forwarded_prompt = stage_in.sanitized_text or input_text

    # --- Stage 2: LLM call -------------------------------------------------
    stage_llm = await _run_llm_stage(
        provider_name=provider_name,
        api_key=api_key,
        model=model,
        prompt=forwarded_prompt,
        system_prompt=system_prompt,
        endpoint_url=endpoint_url,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    stages.append(stage_llm)

    if stage_llm.llm_error is not None:
        # LLM failed — skip output guardrail, surface the error.
        stages.append(_skipped_stage("output_guardrail"))
        return RoundTripTrace(
            request_id=request_id,
            final_decision="BLOCK",  # fail-closed on LLM errors
            final_output=None,
            total_latency_ms=int((time.perf_counter() - run_start) * 1000),
            stages=stages,
        )

    llm_output = stage_llm.output_text or ""

    # --- Stage 3: output guardrail ----------------------------------------
    stage_out = await _run_guardrail_stage(
        evaluator=evaluator,
        kind="output_guardrail",
        text=llm_output,
        policy=policy,
        context="output",
        request_id=request_id,
        org_id=org_id,
    )
    stages.append(stage_out)

    final_decision = stage_out.decision or "ALLOW"
    # If output was transformed, show sanitized text to the user;
    # otherwise show raw. If blocked, no output is shown.
    if stage_out.blocked:
        final_output = None
    elif stage_out.sanitized_text:
        final_output = stage_out.sanitized_text
    else:
        final_output = llm_output

    return RoundTripTrace(
        request_id=request_id,
        final_decision=final_decision,
        final_output=final_output,
        total_latency_ms=int((time.perf_counter() - run_start) * 1000),
        stages=stages,
    )


# ---------------------------------------------------------------------------
# Stage builders
# ---------------------------------------------------------------------------


async def _run_guardrail_stage(
    *,
    evaluator: GuardrailsEvaluator,
    kind: str,
    text: str,
    policy: Dict[str, Any],
    context: str,
    request_id: str,
    org_id: str,
) -> StageResult:
    started = time.time()
    t0 = time.perf_counter()

    eval_request = EvaluationRequest(
        request_id=f"{request_id}-{context}",
        tenant_id=org_id,
        app_id="playground",
        text=text,
    )
    response = await evaluator.evaluate(
        eval_request,
        context=context,
        policy=policy,
    )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    decision = response.decision.value if response.decision else "ALLOW"
    blocked = decision == "BLOCK"

    return StageResult(
        kind=kind,
        started_at=started,
        latency_ms=elapsed_ms,
        blocked=blocked,
        decision=decision,
        risk_score=response.risk_score,
        rule_hits=[
            {
                "rule_id": h.rule_id,
                "severity": getattr(h.severity, "value", str(h.severity)),
                "message": h.message,
            }
            for h in (response.rule_hits or [])
        ],
        detector_results=[
            {
                "detector_name": dr.detector_name,
                "decision": dr.decision,
                "risk_score": dr.risk_score,
                "latency_ms": dr.latency_ms,
                "rule_hits": [
                    {"rule_id": h.rule_id, "message": h.message}
                    for h in (dr.rule_hits or [])
                ],
                "transformed": dr.transformed,
            }
            for dr in (response.detector_results or [])
        ],
        user_message=response.user_message,
        developer_message=response.developer_message,
        sanitized_text=response.sanitized_text,
        input_text=text,
    )


async def _run_llm_stage(
    *,
    provider_name: str,
    api_key: str,
    model: str,
    prompt: str,
    system_prompt: Optional[str],
    endpoint_url: Optional[str],
    max_tokens: int,
    temperature: float,
) -> StageResult:
    started = time.time()
    t0 = time.perf_counter()

    try:
        provider = get_provider(provider_name)
        result: CompletionResult = await provider.complete(
            api_key,
            CompletionRequest(
                input_text=prompt,
                system_prompt=system_prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                endpoint_url=endpoint_url,
            ),
        )
    except LLMCallError as exc:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        # Redact the outer message — it may contain snippets of the
        # provider's raw response which we're about to ship to the browser
        # already, but we keep a defensive max length cap.
        msg = str(exc)[:500]
        logger.info("Playground LLM call failed kind=%s status=%s", exc.kind, exc.status_code)
        return StageResult(
            kind="llm_call",
            started_at=started,
            latency_ms=elapsed_ms,
            blocked=False,
            llm_provider=provider_name,
            llm_model=model,
            input_text=prompt,
            llm_error={"kind": exc.kind, "message": msg, "status_code": exc.status_code},
        )
    except Exception as exc:  # pragma: no cover — belt-and-braces
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.exception("Playground LLM call raised unexpectedly")
        return StageResult(
            kind="llm_call",
            started_at=started,
            latency_ms=elapsed_ms,
            blocked=False,
            llm_provider=provider_name,
            llm_model=model,
            input_text=prompt,
            llm_error={"kind": "other", "message": f"{type(exc).__name__}: {exc}"},
        )

    return StageResult(
        kind="llm_call",
        started_at=started,
        latency_ms=result.latency_ms or int((time.perf_counter() - t0) * 1000),
        blocked=False,
        llm_provider=provider_name,
        llm_model=result.model,
        input_text=prompt,
        output_text=result.text,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.total_tokens,
    )


def _skipped_stage(kind: str, *, provider: Optional[str] = None, model: Optional[str] = None) -> StageResult:
    return StageResult(
        kind=kind,
        started_at=time.time(),
        latency_ms=0,
        skipped=True,
        llm_provider=provider,
        llm_model=model,
    )
