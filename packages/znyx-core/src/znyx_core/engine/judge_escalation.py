"""Judge consensus backend for the escalation engine (P3 unit 5, roadmap §5).

``make_judge_consensus_caller`` returns an escalation-compatible
``(backend, text, timeout_ms) -> DetectorResult`` that runs K independent judge calls
(consensus members) over the (already egress-gated + redacted) text, votes via
``synthesize_consensus``, and writes the audit trail: one ``judge_audit_events`` row per
member (sharing a ``consensus_group_id``, ``is_consensus_result`` = False) plus one
synthesized-result row. The synthesized verdict becomes the escalation's judge-layer
result. If every member fails/malforms, it raises ``BackendUnavailable`` so the strategy
fallback fires.

Egress is the escalation engine's responsibility (it gates once before invoking this
caller); this module makes the provider calls on the redacted text and does not re-gate.
"""
import logging
import uuid
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)

from znyx_core.core.models import Decision, DetectorResult, RuleHit, Severity
from znyx_core.engine.consensus import JudgeVote, synthesize_consensus
from znyx_core.engine.escalation import BackendUnavailable
from znyx_core.engine.quality.judge_evaluator import _run_coro_sync
from znyx_core.llm.judge import JudgeRequest, run_judge


def make_judge_consensus_caller(
    rubric: str, *,
    members: int = 1,
    method: str = "majority",
    provider: str = "openai",
    model: str = "",
    endpoint_url: Optional[str] = None,
    api_key: str = "",
    rubric_version: Optional[str] = None,
    rubric_hash: Optional[str] = None,
    member_weights: Optional[List[float]] = None,
    caller: Optional[Callable] = None,
    audit_sink: Optional[Callable[[dict], None]] = None,
    budget_check: Optional[Callable[[str], bool]] = None,
    detector_key: Optional[str] = None,
    request: Any = None,
    max_tokens: int = 512,
    max_retries: int = 1,
) -> Callable:
    """Build the escalation judge-layer caller. ``members`` judge calls are voted via
    ``method`` (majority|weighted); ``caller`` (the provider caller) is injectable for tests.

    ``detector_key`` attributes the audit rows + budget scope to the owning detector (the
    DetectorBackend carries no key, so without this every escalation judge logged as
    ``"judge"``). ``budget_check`` is the deny-of-wallet gate: when it vetoes (or errors),
    the caller raises ``BackendUnavailable`` so the strategy fallback fires instead of
    spending on the judge."""
    members = max(1, int(members))

    def judge_caller(backend, text: str, timeout_ms: Optional[int], *, egress_gate=None) -> DetectorResult:
        timeout = (timeout_ms or backend.timeout_ms or 20000) / 1000.0
        group_id = uuid.uuid4()
        trace_id = getattr(request, "trace_id", None) if request else None
        env = getattr(request, "env", None) if request else None
        det_key = detector_key or getattr(backend, "detector_key", None) or "judge"

        votes: List[JudgeVote] = []
        denied: Optional[str] = None  # budget/egress denial reason → strategy fallback path
        for i in range(members):
            # Per-member deny-of-wallet gate (fail closed). Checking each member — not once
            # for the batch — means a runtime spend tally that ticks up as members run can
            # stop the batch before it overspends. A denial stops the loop.
            if budget_check is not None:
                try:
                    allowed = budget_check(det_key)
                except Exception:
                    allowed = False
                if not allowed:
                    denied = f"judge budget exceeded for {det_key}"
                    break
            # Per-member F4 egress gate: EACH consensus call is its own audited boundary
            # crossing (one egress event per remote judge call), and the call runs on the
            # gate's redacted text. No gate (in-boundary / tests) → no crossing recorded.
            call_text, crossed, event_id = text, False, None
            if egress_gate is not None:
                prep = egress_gate(text)
                if not prep.proceed:
                    denied = prep.reason  # same denial for every member → stop the batch
                    break
                call_text, crossed, event_id = prep.call_text, bool(prep.decision.is_egress), prep.event_id

            use_model = backend.model_id or model
            use_endpoint = backend.endpoint_url or endpoint_url
            logger.info("judge: %s member %d — calling %s model=%s endpoint=%s",
                        det_key, i, provider, use_model, use_endpoint)
            req = JudgeRequest(
                rubric=rubric, content=call_text, output_kind="detector",
                provider=provider, model=use_model,
                endpoint_url=use_endpoint,
                max_tokens=max_tokens, temperature=0.0, rubric_version=rubric_version,
            )
            try:
                result = _run_coro_sync(lambda r=req: run_judge(
                    r, api_key, caller=caller, timeout=timeout, max_retries=max_retries))
            except Exception as exc:
                logger.warning("judge: %s member %d — exception: %s", det_key, i, exc)
                result = None

            verdict = result.verdict if result else None
            if verdict is None or verdict.decision is None:
                err = result.error if result else "no result"
                raw_preview = (result.raw_text[:200] if result and result.raw_text else "")
                logger.warning("judge: %s member %d — no verdict (error=%s, raw=%r)",
                               det_key, i, err, raw_preview)
                continue  # malformed/failed member — excluded from the vote
            weight = 1.0
            if member_weights and i < len(member_weights):
                weight = member_weights[i]
            votes.append(JudgeVote(
                decision=verdict.decision,
                risk_score=float(verdict.risk_score or 0.0),
                confidence=verdict.confidence, weight=weight,
            ))
            if audit_sink is not None:
                try:
                    audit_sink({
                        "trace_id": trace_id, "env": env, "detector_key": det_key,
                        "judge_model": result.model, "rubric_version": rubric_version,
                        "rubric_hash": rubric_hash, "decision": verdict.decision,
                        "confidence": verdict.confidence,
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                        "total_tokens": result.total_tokens, "latency_ms": result.latency_ms,
                        # the REAL crossing for THIS member + its egress event id
                        "left_runtime_boundary": crossed, "egress_event_id": event_id,
                        "consensus_group_id": group_id, "consensus_method": method,
                        "vote_weight": weight, "member_index": i,
                        "is_consensus_result": False,
                    })
                except Exception:
                    pass

        if not votes:
            msg = denied or "all judge consensus members failed"
            logger.warning("judge: %s — %s", det_key, msg)
            raise BackendUnavailable(msg)

        consensus = synthesize_consensus(votes, method=method)
        logger.info("judge: %s — consensus %s (risk=%.0f, agreement=%s, members=%d)",
                     det_key, consensus.decision, consensus.risk_score,
                     consensus.agreement, consensus.member_count)
        if audit_sink is not None:
            try:
                audit_sink({
                    "trace_id": trace_id, "env": env, "detector_key": det_key,
                    "decision": consensus.decision, "confidence": consensus.confidence,
                    "rubric_version": rubric_version, "rubric_hash": rubric_hash,
                    "consensus_group_id": group_id, "consensus_method": method,
                    # The synthesized verdict is computed locally — NOT a boundary crossing.
                    "is_consensus_result": True, "left_runtime_boundary": False,
                })
            except Exception:
                pass

        try:
            decision = Decision(consensus.decision)
        except ValueError:
            # An unexpected synthesized decision string must not escape as an uncaught
            # ValueError — fall back so the escalation applies the strategy fallback.
            raise BackendUnavailable(f"consensus produced an invalid decision {consensus.decision!r}")
        rule_hits = []
        if decision != Decision.ALLOW:
            rule_hits = [RuleHit(rule_id=f"{det_key}.judge_consensus", severity=Severity.HIGH,
                                 message=f"Judge consensus ({method}, {consensus.member_count} members, "
                                         f"agreement {consensus.agreement}): {consensus.decision}")]
        return DetectorResult(
            decision=decision, risk_score=int(round(consensus.risk_score)),
            rule_hits=rule_hits, confidence=consensus.confidence,
            developer_message=f"judge consensus {method}: {consensus.decision} "
                              f"(agreement={consensus.agreement}, members={consensus.member_count})",
        )

    return judge_caller
