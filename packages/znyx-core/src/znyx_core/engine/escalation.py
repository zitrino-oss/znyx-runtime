"""Escalation engine (F2): run a detector's strategy deterministic → ml → llm.

``run_with_strategy`` takes the already-computed deterministic result and, guided by
the strategy's ``order`` + ``escalate_when`` predicates, optionally calls the next
backend (ml/llm/remote) through the extended RemoteDetector transport. Every attempt
is recorded as a ``LayerResult`` (F1) with the final one flagged ``selected``; the
``fallback`` policy (fail_open / fail_closed / fallback_to_deterministic) governs what
happens when a backend can't be reached, recorded in ``fallback_path``.

F4 will insert the central egress gate (no_external_calls / allowlist / residency)
immediately before any boundary-crossing backend call here.
"""
from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)

from znyx_core.core.labels import normalize_risk
from znyx_core.core.models import Decision, DetectorResult, LayerResult, RuleHit, Severity
from znyx_core.engine.backends import (
    BackendStrategy,
    DetectorBackend,
    should_escalate,
    _DETERMINISTIC,
)
from znyx_core.engine.egress import EgressEvent, is_boundary_crossing, prepare_and_audit_egress


def _caller_accepts_gate(fn) -> bool:
    """True if a judge caller accepts a per-call ``egress_gate`` kwarg (so each consensus
    member is gated/audited individually). A legacy 3-arg caller → False → gate once."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "egress_gate" in params or any(p.kind == p.VAR_KEYWORD for p in params.values())

_ML_MODES = {"local_ml", "local_embedding"}
_LLM_MODES = {"local_llm", "remote_llm"}
# Egress-gate denial reasons that become the fallback_path verbatim (F4). Includes
# the two fail-closed reasons: a configured redactor that couldn't run, and an audit
# write that couldn't be durably persisted — both DENY the outbound call.
_GATE_REASONS = {
    "no_external_calls", "egress_not_allowlisted", "residency_denied",
    "redaction_failed", "egress_audit_unavailable", "egress_audit_unconfigured",
}


def _configured_model_version(backend: DetectorBackend) -> Optional[str]:
    """The model identity we intend to call — known pre-call, used in the audit event
    that is written BEFORE content leaves the boundary (the server-reported
    model_version isn't available until after the call)."""
    if backend.model_id and backend.revision:
        return f"{backend.model_id}@{backend.revision}"
    return backend.model_id


class BackendUnavailable(Exception):
    """An escalated backend could not produce a result (transport failure / no config)."""


def _normalize_kind(mode: str) -> str:
    if mode in _ML_MODES:
        return "ml"
    if mode in _LLM_MODES or mode == "remote_api":
        return "judge"
    return "deterministic"


def _call_backend(backend: DetectorBackend, text: str, timeout_ms: Optional[int]) -> DetectorResult:
    """Invoke a backend via the extended RemoteDetector transport (F0.5).

    Raises BackendUnavailable on a transport failure so the caller can apply the
    strategy fallback. (Tests inject their own backend_caller and never hit this.)
    """
    if not backend.endpoint_url:
        raise BackendUnavailable(f"no endpoint configured for mode {backend.mode}")

    timeout = timeout_ms or backend.timeout_ms

    # P4: a remote_api backend may name a vendor moderation adapter (provider), whose
    # response shape the generic field-path RemoteDetector can't map. Route to it.
    if backend.mode == "remote_api" and backend.provider:
        from znyx_core.detectors.adapters import get_adapter
        adapter = get_adapter(backend.provider)
        if adapter is not None:
            cfg = {
                "endpoint_url": backend.endpoint_url,
                "auth_value": backend.auth_value or "",
                "model_id": backend.model_id,
                "region": backend.region,
            }
            # Merge vendor-specific settings (Azure severity/api-version, Bedrock
            # guardrail id/version + secret, …) without letting them clobber the core keys.
            if isinstance(backend.provider_config, dict):
                for k, v in backend.provider_config.items():
                    cfg.setdefault(k, v)
            try:
                return adapter.evaluate(text, cfg, timeout=(timeout / 1000.0 if timeout else 8.0))
            except Exception as exc:  # transport / mapping failure → strategy fallback
                raise BackendUnavailable(f"remote_api adapter '{backend.provider}' failed: {exc}")

    from znyx_core.detectors.remote import RemoteDetector

    config = {
        "endpoint_url": backend.endpoint_url,
        "task": backend.task,
        "model_id": backend.model_id,
        "model_revision": backend.revision,
        "auth_type": backend.auth_type or "none",
        "auth_value": backend.auth_value or "",
        # fail-open so a transport failure yields a detectable marker (below) rather
        # than a synthetic BLOCK; the strategy fallback then decides the real outcome.
        "fail_open": True,
    }
    if timeout:
        config["timeout_seconds"] = timeout / 1000.0
        config["total_deadline_seconds"] = timeout / 1000.0

    result = RemoteDetector(config).detect(text)
    if "Remote detector unavailable" in (result.developer_message or ""):
        raise BackendUnavailable(result.developer_message or "backend unavailable")
    return result


def _fallback_result(fallback: str, deterministic_result: DetectorResult, mode: str, err: str):
    """Return (result, selected_mode, fallback_path) for a failed escalation."""
    if fallback == "fail_open":
        return (
            DetectorResult(decision=Decision.ALLOW, risk_score=0,
                           developer_message=f"escalation fail-open ({mode}): {err}"),
            mode, f"fail_open:{mode}",
        )
    if fallback == "fail_closed":
        return (
            DetectorResult(
                decision=Decision.BLOCK, risk_score=100,
                rule_hits=[RuleHit(rule_id="escalation.fail_closed", severity=Severity.HIGH,
                                   message=f"Backend unavailable ({mode}); failing closed")],
                developer_message=f"escalation fail-closed ({mode}): {err}",
            ),
            mode, f"fail_closed:{mode}",
        )
    # default: fall back to the deterministic result
    return (deterministic_result, _DETERMINISTIC, f"fallback_to_deterministic:{mode}")


def _additive_merge(det: DetectorResult, ml: DetectorResult) -> DetectorResult:
    """Combine a deterministic result with an additive ML layer's result: worst-of decision,
    max risk, union of rule_hits, and the deterministic layer's redaction/transform text kept
    when present (so e.g. regex PII redaction survives while NER adds its unstructured-PII
    findings). The deterministic decision is never lost — that's the point of additive."""
    from znyx_core.core.labels import decision_rank
    keep = det if decision_rank(det.decision) >= decision_rank(ml.decision) else ml
    return keep.model_copy(update={
        "risk_score": max(det.risk_score, ml.risk_score),
        "rule_hits": list(det.rule_hits) + list(ml.rule_hits),
        "sanitized_text": det.sanitized_text or ml.sanitized_text,
    })


def _layer_from_result(mode: str, result: DetectorResult, selected: bool) -> LayerResult:
    return LayerResult(
        execution_mode=mode,
        decision=result.decision.value if result.decision else None,
        native_score=result.risk_score,
        normalized_score=normalize_risk(result.risk_score, _normalize_kind(mode)),
        confidence=result.confidence,
        calibrated_score=result.calibrated_score,
        label_scores=result.label_scores,
        model_version=result.model_version,
        selected=selected,
    )


def run_with_strategy(
    detector_key: str,
    deterministic_result: DetectorResult,
    strategy: BackendStrategy,
    text: str,
    *,
    backend_caller: Optional[Callable[[DetectorBackend, str, Optional[int]], DetectorResult]] = None,
    judge_caller: Optional[Callable[[DetectorBackend, str, Optional[int]], DetectorResult]] = None,
    request: Any = None,
    egress_sink: Optional[Callable[[EgressEvent], None]] = None,
) -> DetectorResult:
    """Escalate the deterministic result through the strategy's ordered modes.

    Returns a DetectorResult whose scalar decision/scores come from the SELECTED layer,
    with ``layer_results`` holding every attempt (selected flagged), ``execution_mode``
    set to the selected mode, and ``fallback_path`` set if a fallback fired.
    """
    # Resolve at call time (not as a default arg) so tests can monkeypatch _call_backend
    # to exercise the orchestrator → escalation path without real HTTP.
    caller = backend_caller or _call_backend

    det_risk = deterministic_result.risk_score
    det_layer = LayerResult(
        execution_mode=_DETERMINISTIC,
        decision=deterministic_result.decision.value if deterministic_result.decision else None,
        native_score=det_risk, normalized_score=det_risk, selected=True,
    )
    layers: List[LayerResult] = [det_layer]

    selected_result = deterministic_result
    selected_mode = _DETERMINISTIC
    fallback_path: Optional[str] = None

    cur_mode, cur_risk, cur_conf = _DETERMINISTIC, float(det_risk), deterministic_result.confidence

    def _record_fallback(mode: str, reason: str):
        """Apply the strategy fallback and record the failed/blocked attempt layer.
        Returns (selected_result, selected_mode) and sets fallback_path (closure)."""
        nonlocal fallback_path, selected_result, selected_mode
        fb = strategy.fallback or "fallback_to_deterministic"
        fb_result, fb_mode, _generic_path = _fallback_result(fb, deterministic_result, mode, reason)
        # The precise reason (gate reason or transport error) is the fallback_path.
        fallback_path = reason if reason in _GATE_REASONS else _generic_path
        for lyr in layers:
            lyr.selected = False
        layers.append(LayerResult(
            execution_mode=mode,
            decision=fb_result.decision.value if fb_result.decision else None,
            fallback_reason=fallback_path,
            external_egress=False,   # the call was blocked/never made
            selected=(fb_mode == mode),
        ))
        if fb_mode == _DETERMINISTIC:
            det_layer.selected = True
        selected_result, selected_mode = fb_result, fb_mode

    for mode in strategy.order:
        if mode == _DETERMINISTIC:
            continue
        if not should_escalate(strategy.escalate_when, cur_mode, cur_risk, cur_conf):
            logger.debug("escalation: %s — not escalating from %s (risk=%.0f, band=%s)",
                         detector_key, cur_mode, cur_risk, strategy.escalate_when)
            break

        logger.info("escalation: %s — escalating from %s to %s (det_risk=%.0f)",
                     detector_key, cur_mode, mode, cur_risk)

        backend = strategy.backend_for(mode)
        if backend is None:
            _record_fallback(mode, f"no backend configured for mode {mode}")
            logger.warning("escalation: %s — no backend for mode %s, falling back", detector_key, mode)
            break

        # F4 egress sequence (gate → redact → fail-closed audit), shared verbatim with the
        # custom webhook path. The endpoint is resolved to the provider's REAL host so
        # auditing/allowlisting/residency see the true destination (not None → "(unknown)",
        # which the provider would later default to OpenAI/etc.). For a judge consensus
        # caller, hand it a PER-CALL gate so each member is audited as its own crossing (one
        # egress event per remote judge call); ml/legacy modes gate once before the call.
        from znyx_core.llm.providers import effective_endpoint
        eff_endpoint = effective_endpoint(backend.provider, backend.endpoint_url)

        def _gate(t, _ep=eff_endpoint, _b=backend):
            return prepare_and_audit_egress(
                mode, t, endpoint_url=_ep, region=_b.region, in_boundary=_b.in_boundary,
                no_external_calls=strategy.no_external_calls, egress_allowlist=strategy.egress_allowlist,
                allowed_regions=strategy.allowed_regions, redact_pii=strategy.redact_pii,
                redact_secrets=strategy.redact_secrets, detector_key=detector_key, request=request,
                egress_sink=egress_sink, model_version=_configured_model_version(_b),
                pii_config=strategy.redact_pii_config, secrets_config=strategy.redact_secrets_config,
            )

        is_judge = judge_caller is not None and _normalize_kind(mode) == "judge"
        if is_judge and _caller_accepts_gate(judge_caller):
            # The judge caller gates + audits each consensus member call itself.
            try:
                result = judge_caller(backend, text, strategy.timeout_ms, egress_gate=_gate)
            except BackendUnavailable as exc:
                logger.warning("escalation: %s — judge %s unavailable: %s", detector_key, mode, exc)
                _record_fallback(mode, str(exc))
                break
            crossed = is_boundary_crossing(mode, backend.in_boundary)
        else:
            # ml / legacy: gate once before the single backend call.
            prep = _gate(text)
            if not prep.proceed:
                logger.warning("escalation: %s — egress gate denied %s: %s", detector_key, mode, prep.reason)
                _record_fallback(mode, prep.reason)
                break
            use_caller = judge_caller if is_judge else caller
            try:
                result = use_caller(backend, prep.call_text, strategy.timeout_ms)
            except BackendUnavailable as exc:
                logger.warning("escalation: %s — backend %s unavailable: %s", detector_key, mode, exc)
                _record_fallback(mode, str(exc))
                break
            crossed = prep.decision.is_egress

        # successful escalation
        logger.info("escalation: %s — %s returned decision=%s risk=%s",
                     detector_key, mode,
                     result.decision.value if result.decision else None, result.risk_score)
        for lyr in layers:
            lyr.selected = False
        layer = _layer_from_result(mode, result, selected=True)
        layer.external_egress = crossed
        layers.append(layer)
        selected_result, selected_mode = result, mode
        cur_mode, cur_risk, cur_conf = mode, float(result.risk_score), result.confidence

    # Additive: the ML layer augments (worst-of) the deterministic result rather than
    # replacing it. Only when a real ML result was selected (no fallback fired) — a
    # fallback already preserves the deterministic decision on its own.
    if strategy.additive and fallback_path is None and selected_mode != _DETERMINISTIC:
        selected_result = _additive_merge(deterministic_result, selected_result)
        det_layer.selected = True  # both layers contributed to the additive verdict

    return selected_result.model_copy(update={
        "execution_mode": selected_mode,
        "fallback_path": fallback_path,
        # Scalar mirrors the F1 contract: "True if any content left the boundary."
        "external_egress": any(lyr.external_egress for lyr in layers),
        "layer_results": layers,
    })
