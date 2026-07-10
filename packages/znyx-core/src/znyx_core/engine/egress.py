"""Central egress gate + redaction.

Evaluated in `escalation.run_with_strategy` immediately before any boundary-crossing
backend call. "Boundary-crossing" = remote_llm / remote_api always, or a local_*
model mode whose sidecar is NOT in-boundary (a co-located self-host sidecar is not
egress; a hosted/network one is). The gate runs IN ORDER:

  1. runtime_policy.no_external_calls  → skip   (reason "no_external_calls")
  2. host not in per-detector allowlist → deny  (reason "egress_not_allowlisted")
  3. region unknown / not in allowed_regions → deny (reason "residency_denied")
  4. otherwise → allow; redact-before-egress, mark external_egress, emit one
     egress event.

The decision (skip/deny) is synchronous here; the durable egress_events row is written
by the injected sink so the zero-DB runtime never touches control-plane tables.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple
from urllib.parse import urlparse
from uuid import uuid4

_REMOTE_MODES = {"remote_llm", "remote_api"}
_LOCAL_MODEL_MODES = {"local_ml", "local_embedding", "local_llm"}


class RedactionError(Exception):
    """Raised by ``redact_for_egress`` in strict mode when a configured redactor
    cannot complete, so the caller fails closed instead of sending raw content."""


@dataclass
class EgressDecision:
    is_egress: bool                 # does this call cross the trust boundary?
    allowed: bool                   # may the call proceed?
    reason: Optional[str]           # fallback_path reason when not allowed
    host: Optional[str] = None
    region: Optional[str] = None
    redact_pii: bool = False
    redact_secrets: bool = False
    allowlisted: bool = False


@dataclass
class EgressEvent:
    """Metadata-only audit record for one boundary-crossing call (→ egress_events row).

    ``event_id`` is a stable UUID minted at the crossing so a durable sink (spool) can
    be drained into ``egress_events`` idempotently (the row PK)."""
    occurred_at: str
    detector_key: str
    mode: str
    destination_host: Optional[str]
    destination_region: Optional[str] = None
    org_scope: Optional[str] = None
    trace_id: Optional[str] = None
    bytes_out: int = 0
    redacted: bool = False
    allowlisted: bool = False
    model_version: Optional[str] = None
    event_id: Optional[str] = None

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def _host_of(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        return urlparse(url).hostname
    except Exception:
        return None


def _authority_of(url: Optional[str]) -> Optional[str]:
    """host:port (lowercased, credentials stripped) — so a port-specific allowlist
    entry like ``gw.internal:8443`` can be matched exactly while a bare-host entry
    (``gw.internal``) still matches any port."""
    if not url:
        return None
    try:
        netloc = urlparse(url).netloc
    except Exception:
        return None
    if not netloc:
        return None
    return netloc.rsplit("@", 1)[-1].lower()


def is_boundary_crossing(mode: str, in_boundary: Optional[bool]) -> bool:
    """Whether reaching ``mode`` leaves the trust boundary."""
    if mode in _REMOTE_MODES:
        return True
    if mode in _LOCAL_MODEL_MODES:
        # Default in_boundary=True (self-host sidecar) → not egress. A hosted sidecar
        # must declare in_boundary=False to be gated/audited as egress.
        return in_boundary is False
    return False  # local_deterministic never leaves the box


def evaluate_egress(
    mode: str,
    *,
    endpoint_url: Optional[str],
    region: Optional[str],
    in_boundary: Optional[bool],
    no_external_calls: bool,
    egress_allowlist: Optional[list],
    allowed_regions: Optional[list],
    redact_pii: bool,
    redact_secrets: bool,
) -> EgressDecision:
    """Pure gate decision for one backend call (see module docstring for the order)."""
    host = _host_of(endpoint_url)
    if not is_boundary_crossing(mode, in_boundary):
        return EgressDecision(is_egress=False, allowed=True, reason=None, host=host, region=region)

    if no_external_calls:
        return EgressDecision(is_egress=True, allowed=False, reason="no_external_calls",
                              host=host, region=region)

    # Allowlist matches a bare host (any port) OR an exact host:port authority.
    authority = _authority_of(endpoint_url)
    if egress_allowlist is not None and host not in egress_allowlist and authority not in egress_allowlist:
        return EgressDecision(is_egress=True, allowed=False, reason="egress_not_allowlisted",
                              host=host, region=region, allowlisted=False)

    if allowed_regions is not None and (region is None or region not in allowed_regions):
        return EgressDecision(is_egress=True, allowed=False, reason="residency_denied",
                              host=host, region=region, allowlisted=True)

    return EgressDecision(is_egress=True, allowed=True, reason=None, host=host, region=region,
                          redact_pii=redact_pii, redact_secrets=redact_secrets,
                          allowlisted=egress_allowlist is not None)


def redact_for_egress(text: str, redact_pii: bool, redact_secrets: bool,
                      *, strict: bool = False,
                      pii_config: Optional[dict] = None,
                      secrets_config: Optional[dict] = None) -> Tuple[str, bool]:
    """Scrub PII / secrets out of ``text`` before it leaves the boundary. Returns
    (possibly-redacted text, whether anything was redacted). Reuses the deterministic
    PII / secrets detectors' REDACT path.

    ``pii_config`` / ``secrets_config`` carry the ORG's detector config (e.g. the PII
    types it enables, including default-disabled ones like passport/national_id) so
    egress redaction covers exactly what the org redacts in-pipeline — we just force
    ``enabled``/``action=REDACT`` on top. Absent → the detector defaults.

    ``strict`` (the egress path's default) is privacy fail-closed: if a configured
    redactor raises, re-raise ``RedactionError`` so the caller denies the egress
    rather than silently sending unredacted content. ``strict=False`` keeps the
    best-effort behaviour for non-egress callers."""
    redacted = text
    changed = False
    if redact_pii:
        try:
            from znyx_core.detectors.pii import PIIDetector
            cfg = {**(pii_config or {}), "enabled": True, "action": "REDACT"}
            r = PIIDetector(cfg).detect(redacted)
            if r.sanitized_text and r.sanitized_text != redacted:
                redacted, changed = r.sanitized_text, True
        except Exception as exc:  # noqa: BLE001
            if strict:
                raise RedactionError(f"PII redaction failed before egress: {exc}") from exc
    if redact_secrets:
        try:
            from znyx_core.detectors.secrets import SecretsDetector
            cfg = {**(secrets_config or {}), "enabled": True, "action": "REDACT"}
            r = SecretsDetector(cfg).detect(redacted)
            if r.sanitized_text and r.sanitized_text != redacted:
                redacted, changed = r.sanitized_text, True
        except Exception as exc:  # noqa: BLE001
            if strict:
                raise RedactionError(f"secrets redaction failed before egress: {exc}") from exc
    return redacted, changed


@dataclass
class PreparedEgress:
    """Outcome of the per-call egress sequence (gate → redact → fail-closed audit).
    ``proceed`` is True only when the boundary-crossing call may now be made with
    ``call_text``; otherwise ``reason`` is the verbatim fallback_path. ``event_id`` is the
    emitted egress event's id (set only when a boundary-crossing event was recorded) so the
    caller can link the corresponding judge/detector audit row to it."""
    decision: EgressDecision
    call_text: str
    redacted: bool
    proceed: bool
    reason: Optional[str] = None
    event_id: Optional[str] = None


def prepare_and_audit_egress(
    mode: str,
    text: str,
    *,
    endpoint_url: Optional[str],
    region: Optional[str],
    in_boundary: Optional[bool],
    no_external_calls: bool,
    egress_allowlist: Optional[list],
    allowed_regions: Optional[list],
    redact_pii: bool,
    redact_secrets: bool,
    detector_key: str,
    request=None,
    egress_sink=None,
    model_version: Optional[str] = None,
    pii_config: Optional[dict] = None,
    secrets_config: Optional[dict] = None,
) -> PreparedEgress:
    """THE single per-call egress sequence, shared by the model-backed escalation
    engine and the custom webhook-detector path so both fail closed identically:

      1. evaluate_egress (no_external_calls / allowlist / residency) — deny ⇒ stop
      2. redact-before-egress (strict) — a configured redactor that can't run ⇒ stop
      3. fail-closed audit: no sink wired, or a sink that can't durably persist ⇒ stop

    Only if all three pass is ``proceed=True`` returned with the (possibly redacted)
    ``call_text``. The durable audit event is written HERE, before the caller makes the
    boundary-crossing call — so a successful return means the crossing is recorded."""
    decision = evaluate_egress(
        mode, endpoint_url=endpoint_url, region=region, in_boundary=in_boundary,
        no_external_calls=no_external_calls, egress_allowlist=egress_allowlist,
        allowed_regions=allowed_regions, redact_pii=redact_pii, redact_secrets=redact_secrets,
    )
    if not decision.allowed:
        return PreparedEgress(decision, text, False, False, decision.reason)

    call_text, redacted = text, False
    if decision.is_egress and (decision.redact_pii or decision.redact_secrets):
        try:
            call_text, redacted = redact_for_egress(
                text, decision.redact_pii, decision.redact_secrets, strict=True,
                pii_config=pii_config, secrets_config=secrets_config)
        except RedactionError:
            return PreparedEgress(decision, text, False, False, "redaction_failed")

    if decision.is_egress:
        if egress_sink is None:
            return PreparedEgress(decision, text, False, False, "egress_audit_unconfigured")
        event_id = str(uuid4())
        try:
            egress_sink(EgressEvent(
                event_id=event_id,
                occurred_at=datetime.now(timezone.utc).isoformat(),
                detector_key=detector_key,
                mode=mode,
                destination_host=decision.host,
                destination_region=decision.region,
                org_scope=getattr(request, "tenant_id", None),
                trace_id=getattr(request, "trace_id", None),
                # payload TEXT bytes (post-redaction); excludes transport JSON framing
                # + headers, which the gate is deliberately decoupled from.
                bytes_out=len(call_text.encode("utf-8")),
                redacted=redacted,
                allowlisted=decision.allowlisted,
                model_version=model_version,
            ))
        except Exception:  # noqa: BLE001 — any audit failure must fail closed
            return PreparedEgress(decision, text, False, False, "egress_audit_unavailable")
        # Carry the egress event id so the judge/detector audit row can reference it.
        return PreparedEgress(decision, call_text, redacted, True, None, event_id=event_id)

    return PreparedEgress(decision, call_text, redacted, True, None)
