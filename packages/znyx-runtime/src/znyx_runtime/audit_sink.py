"""Egress audit sink (F0.4).

The zero-DB runtime cannot write control-plane tables, but F4 requires that every
boundary-crossing call (remote_llm / remote_api / a hosted inference sidecar) is
audited — and that a *failed* audit write never silently permits an un-audited
egress. This module defines the runtime-side abstraction the F4 egress gate calls
*before* any such call:

    event = EgressAuditEvent(...metadata only...)
    await sink.record(event)        # raises (fail-closed) if it cannot durably record
    # ...only now make the egress call...

Backends (selectable, see ``make_audit_sink``):
  * ``SpoolAuditSink``  — durable append-only JSON-lines spool on local disk
        (default ``~/.znyx/egress-audit.spool``). Works without a database, so it
        is valid in OSS/local mode. The control plane drains the spool into
        ``egress_events`` rows and hash-chain-signs them (F4).
  * ``NoopAuditSink``   — explicit opt-out (records nothing).

Failure semantics: in ``fail_mode="closed"`` (the default) a write failure raises
``AuditWriteError`` so the caller denies the egress. In ``fail_mode="open"`` the
failure is logged and swallowed (the egress proceeds un-audited) — only for
operators who explicitly accept that risk.

Privacy: events carry METADATA ONLY (destination host/region, byte count, redaction
flag, detector key, model/rubric ids) — never raw prompt/response content, mirroring
the telemetry posture. Audit is NOT gated on GDPR telemetry consent (different lawful
basis: security/compliance logging).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_SPOOL = Path.home() / ".znyx" / "egress-audit.spool"


class AuditWriteError(RuntimeError):
    """Raised by a fail-closed sink when an audit event cannot be durably recorded."""


@dataclass
class EgressAuditEvent:
    """A single boundary-crossing audit record. Metadata only — no raw content."""
    occurred_at: str                                  # ISO-8601 timestamp (caller-supplied)
    mode: str                                         # remote_llm | remote_api | local_ml(hosted) | ...
    detector_key: str                                 # e.g. "jailbreak"
    destination_host: str                             # host the call left the boundary to
    org_scope: Optional[str] = None                   # tenant/org identifier (no PII)
    trace_id: Optional[str] = None
    destination_region: Optional[str] = None
    bytes_out: int = 0                                # size of the payload sent (count, not content)
    redacted: bool = False                            # was redact_before_egress applied
    allowlisted: bool = False                         # destination matched the per-detector allowlist
    model_version: Optional[str] = None               # model_id@revision the call targeted
    event_id: Optional[str] = None                    # stable UUID → egress_events PK (idempotent drain)

    def to_dict(self) -> dict:
        return asdict(self)


class AuditSink(ABC):
    """Records egress audit events. ``record`` MUST raise in fail-closed mode if it
    cannot durably persist the event, so the caller can deny the egress."""

    def __init__(self, fail_mode: str = "closed"):
        self.fail_mode = fail_mode if fail_mode in ("closed", "open") else "closed"

    @abstractmethod
    async def record(self, event: EgressAuditEvent) -> None:
        ...

    @abstractmethod
    def record_sync(self, event: EgressAuditEvent) -> None:
        """Synchronous durable record, for the (synchronous) F4 escalation gate which
        cannot await. MUST raise in fail-closed mode if it cannot durably persist, so
        the gate denies the egress."""
        ...

    def _handle_failure(self, exc: Exception) -> None:
        if self.fail_mode == "closed":
            raise AuditWriteError(f"egress audit write failed (fail-closed): {exc}") from exc
        logger.error("egress audit write failed (fail-open — egress NOT audited): %s", exc)


class NoopAuditSink(AuditSink):
    """Records nothing. Explicit opt-out only."""

    async def record(self, event: EgressAuditEvent) -> None:
        return None

    def record_sync(self, event: EgressAuditEvent) -> None:
        return None


class SpoolAuditSink(AuditSink):
    """Durable append-only JSON-lines spool on local disk.

    Each ``record`` appends one JSON line and fsyncs before returning, so a recorded
    event survives a crash. The control plane drains and signs the spool (F4).
    """

    def __init__(self, spool_path: Optional[str] = None, fail_mode: str = "closed"):
        super().__init__(fail_mode=fail_mode)
        self.path = Path(spool_path) if spool_path else _DEFAULT_SPOOL
        self._lock = asyncio.Lock()

    def _append_sync(self, line: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Append + flush + fsync so the record is durable before we return.
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    async def record(self, event: EgressAuditEvent) -> None:
        line = json.dumps(event.to_dict(), separators=(",", ":"), sort_keys=True)
        try:
            async with self._lock:
                await asyncio.to_thread(self._append_sync, line)
        except Exception as exc:  # noqa: BLE001 — any IO failure must hit the fail-mode gate
            self._handle_failure(exc)

    def record_sync(self, event: EgressAuditEvent) -> None:
        line = json.dumps(event.to_dict(), separators=(",", ":"), sort_keys=True)
        try:
            self._append_sync(line)
        except Exception as exc:  # noqa: BLE001 — any IO failure must hit the fail-mode gate
            self._handle_failure(exc)

    def read_all(self) -> List[dict]:
        """Read every spooled event (used by the F4 control-plane drainer / tests).

        Resilient to a single corrupt/partial line (e.g. an interrupted write): a
        line that won't parse is logged and skipped so one bad record can't abort the
        whole drain."""
        if not self.path.exists():
            return []
        out: List[dict] = []
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                logger.warning("skipping unparseable egress spool line: %s", exc)
        return out


def make_audit_sink(
    mode: str = "spool",
    fail_mode: str = "closed",
    spool_path: Optional[str] = None,
) -> AuditSink:
    """Construct an audit sink from runtime configuration."""
    if mode == "noop":
        return NoopAuditSink(fail_mode=fail_mode)
    if mode == "spool":
        return SpoolAuditSink(spool_path=spool_path, fail_mode=fail_mode)
    logger.warning("Unknown audit_sink_mode %r — defaulting to durable spool", mode)
    return SpoolAuditSink(spool_path=spool_path, fail_mode=fail_mode)


def make_audit_egress_sink(sink: "AuditSink"):
    """Adapt an F0.4 ``AuditSink`` to the (synchronous) egress-gate callback the
    escalation engine calls before any boundary-crossing call.

    The escalation engine passes an ``znyx_core.engine.egress.EgressEvent``; we map
    it to an ``EgressAuditEvent`` and ``record_sync`` it. A fail-closed sink raises
    on a failed write, which the gate turns into a denial (no un-audited egress)."""
    def _sink(ev) -> None:
        sink.record_sync(EgressAuditEvent(
            occurred_at=ev.occurred_at,
            mode=ev.mode,
            detector_key=ev.detector_key,
            destination_host=ev.destination_host or "",
            org_scope=ev.org_scope,
            trace_id=ev.trace_id,
            destination_region=ev.destination_region,
            bytes_out=ev.bytes_out,
            redacted=ev.redacted,
            allowlisted=ev.allowlisted,
            model_version=ev.model_version,
            event_id=ev.event_id,
        ))
    return _sink
