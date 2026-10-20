"""Runtime-side judge audit + cached deny-of-wallet.

The stateless runtime can run a co-located judge (a local_llm sidecar reached at a
localhost endpoint in its policy), but it has no DB to write the judge audit trail or to
read CP-wide budget spend. This module is the runtime counterpart of the CP judge bridge,
built on the SAME durable-spool + CP-drain transport used for egress audit:

* ``JudgeAuditSpool`` — a durable append-only JSON-lines spool (fsync per line), mirroring
  ``runtime.audit_sink.SpoolAuditSink`` but carrying judge-call event dicts. The control
  plane drains it into ``judge_audit_events`` rows (``judge_eval_bridge.drain_judge_spool``).
* ``RuntimeJudgeAudit`` — wires that spool as the per-request judge ``audit_sink`` AND keeps
  the running spend in the shared state store (``znyx_core.state_store``) that backs the
  deny-of-wallet check: budget CAPS are delivered in the bundle
  (``policy.runtime_policy.judge_budgets``); spend is a per-budget, per-window counter
  shared by every replica on the same store, read at call time. It is eventually
  consistent with the CP (the drain reconciles it) — the documented trade-off of judging
  on the stateless path. If the store is unreachable in fail-closed mode, judge calls are
  denied (judges are optional; the detector falls back).

Audit here is POST-call (the judge already ran), so a spool write failure is logged and
swallowed rather than denying anything — unlike the egress gate, which is pre-call.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from znyx_core.engine.judge_runtime import (
    JudgeExecutionContext, make_store_budget_check, policy_uses_judge, record_judge_spend,
)
from znyx_core.state_store import StateUnavailable
from znyx_core.utils.spool import append_lines

logger = logging.getLogger(__name__)

_DEFAULT_JUDGE_SPOOL = Path.home() / ".znyx" / "judge-audit.spool"
_JUDGE_PRICE_PER_1K = float(os.getenv("JUDGE_PRICE_PER_1K_TOKENS", "0.01"))


def _estimate_cost(prompt_tokens: int = 0, completion_tokens: int = 0) -> float:
    """Blended token-cost estimate, matching the CP's ``estimate_cost_usd`` so the local
    spend tally and the CP-drained rows agree."""
    total = max(0, int(prompt_tokens or 0)) + max(0, int(completion_tokens or 0))
    return round(total / 1000.0 * _JUDGE_PRICE_PER_1K, 6)


class JudgeAuditSpool:
    """Durable append-only JSON-lines spool of judge-call events. The CP drains it into
    ``judge_audit_events`` (idempotent on a stable id). Mirrors ``SpoolAuditSink``."""

    def __init__(self, spool_path: Optional[str] = None):
        self.path = Path(spool_path) if spool_path else _DEFAULT_JUDGE_SPOOL
        self._lock = threading.Lock()

    def _append_sync(self, line: str) -> None:
        # Locked append: several worker processes share this spool and one rotates it.
        append_lines(self.path, [line], fsync=True)

    def record(self, event: Dict[str, Any]) -> None:
        """Durably append one judge event. Best-effort: a write failure is logged and
        swallowed (the judge call already happened — there's nothing to deny post-hoc)."""
        line = json.dumps(event, separators=(",", ":"), sort_keys=True, default=str)
        try:
            with self._lock:
                self._append_sync(line)
        except Exception as exc:  # noqa: BLE001 — post-call audit; never fail the response
            logger.warning("judge audit spool write failed (event dropped): %s", exc)

    def read_all(self) -> List[Dict[str, Any]]:
        """Every spooled event (used by the CP drainer / tests). A corrupt/partial line is
        skipped so one bad record can't abort the drain."""
        if not self.path.exists():
            return []
        out: List[Dict[str, Any]] = []
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                logger.warning("skipping unparseable judge spool line: %s", exc)
        return out


class RuntimeJudgeAudit:
    """Process-global runtime judge audit + shared-store deny-of-wallet. One instance
    per runtime.

    ``context_for`` builds the per-request ``JudgeExecutionContext``: its audit sink spools
    durably (and adds the call to the shared spend counters) and its budget check reads the
    bundle-delivered caps against the store's current totals."""

    def __init__(self, spool_path: Optional[str] = None, enabled: bool = True):
        self.enabled = enabled
        self.spool = JudgeAuditSpool(spool_path)

    # -- spend tally (feeds the deny-of-wallet) ------------------------------

    def _record(self, event: Dict[str, Any], caps: Optional[List[Dict[str, Any]]] = None,
                org_scope: Optional[str] = None) -> None:
        self.spool.record(event)
        # The synthesized consensus row is excluded from spend (its members already count),
        # matching JudgeAuditService.spend on the CP side.
        if event.get("is_consensus_result") or not caps:
            return
        cost = event.get("cost_usd")
        if cost is None:
            cost = _estimate_cost(event.get("prompt_tokens", 0), event.get("completion_tokens", 0))
        try:
            record_judge_spend(caps, event.get("env") or "*", event.get("detector_key") or "*",
                               float(cost or 0.0), org_scope)
        except StateUnavailable as exc:
            # Post-call: the call already happened; the check fails closed on its own.
            logger.warning("judge spend not recorded (state store unavailable): %s", exc)

    # -- per-request context -------------------------------------------------

    def context_for(self, policy: Dict[str, Any], env: str,
                    org_scope: Optional[str] = None) -> Optional[JudgeExecutionContext]:
        """Build the per-request judge context, or None when judges aren't used / audit is
        disabled. ``org_scope`` (the request tenant/org) is stamped on each spooled event so
        the CP drain can attribute the row, and scopes the shared spend counters; ``env`` is
        stamped when the sink didn't carry it."""
        if not self.enabled or not policy_uses_judge(policy):
            return None
        caps = (policy.get("runtime_policy") or {}).get("judge_budgets") or []

        def _sink(ev: Dict[str, Any]) -> None:
            e = dict(ev)
            if org_scope and not e.get("org_scope"):
                e["org_scope"] = org_scope
            if not e.get("env"):
                e["env"] = env
            self._record(e, caps, org_scope)

        return JudgeExecutionContext(
            audit_sink=_sink,
            budget_check=make_store_budget_check(caps, env, org_scope),
            provider_caller=None,   # local judge reached via its policy endpoint (sidecar)
        )
