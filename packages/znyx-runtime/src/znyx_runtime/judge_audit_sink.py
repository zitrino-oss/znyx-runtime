"""Runtime-side judge audit + cached deny-of-wallet.

The stateless runtime can run a co-located judge (a local_llm sidecar reached at a
localhost endpoint in its policy), but it has no DB to write the judge audit trail or to
read CP-wide budget spend. This module is the runtime counterpart of the CP judge bridge,
built on the SAME durable-spool + CP-drain transport used for egress audit:

* ``JudgeAuditSpool`` — a durable append-only JSON-lines spool (fsync per line), mirroring
  ``runtime.audit_sink.SpoolAuditSink`` but carrying judge-call event dicts. The control
  plane drains it into ``judge_audit_events`` rows (``judge_eval_bridge.drain_judge_spool``).
* ``RuntimeJudgeAudit`` — wires that spool as the per-request judge ``audit_sink`` AND keeps
  a process-local spend counter that backs a *cached* deny-of-wallet check: budget CAPS are
  delivered in the bundle (``policy.runtime_policy.judge_budgets``) and the running spend is
  the runtime's own since-start tally. This is eventually-consistent with the CP (the drain
  reconciles it) — the documented trade-off of judging on the stateless path.

Audit here is POST-call (the judge already ran), so a spool write failure is logged and
swallowed rather than denying anything — unlike the egress gate, which is pre-call.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from znyx_core.engine.judge_runtime import (
    JudgeExecutionContext, make_snapshot_budget_check, policy_uses_judge,
)

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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

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
    """Process-global runtime judge audit + cached budget. One instance per runtime.

    ``context_for`` builds the per-request ``JudgeExecutionContext``: its audit sink spools
    durably (and updates the local spend tally) and its budget check reads the bundle-
    delivered caps against that tally."""

    def __init__(self, spool_path: Optional[str] = None, enabled: bool = True):
        self.enabled = enabled
        self.spool = JudgeAuditSpool(spool_path)
        self._spend: Dict[Tuple[str, str], Dict[str, float]] = {}
        self._lock = threading.Lock()

    # -- spend tally (feeds the cached deny-of-wallet) -----------------------

    def _record(self, event: Dict[str, Any]) -> None:
        self.spool.record(event)
        # The synthesized consensus row is excluded from spend (its members already count),
        # matching JudgeAuditService.spend on the CP side.
        if event.get("is_consensus_result"):
            return
        cost = event.get("cost_usd")
        if cost is None:
            cost = _estimate_cost(event.get("prompt_tokens", 0), event.get("completion_tokens", 0))
        env = event.get("env") or "*"
        det = event.get("detector_key") or "*"
        with self._lock:
            slot = self._spend.setdefault((env, det), {"cost_usd": 0.0, "calls": 0})
            slot["cost_usd"] += float(cost or 0.0)
            slot["calls"] += 1

    def _local_spend(self, cap_env: str, cap_det: str) -> Tuple[float, int]:
        """Running spend across the tally entries a (cap_env, cap_det) budget covers."""
        cost, calls = 0.0, 0
        with self._lock:
            for (env, det), v in self._spend.items():
                if cap_env in ("*", env) and cap_det in ("*", det):
                    cost += v["cost_usd"]
                    calls += v["calls"]
        return cost, calls

    def _snapshot(self, caps: List[Dict[str, Any]], env: str) -> List[Dict[str, Any]]:
        """Merge bundle-delivered caps with the local spend tally into the plain-dict
        snapshot ``budget_allows`` consumes (same shape the CP builds from the DB)."""
        snapshot: List[Dict[str, Any]] = []
        for c in caps or []:
            cenv = c.get("env", "*")
            cdet = c.get("detector_key", "*")
            spend_usd, spend_calls = self._local_spend(cenv, cdet)
            snapshot.append({
                "env": cenv, "detector_key": cdet,
                "max_cost_usd": c.get("max_cost_usd"), "max_calls": c.get("max_calls"),
                "window_seconds": c.get("window_seconds", 86400),
                "spend_usd": spend_usd, "spend_calls": spend_calls,
            })
        return snapshot

    # -- per-request context -------------------------------------------------

    def context_for(self, policy: Dict[str, Any], env: str,
                    org_scope: Optional[str] = None) -> Optional[JudgeExecutionContext]:
        """Build the per-request judge context, or None when judges aren't used / audit is
        disabled. ``org_scope`` (the request tenant/org) is stamped on each spooled event so
        the CP drain can attribute the row; ``env`` is stamped when the sink didn't carry it."""
        if not self.enabled or not policy_uses_judge(policy):
            return None
        caps = (policy.get("runtime_policy") or {}).get("judge_budgets") or []
        snapshot = self._snapshot(caps, env)

        def _sink(ev: Dict[str, Any]) -> None:
            e = dict(ev)
            if org_scope and not e.get("org_scope"):
                e["org_scope"] = org_scope
            if not e.get("env"):
                e["env"] = env
            self._record(e)

        return JudgeExecutionContext(
            audit_sink=_sink,
            budget_check=make_snapshot_budget_check(snapshot, env),
            provider_caller=None,   # local judge reached via its policy endpoint (sidecar)
        )
