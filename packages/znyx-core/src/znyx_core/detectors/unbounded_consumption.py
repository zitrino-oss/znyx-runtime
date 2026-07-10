"""Unbounded-consumption detector (OWASP LLM10).

Caps runaway resource use: per-session cumulative token and cost budgets, agent-loop
iteration (depth) caps, and tool-call depth caps. Stateful per (tenant, app, session) —
mirrors the ``abuse`` detector's in-memory per-tenant state pattern (resets on restart;
use a shared store for multi-instance deployments).

Runs in the ``agent_loop`` stage (per-iteration budget/depth) and in ``input``/``output``
(cumulative token/cost accounting). Budget signals are read from request metadata:

    metadata = {
        "tokens": int,          # tokens consumed by THIS request (caller-reported)
        "cost_usd": float,      # incremental cost of this request
        "agent_step": {"iteration": int, "max_iterations": int, "tool_depth": int},
    }

Token/cost budgets are enforced on caller-REPORTED usage only — the detector does not
estimate tokens from text length (that inflated normal traffic). It is therefore
advisory with respect to a fully-cooperative caller; iteration/tool-depth caps likewise
trust the reported step counters. Once a session crosses a budget it stays blocked for
the remainder of ``session_window_seconds`` (the window then resets).
"""
import math
import time
from typing import Any, Dict, List, Optional

from znyx_core.core.models import Decision, DetectorResult, RuleHit, Severity


def _coerce_nonneg_int(v: Any) -> int:
    """Caller-reported token count → non-negative int. Non-numeric / non-finite (inf, nan)
    coerce to 0 — never raise (``int(float('inf'))`` would OverflowError and crash the
    evaluation pipeline; JSON ``Infinity`` is reachable over the wire)."""
    if v is None:
        return 0
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(f):
        return 0
    return max(0, int(f))


def _coerce_nonneg_float(v: Any) -> float:
    """Caller-reported cost → non-negative finite float (inf/nan/non-numeric → 0.0)."""
    if v is None:
        return 0.0
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(f):
        return 0.0
    return max(0.0, f)


class UnboundedConsumptionDetector:
    """Per-session token/cost budgets + agent-loop depth caps (LLM10)."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.enabled = self.config.get('enabled', False)
        # Default BLOCK (stop the runaway); an org can set WARN to monitor only.
        self.action = (self.config.get('action') or 'BLOCK').upper()
        self.max_session_tokens = self.config.get('max_session_tokens', 200_000)
        self.max_session_cost_usd = self.config.get('max_session_cost_usd', 10.0)
        self.max_iterations = self.config.get('max_iterations', 50)
        self.max_tool_depth = self.config.get('max_tool_depth', 25)
        self.session_window_seconds = self.config.get('session_window_seconds', 3600)
        # key -> {"tokens": int, "cost": float, "first_ts": float}
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._last_cleanup = time.time()

    @staticmethod
    def _key(tenant_id: str, app_id: str, session_id: Optional[str],
             user_id: Optional[str] = None) -> Optional[str]:
        """Per-identity cumulative-budget key: session > user. Returns None when neither is
        known — token/cost budgets are then NOT accumulated (a shared tenant:app bucket
        would cross-contaminate unrelated users); per-request iteration/depth caps still run."""
        if session_id:
            return f"{tenant_id}:{app_id}:s:{session_id}"
        if user_id:
            return f"{tenant_id}:{app_id}:u:{user_id}"
        return None

    def _cleanup(self, now: float) -> None:
        if now - self._last_cleanup < 300:
            return
        stale = [k for k, v in self._sessions.items()
                 if now - v.get('first_ts', now) > self.session_window_seconds]
        for k in stale:
            del self._sessions[k]
        self._last_cleanup = now

    def detect(self, text: str, tenant_id: str = "", app_id: str = "",
               session_id: Optional[str] = None, context: str = "input",
               metadata: Optional[Dict[str, Any]] = None,
               user_id: Optional[str] = None) -> DetectorResult:
        if not self.enabled:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        # Guard against a non-dict metadata (list/str/int) reaching .get() — an internal/SDK
        # caller could pass one (the API path is already protected by Pydantic's Optional[Dict]).
        meta = metadata if isinstance(metadata, dict) else {}
        now = time.time()
        self._cleanup(now)

        rule_hits: List[RuleHit] = []

        # Caller-reported usage only (never estimated from text length — estimating inflated
        # ordinary traffic). 0 is honoured as 0; inf/nan/non-numeric coerce to 0 (no crash).
        req_tokens = _coerce_nonneg_int(meta.get('tokens', meta.get('total_tokens')))
        req_cost = _coerce_nonneg_float(meta.get('cost_usd'))

        # A per-identity key (session > user) enables CUMULATIVE accounting. Without one we
        # cannot attribute usage to an identity, so a shared tenant:app bucket is unsafe (it
        # would let one user's usage block an unrelated user). Instead we check THIS request
        # alone — so a single anonymous runaway is still caught, with no cross-contamination.
        key = self._key(tenant_id, app_id, session_id, user_id)
        if key is not None:
            state = self._sessions.get(key)
            if state is None or (now - state.get('first_ts', now)) > self.session_window_seconds:
                state = {"tokens": 0, "cost": 0.0, "first_ts": now}
                self._sessions[key] = state
            state['tokens'] += req_tokens
            state['cost'] += req_cost
            total_tokens, total_cost = state['tokens'], state['cost']
        else:
            total_tokens, total_cost = req_tokens, req_cost

        if total_tokens > self.max_session_tokens:
            rule_hits.append(RuleHit(
                rule_id="unbounded_consumption.token_budget_exceeded", severity=Severity.HIGH,
                message=f"Token budget exceeded: {total_tokens} > {self.max_session_tokens}",
            ))
        if total_cost > self.max_session_cost_usd:
            rule_hits.append(RuleHit(
                rule_id="unbounded_consumption.cost_budget_exceeded", severity=Severity.HIGH,
                message=f"Cost budget exceeded: ${total_cost:.4f} > ${self.max_session_cost_usd}",
            ))

        # Agent-loop depth / iteration caps (only meaningful when the caller reports them).
        step = meta.get('agent_step') if isinstance(meta.get('agent_step'), dict) else {}
        iteration = step.get('iteration', meta.get('iteration'))
        caller_max = step.get('max_iterations', meta.get('max_iterations'))
        tool_depth = step.get('tool_depth', meta.get('tool_depth'))

        # bool is an int subclass — exclude it so a stray True/False can't be read as depth.
        if isinstance(iteration, int) and not isinstance(iteration, bool):
            effective_cap = self.max_iterations
            if isinstance(caller_max, int) and not isinstance(caller_max, bool) and caller_max > 0:
                effective_cap = min(effective_cap, caller_max)
            if iteration >= effective_cap:
                rule_hits.append(RuleHit(
                    rule_id="unbounded_consumption.iteration_cap_exceeded", severity=Severity.HIGH,
                    message=f"Agent-loop iteration {iteration} reached cap {effective_cap}",
                ))
        if isinstance(tool_depth, int) and not isinstance(tool_depth, bool) and tool_depth > self.max_tool_depth:
            rule_hits.append(RuleHit(
                rule_id="unbounded_consumption.tool_depth_exceeded", severity=Severity.HIGH,
                message=f"Tool-call depth {tool_depth} exceeds cap {self.max_tool_depth}",
            ))

        if not rule_hits:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        risk_score = min(100, len(rule_hits) * 50)
        if self.action == "BLOCK":
            return DetectorResult(
                decision=Decision.BLOCK,
                risk_score=risk_score,
                rule_hits=rule_hits,
                user_message="Request blocked: resource/usage budget exceeded for this session.",
                developer_message=f"unbounded_consumption: {', '.join(h.rule_id for h in rule_hits)}",
            )
        return DetectorResult(
            decision=Decision.WARN,
            risk_score=risk_score,
            rule_hits=rule_hits,
            developer_message=f"unbounded_consumption (advisory): {', '.join(h.rule_id for h in rule_hits)}",
        )
