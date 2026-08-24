"""Unbounded-consumption detector (OWASP LLM06).

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

It also caps THINKING tokens and detects agent loops that spin on the same state,
the two LLM06 surfaces the 2026 edition added:

    metadata = {
        "thinking_tokens": int,   # extended-thinking / reasoning tokens for THIS request
        "agent_step": {"state": "<opaque state key>"},
    }

Reasoning-loop exhaustion is the case a size limit cannot see: a short, ordinary-looking
prompt drives an extended-thinking model into a long reasoning run, so the request passes
every input-size check while burning a large budget. Three signals cover it — a per-request
thinking cap, a cumulative session cap, and a thinking-to-input RATIO, which is the one that
actually characterises the attack (tiny prompt, enormous reasoning).

Loop detection hashes the agent's reported step state and counts repeats within the session
window; the same state recurring past ``max_repeated_states`` is a loop that is making no
progress. It is deliberately confined to requests carrying ``agent_step`` — outside an agent
loop, a user resending identical text is ordinary retry behaviour, not an attack.

Token/cost budgets are enforced on caller-REPORTED usage only — the detector does not
estimate tokens from text length (that inflated normal traffic). It is therefore
advisory with respect to a fully-cooperative caller; iteration/tool-depth caps likewise
trust the reported step counters. Once a session crosses a budget it stays blocked for
the remainder of ``session_window_seconds`` (the window then resets).
"""
import hashlib
import math
import time
from collections import OrderedDict
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
    """Per-session token/cost budgets + agent-loop depth caps (LLM06)."""

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
        # Extended-thinking budgets. Defaults are generous: a legitimate hard reasoning
        # task can run long, so these are set to catch runaway, not deep thought.
        self.max_thinking_tokens_per_request = self.config.get(
            'max_thinking_tokens_per_request', 32_000)
        self.max_session_thinking_tokens = self.config.get(
            'max_session_thinking_tokens', 200_000)
        # Thinking tokens per input token. Only applied above `ratio_min_thinking_tokens`
        # so a 3-token prompt with a 200-token answer cannot trip a 50x ratio.
        self.max_thinking_to_input_ratio = self.config.get('max_thinking_to_input_ratio', 50.0)
        self.ratio_min_thinking_tokens = self.config.get('ratio_min_thinking_tokens', 4_000)
        # Same agent state seen this many times in the window → the loop is not progressing.
        self.max_repeated_states = self.config.get('max_repeated_states', 3)
        # Bounded so a long-lived session cannot grow this without limit.
        self.max_tracked_states = self.config.get('max_tracked_states', 64)
        # key -> {"tokens": int, "cost": float, "thinking": int,
        #         "states": OrderedDict[hash, count], "first_ts": float}
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
        # Extended-thinking usage. Accept both vendor spellings; same coercion rules as
        # tokens, so a JSON Infinity or a string cannot crash the pipeline.
        req_thinking = _coerce_nonneg_int(
            meta.get('thinking_tokens', meta.get('reasoning_tokens')))

        # A per-identity key (session > user) enables CUMULATIVE accounting. Without one we
        # cannot attribute usage to an identity, so a shared tenant:app bucket is unsafe (it
        # would let one user's usage block an unrelated user). Instead we check THIS request
        # alone — so a single anonymous runaway is still caught, with no cross-contamination.
        key = self._key(tenant_id, app_id, session_id, user_id)
        if key is not None:
            state = self._sessions.get(key)
            if state is None or (now - state.get('first_ts', now)) > self.session_window_seconds:
                state = {"tokens": 0, "cost": 0.0, "thinking": 0,
                         "states": OrderedDict(), "first_ts": now}
                self._sessions[key] = state
            state.setdefault('thinking', 0)      # tolerate state from an older process
            # OrderedDict so the loop-state table can evict its oldest entry when full.
            if not isinstance(state.get('states'), OrderedDict):
                state['states'] = OrderedDict(state.get('states') or {})
            state['tokens'] += req_tokens
            state['cost'] += req_cost
            state['thinking'] += req_thinking
            total_tokens, total_cost = state['tokens'], state['cost']
            total_thinking = state['thinking']
        else:
            total_tokens, total_cost = req_tokens, req_cost
            total_thinking = req_thinking

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

        # ── Thinking-token exhaustion (LLM06, 2026) ──────────────────────────────
        # A single request can burn a whole budget on its own, so the per-request cap is
        # checked against THIS request rather than the running total.
        if req_thinking > self.max_thinking_tokens_per_request:
            rule_hits.append(RuleHit(
                rule_id="unbounded_consumption.thinking_tokens_per_request_exceeded",
                severity=Severity.HIGH,
                message=(f"Thinking tokens {req_thinking} exceed per-request cap "
                         f"{self.max_thinking_tokens_per_request}"),
            ))
        if total_thinking > self.max_session_thinking_tokens:
            rule_hits.append(RuleHit(
                rule_id="unbounded_consumption.thinking_token_budget_exceeded",
                severity=Severity.HIGH,
                message=(f"Session thinking-token budget exceeded: {total_thinking} > "
                         f"{self.max_session_thinking_tokens}"),
            ))
        # The characteristic shape of the attack: a small prompt driving a long reasoning
        # run. Requires a known, non-zero input size, and only applies once thinking is
        # already substantial — otherwise a one-word prompt would trip any ratio.
        if (req_thinking >= self.ratio_min_thinking_tokens
                and req_tokens > 0
                and self.max_thinking_to_input_ratio > 0):
            ratio = req_thinking / req_tokens
            if ratio > self.max_thinking_to_input_ratio:
                rule_hits.append(RuleHit(
                    rule_id="unbounded_consumption.thinking_to_input_ratio_exceeded",
                    severity=Severity.HIGH,
                    message=(f"Thinking/input ratio {ratio:.1f}x exceeds "
                             f"{self.max_thinking_to_input_ratio}x "
                             f"({req_thinking} thinking vs {req_tokens} input tokens)"),
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

        # ── State-hash loop detection (LLM06 mitigation, 2026) ───────────────────
        # Confined to agent-loop requests on purpose: outside a loop, identical text is a
        # user retrying, not an agent spinning. Needs a session key too — without one there
        # is no history to compare against, and a shared bucket would collide across users.
        if step and key is not None:
            raw_state = step.get('state')
            if not isinstance(raw_state, str) or not raw_state.strip():
                raw_state = text or ""
            raw_state = raw_state.strip()
            if raw_state:
                digest = hashlib.sha256(raw_state.encode('utf-8', 'ignore')).hexdigest()[:16]
                states: "OrderedDict[str, int]" = self._sessions[key]['states']
                seen = states.get(digest, 0) + 1
                if digest in states:
                    states[digest] = seen
                    # Recently-seen states are the ones worth keeping, so touching an
                    # entry moves it to the end of the eviction order.
                    states.move_to_end(digest)
                else:
                    # EVICT the oldest rather than refuse the new state. Refusing turned a
                    # full table into a bypass: once max_tracked_states distinct states had
                    # been seen, every later state was recomputed as seen == 1 forever, so
                    # an agent could spin on state 65 indefinitely without ever tripping the
                    # rule. Eviction costs history, never detection of an active loop.
                    if len(states) >= self.max_tracked_states:
                        states.popitem(last=False)
                    states[digest] = seen
                if seen >= self.max_repeated_states:
                    rule_hits.append(RuleHit(
                        rule_id="unbounded_consumption.agent_loop_state_repeat",
                        severity=Severity.HIGH,
                        message=(f"Agent loop repeated the same state {seen} times "
                                 f"(cap {self.max_repeated_states}) — not progressing"),
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
