"""Per-request judge wiring shared by the control-plane and runtime composition roots.

The LLM-judge subsystem (consensus caller, deny-of-wallet budgets, audit trail) is built
and unit-tested, but the shared evaluator/orchestrator that run in BOTH the control plane
and the stateless runtime can't import control-plane internals (DB, secret store). This
module is the dependency-free seam between them:

* ``JudgeExecutionContext`` — the per-request bundle a composition root injects into
  ``GuardrailsEvaluator.evaluate(..., judge_ctx=...)``. Every callable is **synchronous**
  (the judge bridge runs on a worker thread), so the audit sink *collects* events that the
  composition root drains afterwards, and the budget check reads a *snapshot* the root
  precomputed — neither touches a DB on the hot path.
* ``budget_allows`` — the pure deny-of-wallet decision over a budget snapshot. The control
  plane builds the snapshot from ``judge_budgets`` rows; the runtime builds it from a
  bundle-delivered snapshot. Same logic, same answer.
* ``policy_uses_judge`` — cheap predicate so a composition root only does the (more
  expensive) judge setup when a request's policy actually uses judges.
* ``build_escalation_judge_caller`` — turns a detector's raw policy config into the
  escalation engine's ``judge_caller`` (multi-judge consensus + audit + budget), sourcing
  the rubric/members/method that ``build_strategy`` drops from the typed backend.

No control-plane imports here — that's the whole point.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Env-var fallback per provider — when the bundle carries no auth_value the runtime
# reads the standard key env var so operators can supply credentials via Docker env.
_PROVIDER_KEY_ENVS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}

# Execution modes that mean "call an LLM judge" (vs ml/deterministic). Mirrors
# escalation._LLM_MODES + remote_api so the predicate and the caller builder agree.
_JUDGE_MODES = ("local_llm", "remote_llm", "remote_api")


@dataclass
class JudgeExecutionContext:
    """Per-request judge wiring injected by a composition root (CP route / runtime).

    All callables are SYNC and safe to call from the evaluator's worker-thread judge
    bridge. A ``None`` field means that capability is off (e.g. no budget configured →
    no veto). ``provider_caller`` lets a root supply the actual provider transport — the
    CP leaves it ``None`` (the judge runtime's default HTTP caller is used), tests inject
    a fake, and the runtime injects a co-located local-model caller.
    """
    audit_sink: Optional[Callable[[dict], None]] = None        # records ONE judge call; org/env stamped at drain
    budget_check: Optional[Callable[[str], bool]] = None       # (detector_key) -> allowed
    rubrics: Optional[Dict[str, str]] = None                   # quality metric -> rubric text (org overrides)
    rubric_versions: Optional[Dict[str, str]] = None           # quality metric -> rubric version
    provider_caller: Optional[Callable] = None                 # llm.judge ProviderCaller (None = default transport)
    consensus_members: int = 1                                 # default member count for escalation judges
    consensus_method: str = "majority"                         # default vote method for escalation judges


# ---------------------------------------------------------------------------
# Deny-of-wallet: pure budget decision over a precomputed snapshot.
# ---------------------------------------------------------------------------
# A snapshot entry is a plain dict (NOT an ORM row) so the same function serves the CP
# (snapshot from judge_budgets) and the runtime (snapshot delivered in the bundle):
#   {"env": str, "detector_key": str, "max_cost_usd": float|None,
#    "max_calls": int|None, "window_seconds": int,
#    "spend_usd": float, "spend_calls": int}
# ``env``/``detector_key`` use "*" as the "applies to all" wildcard, matching JudgeBudget.


def _specificity(env: str, detector_key: str) -> int:
    """Higher = more specific. Mirrors JudgeBudgetService.resolve: env match is worth more
    than detector match so (env, *) beats (*, det)."""
    return (0 if env == "*" else 2) + (0 if detector_key == "*" else 1)


def budget_allows(snapshot: List[Dict[str, Any]], env: str, detector_key: str) -> bool:
    """Deny-of-wallet gate: is another judge call within budget for (env, detector_key)?

    Resolves the most-specific matching budget — (env, det) > (env, *) > (*, det) > (*, *) —
    and denies if its cost or call cap is already met over the window. No matching budget →
    allowed (no cap configured). Pure + synchronous so it runs on the judge worker thread.
    Matches ``JudgeBudgetService.check`` exactly (``>=`` comparison, cost then calls)."""
    candidates = [
        b for b in (snapshot or [])
        if b.get("env") in (env, "*") and b.get("detector_key") in (detector_key, "*")
    ]
    if not candidates:
        return True
    budget = max(candidates, key=lambda b: _specificity(b.get("env", "*"), b.get("detector_key", "*")))
    max_cost = budget.get("max_cost_usd")
    if max_cost is not None and float(budget.get("spend_usd", 0.0)) >= float(max_cost):
        return False
    max_calls = budget.get("max_calls")
    if max_calls is not None and int(budget.get("spend_calls", 0)) >= int(max_calls):
        return False
    return True


def make_snapshot_budget_check(snapshot: List[Dict[str, Any]], env: str) -> Callable[[str], bool]:
    """Bind a budget snapshot + request env into the sync ``(detector_key) -> allowed``
    closure the evaluator/judge path calls. The quality path passes ``judge:<metric>``;
    the escalation path passes the detector key."""
    def _check(detector_key: str) -> bool:
        return budget_allows(snapshot, env, detector_key)
    return _check


# ---------------------------------------------------------------------------
# Judge-usage predicate — keep judge setup off the non-judge hot path.
# ---------------------------------------------------------------------------

def policy_uses_judge(policy: Optional[Dict[str, Any]]) -> bool:
    """True if a resolved policy would invoke an LLM judge: quality judge_mode with a
    judge block, or any enabled detector whose strategy escalates to a judge mode."""
    if not isinstance(policy, dict):
        return False
    qs = policy.get("quality_scoring")
    if isinstance(qs, dict) and qs.get("judge_mode") and isinstance(qs.get("judge"), dict):
        return True
    for cfg in policy.values():
        if not isinstance(cfg, dict) or not cfg.get("enabled", False):
            continue
        strat = cfg.get("strategy")
        if isinstance(strat, dict):
            order = strat.get("order") or []
            if any(m in _JUDGE_MODES for m in order):
                return True
    return False


# ---------------------------------------------------------------------------
# Escalation judge caller — built from a detector's RAW config.
# ---------------------------------------------------------------------------

def judge_params_from_config(policy_key: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract the judge-call parameters for an escalation detector from its raw policy
    config. ``build_strategy`` only copies the typed ``_BACKEND_FIELDS`` into a
    DetectorBackend (model/endpoint/provider), DROPPING the judge-only keys
    (members/method/rubric), so the consensus caller is sourced here from the raw block.

    The rubric is the detector config's explicit ``rubric`` (or ``judge.rubric``), else the
    built-in JUDGE_CANDIDATES rubric for this detector key. Returns None when the config
    has no judge backend or no resolvable rubric (escalation then uses the generic transport)."""
    backends = config.get("backends")
    if not isinstance(backends, dict):
        logger.warning("judge_params: %s — no backends dict in config (keys: %s)",
                       policy_key, list(config.keys()) if isinstance(config, dict) else None)
        return None
    for mode in _JUDGE_MODES:
        block = backends.get(mode)
        if not isinstance(block, dict):
            continue
        # A judge backend is one explicitly flagged judge=True, naming a provider, or simply
        # a local/remote LLM mode (those are judges by definition in this engine).
        if not (block.get("judge") or block.get("provider") or mode in ("local_llm", "remote_llm")):
            continue
        rubric = config.get("rubric")
        if not rubric and isinstance(config.get("judge"), dict):
            rubric = config["judge"].get("rubric")
        if not rubric:
            from znyx_core.engine.judge_catalog import JUDGE_CANDIDATES
            cand = JUDGE_CANDIDATES.get(policy_key) or JUDGE_CANDIDATES.get(f"{policy_key}_judge")
            rubric = cand.rubric if cand else None
        if not rubric:
            return None
        provider = block.get("provider") or "openai"
        api_key = (block.get("auth_value") or block.get("api_key")
                   or os.getenv(_PROVIDER_KEY_ENVS.get(provider, ""), ""))
        return {
            "mode": mode,
            "rubric": rubric,
            "provider": provider,
            "model": block.get("model") or block.get("model_id") or "",
            "endpoint_url": block.get("endpoint_url"),
            "api_key": api_key,
            "members": max(1, int(block.get("members", 1) or 1)),
            "method": block.get("method") or "majority",
        }
    return None


def build_escalation_judge_caller(policy_key: str, config: Dict[str, Any],
                                  judge_ctx: JudgeExecutionContext, request: Any) -> Optional[Callable]:
    """Build the escalation engine's ``judge_caller`` for a detector, or None if the
    detector has no judge backend / rubric (escalation then falls back to the generic
    backend transport). Wires the injected audit sink, budget check, and provider caller."""
    params = judge_params_from_config(policy_key, config)
    if params is None:
        return None
    from znyx_core.engine.judge_escalation import make_judge_consensus_caller
    members = params["members"] if params["members"] > 1 else judge_ctx.consensus_members
    method = params["method"] if params["method"] != "majority" else judge_ctx.consensus_method
    return make_judge_consensus_caller(
        params["rubric"],
        members=members,
        method=method,
        provider=params["provider"],
        model=params["model"],
        endpoint_url=params["endpoint_url"],
        api_key=params["api_key"],
        detector_key=policy_key,
        caller=judge_ctx.provider_caller,
        audit_sink=judge_ctx.audit_sink,
        budget_check=judge_ctx.budget_check,
        request=request,
    )
