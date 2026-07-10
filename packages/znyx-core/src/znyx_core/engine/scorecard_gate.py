"""Scorecard gate — a machine-checkable quality bar a model-backed detector
(``ml_model`` / ``llm_judge``) must clear before it can be published, installed, or used
in *enforcement* (BLOCK/REDACT). "Enforce, don't just display."

Two tiers:
  * **advisory** — required to publish/install a detector at all (runs in WARN).
  * **enforcement** — stricter; required before a detector may BLOCK/REDACT. A detector
    that clears advisory but not enforcement is install-able but pinned to WARN.

Pure + dependency-free (stdlib only) so the SAME gate is evaluated at every enforcement
point: hub publish, org install, policy validation, bundle publish, and runtime
action resolution. A missing metric fails the gate — you can't certify what you didn't
measure.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

ADVISORY = "advisory"
ENFORCEMENT = "enforcement"

class ScorecardGateError(ValueError):
    """A model-backed detector failed its scorecard gate at a publish/install boundary.
    Carries the failing gate details so the API can return a clean 4xx (422)."""

    def __init__(self, message: str, *, detector: str = "", tier: str = "",
                 failures: Optional[list] = None):
        super().__init__(message)
        self.detector = detector
        self.tier = tier
        self.failures = failures or []


# Execution modes that make a detector "model-backed" (everything but pure deterministic).
# A detector whose strategy.order includes one of these is gated.
MODEL_BACKED_MODES = {"local_ml", "local_embedding", "local_llm", "remote_llm", "remote_api"}
# Actions that require the stricter ENFORCEMENT gate (vs advisory).
ENFORCING_ACTIONS = {"BLOCK", "REDACT"}


def is_model_backed(config: Any) -> bool:
    """True if a detector's policy config escalates to a model layer (strategy with a
    model-backed mode in its order) — i.e. it is subject to the scorecard gate."""
    if not isinstance(config, dict):
        return False
    strategy = config.get("strategy")
    if not isinstance(strategy, dict):
        return False
    order = strategy.get("order") or []
    return any(mode in MODEL_BACKED_MODES for mode in order)


def required_tier_for_action(action: Optional[str]) -> str:
    """The gate tier an action requires: BLOCK/REDACT → enforcement, else advisory."""
    return ENFORCEMENT if (action or "").upper() in ENFORCING_ACTIONS else ADVISORY


def model_versions_for(config: Any) -> List[str]:
    """The exact model version(s) a detector's strategy pins — one per model-backed mode
    in `strategy.order`, read from `backends.<mode>.model_id[@revision]`. An unpinned
    mode → the "default" sentinel. The gate must check the scorecard for THESE versions,
    not the latest, so a policy pinned to a failing model can't ride a newer model's
    passing scorecard. Deduped, order-preserving."""
    if not isinstance(config, dict):
        return []
    strategy = config.get("strategy") or {}
    order = strategy.get("order") or []
    backends = config.get("backends") or {}
    out: List[str] = []
    for mode in order:
        if mode not in MODEL_BACKED_MODES:
            continue
        spec = backends.get(mode) if isinstance(backends, dict) else None
        spec = spec or {}
        model_id = spec.get("model_id")
        revision = spec.get("revision")
        version = (f"{model_id}@{revision}" if model_id and revision
                   else (model_id or "default"))
        if version not in out:
            out.append(version)
    return out


def resolve_gated_action(action: Optional[str], enforcement_passed: bool) -> tuple[str, bool]:
    """Runtime action resolution: a BLOCK/REDACT from a model-backed detector whose
    enforcement gate did not pass is downgraded to WARN. Returns (action, downgraded)."""
    act = (action or "").upper()
    if act in ENFORCING_ACTIONS and not enforcement_passed:
        return "WARN", True
    return act, False

# A scorecard validated slightly in the future (clock skew) is tolerated; further out
# is rejected — a far-future date must not be used to dodge the staleness gate.
_CLOCK_SKEW_DAYS = 1.0

# Default bars. Enforcement is strictly tighter. Per-category overrides (below) make
# expert-labelled verticals (healthcare/legal/finance) stricter still.
_GATES: Dict[str, Dict[str, float]] = {
    ADVISORY: {
        "min_f1": 0.60, "min_auroc": 0.70, "max_ece": 0.15, "max_fp_rate": 0.15,
        "max_p95_latency_ms": 1500, "min_samples_per_language": 50, "max_validation_age_days": 365,
    },
    ENFORCEMENT: {
        "min_f1": 0.80, "min_auroc": 0.85, "max_ece": 0.10, "max_fp_rate": 0.05,
        "max_p95_latency_ms": 1000, "min_samples_per_language": 100, "max_validation_age_days": 180,
    },
}

# Category → multiplier-ish overrides (absolute values win). Verticals with expert labels
# demand higher F1/AUROC and lower FP/ECE.
_CATEGORY_OVERRIDES: Dict[str, Dict[str, float]] = {
    "healthcare": {"min_f1": 0.88, "min_auroc": 0.92, "max_fp_rate": 0.02, "max_ece": 0.07},
    "legal": {"min_f1": 0.85, "min_auroc": 0.90, "max_fp_rate": 0.03, "max_ece": 0.08},
    "finance": {"min_f1": 0.85, "min_auroc": 0.90, "max_fp_rate": 0.03, "max_ece": 0.08},
}


@dataclass
class GateResult:
    passed: bool
    tier: str
    failures: List[Dict[str, Any]]   # [{metric, required, actual, op}]
    gate: Dict[str, float]

    def to_dict(self) -> dict:
        return {"passed": self.passed, "tier": self.tier,
                "failures": self.failures, "gate": self.gate}


def gate_for(tier: str, category: Optional[str] = None) -> Dict[str, float]:
    # Reject an unknown tier — this helper sits on enforcement paths, so a typo must NOT
    # silently fall back to the laxer advisory bar.
    if tier not in _GATES:
        raise ValueError(f"unknown scorecard gate tier '{tier}' (expected {ADVISORY!r} or {ENFORCEMENT!r})")
    base = dict(_GATES[tier])
    if category and category in _CATEGORY_OVERRIDES:
        base.update(_CATEGORY_OVERRIDES[category])
    return base


def _fail(failures, metric, required, actual, op):
    failures.append({"metric": metric, "required": required, "actual": actual, "op": op})


def evaluate_gate(scorecard: Dict[str, Any], *, tier: str = ENFORCEMENT,
                  category: Optional[str] = None, now: Optional[datetime] = None) -> GateResult:
    """Evaluate a scorecard dict against the gate for ``tier``/``category``.

    A None/absent metric is a failure (uncertified). ``scorecard`` keys: f1, auroc, ece,
    fp_rate, p95_latency_ms, validated_at (ISO str or datetime), per_language
    ({lang: {samples: n}} or {lang: n})."""
    gate = gate_for(tier, category)
    failures: List[Dict[str, Any]] = []

    def _num(key):
        v = scorecard.get(key)
        # bool is an int subclass — reject it (and NaN/inf) so a bogus metric can't pass.
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            return None
        return float(v)

    for metric, key in (("min_f1", "f1"), ("min_auroc", "auroc")):
        actual = _num(key)
        if actual is None or actual < gate[metric]:
            _fail(failures, key, gate[metric], actual, ">=")
    for metric, key in (("max_ece", "ece"), ("max_fp_rate", "fp_rate")):
        actual = _num(key)
        if actual is None or actual > gate[metric]:
            _fail(failures, key, gate[metric], actual, "<=")

    p95 = _num("p95_latency_ms")
    if p95 is None or p95 > gate["max_p95_latency_ms"]:
        _fail(failures, "p95_latency_ms", gate["max_p95_latency_ms"], p95, "<=")

    # Each covered language must have enough samples; no per-language evidence fails.
    per_lang = scorecard.get("per_language") or {}
    min_lang = _min_lang_samples(per_lang)
    if min_lang is None or min_lang < gate["min_samples_per_language"]:
        _fail(failures, "samples_per_language", gate["min_samples_per_language"], min_lang, ">=")

    age = _validation_age_days(scorecard.get("validated_at"), now)
    # Fail closed on missing, too-old, OR future-dated (beyond clock skew) validation.
    if age is None or age > gate["max_validation_age_days"] or age < -_CLOCK_SKEW_DAYS:
        _fail(failures, "validation_age_days", gate["max_validation_age_days"], age, "0<=age<=max")

    return GateResult(passed=not failures, tier=tier, failures=failures, gate=gate)


def _min_lang_samples(per_language: Dict[str, Any]) -> Optional[int]:
    if not per_language:
        return None
    counts = []
    for v in per_language.values():
        if isinstance(v, dict):
            counts.append(int(v.get("samples", 0)))
        elif isinstance(v, (int, float)):
            counts.append(int(v))
    return min(counts) if counts else None


def _validation_age_days(validated_at, now: Optional[datetime]) -> Optional[float]:
    if not validated_at:
        return None
    if isinstance(validated_at, str):
        try:
            validated_at = datetime.fromisoformat(validated_at)
        except ValueError:
            return None
    if validated_at.tzinfo is None:
        validated_at = validated_at.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return (now - validated_at).total_seconds() / 86400.0
