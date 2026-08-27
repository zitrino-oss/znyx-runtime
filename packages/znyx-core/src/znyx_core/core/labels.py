"""Canonical label / score semantics.

The single source of truth for how ZNYX labels things, so every layer, the aggregator,
the trace UI, scorecards, model cards, and benchmark scoring agree:

  * **Decision set + precedence** — the 5 decisions and their "worst wins" ranking.
  * **Severity ↔ risk bands** — the 3 severities and their 0–100 risk bands.
  * **Confidence bands** — coarse 0–1 confidence buckets.
  * **Score normalization** — mapping each layer's native score onto one 0–100 scale.

This module is the reference for normalizing scores from every layer onto one
shared scale.
"""
from __future__ import annotations

from enum import Enum

from znyx_core.core.models import Decision, Severity


class ConfidenceBand(str, Enum):
    """Coarse confidence buckets for display/triage. Thresholds are inclusive-low."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ── Decision set + precedence ────────────────────────────────────────────────
# The canonical "worst wins" ranking used when aggregating detector results
# (BLOCK > REDACT > TRANSFORM > WARN > ALLOW). The DecisionAggregator imports this so
# there is exactly one ordering in the system.
DECISION_PRECEDENCE = {
    Decision.BLOCK: 5,
    Decision.REDACT: 4,
    Decision.TRANSFORM: 3,
    Decision.WARN: 2,
    Decision.ALLOW: 1,
}

# Human-facing meaning of each decision (model cards / docs / UI tooltips).
DECISION_SEMANTICS = {
    Decision.ALLOW: "No action — content passes unchanged.",
    Decision.WARN: "Allowed, but flagged for observability/telemetry; does not modify or stop.",
    Decision.TRANSFORM: "Content is rewritten (e.g. reformatted to a contract) before passing.",
    Decision.REDACT: "Sensitive spans are masked; the redacted content passes.",
    Decision.BLOCK: "Content is stopped; it does not pass.",
}


def decision_rank(decision: Decision | None) -> int:
    """Precedence rank of a decision (higher = more severe). None/unknown → 0."""
    return DECISION_PRECEDENCE.get(decision, 0) if decision else 0


def most_severe_decision(decisions) -> Decision:
    """The highest-precedence decision in an iterable (empty → ALLOW)."""
    best, best_rank = Decision.ALLOW, 0
    for d in decisions:
        r = decision_rank(d)
        if r > best_rank:
            best, best_rank = d, r
    return best


# ── Severity ↔ risk bands ────────────────────────────────────────────────────
# Representative 0-100 risk for each severity (used when a layer reports only a
# severity, e.g. a deterministic rule hit, and a numeric risk is needed).
SEVERITY_TO_RISK = {
    Severity.LOW: 25,
    Severity.MEDIUM: 60,
    Severity.HIGH: 90,
}

# Inclusive-low boundaries that bucket a 0-100 risk score back into a severity.
RISK_BAND_MEDIUM_MIN = 40
RISK_BAND_HIGH_MIN = 80

# Inclusive-low boundaries that bucket a 0-1 confidence into a ConfidenceBand.
CONFIDENCE_BAND_MEDIUM_MIN = 0.5
CONFIDENCE_BAND_HIGH_MIN = 0.8


def confidence_band(confidence: float | None) -> ConfidenceBand | None:
    """Bucket a 0..1 confidence. None → None (no model confidence available)."""
    if confidence is None:
        return None
    c = min(1.0, max(0.0, float(confidence)))
    if c < CONFIDENCE_BAND_MEDIUM_MIN:
        return ConfidenceBand.LOW
    if c < CONFIDENCE_BAND_HIGH_MIN:
        return ConfidenceBand.MEDIUM
    return ConfidenceBand.HIGH


def severity_to_risk(severity: Severity) -> int:
    return SEVERITY_TO_RISK.get(severity, 60)


def risk_to_severity(risk: float) -> Severity:
    r = min(100.0, max(0.0, float(risk)))
    if r >= RISK_BAND_HIGH_MIN:
        return Severity.HIGH
    if r >= RISK_BAND_MEDIUM_MIN:
        return Severity.MEDIUM
    return Severity.LOW


def normalize_risk(value: float, kind: str = "deterministic") -> int:
    """Map a layer's native score onto the common 0..100 risk scale.

    - ``deterministic`` / ``judge``: already a 0..100 risk score — clamp.
    - ``ml`` / ``embedding`` / ``probability``: a 0..1 probability — ×100 then clamp
      ( "ML probability ×100"). Probabilities are clamped to [0,1] upstream.

    Keeps the aggregator's "worst decision wins" intact while letting each layer
    record both its native and normalized score for divergence analysis.
    """
    if value is None:
        return 0
    v = float(value)
    kind = (kind or "deterministic").lower()
    if kind in ("ml", "embedding", "probability"):
        v *= 100.0
    return int(round(min(100.0, max(0.0, v))))
