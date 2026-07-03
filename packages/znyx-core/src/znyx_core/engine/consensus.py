"""Multi-judge consensus synthesis (P3 unit 5, roadmap §5).

Given N independent judge verdicts (each a decision + risk score + confidence + a vote
weight), synthesize one verdict by **majority** or **weighted** vote. Decision ties break
toward the MOST SEVERE decision (DECISION_PRECEDENCE: BLOCK > REDACT > TRANSFORM > WARN >
ALLOW) — a safe default. The synthesized risk is the weight-weighted average of member
risks; ``agreement`` is the winning decision's vote share (the synthesized confidence).

Pure / dependency-free so it is unit-testable and reusable by the judge escalation caller,
which writes the per-member + synthesized ``judge_audit_events`` rows around it.
"""
from dataclasses import dataclass
from typing import Optional, Sequence

from znyx_core.core.labels import most_severe_decision
from znyx_core.core.models import Decision


@dataclass
class JudgeVote:
    decision: str                     # canonical Decision value
    risk_score: float = 0.0           # 0..100
    confidence: Optional[float] = None
    weight: float = 1.0               # vote weight (weighted method); >= 0


@dataclass
class ConsensusResult:
    decision: str
    risk_score: float
    confidence: float                 # agreement = winning decision's vote share (0..1)
    method: str
    member_count: int
    agreement: float


def synthesize_consensus(votes: Sequence[JudgeVote], method: str = "majority") -> ConsensusResult:
    """Combine member votes into one verdict. ``method`` ∈ {"majority", "weighted"}.

    majority: each member counts once; weighted: each counts for its ``weight``. The
    decision with the highest total tally wins; ties go to the most severe decision."""
    if method not in ("majority", "weighted"):
        raise ValueError(f"unknown consensus method {method!r}")
    if not votes:
        raise ValueError("consensus requires at least one vote")

    def w(v: JudgeVote) -> float:
        if method == "majority":
            return 1.0
        return max(0.0, float(v.weight if v.weight is not None else 1.0))

    # Degenerate weights (all <= 0) would make every decision tie at 0 and tie-break to the
    # most-severe with 0 confidence — silently overriding the real majority. Fall back to an
    # unweighted (count) tally so a unanimous/majority vote is honoured with real agreement.
    effective_method = method
    if method == "weighted" and sum(w(v) for v in votes) <= 0.0:
        effective_method = "majority"

    def ew(v: JudgeVote) -> float:
        return 1.0 if effective_method == "majority" else w(v)

    total_weight = sum(ew(v) for v in votes) or float(len(votes))

    # Tally vote weight per decision.
    tally: dict = {}
    for v in votes:
        tally[v.decision] = tally.get(v.decision, 0.0) + ew(v)

    top_weight = max(tally.values())
    winners = [d for d, wt in tally.items() if wt == top_weight]
    # Tie-break toward the most severe decision (fail-safe).
    if len(winners) == 1:
        decision = winners[0]
    else:
        winner_enums = [Decision(d) for d in winners if d in {x.value for x in Decision}]
        decision = most_severe_decision(winner_enums).value if winner_enums else winners[0]

    # Weight-weighted mean risk across all members, clamped to the 0..100 scale (a member
    # reporting out-of-range risk must not crash the DetectorResult mapping downstream).
    risk = sum(min(100.0, max(0.0, float(v.risk_score or 0.0))) * ew(v) for v in votes) / total_weight
    risk = min(100.0, max(0.0, risk))
    agreement = round(top_weight / total_weight, 4)

    return ConsensusResult(
        decision=decision,
        risk_score=round(risk, 2),
        confidence=agreement,
        method=method,
        member_count=len(votes),
        agreement=agreement,
    )
