from typing import List

from znyx_core.core.models import RuleHit, Severity

# Canonical severity-to-risk mapping used across all detectors
SEVERITY_RISK_SCORES = {
    Severity.LOW: 10,
    Severity.MEDIUM: 30,
    Severity.HIGH: 50,
}


def calculate_risk_score(rule_hits: List[RuleHit]) -> int:
    """Calculate a risk score from a list of rule hits using canonical severity weights.

    The score is the sum of per-hit severity weights, capped at 100.

    Args:
        rule_hits: List of RuleHit objects from a detector.

    Returns:
        Integer risk score in the range [0, 100].
    """
    if not rule_hits:
        return 0
    return min(100, sum(SEVERITY_RISK_SCORES.get(hit.severity, 0) for hit in rule_hits))
