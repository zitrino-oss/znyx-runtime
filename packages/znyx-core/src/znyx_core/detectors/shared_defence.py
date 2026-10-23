"""Reviewed, hash-only signatures distributed in the signed control-plane feed."""
from znyx_core.defence import match
from znyx_core.core.models import Decision, DetectorResult, RuleHit, Severity


class SharedDefenceDetector:
    def __init__(self, config):
        self.config = config

    def detect(self, text):
        if not match(text, self.config):
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)
        return DetectorResult(decision=Decision.BLOCK, risk_score=100, rule_hits=[RuleHit(
            rule_id="shared_defence.attack_fingerprint", severity=Severity.HIGH,
            message="Content matches a reviewed attack fingerprint")])
