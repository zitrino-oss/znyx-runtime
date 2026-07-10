"""System Prompt Leakage detector (OWASP LLM07 — System Prompt Leakage).

Detects when a model's OUTPUT reproduces a registered system prompt. Operators register
their prompts as **keyed shingle-hash fingerprints** (control plane:
``POST /v1/orgs/{org_id}/system-prompts/fingerprint`` → ``system_prompt_fingerprints``,
hash-only). The resolved policy delivers, per org, the HMAC pepper + each fingerprint's
keyed hash set as this detector's config.

Method (privacy-preserving, deterministic): HMAC the output's token shingles with the
same per-org key and count how many match a fingerprint's stored digests. ``match_threshold``
matching shingles (each ≥ ``min_shingle_tokens`` consecutive tokens) trips it — minor
edits only break the few shingles spanning the edit, so near-verbatim leaks still match.
No raw prompt text is ever held by the detector. Embedding-similarity escalation is planned.
"""
from typing import Any, Dict, List

from znyx_core.core.fingerprint import DEFAULT_MIN_SHINGLE_TOKENS, overlap_count
from znyx_core.core.models import Decision, DetectorResult, RuleHit, Severity
from znyx_core.core.risk import calculate_risk_score

# Leaked system prompts are blocked or warned — redaction is not meaningful here.
_ACTION_TO_DECISION = {"BLOCK": Decision.BLOCK, "WARN": Decision.WARN,
                       "ALLOW_WITH_NOTICE": Decision.WARN}


class SystemPromptLeakageDetector:
    """Deterministic system-prompt reproduction detector (LLM07), hash-only matching."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.enabled = self.config.get("enabled", False)
        self.action = (self.config.get("action") or "BLOCK").upper()
        self.match_threshold = max(1, int(self.config.get("match_threshold", 2)))

        key_hex = self.config.get("fingerprint_key")
        try:
            self._key = bytes.fromhex(key_hex) if key_hex else None
        except (ValueError, TypeError):
            self._key = None

        # Each fingerprint: {"hashes": [...], "min_shingle_tokens": int}.
        self._fingerprints: List[Dict[str, Any]] = []
        for fp in (self.config.get("fingerprints") or []):
            if isinstance(fp, dict) and fp.get("hashes"):
                self._fingerprints.append({
                    "hashes": set(fp["hashes"]),
                    "n": int(fp.get("min_shingle_tokens", DEFAULT_MIN_SHINGLE_TOKENS)),
                })

    def detect(self, text: str) -> DetectorResult:
        if not self.enabled or not text or not self._key or not self._fingerprints:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        rule_hits: List[RuleHit] = []
        for idx, fp in enumerate(self._fingerprints):
            if overlap_count(text, self._key, fp["n"], fp["hashes"]) >= self.match_threshold:
                rule_hits.append(RuleHit(
                    rule_id=f"system_prompt_leakage.fingerprint_{idx}",
                    severity=Severity.HIGH,
                    message="Output reproduces a registered system prompt",
                ))

        if not rule_hits:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        risk_score = calculate_risk_score(rule_hits)
        decision = _ACTION_TO_DECISION.get(self.action, Decision.BLOCK)
        dev_msg = f"System prompt leakage: {len(rule_hits)} fingerprint(s) reproduced in output"
        if decision == Decision.BLOCK:
            return DetectorResult(
                decision=Decision.BLOCK, risk_score=risk_score, rule_hits=rule_hits,
                developer_message=dev_msg,
                user_message="I can't share my internal instructions.",
            )
        return DetectorResult(decision=Decision.WARN, risk_score=risk_score,
                              rule_hits=rule_hits, developer_message=dev_msg)
