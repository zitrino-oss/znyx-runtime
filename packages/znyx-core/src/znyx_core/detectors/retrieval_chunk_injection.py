"""Retrieval-chunk injection detector (OWASP LLM01 — indirect prompt injection).

Scans retrieved RAG chunks for instruction-injection markers BEFORE they enter the
model context: "ignore previous instructions", role-switch text, forged chat turns,
prompt-boundary tokens, and tool-invocation lures planted in a document. Deterministic
markers only; the ML classifier escalation is wired via the detector ``strategy``.

Runs in the ``retrieval`` stage (POST /v1/evaluate/retrieval).
"""
import re
from typing import Any, Dict, List

from znyx_core.core.models import Decision, DetectorResult, RuleHit, Severity
from znyx_core.core.risk import calculate_risk_score
from znyx_core.detectors._injection_patterns import scan_injection, scan_patterns

# Retrieved content that tries to make the agent take an action (tool/exec/HTTP) — a
# strong indirect-injection signal beyond the shared instruction-override markers.
_TOOL_LURE_PATTERNS = [
    (r'\b(?:call|invoke|execute|run|use)\s+(?:the\s+)?(?:tool|function|command|api|endpoint)\b', Severity.MEDIUM, "tool_invocation_lure"),
    (r'\b(?:send|post|forward|exfiltrate|upload|transmit)\s+(?:\w+\s+){0,5}(?:to|via)\s+https?://', Severity.HIGH, "exfiltration_lure"),
    (r'\b(?:fetch|retrieve|download|open)\s+https?://', Severity.MEDIUM, "url_fetch_lure"),
]
_COMPILED_LURES = [(re.compile(p, re.IGNORECASE), s, n) for p, s, n in _TOOL_LURE_PATTERNS]


class RetrievalChunkInjectionDetector:
    """Flags prompt-injection planted in retrieved RAG chunks."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.enabled = self.config.get('enabled', False)
        # Default WARN (advisory) — retrieved content is frequently noisy; an org opts
        # into BLOCK once it trusts the signal on its corpus.
        self.action = (self.config.get('action') or 'WARN').upper()
        self.block_threshold = self.config.get('block_threshold', 50)

    def detect(self, text: str) -> DetectorResult:
        if not self.enabled or not text:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        hits: List[RuleHit] = scan_injection(text, "retrieval_chunk_injection")
        seen = {h.rule_id for h in hits}
        hits += scan_patterns(text, _COMPILED_LURES, "retrieval_chunk_injection",
                              "Retrieved chunk action lure: {name}", seen=seen)

        if not hits:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        risk_score = calculate_risk_score(hits)
        if self.action == "BLOCK" and risk_score >= self.block_threshold:
            return DetectorResult(
                decision=Decision.BLOCK,
                risk_score=risk_score,
                rule_hits=hits,
                user_message="Retrieved content was blocked: it contains instructions that look like a prompt-injection attack.",
                developer_message=f"retrieval_chunk_injection: {len(hits)} injection marker(s) in retrieved chunks",
            )
        return DetectorResult(
            decision=Decision.WARN,
            risk_score=risk_score,
            rule_hits=hits,
            developer_message=f"retrieval_chunk_injection: {len(hits)} injection marker(s) in retrieved chunks (advisory)",
        )
