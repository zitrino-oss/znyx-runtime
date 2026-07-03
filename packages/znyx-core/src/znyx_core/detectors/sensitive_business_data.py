"""Sensitive Business Data detector (P1a, OWASP LLM02 — Sensitive Information
Disclosure).

Flags confidential business information leaving (or entering) the model: M&A
activity, internal/unpublished pricing, unreleased roadmap, and customer/account
lists. This is a deterministic dictionary + allowlist control — operators extend the
per-category dictionaries with their own domain terms; a transformer classifier is
added in P2.

Config (all optional, ``DetectorConfig`` base + these):
  - ``categories``: dict ``{category: [phrase, ...]}`` MERGED over the defaults so an
    operator can add domain terms without losing the built-ins. A category mapped to
    ``[]``/``null`` disables that category.
  - ``allowlist``: phrases whose presence suppresses a hit (approved contexts, e.g.
    a public pricing page). Substring, case-insensitive.
  - ``action``: ``WARN`` (default, informational) | ``BLOCK`` | ``REDACT``.
  - ``block_threshold``: risk score at/above which the action is enforced (default 60).
  - ``severity``: per-match severity (default ``medium``).
"""
import re
from typing import Any, Dict, List, Tuple

from znyx_core.core.models import Decision, DetectorResult, RuleHit, Severity
from znyx_core.core.risk import calculate_risk_score

# Conservative, multi-word defaults (single common words like "margin" would over-fire;
# operators add their own domain terms via config["categories"]).
DEFAULT_CATEGORIES: Dict[str, List[str]] = {
    "mergers_acquisitions": [
        "due diligence", "letter of intent", "term sheet", "definitive agreement",
        "merger agreement", "acquisition target", "divestiture", "pre-announcement",
    ],
    "internal_pricing": [
        "wholesale price", "cost basis", "internal pricing", "unpublished pricing",
        "price floor", "discount tier", "rate card", "margin structure",
    ],
    "roadmap": [
        "product roadmap", "internal roadmap", "unreleased feature", "unannounced launch",
        "launch date", "confidential roadmap",
    ],
    "customer_list": [
        "customer list", "client list", "account list", "book of business",
        "customer database", "prospect list",
    ],
}

_ACTION_TO_DECISION = {
    "BLOCK": Decision.BLOCK,
    "REDACT": Decision.REDACT,
    "WARN": Decision.WARN,
    "ALLOW_WITH_NOTICE": Decision.WARN,
}

_SEVERITY = {"low": Severity.LOW, "medium": Severity.MEDIUM, "high": Severity.HIGH}


class SensitiveBusinessDataDetector:
    """Deterministic confidential-business-data detector (LLM02)."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.enabled = self.config.get("enabled", False)
        self.action = (self.config.get("action") or "WARN").upper()
        self.block_threshold = int(self.config.get("block_threshold", 60))
        self.severity = _SEVERITY.get(str(self.config.get("severity", "medium")).lower(), Severity.MEDIUM)

        # Merge operator categories over the defaults (a falsy value disables a category).
        merged: Dict[str, List[str]] = {k: list(v) for k, v in DEFAULT_CATEGORIES.items()}
        for cat, phrases in (self.config.get("categories") or {}).items():
            merged[cat] = list(phrases) if phrases else []
        # Precompile word-boundary, case-insensitive patterns per (category, phrase).
        self._patterns: List[Tuple[str, str, re.Pattern]] = []
        for cat, phrases in merged.items():
            for phrase in phrases:
                if phrase and phrase.strip():
                    self._patterns.append((cat, phrase, re.compile(
                        r"\b" + re.escape(phrase.strip()) + r"\b", re.IGNORECASE)))

        self.allowlist = [a.lower() for a in (self.config.get("allowlist") or []) if a and a.strip()]

    def detect(self, text: str) -> DetectorResult:
        if not self.enabled or not text:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        lowered = text.lower()
        if any(allowed in lowered for allowed in self.allowlist):
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        rule_hits: List[RuleHit] = []
        matched_spans: List[str] = []
        seen: set = set()
        for category, phrase, pattern in self._patterns:
            if pattern.search(text):
                if category not in seen:
                    rule_hits.append(RuleHit(
                        rule_id=f"sensitive_business_data.{category}",
                        severity=self.severity,
                        message=f"Possible confidential business data ({category.replace('_', ' ')})",
                    ))
                    seen.add(category)
                matched_spans.append(phrase)

        if not rule_hits:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        risk_score = calculate_risk_score(rule_hits)
        decision = _ACTION_TO_DECISION.get(self.action, Decision.WARN)
        categories = sorted({h.rule_id.split(".", 1)[1] for h in rule_hits})
        dev_msg = f"Sensitive business data categories: {', '.join(categories)}"

        # Below the enforcement threshold → downgrade an enforcing action to WARN.
        if decision in (Decision.BLOCK, Decision.REDACT) and risk_score < self.block_threshold:
            return DetectorResult(decision=Decision.WARN, risk_score=risk_score,
                                  rule_hits=rule_hits, developer_message=dev_msg)

        if decision == Decision.REDACT:
            sanitized = text
            for phrase in set(matched_spans):
                sanitized = re.sub(r"\b" + re.escape(phrase) + r"\b", "[REDACTED]",
                                   sanitized, flags=re.IGNORECASE)
            return DetectorResult(
                decision=Decision.REDACT, risk_score=risk_score, rule_hits=rule_hits,
                sanitized_text=sanitized, developer_message=dev_msg,
                user_message="Redacted confidential business information.",
            )

        if decision == Decision.BLOCK:
            return DetectorResult(
                decision=Decision.BLOCK, risk_score=risk_score, rule_hits=rule_hits,
                developer_message=dev_msg,
                user_message="This content appears to contain confidential business information.",
            )

        return DetectorResult(decision=Decision.WARN, risk_score=risk_score,
                              rule_hits=rule_hits, developer_message=dev_msg)
