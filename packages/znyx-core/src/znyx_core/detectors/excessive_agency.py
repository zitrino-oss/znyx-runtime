"""Excessive-agency detector (P1b, OWASP LLM06).

Risk-scores a proposed agent plan, or a single agent-loop step's action, by the kind of
actions it takes: destructive/irreversible operations, financial or external
side-effects, privilege/scope escalation, and raw code execution. High-risk plans are
blocked; medium-risk plans WARN (which an org can route to human review via on_fail
remediation — the roadmap's "ASK_HUMAN" maps onto the existing review queue, since the
core Decision set has no dedicated ASK_HUMAN value). LLM-judge escalation is P3.

Runs in the ``agent_plan`` stage (plan JSON) and the ``agent_loop`` stage (a live step's
action). It is deliberately NOT scoped to the ``tool`` stage: that stage carries
tool-RESULT text re-entering context (handled by ``tool_output_injection``), not an
action plan to risk-score.
"""
import json
import re
from typing import Any, Dict, List

from znyx_core.core.models import Decision, DetectorResult, RuleHit, Severity
from znyx_core.core.risk import calculate_risk_score

# action category -> (regex, severity, message)
_ACTION_CATEGORIES = [
    (r'\b(?:delete|remove|drop|truncate|wipe|purge|destroy|erase|rm|rmdir|format|revoke|terminate|shutdown|uninstall|deprovision)\b',
     Severity.HIGH, "destructive_action", "destructive/irreversible operation"),
    (r'\b(?:transfer|wire|pay|payment|purchase|buy|charge|refund|send[_\s]?email|send[_\s]?message|send[_\s]?sms|publish|post[_\s]?to|tweet)\b',
     Severity.HIGH, "external_side_effect", "financial or external side-effect"),
    (r'\b(?:grant|sudo|admin|root|chmod|chown|escalate|impersonate|assume[_\s]?role|disable[_\s]?(?:auth|mfa|security))\b',
     Severity.HIGH, "privilege_escalation", "privilege / scope escalation"),
    (r'\b(?:exec|eval|system|subprocess|spawn|run[_\s]?(?:shell|command|code)|os\.system|/bin/(?:sh|bash))\b',
     Severity.HIGH, "code_execution", "raw code/shell execution"),
    (r'\b(?:modify|update|overwrite|patch|alter|reconfigure|migrate)\b',
     Severity.MEDIUM, "mutating_action", "state-mutating operation"),
]
_COMPILED_ACTIONS = [(re.compile(p, re.IGNORECASE), s, name, msg) for p, s, name, msg in _ACTION_CATEGORIES]

# Broad blast-radius / scope markers that amplify any action's risk. Anchored to a
# scope-like FIELD (target/scope/recipients/…) or a quoted "*" so the common words
# all/global/entire don't false-positive on ordinary prose.
_SCOPE_PATTERNS = [
    (r'(?:target|scope|recipients?|resources?|to|on|against)["\']?\s*[:=]\s*["\']?(?:\*|all|everyone|every[_\s]?(?:one|thing|user)|production|prod|entire|global)\b',
     Severity.MEDIUM, "broad_scope", "broad blast radius / scope"),
    (r'["\']\*["\']',
     Severity.MEDIUM, "wildcard_scope", "wildcard (*) target"),
]
_COMPILED_SCOPE = [(re.compile(p, re.IGNORECASE), s, name, msg) for p, s, name, msg in _SCOPE_PATTERNS]

# Fields in a structured plan/step that name the action being taken.
_ACTION_FIELDS = ("action", "tool", "tool_name", "name", "operation", "op", "command", "method", "function")


_MAX_PLAN_DEPTH = 40  # bound recursion so a hostile deeply-nested plan can't crash us


def _extract_action_strings(plan: Any) -> List[str]:
    """Pull the action-naming fields out of a structured plan for focused matching.
    Recursion is depth-bounded so a maliciously deep plan degrades gracefully instead
    of raising RecursionError (which would 500 / fail open)."""
    out: List[str] = []

    def walk(node: Any, depth: int) -> None:
        if depth > _MAX_PLAN_DEPTH:
            return
        if isinstance(node, dict):
            for f in _ACTION_FIELDS:
                v = node.get(f)
                if isinstance(v, str):
                    out.append(v)
            for v in node.values():
                walk(v, depth + 1)
        elif isinstance(node, list):
            for item in node:
                walk(item, depth + 1)

    walk(plan, 0)
    return out


class ExcessiveAgencyDetector:
    """Risk-scores agent plans / agent-loop step actions for excessive autonomy (LLM06)."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.enabled = self.config.get('enabled', False)
        self.action = (self.config.get('action') or 'WARN').upper()
        # Default 50 so a single HIGH-severity action (destructive / external side-effect /
        # privilege escalation / code execution) reaches BLOCK when action=BLOCK.
        self.block_threshold = self.config.get('block_threshold', 50)
        self.warn_threshold = self.config.get('warn_threshold', 30)
        # Extra org-defined high-risk action keywords (regex-escaped, word-bounded so a
        # short term like "rm" can't substring-match inside an unrelated token).
        extra = self.config.get('high_risk_actions') or []
        self._extra = (re.compile("|".join(rf'\b{re.escape(a)}\b' for a in extra), re.IGNORECASE)
                       if extra else None)

    def detect(self, text: str) -> DetectorResult:
        if not self.enabled or not text:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        # Match against the action-naming fields when the plan is structured JSON, plus
        # the full serialized text as a fallback (covers free-form plans / tool calls).
        haystacks = [text]
        try:
            parsed = json.loads(text)
            haystacks.extend(_extract_action_strings(parsed))
        except (ValueError, TypeError, RecursionError):
            # Unparseable or maliciously-deep JSON → fall back to scanning the raw text.
            parsed = None
        raw = "\n".join(haystacks)
        # Action names are often snake_case / kebab-case (delete_all_files, transfer-funds).
        # `\bkeyword\b` won't match a verb glued to a token by '_' (a word char), so scan a
        # separator-normalized copy too.
        normalized = re.sub(r'[_\-]+', ' ', raw)
        scan_text = raw + "\n" + normalized

        hits: List[RuleHit] = []
        seen: set = set()
        for pattern, severity, name, msg in _COMPILED_ACTIONS:
            if pattern.search(scan_text):
                rid = f"excessive_agency.{name}"
                if rid not in seen:
                    seen.add(rid)
                    hits.append(RuleHit(rule_id=rid, severity=severity, message=f"Excessive agency: {msg}"))
        for pattern, severity, name, msg in _COMPILED_SCOPE:
            if pattern.search(scan_text):
                rid = f"excessive_agency.{name}"
                if rid not in seen:
                    seen.add(rid)
                    hits.append(RuleHit(rule_id=rid, severity=severity, message=f"Excessive agency: {msg}"))
        if self._extra and self._extra.search(scan_text):
            rid = "excessive_agency.org_high_risk_action"
            if rid not in seen:
                hits.append(RuleHit(rule_id=rid, severity=Severity.HIGH,
                                    message="Excessive agency: org-defined high-risk action"))

        if not hits:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        risk_score = calculate_risk_score(hits)
        if self.action == "BLOCK" and risk_score >= self.block_threshold:
            return DetectorResult(
                decision=Decision.BLOCK,
                risk_score=risk_score,
                rule_hits=hits,
                user_message="This action plan was blocked: it requests high-risk or irreversible operations.",
                developer_message=f"excessive_agency: {len(hits)} risk factor(s); score {risk_score} ≥ block_threshold {self.block_threshold}",
            )
        if risk_score >= self.warn_threshold:
            return DetectorResult(
                decision=Decision.WARN,
                risk_score=risk_score,
                rule_hits=hits,
                developer_message=f"excessive_agency: {len(hits)} risk factor(s); score {risk_score} (review recommended)",
            )
        return DetectorResult(decision=Decision.ALLOW, risk_score=risk_score, rule_hits=hits)
