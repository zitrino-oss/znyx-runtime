"""Shared prompt-injection pattern bank (P1b).

Indirect prompt injection (OWASP LLM01) shows up the same way no matter where the
untrusted text comes from — a retrieved RAG chunk, a tool result re-entering context,
a poisoned memory write, or a malicious tool manifest. Rather than copy the same
regexes into four detectors, the markers live here once and each P1b detector calls
``scan_injection`` with its own ``rule_prefix``.

These are deterministic markers only (no ML) — the ML classifier escalation is P2.
"""
import re
from typing import List

from znyx_core.core.models import RuleHit, Severity
from znyx_core.core.text_normalize import match_variants

# Instruction-override / "ignore previous" — the canonical injection tell.
INSTRUCTION_OVERRIDE_PATTERNS = [
    # "ignore previous/all/the above instructions" AND the bare-object form
    # "ignore everything above" / "ignore (all of) the above".
    (r'\b(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+|any\s+|the\s+|your\s+|previous\s+|above\s+|prior\s+|earlier\s+|everything\s+|all\s+of\s+){0,3}(?:instructions?|prompts?|rules?|guidelines?|directions?|context|messages?|above)', Severity.HIGH, "instruction_override"),
    (r'\b(?:ignore|disregard|forget)\s+(?:everything|all)\s+(?:above|before|prior|preceding|that\s+came\s+before)', Severity.HIGH, "instruction_override_above"),
    (r'\b(?:do\s+not|don\'?t)\s+(?:follow|obey|listen\s+to)\s+(?:the\s+|your\s+|any\s+)?(?:previous|above|prior|earlier|system)\s+(?:instructions?|prompts?|rules?)', Severity.HIGH, "instruction_override_negative"),
    (r'\bthese\s+(?:new\s+)?instructions?\s+(?:override|supersede|replace|take\s+precedence)', Severity.HIGH, "instruction_supersede"),
]

# Role / identity switching injected into untrusted content.
ROLE_SWITCH_PATTERNS = [
    # Adverb optional: "you are now a", "you are actually the", AND "you are an admin".
    (r'\byou\s+are\s+(?:now|actually|really)?\s*(?:a|an|the)\s+(?:admin|administrator|root|superuser|developer|unrestricted|jailbroken|new)\b', Severity.HIGH, "role_switch_privileged"),
    (r'\byou\s+are\s+(?:now|actually|really)\s+(?:a|an|the)\b', Severity.HIGH, "role_switch_now"),
    (r'\b(?:from\s+now\s+on|starting\s+now|henceforth)\b[^.]{0,40}\byou\b', Severity.MEDIUM, "role_switch_persistent"),
    (r'\b(?:act|behave|respond|pretend)\s+as\s+(?:if\s+you\s+(?:are|were)\s+)?(?:a|an|the)?\s*(?:DAN|jailbroken|unrestricted|developer\s+mode)', Severity.HIGH, "role_switch_jailbreak_persona"),
    (r'\byour\s+(?:new\s+)?(?:role|task|job|purpose|instruction)\s+is\b', Severity.MEDIUM, "role_reassignment"),
    # Injected chat-role delimiters trying to forge a system/assistant turn. Match at
    # line start OR mid-line after whitespace/quotes/brackets (injection is rarely at col 0).
    (r'(?:^|\n|[\s"\'>\]\)])\s*(?:system|assistant|developer)\s*[:>]\s', Severity.MEDIUM, "forged_role_turn"),
]

# Embedded directives aimed at the model rather than the user.
EMBEDDED_INSTRUCTION_PATTERNS = [
    (r'\bthe\s+(?:assistant|ai|model|llm|bot)\s+(?:must|should|will|needs?\s+to|has\s+to)\b', Severity.MEDIUM, "embedded_directive"),
    (r'\b(?:important|attention|note\s+to\s+(?:self|ai|assistant))\s*[:!]', Severity.LOW, "embedded_attention_marker"),
    (r'\bwhen\s+(?:you\s+)?(?:answer|respond|reply|summari[sz]e)[^.]{0,40}\b(?:include|add|append|say|output|reveal)\b', Severity.MEDIUM, "embedded_output_directive"),
]

# Model/template control tokens used to break the prompt boundary.
PROMPT_BOUNDARY_PATTERNS = [
    (r'</?(?:system|user|assistant|instructions?|prompt)\s*>', Severity.MEDIUM, "boundary_xml_tag"),
    (r'\[/?(?:INST|SYS|SYSTEM)\]', Severity.MEDIUM, "boundary_inst_tag"),
    (r'<\|(?:im_start|im_end|system|user|assistant|endoftext)\|>', Severity.HIGH, "boundary_chatml_token"),
    (r'(?:^|\n)#{3,}\s*(?:system|instruction|new\s+prompt)', Severity.MEDIUM, "boundary_markdown_header"),
]

_DEFAULT_GROUPS = (
    INSTRUCTION_OVERRIDE_PATTERNS,
    ROLE_SWITCH_PATTERNS,
    EMBEDDED_INSTRUCTION_PATTERNS,
    PROMPT_BOUNDARY_PATTERNS,
)

# Pre-compile once at import.
_COMPILED_DEFAULT = [
    (re.compile(pat, re.IGNORECASE), sev, name)
    for group in _DEFAULT_GROUPS
    for pat, sev, name in group
]


def scan_patterns(text: str, compiled, rule_prefix: str, message_fmt: str,
                  seen: set = None) -> List[RuleHit]:
    """Scan ``text`` (and its evasion-normalized variants) against a compiled pattern
    bank, returning deduped RuleHits. Matching every variant from
    :func:`match_variants` defeats homoglyph / zero-width / leetspeak evasion that would
    otherwise walk past a single confusable character.

    Args:
        text: untrusted content to scan.
        compiled: pre-compiled ``[(regex, severity, name)]`` list.
        rule_prefix: detector key namespacing ``rule_id``.
        message_fmt: format string with a ``{name}`` placeholder for the human message.
        seen: optional shared dedup set (so a caller can run several banks without dupes).
    """
    if not text:
        return []
    variants = match_variants(text)
    hits: List[RuleHit] = []
    if seen is None:
        seen = set()
    for pattern, severity, name in compiled:
        if any(pattern.search(v) for v in variants):
            rule_id = f"{rule_prefix}.{name}"
            if rule_id not in seen:
                seen.add(rule_id)
                hits.append(RuleHit(rule_id=rule_id, severity=severity,
                                    message=message_fmt.format(name=name.replace('_', ' '))))
    return hits


def scan_injection(text: str, rule_prefix: str, compiled=None) -> List[RuleHit]:
    """Scan ``text`` for prompt-injection markers (evasion-normalized), deduped RuleHits."""
    return scan_patterns(text, compiled or _COMPILED_DEFAULT, rule_prefix,
                         "Prompt-injection marker: {name}")
