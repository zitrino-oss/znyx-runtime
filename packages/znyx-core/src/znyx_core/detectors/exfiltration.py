"""
Context/Memory Exfiltration Detector

Detects attempts to:
- Reveal system prompts or hidden instructions
- Dump conversation history or memory
- Access cross-user/org data
"""
import re
from typing import List, Dict, Any
from znyx_core.core.models import DetectorResult, RuleHit, Severity, Decision
from znyx_core.core.risk import calculate_risk_score


class ExfiltrationDetector:
    """Detects context and memory exfiltration attempts"""

    # Patterns for system prompt revelation
    SYSTEM_PROMPT_PATTERNS = [
        (r'\b(show|reveal|print|display|tell|give|output|find|get|extract|retrieve|fetch|access)\s+(?:me\s+)?(?:your|the|my)?\s*(?:secret\s+)?(?:system|developer|hidden|initial|original|internal|server)?\s*(?:api\s+)?(?:keys?|credentials?|tokens?|secrets?|prompt|instructions?|rules?|guidelines?)\s+(?:from|in|of)\s+(?:your|the)\s+(?:system|server|prompt|config)', Severity.HIGH, "System credential/prompt extraction"),
        (r'\b(show|reveal|print|display|tell|give|output|find|get|extract|retrieve|fetch|access)\s+(?:me\s+)?(?:your|the)\s+(?:system|developer|hidden|initial|original)\s+(?:prompt|instructions?|rules?|guidelines?)', Severity.HIGH, "System prompt revelation attempt"),
        (r'\bwhat\s+(?:are|is|were)\s+your\s+(?:system|initial|original|hidden)\s+(?:prompt|instructions?|rules?)', Severity.HIGH, "System prompt query"),
        (r'\bignore\s+(?:previous|all|above)\s+instructions?\s+and\s+(?:show|reveal|print)', Severity.HIGH, "Instruction override + revelation"),
        (r'\b(?:secret|internal|server)\s+(?:\w+\s+)?(?:api\s+)?(?:keys?|credentials?|tokens?|passwords?|secrets?)', Severity.HIGH, "Secret credential reference"),
    ]

    # Patterns for memory/history dumps
    MEMORY_DUMP_PATTERNS = [
        (r'\b(dump|export|show|print|list|reveal)\s+(?:me\s+)?(?:all|entire|full|complete)?\s*(?:the|our)?\s*(?:conversation|chat|memory|history|messages?)', Severity.HIGH, "memory_dump attempt"),
        (r'\bshow\s+(?:me\s+)?(?:(?:all|entire|our|the)\s+){1,2}(?:conversation|chat|history)', Severity.HIGH, "memory_dump conversation request"),
        (r'\bprint\s+(?:the\s+)?(?:full\s+)?(?:memory|chat\s+log|conversation\s+log|history)', Severity.HIGH, "memory_dump log access"),
        # Context window / hidden content dumps
        (r'\b(?:output|dump|print|show|list|repeat)\s+(?:everything|all)\s+(?:in|from|of)?\s*(?:your\s+)?(?:context\s+(?:window)?|memory|training)', Severity.HIGH, "context_window dump attempt"),
        (r'\bcontext\s+window\b', Severity.HIGH, "context window access attempt"),
        # Repeat-above tricks (prompt injection)
        (r'\brepeat\s+(?:the\s+)?(?:text|content|instructions?|rules?|prompt|message)\s+(?:above|before|prior)\b', Severity.HIGH, "repeat above content attempt"),
        (r'\b(?:rules?|instructions?|guidelines?|prompt)\s+(?:I|the\s+user)\s+(?:cannot|can\'t|couldn\'t)\s+see\b', Severity.HIGH, "hidden instruction access attempt"),
        # Previous / hidden conversation turns
        (r'\b(?:show|reveal|give|display|tell)\s+(?:me\s+)?(?:the\s+)?(?:previous|hidden|invisible|secret)\s+(?:conversation|message|turn|context|prompt)\b', Severity.HIGH, "hidden conversation turn access"),
        (r'\bthe\s+user\s+(?:couldn\'t|cannot|can\'t|could\s+not)\s+see\b', Severity.HIGH, "user-invisible content access"),
    ]

    # Patterns for internal data and tool enumeration
    ENUMERATION_PATTERNS = [
        (r'\b(?:list|show|enumerate|find|get)\s+(?:all\s+)?(?:training|internal)\s+(?:data\s+)?(?:sources?|files?|models?|datasets?|endpoints?)', Severity.HIGH, "internal data source enumeration"),
        (r'\btraining\s+data\s+sources?\b', Severity.HIGH, "training data access attempt"),
        (r'\bwhat\s+(?:internal\s+)?(?:tools?|functions?|capabilities?|apis?|integrations?|plugins?)\s+(?:do\s+you\s+have|are\s+available|can\s+you\s+(?:use|access|call))\b', Severity.MEDIUM, "tool enumeration attempt"),
        (r'\b(?:list|enumerate|show)\s+(?:all\s+)?(?:available\s+)?(?:tools?|functions?|capabilities?|plugins?|integrations?)\b', Severity.MEDIUM, "capability enumeration attempt"),
    ]

    # Patterns for cross-user data access
    CROSS_USER_PATTERNS = [
        (r'\b(show|list|display|reveal)\s+(?:me\s+)?(?:all|other)\s+(?:other\s+)?(?:user|tenant|customer|client|account)s?\'?\s*(?:chats?|data|messages?|conversations?)', Severity.HIGH, "cross_user data access"),
        (r'\baccess\s+(?:other|another)\s+(?:user|tenant|account)\'?\s*(?:data|information)?', Severity.HIGH, "cross_user access attempt"),
        (r'\blist\s+(?:me\s+)?all\s+(?:tenants?|organizations?|accounts?)', Severity.HIGH, "cross_user organization enumeration"),
        (r'\breveal\s+(?:me\s+)?all\s+(?:customer|client)\s+(?:conversations?|chats?|messages?)', Severity.HIGH, "cross_user data revelation"),
    ]

    # Patterns for data exfiltration via external URLs
    URL_EXFILTRATION_PATTERNS = [
        (r'\b(?:send|post|transmit|forward|upload|exfiltrate|transfer)\s+(?:\w+\s+){0,5}(?:to|via)\s+https?://', Severity.HIGH, "Data exfiltration via URL"),
        (r'\bsend\s+(?:a\s+)?(?:POST|GET|PUT)\s+request\s+to\s+', Severity.HIGH, "HTTP request exfiltration attempt"),
        (r'\b(?:format|encode|convert)\s+(?:\w+\s+){0,3}(?:into|as|to)\s+(?:a\s+)?(?:JSON|XML|CSV)\s+(?:\w+\s+){0,3}(?:and\s+)?(?:send|post|transmit)', Severity.HIGH, "Data formatting + exfiltration"),
    ]

    # Patterns for policy/config revelation
    POLICY_PATTERNS = [
        (r'\b(show|reveal|display|print)\s+(?:me\s+)?(?:your|the)\s+(?:policy|policies|configuration|config|settings?)', Severity.MEDIUM, "Policy revelation attempt"),
        (r'\bwhat\s+(?:are|is)\s+(?:your|the)\s+(?:policy|policies|rules?|restrictions?|confidential|internal)\s+(?:instructions?|guidelines?|rules?|settings?)?', Severity.MEDIUM, "Policy query"),
        (r'\b(?:confidential|internal|private|hidden)\s+(?:\w+\s+)?(?:instructions?|guidelines?|rules?)', Severity.MEDIUM, "Confidential instructions query"),
        (r'\breveal\s+(?:me\s+)?(?:your|the)\s+(?:internal|confidential|hidden)\s+(?:rules?|instructions?|guidelines?)', Severity.MEDIUM, "Internal rules revelation"),
    ]

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize exfiltration detector.

        Args:
            config: Configuration dict with keys:
                - enabled: bool (default: True)
                - block_threshold: int (default: 40) - risk score threshold for blocking
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        self.block_threshold = config.get('block_threshold', 40)

        # Compile all patterns
        self.compiled_patterns = []
        for patterns_list in [
            self.SYSTEM_PROMPT_PATTERNS,
            self.MEMORY_DUMP_PATTERNS,
            self.ENUMERATION_PATTERNS,
            self.CROSS_USER_PATTERNS,
            self.URL_EXFILTRATION_PATTERNS,
            self.POLICY_PATTERNS
        ]:
            for pattern, severity, message in patterns_list:
                self.compiled_patterns.append(
                    (re.compile(pattern, re.IGNORECASE), severity, message)
                )

    def detect(self, text: str) -> DetectorResult:
        """
        Detect exfiltration attempts in text.

        Args:
            text: Input text to scan

        Returns:
            DetectorResult with BLOCK if high-confidence exfiltration detected
        """
        if not self.enabled:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        rule_hits: List[RuleHit] = []

        # Check each pattern
        for pattern, severity, message in self.compiled_patterns:
            if pattern.search(text):
                # Avoid duplicate hits for same pattern
                rule_id = f"exfiltration.{message.lower().replace(' ', '_')[:30]}"
                if not any(hit.rule_id == rule_id for hit in rule_hits):
                    rule_hits.append(RuleHit(
                        rule_id=rule_id,
                        severity=severity,
                        message=message
                    ))

        # No exfiltration detected
        if not rule_hits:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        # Calculate risk score
        risk_score = self._calculate_risk_score(rule_hits)

        # Block if above threshold
        if risk_score >= self.block_threshold:
            return DetectorResult(
                decision=Decision.BLOCK,
                risk_score=risk_score,
                rule_hits=rule_hits,
                user_message="I can't help with requests to reveal internal instructions or private data.",
                developer_message=f"Exfiltration attempt detected: {len(rule_hits)} pattern(s) matched"
            )
        else:
            # Below threshold - warn
            return DetectorResult(
                decision=Decision.WARN,
                risk_score=risk_score,
                rule_hits=rule_hits,
                developer_message=f"Potential exfiltration patterns detected: {len(rule_hits)} match(es)"
            )

    @staticmethod
    def _calculate_risk_score(rule_hits: List[RuleHit]) -> int:
        """Calculate risk score based on severity"""
        return calculate_risk_score(rule_hits)
