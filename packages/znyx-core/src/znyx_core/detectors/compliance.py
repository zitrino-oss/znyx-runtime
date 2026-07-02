"""
Regulatory Compliance Detector.

Enforces industry-specific regulatory requirements:
- Mandatory disclaimers (medical, financial, legal)
- AI disclosure statements
- Prohibited claims (guaranteed outcomes, unapproved medical claims)
- Jurisdiction-aware content rules

Unique APPEND action: can append required disclaimers to output.
"""
import re
import logging
from typing import Any, Dict, List, Tuple

from znyx_core.core.models import DetectorResult, RuleHit, Severity, Decision
from znyx_core.core.risk import calculate_risk_score

logger = logging.getLogger(__name__)


# ── Industry Rule Packs ────────────────────────────────────────────────────

# Each rule: (topic_keywords, prohibited_patterns, required_disclaimer_patterns, disclaimer_text)

class _IndustryRule:
    """A single compliance rule: topic detection → check requirements."""

    def __init__(
        self,
        name: str,
        topic_keywords: List[str],
        prohibited_patterns: List[Tuple[re.Pattern, str]],
        required_patterns: List[re.Pattern],
        disclaimer_text: str,
    ):
        self.name = name
        self.topic_keywords = [k.lower() for k in topic_keywords]
        self.prohibited_patterns = prohibited_patterns
        self.required_patterns = required_patterns
        self.disclaimer_text = disclaimer_text

    def topic_matches(self, text_lower: str) -> bool:
        """Check if the text discusses this rule's topic."""
        matches = sum(1 for kw in self.topic_keywords if kw in text_lower)
        return matches >= 1  # Require at least 1 keyword match

    def has_disclaimer(self, text_lower: str) -> bool:
        """Check if any required disclaimer pattern is present."""
        return any(p.search(text_lower) for p in self.required_patterns)


# Healthcare rules
_HEALTHCARE_RULES: List[_IndustryRule] = [
    _IndustryRule(
        name="medical_advice",
        topic_keywords=[
            "diagnosis", "diagnose", "treatment", "medication", "dosage",
            "symptom", "symptoms", "prescribe", "prescription", "medical",
            "disease", "condition", "therapy", "surgery", "prognosis",
            "side effect", "drug", "clinical", "patient",
        ],
        prohibited_patterns=[
            (re.compile(r"\b(?:you (?:have|definitely have|clearly have|are suffering from))\s+\w+", re.I),
             "Diagnostic claim without qualification"),
            (re.compile(r"\b(?:take|use|consume)\s+\d+\s*(?:mg|ml|mcg|tablets?|pills?|capsules?)\b", re.I),
             "Specific dosage recommendation"),
            (re.compile(r"\b(?:guaranteed|certain|definite)\s+(?:cure|treatment|recovery|remedy)\b", re.I),
             "Guaranteed medical outcome claim"),
            (re.compile(r"\b(?:stop taking|discontinue|don't take)\s+(?:your|the)\s+(?:medication|medicine|pills?)\b", re.I),
             "Instruction to stop prescribed medication"),
        ],
        required_patterns=[
            re.compile(r"consult\s+(?:a|your)\s+(?:doctor|physician|healthcare|medical)\s+(?:professional|provider|practitioner)", re.I),
            re.compile(r"not\s+(?:a\s+)?(?:substitute\s+for|replacement\s+for)\s+(?:professional|medical)\s+(?:medical\s+)?advice", re.I),
            re.compile(r"seek\s+(?:professional\s+)?medical\s+(?:advice|attention|help)", re.I),
            re.compile(r"for\s+informational\s+purposes\s+only", re.I),
        ],
        disclaimer_text="Please consult a qualified healthcare professional before making any medical decisions. This information is for educational purposes only and is not a substitute for professional medical advice.",
    ),
]

_FINANCE_RULES: List[_IndustryRule] = [
    _IndustryRule(
        name="financial_advice",
        topic_keywords=[
            "investment", "invest", "stock", "bond", "portfolio",
            "returns", "dividend", "trading", "market", "financial",
            "retirement", "401k", "ira", "mutual fund", "etf",
            "crypto", "bitcoin", "forex", "securities", "wealth",
        ],
        prohibited_patterns=[
            (re.compile(r"\b(?:guaranteed|certain|definite|sure)\s+(?:returns?|profit|gains?|income)\b", re.I),
             "Guaranteed returns claim"),
            (re.compile(r"\b(?:guaranteed|certain|definite|sure)\s+(?:investment|strategy|way|method)\b", re.I),
             "Guaranteed investment strategy claim"),
            (re.compile(r"\b(?:double|triple|10x|multiply)\s+(?:your|my|their)\s+(?:money|investment|savings|wealth)\b", re.I),
             "Unrealistic return claim"),
            (re.compile(r"\bto\s+(?:double|triple|10x)\s+(?:your|my|their)?\s*(?:money|investment|savings)\b", re.I),
             "Unrealistic return claim"),
            (re.compile(r"\b(?:risk[- ]free|no[- ]risk|zero[- ]risk)\s+(?:investment|opportunity|return)\b", re.I),
             "Risk-free investment claim"),
            (re.compile(r"\b(?:will|going to|definitely)\s+(?:go up|increase|rise|double|triple|10x)\b", re.I),
             "Specific price prediction"),
            (re.compile(r"\b(?:guaranteed|certain|definite)\s+(?:to\s+)?(?:rise|increase|go up|double|triple|outperform)\b", re.I),
             "Guaranteed price movement claim"),
            (re.compile(r"\b(?:can't lose|impossible to lose|never lose)\b", re.I),
             "Misleading loss-prevention claim"),
            (re.compile(r"\b(?:buy|sell|short)\s+(?:now|immediately|today|right away)\b.{0,20}\b(?:before|while)\b", re.I),
             "False urgency in financial recommendation"),
            (re.compile(r"\bwhich\s+(?:specific\s+)?stocks?\s+to\s+buy\b", re.I),
             "Specific stock-picking advice"),
            (re.compile(r"\b(?:tell|advise|recommend)\s+(?:me|you|them)\s+(?:which\s+)?(?:stocks?|shares?|crypto|coins?)\s+to\s+(?:buy|sell|invest)\b", re.I),
             "Specific investment recommendation"),
        ],
        required_patterns=[
            re.compile(r"past\s+performance\s+(?:does\s+not|is\s+no)\s+guarantee", re.I),
            re.compile(r"not\s+(?:a\s+)?(?:financial|investment)\s+advice", re.I),
            re.compile(r"consult\s+(?:a|your)\s+(?:financial\s+)?(?:advisor|adviser|professional|planner)", re.I),
            re.compile(r"involves?\s+(?:risk|risks)", re.I),
            re.compile(r"may\s+(?:lose|result in (?:a )?loss)", re.I),
        ],
        disclaimer_text="This is not financial advice. Investing involves risk, including the possible loss of principal. Past performance does not guarantee future results. Consult a qualified financial advisor before making investment decisions.",
    ),
]

_LEGAL_RULES: List[_IndustryRule] = [
    _IndustryRule(
        name="legal_advice",
        topic_keywords=[
            "lawsuit", "sue", "legal", "court", "attorney", "lawyer",
            "liability", "damages", "statute", "regulation", "contract",
            "litigation", "settlement", "jurisdiction", "rights",
            "defendant", "plaintiff", "verdict", "appeal",
        ],
        prohibited_patterns=[
            (re.compile(r"\b(?:you should|you must|you need to)\s+(?:sue|file|litigate|take legal action)\b", re.I),
             "Specific legal action recommendation"),
            (re.compile(r"\b(?:you will|you'll|you are going to)\s+(?:win|lose|prevail)\b", re.I),
             "Outcome prediction in legal matter"),
            (re.compile(r"\b(?:definitely|certainly|clearly)\s+(?:liable|guilty|negligent|at fault)\b", re.I),
             "Definitive legal conclusion"),
        ],
        required_patterns=[
            re.compile(r"not\s+(?:a\s+)?(?:legal|substitute for legal)\s+advice", re.I),
            re.compile(r"consult\s+(?:a|an|your)\s+(?:attorney|lawyer|legal\s+professional|legal\s+counsel)", re.I),
            re.compile(r"for\s+(?:general\s+)?informational\s+purposes", re.I),
            re.compile(r"seek\s+(?:qualified\s+)?legal\s+(?:advice|counsel|representation)", re.I),
        ],
        disclaimer_text="This is not legal advice. Please consult a qualified attorney for advice specific to your situation. Laws vary by jurisdiction.",
    ),
]

_INSURANCE_RULES: List[_IndustryRule] = [
    _IndustryRule(
        name="insurance_advice",
        topic_keywords=[
            "insurance", "coverage", "policy", "premium", "deductible",
            "claim", "underwriting", "actuary", "beneficiary", "rider",
        ],
        prohibited_patterns=[
            (re.compile(r"\b(?:guaranteed|certain|definite)\s+(?:coverage|payout|approval)\b", re.I),
             "Guaranteed insurance outcome claim"),
            (re.compile(r"\b(?:you don't need|skip|avoid)\s+(?:insurance|coverage)\b", re.I),
             "Recommendation against insurance coverage"),
        ],
        required_patterns=[
            re.compile(r"consult\s+(?:a|your)\s+(?:insurance|licensed)\s+(?:agent|broker|professional|advisor)", re.I),
            re.compile(r"(?:coverage|policies?)\s+(?:vary|differ|depend)", re.I),
            re.compile(r"review\s+(?:your|the)\s+(?:policy|terms|conditions)", re.I),
        ],
        disclaimer_text="Insurance coverage varies by provider, policy, and jurisdiction. Consult a licensed insurance agent or broker for advice specific to your needs.",
    ),
]

_INDUSTRY_RULES: Dict[str, List[_IndustryRule]] = {
    "healthcare": _HEALTHCARE_RULES,
    "finance": _FINANCE_RULES,
    "legal": _LEGAL_RULES,
    "insurance": _INSURANCE_RULES,
}

# Maps frontend 'framework' values to backend 'industry' keys
_FRAMEWORK_MAP: Dict[str, str] = {
    "hipaa":   "healthcare",
    "gdpr":    "healthcare",
    "pci_dss": "finance",
    "ferpa":   "healthcare",
    "mixed":   "all",
    "all":     "all",
}

# General rules (applicable to all industries)
_GENERAL_PROHIBITED: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:100%|absolutely|completely)\s+(?:guaranteed|certain|safe|risk[- ]free)\b", re.I),
     "Absolute guarantee claim"),
    (re.compile(r"\b(?:act now|hurry|limited time|offer expires|only \d+ left)\b", re.I),
     "False urgency or scarcity tactic"),
]


class ComplianceDetector:
    """Enforces industry-specific regulatory compliance in LLM outputs."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", False)
        self.action = config.get("action", "WARN")  # WARN, BLOCK, APPEND
        raw_framework = config.get("framework") or config.get("industry", "all")
        self.industry = _FRAMEWORK_MAP.get(raw_framework, raw_framework)
        self.ai_disclosure = config.get("ai_disclosure", False)
        self.ai_disclosure_text = config.get(
            "ai_disclosure_text",
            "This response was generated by AI and may not be fully accurate."
        )

        # Custom rules
        self.custom_rules: List[_IndustryRule] = []
        for rule_cfg in config.get("custom_rules", []):
            try:
                prohibited = [
                    (re.compile(p["pattern"], re.I), p.get("message", "Custom prohibited pattern"))
                    for p in rule_cfg.get("prohibited_patterns", [])
                ]
                required = [
                    re.compile(p, re.I) for p in rule_cfg.get("required_text", [])
                ]
                self.custom_rules.append(_IndustryRule(
                    name=rule_cfg.get("name", "custom"),
                    topic_keywords=rule_cfg.get("topic_keywords", []),
                    prohibited_patterns=prohibited,
                    required_patterns=required,
                    disclaimer_text=rule_cfg.get("disclaimer_text", ""),
                ))
            except (re.error, KeyError, TypeError):
                continue

        # Custom prohibited claims (simple patterns)
        self.prohibited_claims: List[Tuple[re.Pattern, str]] = []
        for claim in config.get("prohibited_claims", []):
            if isinstance(claim, str):
                try:
                    self.prohibited_claims.append(
                        (re.compile(claim, re.I), f"Prohibited claim pattern: {claim}")
                    )
                except re.error:
                    continue

        # Custom required disclaimers
        self.required_disclaimers: List[Dict] = config.get("required_disclaimers", [])

    def detect(self, text: str) -> DetectorResult:
        if not self.enabled:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        if not text or len(text.strip()) < 20:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        text_lower = text.lower()
        rule_hits: List[RuleHit] = []
        disclaimers_to_append: List[str] = []

        # Check industry-specific rules
        if self.industry == "all":
            rules = [r for ruleset in _INDUSTRY_RULES.values() for r in ruleset] + self.custom_rules
        else:
            rules = _INDUSTRY_RULES.get(self.industry, []) + self.custom_rules

        for rule in rules:
            if not rule.topic_matches(text_lower):
                continue

            # Check prohibited patterns
            for pattern, message in rule.prohibited_patterns:
                if pattern.search(text):
                    rule_hits.append(RuleHit(
                        rule_id=f"compliance.prohibited_claim.{rule.name}",
                        severity=Severity.HIGH,
                        message=message,
                    ))

            # Check required disclaimers (skip if disclaimer already present in text)
            if not rule.has_disclaimer(text_lower) and rule.disclaimer_text:
                # Avoid duplicate disclaimers if text already contains it
                if rule.disclaimer_text.lower() not in text_lower:
                    rule_hits.append(RuleHit(
                        rule_id=f"compliance.missing_disclaimer.{rule.name}",
                        severity=Severity.MEDIUM,
                        message=f"Missing required disclaimer for {rule.name} content",
                    ))
                    if rule.disclaimer_text not in disclaimers_to_append:
                        disclaimers_to_append.append(rule.disclaimer_text)

        # General prohibited patterns
        for pattern, message in _GENERAL_PROHIBITED:
            if pattern.search(text):
                rule_hits.append(RuleHit(
                    rule_id="compliance.general_prohibited",
                    severity=Severity.MEDIUM,
                    message=message,
                ))

        # Custom prohibited claims
        for pattern, message in self.prohibited_claims:
            if pattern.search(text):
                rule_hits.append(RuleHit(
                    rule_id="compliance.custom_prohibited",
                    severity=Severity.HIGH,
                    message=message,
                ))

        # AI disclosure check
        if self.ai_disclosure:
            ai_markers = [
                "generated by ai", "ai-generated", "artificial intelligence",
                "language model", "ai assistant", "automated response",
            ]
            has_disclosure = any(m in text_lower for m in ai_markers)
            if not has_disclosure:
                rule_hits.append(RuleHit(
                    rule_id="compliance.missing_ai_disclosure",
                    severity=Severity.LOW,
                    message="AI disclosure statement missing from response",
                ))
                disclaimers_to_append.append(self.ai_disclosure_text)

        if not rule_hits:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        # Risk score
        risk_score = calculate_risk_score(rule_hits)

        # Determine decision and handle APPEND action
        sanitized_text = None
        if self.action == "APPEND" and disclaimers_to_append:
            # Append disclaimers to the output
            combined_disclaimer = "\n\n---\n" + "\n\n".join(
                f"**Disclaimer:** {d}" for d in disclaimers_to_append
            )
            sanitized_text = text + combined_disclaimer
            decision = Decision.TRANSFORM
        elif self.action == "BLOCK":
            decision = Decision.BLOCK
        else:
            decision = Decision.WARN

        return DetectorResult(
            decision=decision,
            risk_score=risk_score,
            rule_hits=rule_hits,
            sanitized_text=sanitized_text,
            user_message="The response requires compliance review." if decision == Decision.BLOCK else None,
            developer_message=f"compliance: {len(rule_hits)} compliance issues ({self.industry} industry)",
        )
