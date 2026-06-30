"""
Bias / Fairness Detector.

Detects demographic bias across protected attributes in LLM outputs.
Catches subtle differential treatment, stereotyping, and exclusionary language
that wouldn't trigger a toxicity detector.
"""
import re
import logging
from typing import Any, Dict, List, Tuple

from app.shared.core.models import DetectorResult, RuleHit, Severity, Decision
from app.shared.core.risk import calculate_risk_score

logger = logging.getLogger(__name__)

# ── Negation context detection ─────────────────────────────────────────────

# Phrases that negate or challenge the stereotype that follows.
# We check the text immediately before a match (up to 60 chars) for these.
_NEGATION_PREFIX_RE = re.compile(
    r"(?:"
    r"(?:it(?:'s| is)\s+(?:wrong|incorrect|false|a myth|a stereotype|untrue|inaccurate)\s+(?:to say|to claim|to think|to assume|to suggest|that))"
    r"|(?:we should(?:n't| not)\s+(?:say|assume|think|claim|suggest|believe))"
    r"|(?:(?:should|we)\s+avoid\s+(?:saying|assuming|thinking|claiming|suggesting))"
    r"|(?:(?:it is|it's)\s+(?:not true|unfair|biased|prejudiced|stereotypical)\s+(?:to say|to claim|that))"
    r"|(?:not\s+(?:true|accurate|correct|fair)\s+(?:that|to say))"
    r"|(?:don(?:'t|t)\s+(?:say|think|assume|believe|claim)\s+(?:that)?)"
    r"|(?:(?:they|these|this|those|such\s+claims?)\s+(?:are|is)\s+(?:not|NOT|n't))"
    r"|(?:(?:statement|claim|idea|notion|view|belief)\s+(?:that\s+)?(?:women|men|people|anyone)\s+(?:are|is)\s+(?:not|NOT|n't))"
    r")(?:\s+that)?\s*$",
    re.I,
)


# ── Pattern databases per protected attribute ──────────────────────────────

# Each entry: (compiled regex, severity, message template)
# Patterns are designed to catch *bias* not *hate speech* (toxicity handles that)

_GENDER_PATTERNS: List[Tuple[re.Pattern, Severity, str]] = [
    # Stereotyping
    (re.compile(r"\b(?:women|females|girls)\s+(?:are|tend to be|are typically|are naturally)\s+(?:emotional|irrational|nurturing|submissive|indecisive|gossipy|sensitive|bad at)", re.I),
     Severity.HIGH, "Gender stereotype about women: {match}"),
    (re.compile(r"\b(?:men|males|boys)\s+(?:are|tend to be|are typically|are naturally)\s+(?:logical|rational|strong|dominant|aggressive|better at|natural leaders)", re.I),
     Severity.HIGH, "Gender stereotype about men: {match}"),
    # Job assumptions
    (re.compile(r"\b(?:she|her|woman|female)\b.{0,30}\b(?:nurse|secretary|assistant|receptionist|teacher|caregiver)\b", re.I),
     Severity.MEDIUM, "Gendered job assumption: {match}"),
    (re.compile(r"\b(?:he|his|man|male)\b.{0,30}\b(?:engineer|ceo|executive|leader|surgeon|developer|architect)\b.{0,30}\b(?:while|whereas|but|and)\b.{0,30}\b(?:she|her|woman|female)\b.{0,30}\b(?:support|assist|help|admin)", re.I),
     Severity.HIGH, "Differential role assignment by gender: {match}"),
    # Trait assumptions
    (re.compile(r"\b(?:women|girls|females)\s+(?:shouldn't|should not|can't|cannot|don't|aren't able to)\s+\w+", re.I),
     Severity.HIGH, "Limiting assumption about women: {match}"),
    (re.compile(r"\b(?:men|boys|males)\s+(?:shouldn't|should not|don't|can't)\s+(?:cry|show emotion|be sensitive|be nurturing)", re.I),
     Severity.MEDIUM, "Gender role enforcement: {match}"),
    (re.compile(r"\b(?:women|females|girls)\s+(?:are not|aren't|are never)\s+(?:suited|fit|good enough|capable|qualified)\s+(?:for|to)\b", re.I),
     Severity.HIGH, "Gender capability limitation: {match}"),
]

_RACE_PATTERNS: List[Tuple[re.Pattern, Severity, str]] = [
    # Stereotyping by race/ethnicity
    (re.compile(r"\b(?:asian|asians)\s+(?:are|tend to be)\s+(?:good at math|smart|hardworking|quiet|submissive|model minority)", re.I),
     Severity.HIGH, "Racial stereotype: {match}"),
    (re.compile(r"\b(?:black|african.american)\s+(?:people|individuals|communities)\s+(?:are|tend to be)\s+(?:athletic|aggressive|loud|lazy|criminal|dangerous)", re.I),
     Severity.HIGH, "Racial stereotype: {match}"),
    (re.compile(r"\b(?:hispanic|latino|latina)\s+(?:people|individuals|communities)\s+(?:are|tend to be)\s+(?:lazy|illegal|uneducated|hot.tempered)", re.I),
     Severity.HIGH, "Racial stereotype: {match}"),
    (re.compile(r"\b(?:white|caucasian)\s+(?:people|individuals)\s+(?:are|tend to be)\s+(?:superior|smarter|more civilized|normal)", re.I),
     Severity.HIGH, "Racial superiority claim: {match}"),
    # Differential treatment in professional contexts
    (re.compile(r"\b(?:diverse|minority)\s+(?:candidates?|applicants?|employees?)\s+(?:may not|might not|probably won't|tend to lack|often struggle)", re.I),
     Severity.HIGH, "Differential professional assumption by race: {match}"),
    # Racialized language
    (re.compile(r"\b(?:articulate|well-spoken)\b.{0,20}\b(?:for a|considering|despite being)\b", re.I),
     Severity.MEDIUM, "Potentially racialized qualifier: {match}"),
    # Vague ethnic group generalizations (no named race required)
    (re.compile(r"\b(?:that|this|those|certain)\s+(?:ethnic|racial)\s+(?:background|group|community|ethnicity)\b.{0,60}\b(?:tend to be|are typically|are generally|are)\s+(?:less|not|unreliable|untrustworthy|problematic|risky|dangerous)", re.I),
     Severity.HIGH, "Ethnic group generalization: {match}"),
]

_AGE_PATTERNS: List[Tuple[re.Pattern, Severity, str]] = [
    (re.compile(r"\b(?:older|elderly|senior)\s+(?:workers?|employees?|people|individuals)\s+(?:can't|cannot|aren't able to|struggle to|are too slow to)\s+(?:learn|adapt|keep up|use technology)", re.I),
     Severity.HIGH, "Age discrimination: {match}"),
    (re.compile(r"\b(?:too old|past their prime|over the hill)\s+(?:to|for)\b", re.I),
     Severity.HIGH, "Ageist language: {match}"),
    (re.compile(r"\b(?:young|younger)\s+(?:workers?|employees?|people)\s+(?:are|tend to be)\s+(?:lazy|entitled|unreliable|irresponsible|immature)", re.I),
     Severity.MEDIUM, "Age stereotype about younger people: {match}"),
    (re.compile(r"\b(?:millennials?|gen.?z|boomers?)\s+(?:are|always|never|can't|don't)", re.I),
     Severity.MEDIUM, "Generational stereotype: {match}"),
    (re.compile(r"\b(?:people|workers?|employees?|candidates?|anyone)\s+(?:over|above|past)\s+\d+\s+(?:should\s+not\s+be|shouldn't\s+be|can't\s+be|cannot\s+be|must\s+not\s+be)\s+(?:hired|employed|considered|recruited)", re.I),
     Severity.HIGH, "Age-based hiring discrimination: {match}"),
    (re.compile(r"\b(?:people|workers?|employees?|candidates?|those)\s+(?:over|above|past|aged?)\s+\d+\s+.{0,60}(?:can't|cannot|won't|are unable to)\s+(?:learn|adapt|keep up|use technology|pick up)", re.I),
     Severity.HIGH, "Age-based capability assumption: {match}"),
]

_DISABILITY_PATTERNS: List[Tuple[re.Pattern, Severity, str]] = [
    (re.compile(r"\b(?:disabled|handicapped)\s+(?:people|individuals|workers?|employees?)\s+(?:can't|cannot|aren't able to|will never|shouldn't)\b", re.I),
     Severity.HIGH, "Ableist assumption of limitation: {match}"),
    (re.compile(r"\b(?:suffering from|afflicted with|victim of|confined to)\s+(?:a disability|a wheelchair|autism|blindness|deafness)", re.I),
     Severity.MEDIUM, "Disempowering disability language: {match}"),
    (re.compile(r"\b(?:normal|healthy)\s+(?:people|individuals)\b.{0,20}\b(?:unlike|compared to|versus)\b.{0,20}\b(?:disabled|handicapped)", re.I),
     Severity.MEDIUM, "Othering disability language: {match}"),
    (re.compile(r"\b(?:mentally?\s+(?:retarded|deficient|defective)|crippled|lame|deaf to reason|blind to)\b", re.I),
     Severity.HIGH, "Ableist slur or metaphor: {match}"),
]

_RELIGION_PATTERNS: List[Tuple[re.Pattern, Severity, str]] = [
    (re.compile(r"\b(?:muslims?|islam)\s+(?:are|is|promotes?)\s+(?:violent|terrorist|extremist|backwards|oppressive)", re.I),
     Severity.HIGH, "Religious stereotype: {match}"),
    (re.compile(r"\b(?:jews?|jewish people)\s+(?:are|control|run|own)\s+(?:greedy|cheap|banks?|media|the world)", re.I),
     Severity.HIGH, "Religious stereotype: {match}"),
    (re.compile(r"\b(?:christians?|christianity)\s+(?:are|is)\s+(?:ignorant|stupid|anti-science|backwards|bigoted)", re.I),
     Severity.HIGH, "Religious stereotype: {match}"),
    (re.compile(r"\b(?:hindus?|hinduism)\s+(?:are|is)\s+(?:primitive|idol.worship|backwards|caste)", re.I),
     Severity.HIGH, "Religious stereotype: {match}"),
]

# ── Industry-specific presets ──────────────────────────────────────────────

_HR_PATTERNS: List[Tuple[re.Pattern, Severity, str]] = [
    # Hiring bias
    (re.compile(r"\b(?:culture fit|cultural fit)\b.{0,40}\b(?:might not|probably not|unlikely|doesn't seem)", re.I),
     Severity.HIGH, "Potential proxy bias in hiring via 'culture fit': {match}"),
    (re.compile(r"\b(?:maternity|pregnancy|pregnant|children|family planning)\b.{0,30}\b(?:risk|concern|liability|problem|issue)", re.I),
     Severity.HIGH, "Pregnancy/family status bias in employment: {match}"),
    (re.compile(r"\b(?:name|names?)\s+(?:sounds?|seems?|appears?|looks?)\s+(?:foreign|ethnic|unusual|hard to pronounce)", re.I),
     Severity.HIGH, "Name-based bias: {match}"),
    # Compensation bias
    (re.compile(r"\b(?:she|her|women|female)\b.{0,30}\b(?:lower|less|reduced)\s+(?:salary|compensation|pay|rate)", re.I),
     Severity.HIGH, "Gender-based compensation differential: {match}"),
    (re.compile(r"\b(?:overqualified)\b.{0,30}\b(?:age|senior|older|experienced)", re.I),
     Severity.MEDIUM, "Age-coded 'overqualified' language: {match}"),
]

_FINANCE_PATTERNS: List[Tuple[re.Pattern, Severity, str]] = [
    # Lending bias
    (re.compile(r"\b(?:neighborhood|zip.?code|area|location)\b.{0,30}\b(?:higher risk|risky|declined|deny|reject)", re.I),
     Severity.HIGH, "Potential redlining/geographic bias in lending: {match}"),
    (re.compile(r"\b(?:single mother|single parent|unmarried)\b.{0,30}\b(?:risk|risky|unreliable|default)", re.I),
     Severity.HIGH, "Marital/family status bias in financial assessment: {match}"),
    (re.compile(r"\b(?:immigrant|foreign|non-citizen)\b.{0,30}\b(?:higher risk|risky|untrustworthy|unreliable)", re.I),
     Severity.HIGH, "National origin bias in financial assessment: {match}"),
]


class BiasDetector:
    """Detects demographic bias and unfair treatment in text."""

    # Attribute name → pattern list mapping
    _ATTRIBUTE_PATTERNS = {
        "gender": _GENDER_PATTERNS,
        "race": _RACE_PATTERNS,
        "age": _AGE_PATTERNS,
        "disability": _DISABILITY_PATTERNS,
        "religion": _RELIGION_PATTERNS,
    }

    _INDUSTRY_PATTERNS = {
        "hr": _HR_PATTERNS,
        "finance": _FINANCE_PATTERNS,
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", False)
        self.action = config.get("action", "WARN")
        self.sensitivity = config.get("sensitivity", "medium")  # low, medium, high

        # Which attributes to check
        requested = config.get("protected_attributes", list(self._ATTRIBUTE_PATTERNS.keys()))
        self.active_attributes = [a for a in requested if a in self._ATTRIBUTE_PATTERNS]

        # Industry preset
        self.industry_preset = config.get("industry_preset", "none")

        # Custom patterns
        self.custom_patterns: List[Tuple[re.Pattern, Severity, str]] = []
        for pat_cfg in config.get("custom_patterns", []):
            try:
                self.custom_patterns.append((
                    re.compile(pat_cfg["pattern"], re.I),
                    Severity(pat_cfg.get("severity", "medium")),
                    pat_cfg.get("message", "Custom bias pattern matched: {match}"),
                ))
            except (re.error, KeyError, ValueError):
                continue

    def _get_severity_filter(self) -> set:
        """Return which severities to report based on sensitivity setting."""
        if self.sensitivity == "high":
            return {Severity.LOW, Severity.MEDIUM, Severity.HIGH}
        elif self.sensitivity == "low":
            return {Severity.HIGH}
        else:  # medium (default)
            return {Severity.MEDIUM, Severity.HIGH}

    @staticmethod
    def _is_negated(text: str, match_start: int) -> bool:
        """Check whether the text preceding a match contains negation context.

        Looks at up to 80 characters before *match_start* for phrases that
        challenge, deny, or refute the stereotype (e.g. "it's wrong to say
        that", "are not", "we shouldn't assume").
        """
        prefix_start = max(0, match_start - 80)
        prefix = text[prefix_start:match_start]
        return _NEGATION_PREFIX_RE.search(prefix) is not None

    def _scan_patterns(
        self, text: str, patterns: List[Tuple[re.Pattern, Severity, str]]
    ) -> List[RuleHit]:
        """Scan text against a list of patterns, return rule hits.

        If a match is preceded by negation context the hit severity is
        downgraded: HIGH → LOW, MEDIUM/LOW → skipped entirely.  This avoids
        false-positives on sentences that *challenge* stereotypes.
        """
        hits: List[RuleHit] = []
        severity_filter = self._get_severity_filter()

        for pattern, severity, msg_template in patterns:
            if severity not in severity_filter:
                continue
            for match in pattern.finditer(text):
                effective_severity = severity
                if self._is_negated(text, match.start()):
                    # Downgrade: HIGH → LOW, anything else → skip
                    if severity == Severity.HIGH:
                        effective_severity = Severity.LOW
                    else:
                        continue
                    # After downgrade the hit may fall outside the filter
                    if effective_severity not in severity_filter:
                        continue

                matched_text = match.group(0)[:120]
                hits.append(RuleHit(
                    rule_id=f"bias.{effective_severity.value}_match",
                    severity=effective_severity,
                    message=msg_template.format(match=matched_text),
                ))
        return hits

    def detect(self, text: str) -> DetectorResult:
        if not self.enabled:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        if not text or len(text.strip()) < 10:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        all_hits: List[RuleHit] = []

        # Check each active protected attribute
        for attr in self.active_attributes:
            patterns = self._ATTRIBUTE_PATTERNS[attr]
            hits = self._scan_patterns(text, patterns)
            # Tag rule_id with attribute name
            for hit in hits:
                hit.rule_id = f"bias.{attr}_{hit.severity.value}"
            all_hits.extend(hits)

        # Industry-specific patterns
        if self.industry_preset in self._INDUSTRY_PATTERNS:
            industry_hits = self._scan_patterns(
                text, self._INDUSTRY_PATTERNS[self.industry_preset]
            )
            for hit in industry_hits:
                hit.rule_id = f"bias.{self.industry_preset}_{hit.severity.value}"
            all_hits.extend(industry_hits)

        # Custom patterns
        if self.custom_patterns:
            custom_hits = self._scan_patterns(text, self.custom_patterns)
            for hit in custom_hits:
                hit.rule_id = f"bias.custom_{hit.severity.value}"
            all_hits.extend(custom_hits)

        if not all_hits:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        # Calculate risk score
        risk_score = calculate_risk_score(all_hits)

        decision = Decision.BLOCK if self.action == "BLOCK" else Decision.WARN

        return DetectorResult(
            decision=decision,
            risk_score=risk_score,
            rule_hits=all_hits,
            user_message="The response may contain biased or discriminatory content.",
            developer_message=f"bias: {len(all_hits)} bias patterns detected across protected attributes",
        )
