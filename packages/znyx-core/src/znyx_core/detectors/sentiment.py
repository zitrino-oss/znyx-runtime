"""
Sentiment / Tone Detector.

Analyzes output sentiment and tone to enforce brand-appropriate responses.
Detects sarcasm, passive-aggression, condescension, dismissiveness, and
excessive negativity using lexicon-based scoring and tone pattern matching.

Different from Toxicity: catches responses that aren't hateful/threatening
but damage brand trust (e.g., a customer service bot being dismissive).
"""
import re
import logging
from typing import Any, Dict, List, Tuple

from znyx_core.core.models import DetectorResult, RuleHit, Severity, Decision
from znyx_core.core.risk import calculate_risk_score

logger = logging.getLogger(__name__)


# ── Tone patterns ──────────────────────────────────────────────────────────
# Each: (regex, tone_category, severity, message)

TonePattern = Tuple[re.Pattern, str, Severity, str]

_SARCASM_PATTERNS: List[TonePattern] = [
    (re.compile(r"\boh\s+(?:sure|great|wonderful|fantastic|brilliant|perfect)\b", re.I),
     "sarcastic", Severity.MEDIUM, "Sarcastic marker: '{match}'"),
    (re.compile(r"\bright,?\s+because\b", re.I),
     "sarcastic", Severity.MEDIUM, "Sarcastic 'right, because' pattern"),
    (re.compile(r"\bhow\s+(?:wonderful|lovely|delightful|nice)\s+(?:that|of)\b", re.I),
     "sarcastic", Severity.MEDIUM, "Sarcastic faux-positive"),
    (re.compile(r"\bwow,?\s+(?:just|really)\b", re.I),
     "sarcastic", Severity.LOW, "Potentially sarcastic 'wow'"),
    (re.compile(r"\byeah,?\s+(?:no|right|sure)\b", re.I),
     "sarcastic", Severity.MEDIUM, "Dismissive-sarcastic response"),
    (re.compile(r"\bshocker\b|\bwhat a surprise\b|\bcolor me shocked\b", re.I),
     "sarcastic", Severity.MEDIUM, "Sarcastic surprise expression"),
    (re.compile(r"\bgood luck with that\b", re.I),
     "sarcastic", Severity.MEDIUM, "Dismissive-sarcastic 'good luck with that'"),
]

_CONDESCENDING_PATTERNS: List[TonePattern] = [
    (re.compile(r"\bas\s+i\s+(?:already|just|previously)\s+(?:explained|said|told|mentioned)\b", re.I),
     "condescending", Severity.MEDIUM, "Condescending repetition marker: '{match}'"),
    (re.compile(r"\b(?:obviously|clearly|evidently)\b", re.I),
     "condescending", Severity.LOW, "Potentially condescending 'obviously/clearly'"),
    (re.compile(r"\bit(?:'s| is)\s+(?:really\s+)?(?:not\s+that|quite\s+)?(?:hard|difficult|complicated|complex)\b", re.I),
     "condescending", Severity.MEDIUM, "Minimizing difficulty: '{match}'"),
    (re.compile(r"\blet\s+me\s+(?:simplify|dumb\s+(?:it|this)\s+down|break\s+(?:it|this)\s+down\s+for\s+you)\b", re.I),
     "condescending", Severity.HIGH, "Condescending simplification: '{match}'"),
    (re.compile(r"\b(?:surely|certainly)\s+you\s+(?:know|understand|realize|can see)\b", re.I),
     "condescending", Severity.MEDIUM, "Condescending assumption of knowledge"),
    (re.compile(r"\bI\s+would\s+have\s+thought\s+(?:that|you)\b", re.I),
     "condescending", Severity.MEDIUM, "Condescending expectation expression"),
    (re.compile(r"\b(?:any|every)\s+(?:child|beginner|novice)\s+(?:would|could|can)\s+(?:understand|do|figure)\b", re.I),
     "condescending", Severity.HIGH, "Comparing to beginner/child: '{match}'"),
]

_PASSIVE_AGGRESSIVE_PATTERNS: List[TonePattern] = [
    (re.compile(r"\bfine\.\s*$", re.I | re.M),
     "passive_aggressive", Severity.MEDIUM, "Curt 'Fine.' response"),
    (re.compile(r"\bwhatever\s+you\s+(?:say|think|want|prefer)\b", re.I),
     "passive_aggressive", Severity.MEDIUM, "Passive-aggressive dismissal: '{match}'"),
    (re.compile(r"\bif\s+that(?:'s| is)\s+what\s+you\s+(?:think|want|believe)\b", re.I),
     "passive_aggressive", Severity.MEDIUM, "Passive-aggressive concession"),
    (re.compile(r"\bi\s+guess\s+(?:so|that's|you're|we)\b", re.I),
     "passive_aggressive", Severity.LOW, "Reluctant agreement: '{match}'"),
    (re.compile(r"\bno\s+worries\b.{0,20}\b(?:but|however|though)\b", re.I),
     "passive_aggressive", Severity.LOW, "'No worries' followed by contradiction"),
    (re.compile(r"\bwith\s+all\s+due\s+respect\b", re.I),
     "passive_aggressive", Severity.LOW, "'With all due respect' (often precedes disrespectful statement)"),
    (re.compile(r"\bper\s+my\s+(?:last|previous)\s+(?:email|message|note)\b", re.I),
     "passive_aggressive", Severity.MEDIUM, "Passive-aggressive reference to prior communication"),
]

_DISMISSIVE_PATTERNS: List[TonePattern] = [
    (re.compile(r"\bthat(?:'s| is)\s+not\s+(?:my|our)\s+(?:problem|concern|issue)\b", re.I),
     "dismissive", Severity.HIGH, "Dismissive of user's concern: '{match}'"),
    (re.compile(r"\bdon(?:'t| not)\s+worry\s+about\s+(?:it|that)\b", re.I),
     "dismissive", Severity.LOW, "Potentially dismissive 'don't worry about it'"),
    (re.compile(r"\bit\s+doesn(?:'t| not)\s+(?:matter|make a difference)\b", re.I),
     "dismissive", Severity.MEDIUM, "Dismissive 'it doesn't matter'"),
    (re.compile(r"\b(?:just|simply)\s+(?:deal\s+with\s+it|get\s+over\s+it|move\s+on)\b", re.I),
     "dismissive", Severity.HIGH, "Dismissive of concern: '{match}'"),
    (re.compile(r"\bnot\s+(?:my|our)\s+(?:job|responsibility|department)\b", re.I),
     "dismissive", Severity.HIGH, "Rejecting responsibility: '{match}'"),
    (re.compile(r"\b(?:you should have|you should've|why didn't you)\b", re.I),
     "dismissive", Severity.MEDIUM, "Blaming the user: '{match}'"),
]

# Combined into a category → patterns mapping
_ALL_TONE_PATTERNS: Dict[str, List[TonePattern]] = {
    "sarcastic": _SARCASM_PATTERNS,
    "condescending": _CONDESCENDING_PATTERNS,
    "passive_aggressive": _PASSIVE_AGGRESSIVE_PATTERNS,
    "dismissive": _DISMISSIVE_PATTERNS,
}

# ── Compact sentiment lexicon (core negative terms with valence) ───────────
# valence: -1.0 (very negative) to +1.0 (very positive)
# This is a minimal set; production would use VADER or similar (~7500 words)

_SENTIMENT_LEXICON: Dict[str, float] = {
    # ── Strong negative ──
    "terrible": -0.8, "horrible": -0.9, "awful": -0.8, "dreadful": -0.7,
    "disgusting": -0.8, "pathetic": -0.7, "useless": -0.7, "worthless": -0.9,
    "stupid": -0.7, "idiotic": -0.8, "ridiculous": -0.5, "absurd": -0.5,
    "incompetent": -0.7, "failure": -0.6, "disaster": -0.7, "catastrophe": -0.8,
    "nightmare": -0.7, "miserable": -0.7, "hopeless": -0.8, "pointless": -0.6,
    "waste": -0.5, "rubbish": -0.6, "garbage": -0.6, "trash": -0.6,
    "unacceptable": -0.6, "inexcusable": -0.7, "outrageous": -0.6,
    "frustrating": -0.5, "annoying": -0.5, "irritating": -0.5,
    "disappointing": -0.5, "unfortunate": -0.3, "regrettable": -0.4,
    "hate": -0.8, "despise": -0.8, "loathe": -0.8, "detest": -0.8,
    "never": -0.2, "nothing": -0.2, "nobody": -0.2, "nowhere": -0.2,
    "impossible": -0.4, "wrong": -0.4, "bad": -0.5, "worse": -0.6,
    "worst": -0.8, "poor": -0.4, "ugly": -0.5, "dumb": -0.6,
    # ── Mild negative ──
    "mediocre": -0.3, "bland": -0.3, "boring": -0.4, "tedious": -0.4,
    "confusing": -0.4, "clumsy": -0.4, "broken": -0.5, "flawed": -0.4,
    "lacking": -0.3, "inadequate": -0.5, "inferior": -0.5, "subpar": -0.4,
    "painful": -0.5, "stressful": -0.4, "exhausting": -0.4, "tiresome": -0.4,
    "offensive": -0.7, "hostile": -0.6, "rude": -0.6, "cruel": -0.7,
    "angry": -0.5, "furious": -0.7, "enraged": -0.8, "livid": -0.7,
    "depressing": -0.6, "grim": -0.5, "bleak": -0.5, "gloomy": -0.4,
    "scary": -0.4, "frightening": -0.5, "alarming": -0.5, "shocking": -0.5,
    "shameful": -0.6, "disgraceful": -0.7, "appalling": -0.7, "atrocious": -0.8,
    "abysmal": -0.8, "horrendous": -0.8, "ghastly": -0.7, "vile": -0.8,
    # ── Positive ──
    "good": 0.5, "great": 0.7, "excellent": 0.8, "amazing": 0.8,
    "wonderful": 0.8, "fantastic": 0.8, "outstanding": 0.9, "superb": 0.8,
    "brilliant": 0.8, "magnificent": 0.8, "perfect": 0.9, "beautiful": 0.7,
    "impressive": 0.7, "remarkable": 0.7, "exceptional": 0.8, "splendid": 0.7,
    "love": 0.7, "adore": 0.7, "enjoy": 0.5, "appreciate": 0.5,
    "pleased": 0.5, "satisfied": 0.5, "delighted": 0.7, "thrilled": 0.7,
    "happy": 0.6, "glad": 0.5, "joyful": 0.7, "cheerful": 0.6,
    "helpful": 0.5, "useful": 0.5, "valuable": 0.5, "effective": 0.5,
    "efficient": 0.5, "reliable": 0.5, "trustworthy": 0.6, "dependable": 0.5,
    "elegant": 0.6, "graceful": 0.5, "charming": 0.5, "lovely": 0.6,
    "awesome": 0.7, "incredible": 0.7, "marvelous": 0.7, "terrific": 0.7,
    "pleasant": 0.4, "nice": 0.4, "fine": 0.3, "decent": 0.3,
    "generous": 0.5, "kind": 0.5, "gentle": 0.4, "warm": 0.4,
    "exciting": 0.6, "inspiring": 0.6, "motivating": 0.5, "uplifting": 0.6,
    "confident": 0.5, "optimistic": 0.5, "hopeful": 0.4, "promising": 0.4,
    "innovative": 0.5, "creative": 0.5, "clever": 0.5, "smart": 0.5,
    "success": 0.6, "triumph": 0.7, "achievement": 0.6, "victory": 0.7,
    "recommend": 0.5, "praise": 0.6, "commend": 0.6, "celebrate": 0.6,
    # ── Neutral-ish modifiers ──
    "okay": 0.1, "alright": 0.1, "average": 0.0, "normal": 0.0,
}

_NEGATION_WORDS = {"not", "no", "never", "neither", "nor", "don't", "doesn't",
                   "didn't", "won't", "wouldn't", "couldn't", "shouldn't",
                   "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't"}

_INTENSIFIERS = {"very", "extremely", "incredibly", "absolutely", "totally",
                 "completely", "utterly", "really", "truly", "exceptionally"}


def _compute_sentiment(text: str, custom_lexicon: Dict[str, float] = None) -> float:
    """Compute a simple sentiment score from -1.0 to +1.0."""
    lexicon = dict(_SENTIMENT_LEXICON)
    if custom_lexicon:
        lexicon.update(custom_lexicon)

    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0

    total_valence = 0.0
    count = 0

    for i, word in enumerate(words):
        if word not in lexicon:
            continue

        valence = lexicon[word]

        # Check for negation in preceding 3 words
        preceding = words[max(0, i - 3):i]
        if any(w in _NEGATION_WORDS for w in preceding):
            valence *= -0.5  # Flip and reduce intensity

        # Check for intensifiers
        if any(w in _INTENSIFIERS for w in preceding):
            valence *= 1.5

        total_valence += valence
        count += 1

    if count == 0:
        return 0.0

    # Normalize to -1.0 to +1.0 range
    avg = total_valence / count
    return max(-1.0, min(1.0, avg))


class SentimentDetector:
    """Analyzes output tone and sentiment for brand-appropriate communication."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", False)
        self.action = config.get("action", "WARN")
        self.blocked_tones: List[str] = config.get(
            "blocked_tones",
            ["sarcastic", "condescending", "dismissive", "passive_aggressive"],
        )
        self.min_sentiment_score = config.get("min_sentiment_score", -0.3)
        self.custom_lexicon: Dict[str, float] = config.get("custom_lexicon", {})

    def detect(self, text: str) -> DetectorResult:
        if not self.enabled:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        if not text or len(text.strip()) < 15:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        rule_hits: List[RuleHit] = []

        # 1. Check tone patterns
        for tone, patterns in _ALL_TONE_PATTERNS.items():
            if tone not in self.blocked_tones:
                continue
            for pattern, tone_cat, severity, msg_template in patterns:
                for match in pattern.finditer(text):
                    matched_text = match.group(0)[:80]
                    rule_hits.append(RuleHit(
                        rule_id=f"sentiment.{tone_cat}",
                        severity=severity,
                        message=msg_template.format(match=matched_text),
                    ))

        # 2. Check overall sentiment
        sentiment = _compute_sentiment(text, self.custom_lexicon)
        if sentiment < self.min_sentiment_score:
            rule_hits.append(RuleHit(
                rule_id="sentiment.excessively_negative",
                severity=Severity.MEDIUM,
                message=f"Excessively negative sentiment (score={sentiment:.2f}, threshold={self.min_sentiment_score})",
            ))

        if not rule_hits:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        # Risk score
        risk_score = calculate_risk_score(rule_hits)

        decision = Decision.BLOCK if self.action == "BLOCK" else Decision.WARN

        return DetectorResult(
            decision=decision,
            risk_score=risk_score,
            rule_hits=rule_hits,
            developer_message=f"sentiment: {len(rule_hits)} tone issues detected (sentiment={sentiment:.2f})",
        )
