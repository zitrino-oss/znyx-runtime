"""
Language Detection Detector.

Detects the language of input/output and enforces allowed/blocked language policies.
Prevents language-switching bypass attacks where attackers switch to languages
with weaker safety training.

Uses Unicode range analysis + trigram frequency matching (zero external deps).
"""
import re
import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from znyx_core.core.models import DetectorResult, RuleHit, Severity, Decision

logger = logging.getLogger(__name__)


# ── Unicode script ranges ──────────────────────────────────────────────────

_SCRIPT_RANGES: List[Tuple[str, int, int]] = [
    ("latin", 0x0000, 0x024F),
    ("cyrillic", 0x0400, 0x04FF),
    ("greek", 0x0370, 0x03FF),
    ("arabic", 0x0600, 0x06FF),
    ("hebrew", 0x0590, 0x05FF),
    ("devanagari", 0x0900, 0x097F),
    ("tamil", 0x0B80, 0x0BFF),
    ("thai", 0x0E00, 0x0E7F),
    ("cjk", 0x4E00, 0x9FFF),
    ("hangul", 0xAC00, 0xD7AF),
    ("katakana", 0x30A0, 0x30FF),
    ("hiragana", 0x3040, 0x309F),
    ("bengali", 0x0980, 0x09FF),
    ("gujarati", 0x0A80, 0x0AFF),
    ("telugu", 0x0C00, 0x0C7F),
    ("kannada", 0x0C80, 0x0CFF),
    ("malayalam", 0x0D00, 0x0D7F),
    ("georgian", 0x10A0, 0x10FF),
    ("armenian", 0x0530, 0x058F),
    ("ethiopic", 0x1200, 0x137F),
]

# Script → likely language(s) mapping
_SCRIPT_LANGUAGES: Dict[str, List[str]] = {
    "cyrillic": ["ru", "uk", "bg", "sr"],
    "greek": ["el"],
    "arabic": ["ar", "fa", "ur"],
    "hebrew": ["he"],
    "devanagari": ["hi", "mr", "ne"],
    "tamil": ["ta"],
    "thai": ["th"],
    "cjk": ["zh", "ja"],
    "hangul": ["ko"],
    "katakana": ["ja"],
    "hiragana": ["ja"],
    "bengali": ["bn"],
    "gujarati": ["gu"],
    "telugu": ["te"],
    "kannada": ["kn"],
    "malayalam": ["ml"],
    "georgian": ["ka"],
    "armenian": ["hy"],
    "ethiopic": ["am"],
}


# ── Trigram frequency profiles for Latin-script languages ──────────────────
# Top 30 trigrams per language (enough for discrimination)

_TRIGRAM_PROFILES: Dict[str, List[str]] = {
    "en": ["the", "and", "ing", "ion", "tio", "ent", "ati", "for", "her", "ter",
           "hat", "tha", "ere", "ate", "his", "con", "res", "ver", "all", "ons",
           "nce", "men", "ith", "ted", "ers", "pro", "thi", "wit", "are", "ess"],
    "es": ["que", "ión", "ent", "aci", "ció", "con", "ado", "los", "est", "las",
           "por", "des", "nte", "era", "res", "ien", "men", "ion", "par", "com",
           "sta", "tos", "cia", "tra", "nto", "ard", "ali", "ado", "tas", "ter"],
    "fr": ["les", "ent", "que", "ion", "des", "ait", "eur", "ant", "tio", "ons",
           "men", "par", "est", "con", "com", "pas", "ais", "res", "ter", "ell",
           "ous", "dan", "ien", "eme", "our", "pou", "tte", "ire", "nte", "uit"],
    "de": ["ein", "sch", "che", "der", "die", "und", "den", "ich", "ung", "eit",
           "ver", "gen", "ber", "ach", "ter", "ent", "ine", "ges", "auf", "ren",
           "aus", "hen", "ier", "ste", "nic", "cht", "erd", "tte", "war", "ers"],
    "it": ["che", "ell", "ion", "per", "ent", "con", "ato", "del", "azi", "tti",
           "one", "men", "gli", "are", "ali", "com", "nte", "ere", "tta", "ato",
           "sta", "ver", "par", "ato", "ter", "pre", "tra", "str", "tto", "eri"],
    "pt": ["que", "ção", "ent", "ado", "est", "con", "dos", "com", "nte", "par",
           "ões", "men", "são", "des", "era", "ter", "sta", "res", "ais", "oss"],
    "nl": ["een", "van", "het", "aar", "ver", "oor", "den", "die", "ing", "and",
           "erd", "ter", "ijn", "dat", "sch", "ren", "gen", "ond", "ste", "eri"],
    "pl": ["nie", "prz", "owa", "rze", "ych", "ści", "ani", "sta", "eni", "prz",
           "icz", "wie", "kie", "cze", "osc", "nia", "ych", "ego", "est", "iec"],
    "tr": ["lar", "bir", "eri", "ler", "ını", "dan", "ara", "anı", "dir", "ile",
           "yen", "ası", "nda", "ind", "rin", "lik", "rak", "ard", "aya", "edi"],
    "ro": ["are", "ent", "rea", "ate", "con", "ulu", "ări", "rea", "tat", "est",
           "pen", "ent", "lor", "din", "pre", "car", "int", "ori", "cer", "tel"],
    "sv": ["och", "för", "att", "det", "som", "den", "var", "med", "har", "inte",
           "lig", "ade", "ter", "ing", "ern", "und", "ens", "tig", "ver", "sta"],
    "da": ["der", "det", "for", "med", "den", "som", "har", "til", "var", "ikke",
           "ere", "ige", "hed", "ens", "lig", "ell", "gen", "age", "nde", "ste"],
    "fi": ["nen", "ist", "tta", "sta", "ise", "een", "ais", "ssa", "lla", "iin",
           "taa", "sti", "nut", "kse", "ään", "oli", "tai", "uut", "tta", "min"],
    "ja": [],  # Detected via script (hiragana/katakana/CJK)
    "zh": [],  # Detected via script (CJK)
    "ko": [],  # Detected via script (hangul)
    "ru": [],  # Detected via script (cyrillic)
    "ar": [],  # Detected via script (arabic)
    "hi": [],  # Detected via script (devanagari)
}


def _get_trigrams(text: str) -> Counter:
    """Extract character trigram frequencies from text."""
    text = re.sub(r"[^a-záàâãäåæçèéêëìíîïðñòóôõöùúûüýþÿœšžğışöüçěřůąćęłńóśźżăîâșțéèêëàâùûôïüöäß]", "", text.lower())
    trigrams = Counter()
    for i in range(len(text) - 2):
        trigrams[text[i:i+3]] += 1
    return trigrams


def _profile_similarity(text_trigrams: Counter, profile: List[str]) -> float:
    """Compute similarity between text trigrams and a language profile."""
    if not profile or not text_trigrams:
        return 0.0
    profile_set = set(profile)
    text_top = set(t for t, _ in text_trigrams.most_common(60))
    if not text_top:
        return 0.0
    overlap = len(text_top & profile_set)
    return overlap / max(len(profile_set), 1)


def _detect_scripts(text: str) -> Dict[str, int]:
    """Count characters per Unicode script block."""
    counts: Dict[str, int] = Counter()
    for char in text:
        cp = ord(char)
        for script, start, end in _SCRIPT_RANGES:
            if start <= cp <= end:
                counts[script] += 1
                break
    return dict(counts)


class LanguageDetector:
    """Detects and enforces language policies."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", False)
        self.allowed_languages: Optional[List[str]] = config.get("allowed_languages", None)
        self.blocked_languages: List[str] = config.get("blocked_languages", [])
        self.action = config.get("action", "BLOCK")
        self.detect_mixed = config.get("detect_mixed", False)
        self.min_text_length = config.get("min_text_length", 20)

    def _identify_language(self, text: str) -> Tuple[str, float]:
        """
        Identify the primary language of text.
        Returns (language_code, confidence).
        """
        # Step 1: Script-based detection (for non-Latin scripts)
        scripts = _detect_scripts(text)
        total_chars = sum(scripts.values())
        if total_chars == 0:
            return "unknown", 0.0

        # Find dominant non-Latin script
        for script, count in sorted(scripts.items(), key=lambda x: -x[1]):
            if script == "latin":
                continue
            ratio = count / total_chars
            if ratio > 0.3 and script in _SCRIPT_LANGUAGES:
                # Dominant non-Latin script
                langs = _SCRIPT_LANGUAGES[script]
                return langs[0], min(ratio + 0.2, 1.0)

        # Step 2: Trigram-based detection (for Latin-script languages)
        text_trigrams = _get_trigrams(text)
        if not text_trigrams:
            return "unknown", 0.0

        best_lang = "unknown"
        best_score = 0.0

        for lang, profile in _TRIGRAM_PROFILES.items():
            if not profile:
                continue
            score = _profile_similarity(text_trigrams, profile)
            if score > best_score:
                best_score = score
                best_lang = lang

        return best_lang, best_score

    def _detect_mixed_languages(self, text: str) -> List[str]:
        """Detect if text contains multiple languages."""
        scripts = _detect_scripts(text)
        total = sum(scripts.values())
        if total == 0:
            return []

        # Find scripts that represent >20% of content
        significant = []
        for script, count in scripts.items():
            if count / total > 0.2:
                if script in _SCRIPT_LANGUAGES:
                    significant.extend(_SCRIPT_LANGUAGES[script])
                elif script == "latin":
                    significant.append("latin")

        return significant

    def detect(self, text: str) -> DetectorResult:
        if not self.enabled:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        if not text or len(text.strip()) < self.min_text_length:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        rule_hits: List[RuleHit] = []

        # Detect primary language
        lang, confidence = self._identify_language(text)

        # Check blocked languages
        if lang in self.blocked_languages:
            rule_hits.append(RuleHit(
                rule_id="language.blocked_language",
                severity=Severity.HIGH,
                message=f"Blocked language detected: {lang} (confidence={confidence:.2f})",
            ))

        # Check allowed languages
        if self.allowed_languages is not None and lang != "unknown":
            if lang not in self.allowed_languages:
                rule_hits.append(RuleHit(
                    rule_id="language.not_in_allowed",
                    severity=Severity.HIGH,
                    message=f"Language '{lang}' not in allowed list: {self.allowed_languages}",
                ))

        # Check for mixed-language content
        if self.detect_mixed:
            mixed = self._detect_mixed_languages(text)
            if len(mixed) > 1:
                rule_hits.append(RuleHit(
                    rule_id="language.mixed_content",
                    severity=Severity.MEDIUM,
                    message=f"Mixed-language content detected: {', '.join(set(mixed))}",
                ))

        if not rule_hits:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        # Risk score
        high_count = sum(1 for h in rule_hits if h.severity == Severity.HIGH)
        risk_score = min(100, high_count * 60 + (len(rule_hits) - high_count) * 20)

        decision = Decision.BLOCK if self.action == "BLOCK" else Decision.WARN

        return DetectorResult(
            decision=decision,
            risk_score=risk_score,
            rule_hits=rule_hits,
            developer_message=f"language: detected '{lang}' (confidence={confidence:.2f})",
        )
