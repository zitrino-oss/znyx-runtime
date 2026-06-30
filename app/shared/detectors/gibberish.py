"""
Gibberish / Adversarial Input Detector.

Detects nonsensical, high-entropy, or adversarially crafted inputs designed
to confuse the model or waste compute. Checks:
- Shannon entropy (character-level)
- Consonant-vowel ratio
- Invisible Unicode characters
- Special character ratio
- Token repetition (stuffing)
- Word recognition (basic vocabulary check)
"""
import math
import re
import logging
from collections import Counter
from typing import Any, Dict, List, Set

from app.shared.core.models import DetectorResult, RuleHit, Severity, Decision
from app.shared.core.risk import calculate_risk_score

logger = logging.getLogger(__name__)

# ── Invisible / zero-width Unicode characters ──────────────────────────────

_INVISIBLE_CHARS: Set[int] = {
    0x200B,  # Zero Width Space
    0x200C,  # Zero Width Non-Joiner
    0x200D,  # Zero Width Joiner
    0x200E,  # Left-to-Right Mark
    0x200F,  # Right-to-Left Mark
    0x202A,  # Left-to-Right Embedding
    0x202B,  # Right-to-Left Embedding
    0x202C,  # Pop Directional Formatting
    0x202D,  # Left-to-Right Override
    0x202E,  # Right-to-Left Override
    0x2060,  # Word Joiner
    0x2061,  # Function Application
    0x2062,  # Invisible Times
    0x2063,  # Invisible Separator
    0x2064,  # Invisible Plus
    0xFEFF,  # Zero Width No-Break Space (BOM)
    0x00AD,  # Soft Hyphen
    0x034F,  # Combining Grapheme Joiner
    0x061C,  # Arabic Letter Mark
    0x180E,  # Mongolian Vowel Separator
}

# Variation selectors (U+FE00 - U+FE0F, U+E0100 - U+E01EF)
_INVISIBLE_CHARS.update(range(0xFE00, 0xFE10))

# ── Basic English vocabulary (top ~3000 words for coherence checking) ──────

# This is a compact set of the most common English words.
# In practice, a production system would use a larger list (~10K words).
_COMMON_WORDS: Set[str] = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see",
    "other", "than", "then", "now", "look", "only", "come", "its", "over",
    "think", "also", "back", "after", "use", "two", "how", "our", "work",
    "first", "well", "way", "even", "new", "want", "because", "any", "these",
    "give", "day", "most", "us", "are", "was", "is", "has", "had", "been",
    "did", "does", "being", "more", "very", "much", "many", "own", "each",
    "still", "find", "here", "thing", "man", "world", "life", "hand", "part",
    "child", "eye", "woman", "place", "right", "old", "great", "help", "long",
    "line", "turn", "move", "live", "real", "left", "same", "state", "high",
    "keep", "home", "small", "end", "last", "never", "point", "start", "city",
    "run", "ask", "open", "try", "read", "need", "too", "land", "story",
    "should", "must", "shall", "may", "might", "let", "down", "off", "got",
    "before", "where", "while", "through", "between", "under", "again", "once",
    "head", "best", "change", "name", "play", "tell", "might", "number", "group",
    "side", "water", "room", "young", "house", "case", "system", "every", "age",
    "why", "call", "put", "set", "show", "big", "few", "sure", "little", "lot",
    "going", "done", "said", "went", "made", "came", "found", "data", "code",
    "model", "input", "output", "error", "test", "user", "file", "function",
    "class", "type", "value", "list", "key", "text", "string", "please",
    "thank", "yes", "hello", "help", "question", "answer", "information",
    "however", "therefore", "although", "since", "until", "whether", "both",
    "during", "those", "above", "below", "always", "often", "sometimes",
    "already", "perhaps", "quite", "rather", "enough", "almost", "another",
    "different", "important", "possible", "available", "specific", "general",
    # Extended common words for better recognition
    "program", "process", "report", "level", "office", "door", "health",
    "person", "art", "war", "history", "party", "result", "problem",
    "large", "number", "company", "area", "market", "service", "president",
    "member", "power", "law", "money", "idea", "control", "example",
    "evidence", "study", "death", "body", "blood", "face", "town", "family",
    "sense", "mind", "matter", "action", "table", "letter", "music",
    "cause", "reason", "force", "moment", "period", "plan", "building",
    "street", "court", "paper", "field", "week", "month", "word", "space",
    "team", "night", "morning", "foot", "car", "game", "science", "school",
    "rate", "cost", "price", "language", "class", "issue", "effect",
    "today", "anything", "something", "nothing", "everything", "everyone",
    "anyone", "someone", "yourself", "themselves", "itself", "myself",
    "together", "against", "within", "without", "toward", "across",
    "nothing", "likely", "simply", "early", "nearly", "later", "hard",
    "order", "several", "public", "local", "social", "political", "human",
    "natural", "free", "clear", "true", "whole", "strong", "close",
    "full", "short", "easy", "ready", "simple", "support", "include",
    "seem", "hold", "stand", "bring", "follow", "begin", "feel", "mean",
    "continue", "learn", "leave", "speak", "allow", "lead", "spend",
    "grow", "offer", "remember", "believe", "receive", "provide", "create",
    "sell", "require", "develop", "produce", "carry", "build", "describe",
    "remain", "expect", "cover", "reach", "suggest", "raise", "pass",
    "accept", "decide", "reduce", "explain", "agree", "consider", "remove",
    "serve", "watch", "compare", "apply", "report", "note", "form",
    "press", "serve", "appear", "record", "wish", "contain", "manage",
    "design", "maintain", "concern", "affect", "achieve", "indicate",
    "technology", "computer", "internet", "software", "digital", "network",
    "security", "system", "server", "application", "database", "machine",
    "algorithm", "platform", "device", "interface", "document", "project",
    "method", "approach", "analysis", "strategy", "response", "message",
    "version", "feature", "setting", "option", "update", "content",
    "account", "access", "request", "permission", "policy", "config",
}

_VOWELS = set("aeiouAEIOU")
_CONSONANTS = set("bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ")


def _shannon_entropy(text: str) -> float:
    """Calculate Shannon entropy (bits per character)."""
    if not text:
        return 0.0
    freq = Counter(text)
    total = len(text)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _consonant_vowel_ratio(text: str) -> float:
    """Calculate consonant-to-vowel ratio. Returns 0 if no vowels."""
    vowels = sum(1 for c in text if c in _VOWELS)
    consonants = sum(1 for c in text if c in _CONSONANTS)
    if vowels == 0:
        return float(consonants) if consonants > 0 else 0.0
    return consonants / vowels


def _count_invisible_chars(text: str) -> int:
    """Count invisible Unicode characters in text."""
    return sum(1 for c in text if ord(c) in _INVISIBLE_CHARS)


def _special_char_ratio(text: str) -> float:
    """Ratio of non-alphanumeric, non-space characters to total length."""
    if not text:
        return 0.0
    special = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return special / len(text)


def _find_repeated_tokens(text: str, threshold: int) -> List[str]:
    """Find tokens repeated more than threshold times."""
    tokens = text.lower().split()
    counts = Counter(tokens)
    return [token for token, count in counts.items() if count >= threshold and len(token) > 2]


def _word_recognition_ratio(text: str) -> float:
    """Ratio of recognized words to total words."""
    words = re.findall(r"[a-zA-Z]{2,}", text.lower())
    if not words:
        return 0.0
    recognized = sum(1 for w in words if w in _COMMON_WORDS)
    return recognized / len(words)


class GibberishDetector:
    """Detects gibberish, adversarial, and nonsensical input."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)
        self.action = config.get("action", "BLOCK")
        self.entropy_threshold = config.get("entropy_threshold", 5.5)
        self.max_special_char_ratio = config.get("max_special_char_ratio", 0.4)
        self.detect_invisible_chars = config.get("detect_invisible_chars", True)
        self.detect_token_stuffing = config.get("detect_token_stuffing", True)
        self.repetition_threshold = config.get("repetition_threshold", 5)
        self.min_text_length = config.get("min_text_length", 10)

    def detect(self, text: str) -> DetectorResult:
        if not self.enabled:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        if not text or len(text.strip()) < self.min_text_length:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        rule_hits: List[RuleHit] = []

        # 1. Invisible character detection
        if self.detect_invisible_chars:
            invisible_count = _count_invisible_chars(text)
            if invisible_count > 0:
                rule_hits.append(RuleHit(
                    rule_id="gibberish.invisible_chars",
                    severity=Severity.HIGH,
                    message=f"Found {invisible_count} invisible/zero-width Unicode characters (potential injection)",
                ))

        # 2. Shannon entropy check
        # Strip whitespace for entropy calculation (whitespace reduces entropy artificially)
        stripped = text.replace(" ", "").replace("\n", "").replace("\t", "")
        if len(stripped) > 20:
            entropy = _shannon_entropy(stripped)
            if entropy > self.entropy_threshold:
                rule_hits.append(RuleHit(
                    rule_id="gibberish.high_entropy",
                    severity=Severity.MEDIUM,
                    message=f"High character entropy ({entropy:.2f} bits/char) - text may be random or adversarial",
                ))

        # 3. Special character ratio
        spec_ratio = _special_char_ratio(text)
        if spec_ratio > self.max_special_char_ratio:
            rule_hits.append(RuleHit(
                rule_id="gibberish.excessive_special_chars",
                severity=Severity.MEDIUM,
                message=f"Excessive special characters ({spec_ratio:.0%} of text)",
            ))

        # 4. Token stuffing detection
        if self.detect_token_stuffing:
            repeated = _find_repeated_tokens(text, self.repetition_threshold)
            if repeated:
                rule_hits.append(RuleHit(
                    rule_id="gibberish.token_stuffing",
                    severity=Severity.MEDIUM,
                    message=f"Token stuffing detected: {', '.join(repeated[:5])} repeated ≥{self.repetition_threshold} times",
                ))

        # 5. Consonant-vowel ratio (only for Latin-script text)
        alpha_text = re.sub(r"[^a-zA-Z]", "", text)
        if len(alpha_text) > 20:
            cv_ratio = _consonant_vowel_ratio(alpha_text)
            if cv_ratio > 6.0 or (cv_ratio > 0 and cv_ratio < 0.3):
                rule_hits.append(RuleHit(
                    rule_id="gibberish.incoherent",
                    severity=Severity.MEDIUM,
                    message=f"Abnormal consonant/vowel ratio ({cv_ratio:.1f}) - text may be incoherent",
                ))

        # 6. Word recognition (only for Latin-script text)
        if len(alpha_text) > 30:
            recognition = _word_recognition_ratio(text)
            if recognition < 0.15:
                rule_hits.append(RuleHit(
                    rule_id="gibberish.no_recognizable_words",
                    severity=Severity.HIGH,
                    message=f"Only {recognition:.0%} of words are recognizable - text appears to be gibberish",
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
            user_message="Your input appears to be invalid or malformed.",
            developer_message=f"gibberish: {len(rule_hits)} adversarial input indicators detected",
        )
