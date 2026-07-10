"""Shared adversarial-text normalization for evasion-resistant keyword matching.

Detectors that match untrusted text against keyword/marker banks (toxicity, the
prompt-injection bank) must defend against the same evasion class: Unicode homoglyphs,
zero-width characters, fullwidth forms, and leetspeak. This module is the single source
of truth for those normalizers so the detectors stay consistent — previously only
toxicity normalized, letting a single confusable character walk past every injection
detector.
"""
import re
import unicodedata
from typing import Dict, List

# Visually-similar characters → ASCII equivalents (Cyrillic/IPA/fullwidth/dashes).
UNICODE_HOMOGLYPHS: Dict[str, str] = {
    "а": "a", "е": "e", "о": "o", "р": "p",  # Cyrillic
    "с": "c", "у": "y", "і": "i", "ј": "j",
    "ѕ": "s", "һ": "h", "Ѡ": "o",
    "ɑ": "a", "ɡ": "g", "ɪ": "i",  # IPA
    "ᴀ": "a", "ᴄ": "c", "ᴅ": "d", "ᴇ": "e",
    "‐": "-", "‑": "-", "‒": "-", "–": "-",  # dashes
    "ａ": "a", "ｂ": "b", "ｃ": "c", "ｄ": "d",  # fullwidth
    "ｅ": "e", "ｆ": "f", "ｇ": "g", "ｈ": "h",
    "ｉ": "i", "ｊ": "j", "ｋ": "k", "ｌ": "l",
    "ｍ": "m", "ｎ": "n", "ｏ": "o", "ｐ": "p",
    "ｑ": "q", "ｒ": "r", "ｓ": "s", "ｔ": "t",
    "ｕ": "u", "ｖ": "v", "ｗ": "w", "ｘ": "x",
    "ｙ": "y", "ｚ": "z",
}

# Zero-width / invisible code points injected to split keywords.
_ZERO_WIDTH_RE = re.compile("[​‌‍⁠﻿­]")

# Conservative leetspeak substitutions (only unambiguous ones).
_LEETSPEAK_MAP = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t",
                  "@": "a", "$": "s", "!": "i"}
_LEET_RE = re.compile("|".join(re.escape(k) for k in _LEETSPEAK_MAP))


def normalize_unicode(text: str) -> str:
    """NFKD-decompose, then map residual homoglyphs/confusables to ASCII."""
    text = unicodedata.normalize("NFKD", text)
    return "".join(UNICODE_HOMOGLYPHS.get(ch, ch) for ch in text)


def strip_zero_width(text: str) -> str:
    return _ZERO_WIDTH_RE.sub("", text)


def collapse_leetspeak(text: str) -> str:
    return _LEET_RE.sub(lambda m: _LEETSPEAK_MAP[m.group(0)], text)


def normalize_for_matching(text: str) -> str:
    """Aggressive normalization for evasion-resistant matching: strip zero-width,
    NFKD + homoglyph→ASCII, then collapse leetspeak. Callers should match BOTH the raw
    text and this normalized variant so normalization can never mask a legitimate hit."""
    if not text:
        return text
    return collapse_leetspeak(normalize_unicode(strip_zero_width(text)))


def match_variants(text: str) -> List[str]:
    """The text variants to run keyword regexes against: the raw text plus the
    normalized variant (deduped). Matching every variant defeats homoglyph/zero-width/
    leetspeak evasion without losing matches that only exist in the raw form."""
    if not text:
        return [text] if text is not None else []
    normalized = normalize_for_matching(text)
    return [text] if normalized == text else [text, normalized]
