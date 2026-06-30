"""
Copyright / Intellectual Property Detector.

Detects when LLM outputs contain verbatim or near-verbatim copyrighted content:
- Book opening lines and famous passages
- Song lyrics (chorus/verse repetition patterns)
- Code with restrictive licenses (GPL, AGPL, SSPL)
- N-gram fingerprinting against custom signature databases

Uses zero external dependencies - pure Python hashing and pattern matching.
"""
import hashlib
import re
import logging
from typing import Any, Dict, List, Set, Tuple

from app.shared.core.models import DetectorResult, RuleHit, Severity, Decision
from app.shared.core.risk import calculate_risk_score

logger = logging.getLogger(__name__)


# ── Known copyrighted content signatures ───────────────────────────────────
# These are opening lines / famous passages from frequently-litigated works.
# Each entry is a normalized text fragment → (work title, type).

_KNOWN_PASSAGES: Dict[str, Tuple[str, str]] = {
    # Book openings (normalized lowercase, stripped punctuation)
    "it was the best of times it was the worst of times": ("A Tale of Two Cities - Charles Dickens", "book"),
    "call me ishmael some years ago never mind how long precisely": ("Moby Dick - Herman Melville", "book"),
    "it is a truth universally acknowledged that a single man in possession": ("Pride and Prejudice - Jane Austen", "book"),
    "all happy families are alike each unhappy family is unhappy in its own way": ("Anna Karenina - Leo Tolstoy", "book"),
    "in a hole in the ground there lived a hobbit": ("The Hobbit - J.R.R. Tolkien", "book"),
    "it was a bright cold day in april and the clocks were striking thirteen": ("1984 - George Orwell", "book"),
    "mr and mrs dursley of number four privet drive were proud to say": ("Harry Potter - J.K. Rowling", "book"),
    "in my younger and more vulnerable years my father gave me some advice": ("The Great Gatsby - F. Scott Fitzgerald", "book"),
    "all children except one grow up": ("Peter Pan - J.M. Barrie", "book"),
    "the man in black fled across the desert and the gunslinger followed": ("The Dark Tower - Stephen King", "book"),
    "last night i dreamt i went to manderley again": ("Rebecca - Daphne du Maurier", "book"),
    "it was a pleasure to burn": ("Fahrenheit 451 - Ray Bradbury", "book"),
    "the sky above the port was the color of television tuned to a dead channel": ("Neuromancer - William Gibson", "book"),
    "far out in the uncharted backwaters of the unfashionable end of the western spiral arm": ("Hitchhiker's Guide - Douglas Adams", "book"),
    "when gregor samsa woke up one morning from unsettling dreams he found himself": ("The Metamorphosis - Franz Kafka", "book"),
    "whether i shall turn out to be the hero of my own life": ("David Copperfield - Charles Dickens", "book"),
    "you dont know about me without you have read a book by the name of": ("Adventures of Huckleberry Finn - Mark Twain", "book"),
    "happy families are all alike every unhappy family is unhappy in its own way": ("Anna Karenina - Leo Tolstoy", "book"),
    "once upon a time and a very good time it was there was a moocow": ("A Portrait of the Artist as a Young Man - James Joyce", "book"),
    "someone must have slandered josef k for one morning without having done anything": ("The Trial - Franz Kafka", "book"),
    "if you really want to hear about it the first thing youll probably want to know": ("The Catcher in the Rye - J.D. Salinger", "book"),
    "many years later as he faced the firing squad colonel aureliano buendia was to remember": ("One Hundred Years of Solitude - Gabriel Garcia Marquez", "book"),
    "lolita light of my life fire of my loins my sin my soul": ("Lolita - Vladimir Nabokov", "book"),
    "miss brooke had that kind of beauty which seems to be thrown into relief by poor dress": ("Middlemarch - George Eliot", "book"),
    "i am an invisible man no i am not a spook like those who haunted edgar allan poe": ("Invisible Man - Ralph Ellison", "book"),
    "the primroses were over toward the edge of the wood where the ground became open": ("Watership Down - Richard Adams", "book"),
    "we were somewhere around barstow on the edge of the desert when the drugs began to take hold": ("Fear and Loathing in Las Vegas - Hunter S. Thompson", "book"),
    "i am a sick man i am a spiteful man i am an unattractive man": ("Notes from Underground - Fyodor Dostoevsky", "book"),
    "riverrun past eve and adams from swerve of shore to bend of bay": ("Finnegans Wake - James Joyce", "book"),
    "stately plump buck mulligan came from the stairhead bearing a bowl of lather": ("Ulysses - James Joyce", "book"),
    "mother died today or maybe yesterday i cant be sure": ("The Stranger - Albert Camus", "book"),
    "all this happened more or less the war parts anyway are pretty much true": ("Slaughterhouse-Five - Kurt Vonnegut", "book"),
    "ships at a distance have every mans wish on board for some they come in with the tide": ("Their Eyes Were Watching God - Zora Neale Hurston", "book"),
    "i had the story bit by bit from various people and as generally happens in such cases": ("The Great Gatsby - F. Scott Fitzgerald", "book"),
}

# ── Code license patterns ─────────────────────────────────────────────────

_LICENSE_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
    # GPL family
    (re.compile(r"GNU\s+General\s+Public\s+License", re.I),
     "GPL", "GNU General Public License detected"),
    (re.compile(r"GPL[-\s]?(?:v|version\s*)?[23](?:\.0)?(?:\s*(?:or\s+later|\+))?", re.I),
     "GPL", "GPL license reference detected"),
    (re.compile(r"SPDX-License-Identifier:\s*GPL", re.I),
     "GPL", "SPDX GPL license identifier"),

    # AGPL
    (re.compile(r"GNU\s+Affero\s+General\s+Public\s+License", re.I),
     "AGPL-3.0", "GNU Affero GPL detected"),
    (re.compile(r"AGPL[-\s]?(?:v|version\s*)?3(?:\.0)?", re.I),
     "AGPL-3.0", "AGPL license reference detected"),
    (re.compile(r"SPDX-License-Identifier:\s*AGPL", re.I),
     "AGPL-3.0", "SPDX AGPL license identifier"),

    # LGPL
    (re.compile(r"GNU\s+Lesser\s+General\s+Public\s+License", re.I),
     "LGPL", "GNU Lesser GPL detected"),

    # SSPL
    (re.compile(r"Server\s+Side\s+Public\s+License", re.I),
     "SSPL-1.0", "SSPL license detected"),
    (re.compile(r"SSPL[-\s]?(?:v|version\s*)?1", re.I),
     "SSPL-1.0", "SSPL license reference detected"),

    # MPL (copyleft-lite)
    (re.compile(r"Mozilla\s+Public\s+License", re.I),
     "MPL", "Mozilla Public License detected"),

    # Generic copyright headers
    (re.compile(r"Copyright\s+\(c\)\s+\d{4}", re.I),
     "copyright_header", "Copyright header with year detected"),
    (re.compile(r"All\s+rights\s+reserved", re.I),
     "all_rights_reserved", "All rights reserved notice"),
]

# ── Lyrics detection ──────────────────────────────────────────────────────

# Heuristic: lyrics often have short lines with rhyming patterns and repetition
_LYRICS_INDICATORS = [
    re.compile(r"(?:verse|chorus|bridge|pre-chorus|outro|intro)\s*(?:\d|:)", re.I),
    re.compile(r"\b(?:oh|ooh|yeah|baby|la la|na na)\b", re.I),
]


def _normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _ngram_fingerprints(text: str, n: int = 8) -> Set[str]:
    """Generate n-gram fingerprints (hashes of n consecutive words)."""
    words = _normalize_text(text).split()
    if len(words) < n:
        return set()
    fingerprints = set()
    for i in range(len(words) - n + 1):
        ngram = " ".join(words[i:i + n])
        # Non-cryptographic content fingerprint for passage matching, not a
        # security control — usedforsecurity=False documents that intent.
        fp = hashlib.md5(ngram.encode(), usedforsecurity=False).hexdigest()[:12]
        fingerprints.add(fp)
    return fingerprints


def _check_known_passages(text: str) -> List[Tuple[str, str]]:
    """Check if text contains known copyrighted passages."""
    normalized = _normalize_text(text)
    matches = []
    for passage, (work, content_type) in _KNOWN_PASSAGES.items():
        if passage in normalized:
            matches.append((work, content_type))
    return matches


def _detect_repetition(text: str, threshold: int = 8) -> List[str]:
    """Find sequences of consecutive words that repeat (lyrics/content repetition)."""
    words = text.lower().split()
    repeated = []

    # Sliding window: check if any window of `threshold` words appears more than once
    seen = {}
    for i in range(len(words) - threshold + 1):
        window = " ".join(words[i:i + threshold])
        if window in seen:
            if window not in repeated:
                repeated.append(window)
        else:
            seen[window] = i

    return repeated


class CopyrightDetector:
    """Detects copyrighted content and license violations in LLM output."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", False)
        self.action = config.get("action", "WARN")
        self.verbatim_threshold = config.get("verbatim_threshold", 8)
        self.check_code_licenses = config.get("check_code_licenses", True)
        self.check_lyrics = config.get("check_lyrics", True)
        self.check_books = config.get("check_books", True)

        # License blocklist
        self.code_license_blocklist: Set[str] = set(config.get(
            "code_license_blocklist",
            ["GPL-2.0", "GPL-3.0", "AGPL-3.0", "SSPL-1.0"],
        ))

        # Custom signatures (user-provided fingerprints)
        self.custom_signatures: Set[str] = set()
        custom_sigs = config.get("custom_signatures", [])
        for sig in custom_sigs:
            if isinstance(sig, str):
                # Treat as text - generate fingerprints
                self.custom_signatures.update(_ngram_fingerprints(sig, self.verbatim_threshold))

    def detect(self, text: str) -> DetectorResult:
        if not self.enabled:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        if not text or len(text.strip()) < 20:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        rule_hits: List[RuleHit] = []

        # 1. Check known book passages
        if self.check_books:
            known_matches = _check_known_passages(text)
            for work, content_type in known_matches:
                rule_hits.append(RuleHit(
                    rule_id=f"copyright.known_work_match",
                    severity=Severity.HIGH,
                    message=f"Known copyrighted {content_type} passage detected: {work}",
                ))

        # 2. Check code license violations
        if self.check_code_licenses:
            for pattern, license_id, message in _LICENSE_PATTERNS:
                if pattern.search(text):
                    # Check if this license is in the blocklist
                    blocked = any(
                        license_id.startswith(bl) or bl.startswith(license_id)
                        for bl in self.code_license_blocklist
                    )
                    if blocked or license_id in ("copyright_header", "all_rights_reserved"):
                        severity = Severity.HIGH if blocked else Severity.MEDIUM
                        rule_hits.append(RuleHit(
                            rule_id="copyright.license_violation",
                            severity=severity,
                            message=f"{message} (license: {license_id})",
                        ))

        # 3. Check for lyrics patterns
        if self.check_lyrics:
            lyrics_indicators = sum(
                1 for pattern in _LYRICS_INDICATORS if pattern.search(text)
            )
            # Also check for repetition (common in lyrics)
            repeated = _detect_repetition(text, min(self.verbatim_threshold, 6))
            if lyrics_indicators >= 1 and repeated:
                rule_hits.append(RuleHit(
                    rule_id="copyright.lyrics_match",
                    severity=Severity.MEDIUM,
                    message=f"Possible song lyrics detected (structural indicators + repetition)",
                ))

        # 4. Custom signature matching
        if self.custom_signatures:
            text_fps = _ngram_fingerprints(text, self.verbatim_threshold)
            overlap = text_fps & self.custom_signatures
            if overlap:
                rule_hits.append(RuleHit(
                    rule_id="copyright.verbatim_match",
                    severity=Severity.HIGH,
                    message=f"Verbatim match with protected content ({len(overlap)} matching n-gram fingerprints)",
                ))

        # 5. General verbatim repetition check (content that repeats itself - possible copy-paste)
        if self.verbatim_threshold > 0:
            repeated = _detect_repetition(text, self.verbatim_threshold)
            if len(repeated) >= 3:
                rule_hits.append(RuleHit(
                    rule_id="copyright.verbatim_match",
                    severity=Severity.LOW,
                    message=f"Significant content repetition detected ({len(repeated)} repeated sequences)",
                ))

        if not rule_hits:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        # Deduplicate
        seen_ids = set()
        unique_hits = []
        for hit in rule_hits:
            key = (hit.rule_id, hit.message)
            if key not in seen_ids:
                seen_ids.add(key)
                unique_hits.append(hit)

        # Risk score
        risk_score = calculate_risk_score(unique_hits)

        decision = Decision.BLOCK if self.action == "BLOCK" else Decision.WARN

        return DetectorResult(
            decision=decision,
            risk_score=risk_score,
            rule_hits=unique_hits,
            user_message="The response may contain copyrighted or license-restricted content.",
            developer_message=f"copyright: {len(unique_hits)} potential IP issues detected",
        )
