"""Exact text edits: (start, end, replacement) against an evaluated text.

Detectors that rewrite text report the edits they made alongside their
``sanitized_text`` (``DetectorResult.text_edits``), so a consumer can tell which
characters of the original each replacement covers. Streaming evaluation needs
that to release a window up to a point that never falls inside a redaction.

The edits are informational: a consumer verifies them with ``apply_edits`` and
falls back to comparing the strings when they do not reproduce the output.
"""
import re
from typing import Iterable, List, Optional, Pattern, Sequence, Tuple

Edit = Tuple[int, int, str]


def apply_edits(text: str, edits: Sequence[Edit]) -> str:
    """Apply sorted, disjoint edits to ``text``."""
    out: List[str] = []
    pos = 0
    for start, end, replacement in edits:
        out.append(text[pos:start])
        out.append(replacement)
        pos = end
    out.append(text[pos:])
    return "".join(out)


def normalise_edits(edits: Iterable[Edit], length: int) -> Optional[List[Edit]]:
    """Return the edits sorted, or None if any is out of range or they overlap."""
    ordered = sorted(edits, key=lambda e: (e[0], e[1]))
    last = 0
    for start, end, _replacement in ordered:
        if not (0 <= start <= end <= length) or start < last:
            return None
        last = end
    return ordered


def edits_for_patterns(text: str, patterns: Iterable[Tuple[Pattern, str]]) -> List[Edit]:
    """Edits replacing every match of each (pattern, replacement) in ``text``,
    found in one pass over the original text. Overlaps resolve leftmost, then
    longest, as the PII redactor does."""
    matches: List[Edit] = []
    for pattern, replacement in patterns:
        for m in pattern.finditer(text):
            if m.end() > m.start():
                matches.append((m.start(), m.end(), replacement))
    matches.sort(key=lambda e: (e[0], -(e[1] - e[0])))
    kept: List[Edit] = []
    last = 0
    for start, end, replacement in matches:
        if start >= last:
            kept.append((start, end, replacement))
            last = end
    return kept


def literal_pattern(value: str, word_boundary: bool = True, ignore_case: bool = True) -> Pattern:
    """A compiled pattern for a literal value, optionally anchored on word boundaries."""
    body = re.escape(value)
    if word_boundary:
        body = r"\b" + body + r"\b"
    return re.compile(body, re.IGNORECASE if ignore_case else 0)
