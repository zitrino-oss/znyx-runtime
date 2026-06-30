"""Fluency scorer - measures grammatical quality and readability.

Evaluates sentence length distribution, incomplete sentences,
punctuation patterns, and word-level repetition.
"""
import re
import statistics
from typing import Dict, Any, List, Optional

from app.shared.core.models import QualityScore


def _sentence_split(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _word_count(text: str) -> int:
    return len(text.split())


def _sentence_length_score(sentences: List[str]) -> float:
    """Penalize very short (<3 words) or very long (>80 words) sentences."""
    if not sentences:
        return 0.5

    lengths = [_word_count(s) for s in sentences]
    good = sum(1 for l in lengths if 3 <= l <= 80)
    base_score = good / len(lengths)

    # Bonus for reasonable variance (not all same length)
    if len(lengths) > 1:
        std = statistics.stdev(lengths)
        mean = statistics.mean(lengths)
        # CV between 0.2 and 0.8 is good
        cv = std / mean if mean > 0 else 0
        variance_bonus = 0.1 if 0.2 <= cv <= 0.8 else 0.0
    else:
        variance_bonus = 0.0

    return min(base_score + variance_bonus, 1.0)


def _incomplete_sentence_score(sentences: List[str]) -> float:
    """Check for incomplete sentences (no terminal punctuation, starts with lowercase)."""
    if not sentences:
        return 0.5

    complete = 0
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        has_terminal = bool(re.search(r"[.!?]$", sent))
        starts_proper = sent[0].isupper() or sent[0].isdigit() or sent[0] in '"\'('
        if has_terminal and starts_proper:
            complete += 1
        elif has_terminal:
            complete += 0.5  # partial credit
    return complete / len(sentences)


def _punctuation_score(text: str) -> float:
    """Score proper punctuation usage."""
    # Check for common punctuation issues
    issues = 0
    total_checks = 4

    # Double spaces
    if "  " in text:
        issues += 1
    # Missing space after punctuation
    if re.search(r"[.!?,;:][a-zA-Z]", text):
        issues += 1
    # Multiple consecutive punctuation (except ... and !!)
    if re.search(r"[.!?,;:]{3,}", text.replace("...", "")):
        issues += 1
    # Unbalanced quotes or parentheses
    if text.count('"') % 2 != 0 or text.count('(') != text.count(')'):
        issues += 1

    return (total_checks - issues) / total_checks


def _word_repetition_score(sentences: List[str]) -> float:
    """Penalize word-level repetition within individual sentences."""
    if not sentences:
        return 0.5

    scores = []
    for sent in sentences:
        words = re.findall(r"[a-z]+", sent.lower())
        if len(words) <= 3:
            scores.append(1.0)
            continue
        unique = len(set(words))
        scores.append(unique / len(words))

    return sum(scores) / len(scores)


def score_fluency(
    output_text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> QualityScore:
    """Score fluency of output text."""
    sentences = _sentence_split(output_text)

    if not sentences:
        return QualityScore(metric="fluency", score=0.5, details="No sentences detected.")

    sent_length = _sentence_length_score(sentences)
    incomplete = _incomplete_sentence_score(sentences)
    punctuation = _punctuation_score(output_text)
    word_rep = _word_repetition_score(sentences)

    score = 0.3 * sent_length + 0.3 * incomplete + 0.2 * punctuation + 0.2 * word_rep
    score = min(max(score, 0.0), 1.0)

    return QualityScore(
        metric="fluency",
        score=round(score, 3),
        details=f"sent_length={sent_length:.2f}, completeness={incomplete:.2f}, punctuation={punctuation:.2f}, word_variety={word_rep:.2f}",
        sub_scores={
            "sentence_length": round(sent_length, 3),
            "completeness": round(incomplete, 3),
            "punctuation": round(punctuation, 3),
            "word_variety": round(word_rep, 3),
        },
    )
