"""Intent resolution scorer - measures whether the output addresses the input's intent.

Classifies the input question type and checks if the output contains the
expected answer pattern.
"""
import re
from typing import Dict, Any, Optional

from znyx_core.core.models import QualityScore

# Question type patterns
_QUESTION_TYPES = [
    ("list", re.compile(r"\b(list|enumerate|name|give me|provide)\b.*\b(\d+)\b", re.I)),
    ("count", re.compile(r"\bhow many\b", re.I)),
    ("yes_no", re.compile(r"^(is|are|was|were|do|does|did|can|could|will|would|should|has|have)\b", re.I)),
    ("who", re.compile(r"\bwho\b", re.I)),
    ("when", re.compile(r"\bwhen\b", re.I)),
    ("where", re.compile(r"\bwhere\b", re.I)),
    ("why", re.compile(r"\bwhy\b", re.I)),
    ("how", re.compile(r"\bhow\b(?!\s+many)", re.I)),
    ("what", re.compile(r"\bwhat\b", re.I)),
    ("define", re.compile(r"\b(define|explain|describe|what is|what are)\b", re.I)),
    ("compare", re.compile(r"\b(compare|difference|versus|vs\.?|better)\b", re.I)),
]


def _classify_question(text: str) -> tuple:
    """Return (question_type, match) for the input."""
    for qtype, pattern in _QUESTION_TYPES:
        match = pattern.search(text)
        if match:
            return qtype, match
    return "general", None


def _check_list_answer(output: str, expected_count: int) -> float:
    """Check if output contains at least expected_count list items."""
    # Count numbered items (1. 2. 3.) or bullet points (- *)
    numbered = len(re.findall(r"^\s*\d+[.)]\s", output, re.M))
    bulleted = len(re.findall(r"^\s*[-*]\s", output, re.M))
    # Also count comma-separated items in a single sentence
    item_count = max(numbered, bulleted)
    if item_count == 0:
        # Fallback: count sentences as items
        item_count = len(re.split(r"[.!?]\s", output))

    if item_count >= expected_count:
        return 1.0
    return item_count / expected_count if expected_count > 0 else 0.5


def _check_yes_no_answer(output: str) -> float:
    """Check if output contains an affirmative or negative answer."""
    lower = output.lower()[:200]  # check beginning
    affirmative = bool(re.search(r"\b(yes|correct|indeed|absolutely|certainly|right|true)\b", lower))
    negative = bool(re.search(r"\b(no|not|incorrect|false|neither|never)\b", lower))
    if affirmative or negative:
        return 1.0
    return 0.3


def _check_entity_answer(output: str, qtype: str) -> float:
    """Check if output contains entity-like content matching question type."""
    lower = output.lower()[:500]
    if qtype == "who":
        # Look for person names (capitalized sequences)
        has_name = bool(re.search(r"[A-Z][a-z]+ [A-Z][a-z]+", output[:500]))
        return 1.0 if has_name else 0.4
    if qtype == "when":
        has_date = bool(re.search(r"\b(\d{4}|\d{1,2}/\d{1,2}|january|february|march|april|may|june|july|august|september|october|november|december)\b", lower))
        return 1.0 if has_date else 0.4
    if qtype == "where":
        # Capitalized place names
        has_place = bool(re.search(r"[A-Z][a-z]+(,\s*[A-Z][a-z]+)?", output[:500]))
        return 0.8 if has_place else 0.4
    return 0.6


def _check_how_answer(output: str) -> float:
    """Check if output contains instructional/step content."""
    # Look for step markers, numbered instructions, imperatives
    has_steps = bool(re.search(r"(step \d|first,|1[.)]\s|to do this|you can|you should)", output.lower()[:500]))
    return 1.0 if has_steps else 0.5


def _keyword_recall(input_text: str, output_text: str) -> float:
    """Fraction of input content words present in output."""
    stopwords = {"a", "an", "the", "is", "are", "was", "were", "be", "in", "to",
                 "for", "of", "on", "at", "by", "and", "or", "but", "not", "with",
                 "how", "what", "when", "where", "who", "why", "do", "does", "did",
                 "can", "could", "will", "would", "should", "i", "me", "my", "you",
                 "your", "we", "they", "it", "this", "that"}
    input_words = set(re.findall(r"[a-z0-9]+", input_text.lower())) - stopwords
    output_words = set(re.findall(r"[a-z0-9]+", output_text.lower()))
    if not input_words:
        return 1.0
    return len(input_words & output_words) / len(input_words)


def score_intent_resolution(
    input_text: str,
    output_text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> QualityScore:
    """Score how well the output resolves the input's intent."""
    qtype, match = _classify_question(input_text)

    # Pattern-specific check
    if qtype == "list" and match:
        try:
            expected = int(match.group(2))
        except (IndexError, ValueError):
            expected = 3
        pattern_score = _check_list_answer(output_text, expected)
    elif qtype == "yes_no":
        pattern_score = _check_yes_no_answer(output_text)
    elif qtype in ("who", "when", "where"):
        pattern_score = _check_entity_answer(output_text, qtype)
    elif qtype == "how":
        pattern_score = _check_how_answer(output_text)
    elif qtype == "compare":
        # Look for comparison structure
        has_compare = bool(re.search(r"(while|whereas|on the other hand|in contrast|however|both|differ)", output_text.lower()))
        pattern_score = 1.0 if has_compare else 0.4
    else:
        pattern_score = 0.6  # general question, partial credit

    recall = _keyword_recall(input_text, output_text)

    score = 0.6 * pattern_score + 0.4 * recall
    score = min(max(score, 0.0), 1.0)

    return QualityScore(
        metric="intent_resolution",
        score=round(score, 3),
        details=f"question_type={qtype}, pattern_score={pattern_score:.2f}, keyword_recall={recall:.2f}",
        sub_scores={
            "question_type_score": round(pattern_score, 3),
            "keyword_recall": round(recall, 3),
        },
    )
