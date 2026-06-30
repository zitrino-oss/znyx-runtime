"""Relevance scorer - measures how relevant the output is to the input query.

Uses TF-IDF-style cosine similarity and entity recall.
"""
import math
import re
from collections import Counter
from typing import Dict, Any, List, Optional

from app.shared.core.models import QualityScore

_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could of in to for on with "
    "at by from as into through during before after above below between "
    "and but or nor not no so yet both either neither each every all "
    "any few more most other some such that this these those i me my "
    "we our you your he him his she her it its they them their what "
    "which who whom how when where why if then than too very also just "
    "about up out there here only own same".split()
)


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _cosine_similarity(vec_a: Counter, vec_b: Counter) -> float:
    """Compute cosine similarity between two term-frequency vectors."""
    common = set(vec_a.keys()) & set(vec_b.keys())
    if not common:
        return 0.0
    dot = sum(vec_a[k] * vec_b[k] for k in common)
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _extract_entities(text: str) -> set:
    """Simple entity extraction: capitalized multi-word sequences and quoted terms."""
    entities = set()
    # Capitalized sequences (2+ words)
    for match in re.finditer(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", text):
        entities.add(match.group().lower())
    # Single capitalized words (not at sentence start)
    for match in re.finditer(r"(?<=[.!?]\s)[A-Z][a-z]{2,}", text):
        entities.add(match.group().lower())
    # Quoted terms
    for match in re.finditer(r'"([^"]{2,})"', text):
        entities.add(match.group(1).lower())
    return entities


def score_relevance(
    input_text: str,
    output_text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> QualityScore:
    """Score relevance of output to input query."""
    input_tokens = _tokenize(input_text)
    output_tokens = _tokenize(output_text)

    if not input_tokens or not output_tokens:
        return QualityScore(metric="relevance", score=0.5, details="Insufficient text for relevance scoring.")

    # TF-based cosine similarity
    input_freq = Counter(input_tokens)
    output_freq = Counter(output_tokens)
    cosine = _cosine_similarity(input_freq, output_freq)

    # Entity recall: entities from input appearing in output
    input_entities = _extract_entities(input_text)
    if input_entities:
        output_lower = output_text.lower()
        recalled = sum(1 for e in input_entities if e in output_lower)
        entity_recall = recalled / len(input_entities)
    else:
        entity_recall = 1.0  # no entities to recall

    # Keyword recall: fraction of input content words in output
    input_set = set(input_tokens)
    output_set = set(output_tokens)
    keyword_recall = len(input_set & output_set) / len(input_set) if input_set else 0.0

    # Weighted combination
    score = 0.5 * cosine + 0.25 * entity_recall + 0.25 * keyword_recall
    score = min(max(score, 0.0), 1.0)

    return QualityScore(
        metric="relevance",
        score=round(score, 3),
        details=f"Cosine={cosine:.2f}, entity_recall={entity_recall:.2f}, keyword_recall={keyword_recall:.2f}",
        sub_scores={
            "cosine_similarity": round(cosine, 3),
            "entity_recall": round(entity_recall, 3),
            "keyword_recall": round(keyword_recall, 3),
        },
    )
