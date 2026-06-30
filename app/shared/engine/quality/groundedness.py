"""Groundedness scorer - measures how well output claims are grounded in source context.

Uses token overlap between output sentences and grounding sources.
Score = proportion of well-grounded claims (overlap ratio > threshold).
"""
import re
from typing import Dict, Any, List, Optional

from app.shared.core.models import QualityScore

# Words too common to be informative for overlap
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

_GROUNDING_THRESHOLD = 0.3  # minimum overlap ratio to count as grounded


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, remove stopwords."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _sentence_split(text: str) -> List[str]:
    """Split text into sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def _overlap_ratio(claim_tokens: List[str], source_tokens: set) -> float:
    if not claim_tokens:
        return 0.0
    matched = sum(1 for t in claim_tokens if t in source_tokens)
    return matched / len(claim_tokens)


def score_groundedness(
    output_text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> QualityScore:
    """Score groundedness of output against provided source context."""
    metadata = metadata or {}

    # Gather grounding sources
    sources = []
    if "source_context" in metadata:
        sources.append(str(metadata["source_context"]))
    if "grounding_sources" in metadata:
        gs = metadata["grounding_sources"]
        if isinstance(gs, list):
            sources.extend(str(s) for s in gs)
        else:
            sources.append(str(gs))

    if not sources:
        return QualityScore(
            metric="groundedness",
            score=1.0,
            details="No grounding sources provided; score defaults to 1.0.",
        )

    # Build source token set
    source_tokens = set()
    for src in sources:
        source_tokens.update(_tokenize(src))

    # Score each output sentence
    sentences = _sentence_split(output_text)
    if not sentences:
        return QualityScore(metric="groundedness", score=1.0, details="Output too short to evaluate.")

    grounded_count = 0
    for sent in sentences:
        tokens = _tokenize(sent)
        if _overlap_ratio(tokens, source_tokens) >= _GROUNDING_THRESHOLD:
            grounded_count += 1

    score = grounded_count / len(sentences)
    return QualityScore(
        metric="groundedness",
        score=round(score, 3),
        details=f"{grounded_count}/{len(sentences)} sentences grounded in source context.",
        sub_scores={"grounded_sentences": grounded_count, "total_sentences": len(sentences)},
    )
