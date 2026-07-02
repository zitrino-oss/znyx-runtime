"""Coherence scorer - measures logical flow and connectedness of output text.

Evaluates discourse markers, sentence-to-sentence topic continuity,
repetition penalty, and paragraph structure.
"""
import re
from typing import Dict, Any, List, Optional

from znyx_core.core.models import QualityScore

_DISCOURSE_MARKERS = frozenset(
    "however therefore furthermore moreover additionally consequently "
    "meanwhile nevertheless nonetheless accordingly similarly likewise "
    "conversely alternatively specifically particularly notably indeed "
    "in addition in contrast on the other hand as a result for example "
    "for instance in particular that said first second third finally "
    "next then also thus hence because since although though while".split()
)

_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could of in to for on with "
    "at by from as into through during before after above below between "
    "and but or not so yet i me my we our you your he him his she her "
    "it its they them their".split()
)


def _sentence_split(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 5]


def _content_tokens(text: str) -> set:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}


def _discourse_marker_score(sentences: List[str]) -> float:
    """Fraction of sentences containing discourse markers."""
    if len(sentences) <= 1:
        return 1.0
    count = 0
    for sent in sentences[1:]:  # skip first sentence
        lower = sent.lower()
        if any(marker in lower for marker in _DISCOURSE_MARKERS):
            count += 1
    return count / (len(sentences) - 1)


def _topic_continuity_score(sentences: List[str]) -> float:
    """Average token overlap between adjacent sentences."""
    if len(sentences) <= 1:
        return 1.0
    overlaps = []
    for i in range(1, len(sentences)):
        prev_tokens = _content_tokens(sentences[i - 1])
        curr_tokens = _content_tokens(sentences[i])
        union = prev_tokens | curr_tokens
        if union:
            overlaps.append(len(prev_tokens & curr_tokens) / len(union))
        else:
            overlaps.append(0.0)
    return sum(overlaps) / len(overlaps) if overlaps else 0.0


def _repetition_penalty(sentences: List[str]) -> float:
    """Penalize repeated sentences. Returns 1.0 (no repetition) to 0.0 (all repeated)."""
    if len(sentences) <= 1:
        return 1.0
    # Fingerprint each sentence by sorted content tokens
    fingerprints = []
    for sent in sentences:
        fp = frozenset(_content_tokens(sent))
        fingerprints.append(fp)

    unique = len(set(fingerprints))
    return unique / len(fingerprints)


def score_coherence(
    output_text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> QualityScore:
    """Score coherence of output text."""
    sentences = _sentence_split(output_text)

    if len(sentences) <= 1:
        return QualityScore(metric="coherence", score=0.8, details="Single sentence; coherence assumed.")

    discourse = _discourse_marker_score(sentences)
    continuity = _topic_continuity_score(sentences)
    repetition = _repetition_penalty(sentences)

    # Weighted combination
    score = 0.3 * discourse + 0.4 * continuity + 0.3 * repetition
    score = min(max(score, 0.0), 1.0)

    return QualityScore(
        metric="coherence",
        score=round(score, 3),
        details=f"discourse={discourse:.2f}, continuity={continuity:.2f}, repetition_penalty={repetition:.2f}",
        sub_scores={
            "discourse_markers": round(discourse, 3),
            "topic_continuity": round(continuity, 3),
            "repetition_penalty": round(repetition, 3),
        },
    )
