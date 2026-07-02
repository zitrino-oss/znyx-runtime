"""Groundedness scorer - measures how well output claims are grounded in source context.

Default: token overlap between output sentences and grounding sources (no deps).
P2 upgrade: when an ``nli_scorer`` is provided (the F3 inference service's NLI/cross-
encoder task, wired by the caller), each atomic claim is checked for *entailment* by the
sources and per-claim ``evidence_spans`` are returned. Falls back to token overlap when
no scorer is supplied or it errors — preserving current behaviour when the service is
absent (same optional posture as sentence-transformers in hallucination.py).

``nli_scorer`` signature: ``(premise: str, hypotheses: list[str]) -> list[float]``
returning the entailment probability (0..1) of each hypothesis given the premise.
"""
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from znyx_core.core.models import QualityScore

logger = logging.getLogger(__name__)

_NLI_ENTAILMENT_THRESHOLD = 0.5  # claim is grounded if best source entails it at >= this

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


def _gather_sources(metadata: Dict[str, Any]) -> List[tuple]:
    """Collect grounding sources as ``(source_id, text)`` pairs.

    Accepts ``source_context`` (str) and ``grounding_sources`` (str | list of
    str | list of dict with id/source_id/url + text/content/body)."""
    sources: List[tuple] = []
    if "source_context" in metadata:
        sources.append(("source_context", str(metadata["source_context"])))

    gs = metadata.get("grounding_sources")
    if gs is not None:
        items = gs if isinstance(gs, list) else [gs]
        for i, item in enumerate(items):
            if isinstance(item, dict):
                sid = item.get("source_id") or item.get("id") or item.get("url") or f"source_{i}"
                text = item.get("text") or item.get("content") or item.get("body") or ""
                sources.append((str(sid), str(text)))
            else:
                sources.append((f"source_{i}", str(item)))
    return [(sid, txt) for sid, txt in sources if txt.strip()]


def _score_with_nli(
    claims: List[str],
    sources: List[tuple],
    nli_scorer: Callable[[str, List[str]], List[float]],
) -> QualityScore:
    """Entailment-based groundedness: each claim is grounded if any source
    entails it above threshold. Returns per-claim ``evidence_spans``.

    May raise if the scorer misbehaves — the caller falls back to token overlap."""
    best_support = [0.0] * len(claims)
    best_source = [None] * len(claims)

    for sid, src_text in sources:
        probs = nli_scorer(src_text, claims)
        if len(probs) != len(claims):
            raise ValueError(
                f"nli_scorer returned {len(probs)} probs for {len(claims)} claims"
            )
        for i, p in enumerate(probs):
            p = float(p)
            if p > best_support[i]:
                best_support[i] = p
                best_source[i] = sid

    grounded = sum(1 for s in best_support if s >= _NLI_ENTAILMENT_THRESHOLD)
    evidence_spans = [
        {
            "claim": claims[i],
            "source_id": best_source[i],
            "support": round(best_support[i], 3),
        }
        for i in range(len(claims))
    ]
    score = grounded / len(claims)
    return QualityScore(
        metric="groundedness",
        score=round(score, 3),
        details=f"NLI: {grounded}/{len(claims)} claims entailed by source context.",
        sub_scores={"grounded_claims": float(grounded), "total_claims": float(len(claims))},
        evidence_spans=evidence_spans,
    )


def score_groundedness(
    output_text: str,
    metadata: Optional[Dict[str, Any]] = None,
    nli_scorer: Optional[Callable[[str, List[str]], List[float]]] = None,
) -> QualityScore:
    """Score groundedness of output against provided source context.

    When ``nli_scorer`` is supplied (F3 inference NLI task), score by per-claim
    entailment and return ``evidence_spans``. Otherwise — or if the scorer errors
    — fall back to the deterministic token-overlap heuristic (no deps)."""
    metadata = metadata or {}

    sources = _gather_sources(metadata)
    if not sources:
        return QualityScore(
            metric="groundedness",
            score=1.0,
            details="No grounding sources provided; score defaults to 1.0.",
        )

    # Split output into atomic claims (sentence granularity).
    claims = _sentence_split(output_text)
    if not claims:
        return QualityScore(metric="groundedness", score=1.0, details="Output too short to evaluate.")

    # Preferred path: NLI entailment via the inference service.
    if nli_scorer is not None:
        try:
            return _score_with_nli(claims, sources, nli_scorer)
        except Exception as exc:  # noqa: BLE001 — degrade to token overlap, never fail the request
            logger.warning("NLI groundedness scorer failed (%s); falling back to token overlap", exc)

    # Fallback: token overlap against the union of source tokens.
    source_tokens = set()
    for _sid, src_text in sources:
        source_tokens.update(_tokenize(src_text))

    grounded_count = 0
    for claim in claims:
        tokens = _tokenize(claim)
        if _overlap_ratio(tokens, source_tokens) >= _GROUNDING_THRESHOLD:
            grounded_count += 1

    score = grounded_count / len(claims)
    return QualityScore(
        metric="groundedness",
        score=round(score, 3),
        details=f"{grounded_count}/{len(claims)} sentences grounded in source context.",
        sub_scores={
            "grounded_sentences": float(grounded_count),
            "total_sentences": float(len(claims)),
        },
    )
