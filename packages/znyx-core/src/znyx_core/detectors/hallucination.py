"""
Hallucination / Grounding Detector.

Verifies LLM output claims are grounded in provided source documents.
Supports two methods:
  - token_overlap (default): fast, zero-dependency word overlap comparison
  - embedding: cosine similarity via sentence-transformers (optional install)
"""
import math
import re
import string
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from znyx_core.core.models import DetectorResult, RuleHit, Severity, Decision

logger = logging.getLogger(__name__)

# Common English stopwords (kept small - no external deps)
STOPWORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "because", "but", "and", "or", "if", "while", "that", "this",
    "these", "those", "it", "its", "i", "me", "my", "we", "our", "you",
    "your", "he", "him", "his", "she", "her", "they", "them", "their",
    "what", "which", "who", "whom", "also", "about",
}

# Sentence boundary pattern
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, remove stopwords."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    sentences = _SENTENCE_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _token_overlap(claim_tokens: List[str], source_tokens: Set[str]) -> float:
    """Compute overlap ratio of claim tokens present in source tokens."""
    if not claim_tokens:
        return 1.0  # empty claim is trivially grounded
    matches = sum(1 for t in claim_tokens if t in source_tokens)
    return matches / len(claim_tokens)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class HallucinationDetector:
    """Detects ungrounded claims in LLM output by comparing against source context."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", False)
        self.method = config.get("method", "token_overlap")
        self.grounding_threshold = config.get("grounding_threshold", 0.5)
        self.action = config.get("action", "WARN")
        self.source_field = config.get("source_field", "source_context")
        self.min_claim_words = config.get("min_claim_words", 3)

        # Grounding sources - accept multiple aliases: source_context, grounding_sources, context_documents
        raw_sources = (config.get("source_context", "") or
                       config.get("grounding_sources", []) or
                       config.get("context_documents", []))
        if isinstance(raw_sources, str):
            self.sources = [raw_sources] if raw_sources.strip() else []
        elif isinstance(raw_sources, list):
            self.sources = [str(s) for s in raw_sources if str(s).strip()]
        else:
            self.sources = []

        # Pre-tokenize sources for token_overlap method
        self._source_token_sets: List[Set[str]] = []
        for src in self.sources:
            self._source_token_sets.append(set(_tokenize(src)))

        # All source tokens combined (for fast lookup)
        self._all_source_tokens: Set[str] = set()
        for ts in self._source_token_sets:
            self._all_source_tokens.update(ts)

        # Embedding model (lazy loaded)
        self._embed_model = None

        # NLI groundedness (P2/F3): an entailment scorer (premise, hypotheses) -> list[float].
        # Auto-wired by the orchestrator from the detector's `nli` config block (set as an
        # instance attribute post-construction); also accepts a directly-injected scorer via
        # config for tests. None → deterministic token-overlap / embedding path. A claim is
        # grounded when the best source entails it at >= `min_nli_entailment`.
        self.nli_scorer = config.get("nli_scorer")
        self.min_nli_entailment = float(config.get("min_nli_entailment", 0.5))

    def _get_embed_model(self):
        """Lazy-load sentence-transformers model."""
        if self._embed_model is not None:
            return self._embed_model
        try:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            return self._embed_model
        except ImportError:
            logger.warning(
                "sentence-transformers not installed - falling back to token_overlap. "
                "Install with: pip install sentence-transformers"
            )
            return None

    def _check_claim_token_overlap(self, claim: str) -> Tuple[float, str]:
        """Check a single claim using token overlap. Returns (score, best_source_snippet)."""
        claim_tokens = _tokenize(claim)
        if len(claim_tokens) < self.min_claim_words:
            return 1.0, ""  # skip very short fragments

        best_score = 0.0
        # Check against each source independently
        for source_tokens in self._source_token_sets:
            score = _token_overlap(claim_tokens, source_tokens)
            best_score = max(best_score, score)

        # Also check combined
        combined_score = _token_overlap(claim_tokens, self._all_source_tokens)
        best_score = max(best_score, combined_score)

        return best_score, ""

    def _nli_claim_scores(self, claims: List[str]) -> Optional[List[float]]:
        """Best entailment probability per claim across all sources, via the NLI scorer.
        One call per source (claims as hypotheses). Returns None on any error or contract
        violation so the caller degrades to token overlap — never fail the request."""
        if self.nli_scorer is None or not claims:
            return None
        try:
            best = [0.0] * len(claims)
            for src in self.sources:
                probs = self.nli_scorer(src, claims)
                if len(probs) != len(claims):
                    raise ValueError(
                        f"nli_scorer returned {len(probs)} probs for {len(claims)} claims")
                for i, p in enumerate(probs):
                    best[i] = max(best[i], float(p))
            return best
        except Exception as exc:  # noqa: BLE001 — degrade to token overlap, never fail
            logger.warning("NLI hallucination scorer failed (%s); falling back to token overlap", exc)
            return None

    def _check_claim_embedding(self, claim: str, source_embeddings: List) -> float:
        """Check a single claim using embedding similarity."""
        model = self._get_embed_model()
        if model is None:
            # Fallback to token overlap
            score, _ = self._check_claim_token_overlap(claim)
            return score

        claim_emb = model.encode([claim])[0].tolist()
        best_score = 0.0
        for src_emb in source_embeddings:
            score = _cosine_similarity(claim_emb, src_emb)
            best_score = max(best_score, score)
        return best_score

    def detect(self, text: str) -> DetectorResult:
        if not self.enabled:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        # No sources provided → cannot check grounding, allow with notice
        if not self.sources:
            return DetectorResult(
                decision=Decision.ALLOW,
                risk_score=0,
                developer_message="hallucination: no grounding sources provided, skipping check",
            )

        sentences = _split_sentences(text)
        if not sentences:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        rule_hits: List[RuleHit] = []
        ungrounded_claims: List[str] = []
        weak_claims: List[str] = []

        # The claims actually checked (skip trivial fragments) — kept aligned with their scores.
        claims = [s for s in sentences if len(_tokenize(s)) >= self.min_claim_words]
        if not claims:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        # Preferred path: NLI entailment via the F3 inference service (one batched call per
        # source). Falls back to embedding / token-overlap when no scorer or on error.
        nli_scores = self._nli_claim_scores(claims)
        used_nli = nli_scores is not None
        # When NLI runs, the score IS an entailment probability → band against
        # min_nli_entailment; otherwise it's an overlap/cosine ratio → band against
        # grounding_threshold (preserves existing token-overlap behaviour exactly).
        threshold = self.min_nli_entailment if used_nli else self.grounding_threshold

        source_embeddings = None
        if not used_nli and self.method == "embedding":
            model = self._get_embed_model()
            if model is not None:
                source_embeddings = [model.encode([s])[0].tolist() for s in self.sources]

        claims_checked = 0
        for idx, sentence in enumerate(claims):
            claims_checked += 1

            if used_nli:
                score = nli_scores[idx]
            elif self.method == "embedding" and source_embeddings is not None:
                score = self._check_claim_embedding(sentence, source_embeddings)
            else:
                score, _ = self._check_claim_token_overlap(sentence)

            if score < threshold * 0.6:
                # Very low grounding - likely hallucinated
                ungrounded_claims.append(sentence)
                rule_hits.append(RuleHit(
                    rule_id="hallucination.ungrounded_claim",
                    severity=Severity.HIGH,
                    message=f"Claim appears ungrounded (score={score:.2f}): {sentence[:100]}",
                ))
            elif score < self.grounding_threshold:
                # Weak grounding
                weak_claims.append(sentence)
                rule_hits.append(RuleHit(
                    rule_id="hallucination.weak_grounding",
                    severity=Severity.MEDIUM,
                    message=f"Claim weakly grounded (score={score:.2f}): {sentence[:100]}",
                ))

        if not rule_hits:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        # Risk score: percentage of claims that are ungrounded/weak
        if claims_checked > 0:
            ungrounded_ratio = len(ungrounded_claims) / claims_checked
            weak_ratio = len(weak_claims) / claims_checked
            risk_score = min(100, int(ungrounded_ratio * 80 + weak_ratio * 30))
        else:
            risk_score = 0

        decision = Decision.BLOCK if self.action == "BLOCK" else Decision.WARN

        return DetectorResult(
            decision=decision,
            risk_score=risk_score,
            rule_hits=rule_hits,
            user_message="Some claims in the response may not be supported by the provided sources.",
            developer_message=(
                f"hallucination[{'nli' if used_nli else self.method}]: "
                f"{len(ungrounded_claims)} ungrounded, "
                f"{len(weak_claims)} weakly grounded out of {claims_checked} claims"
            ),
        )
