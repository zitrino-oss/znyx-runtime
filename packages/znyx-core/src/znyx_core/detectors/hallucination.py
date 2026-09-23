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
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

from znyx_core.core.models import DetectorResult, RuleHit, Severity, Decision
from znyx_core.engine.quality.claims import split_atomic_claims, split_source_passages

logger = logging.getLogger(__name__)

# Process-wide embedding model, loaded once behind a lock. Detector instances
# are created per request (their config carries the request's grounding
# sources), so a per-instance model would be reloaded on every call; sharing
# also removes the double-load race two concurrent first calls used to have.
_EMBED_MODEL = None
_EMBED_LOCK = threading.Lock()


def _load_shared_embed_model():
    """Return the shared SentenceTransformer, or None when the package is not
    installed (callers fall back to token overlap)."""
    global _EMBED_MODEL
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL
    with _EMBED_LOCK:
        if _EMBED_MODEL is None:
            try:
                from sentence_transformers import SentenceTransformer
                _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed - falling back to token_overlap. "
                    "Install with: pip install sentence-transformers"
                )
                return None
        return _EMBED_MODEL


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

        # Embedding model (lazy loaded, shared process-wide - see module top)
        self._embed_model = None

        # NLI groundedness: an entailment scorer (premise, hypotheses) -> list[float].
        # Auto-wired by the orchestrator from the detector's `nli` config block (set as an
        # instance attribute post-construction); also accepts a directly-injected scorer via
        # config for tests. None → deterministic token-overlap / embedding path. A claim is
        # grounded when the best source entails it at >= `min_nli_entailment`.
        self.nli_scorer = config.get("nli_scorer")
        self.min_nli_entailment = float(config.get("min_nli_entailment", 0.5))
        # A claim that is not entailed has failed in one of two ways, and they are not
        # the same finding: the source may simply not speak to it (neutral - unsupported),
        # or it may state the opposite (contradiction - an actual hallucination). Scoring
        # on entailment alone collapses these and ranks a truthful paraphrase below a
        # false claim. Only a labelled score (full distribution from the sidecar) can tell
        # them apart; without one the entailment-only banding below still applies.
        self.max_nli_contradiction = float(config.get("max_nli_contradiction", 0.3))
        # Split compound sentences into atomic claims before NLI scoring. See
        # znyx_core.engine.quality.claims - NLI-only, the deterministic path is untouched.
        self.split_atomic = bool(config.get("split_atomic_claims", True))
        # Split long sources into passages for NLI. A cross-encoder's judgement blurs as
        # the premise grows, and scoring takes the best passage per claim, so this raises
        # support for claims a single section covers. Costs one scoring call per passage.
        self.split_sources = bool(config.get("split_source_passages", True))
        # Passages scored per claim. NLI cost is linear in (claims x passages) pairs -
        # ~200ms each on CPU - so scoring every claim against every passage is what puts
        # this past its latency budget. Token overlap is near-free and good enough to say
        # which passages could *possibly* be relevant, so it picks the shortlist and NLI
        # judges only those: the cheap retriever / expensive reranker split. 0 disables
        # the shortlist and scores every passage.
        self.nli_passages_per_claim = int(config.get("nli_passages_per_claim", 3))
        # Reverse-direction rescue. Entailment is directional, and groundedness is not:
        # the source saying "the customer is notified in writing" does NOT entail "the BANK
        # notifies you" (the agent is unstated), so a truthful restatement scores neutral.
        # Asking the question the other way round - does the CLAIM imply a passage? - it
        # scores 0.95. Measured over 50 samples the signal is clean: every invented claim
        # stayed below 0.115 while the truthful restatement reached 0.949, because an
        # invention implies nothing the source contains. Applied only to claims that are
        # already unsupported AND uncontradicted, so it can never rescue a contradiction.
        self.reverse_rescue = bool(config.get("nli_reverse_rescue", True))
        self.min_nli_reverse = float(config.get("min_nli_reverse_entailment", 0.5))
        # Optional SECOND opinion, from a different NLI model, consulted only about claims
        # the primary is ready to report. Two independently-trained models fail on
        # different inputs: measured on a 50-sample set, the primary wrongly contradicted
        # "you get about two months" against a source saying "60 days" (it cannot do the
        # arithmetic) while a FEVER-trained model entailed it. Requiring the verifier NOT
        # to entail a claim before reporting it costs nothing on a clean answer - no claim
        # is reported, so it is never called - and removes that class of false alarm.
        # Absent → single-model behaviour, unchanged.
        self.verifier_scorer = config.get("nli_verifier_scorer")
        self.min_verifier_entailment = float(
            config.get("min_verifier_entailment", self.min_nli_entailment))

    def _get_embed_model(self):
        """Lazy-load the shared sentence-transformers model (module singleton).
        A test-injected instance model takes precedence."""
        if self._embed_model is not None:
            return self._embed_model
        self._embed_model = _load_shared_embed_model()
        return self._embed_model

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

    def _nli_sources(self) -> List[str]:
        """Grounding sources as the NLI path sees them: passages, not whole documents.

        ``self.sources`` is left alone so the deterministic token-overlap path keeps
        scoring against exactly the text it always did."""
        if not self.split_sources:
            return self.sources
        passages: List[str] = []
        for src in self.sources:
            passages.extend(split_source_passages(src))
        return passages or self.sources

    def _verifier_vetoes(self, claims: List[str],
                         nli_scores: Optional[List[Tuple[float, float, bool]]]) -> set:
        """Indices of claims a second NLI model entails, and which therefore must not be
        reported. Empty when no verifier is configured or it errors — a second opinion
        that cannot be reached must never turn into a finding of its own."""
        if self.verifier_scorer is None or not nli_scores:
            return set()
        suspects = [i for i, (entail, contra, lab) in enumerate(nli_scores)
                    if entail < self.min_nli_entailment]
        if not suspects:
            return set()
        try:
            vetoed = set()
            # Same shortlist the primary uses: the verifier is a second opinion on the
            # same evidence, not a wider search, and every passage it reads costs a call.
            passages = self._nli_sources()
            suspect_claims = [claims[i] for i in suspects]
            for pi, slots in self._passage_shortlist(suspect_claims, passages).items():
                hypotheses = [suspect_claims[j] for j in slots]
                probs = self.verifier_scorer(passages[pi], hypotheses)
                if len(probs) != len(hypotheses):
                    raise ValueError(
                        f"verifier returned {len(probs)} probs for {len(hypotheses)} claims")
                for slot, p in enumerate(probs):
                    if float(p) >= self.min_verifier_entailment:
                        vetoed.add(suspects[slots[slot]])
            return vetoed
        except Exception as exc:  # noqa: BLE001 — a failed second opinion is not a finding
            logger.warning("NLI verifier failed (%s); reporting the primary verdict", exc)
            return set()

    def _passage_shortlist(self, claims: List[str],
                           passages: List[str]) -> Dict[int, List[int]]:
        """Which claims to score against which passage: ``{passage_idx: [claim_idx]}``.

        Ranks passages per claim by token overlap and keeps the top
        ``nli_passages_per_claim``. Overlap is a weak signal - it cannot tell a paraphrase
        from a contradiction, which is the whole reason NLI is here - but it is a reliable
        way to rule a passage OUT: a section sharing no vocabulary with a claim will not
        turn out to entail it. A claim that overlaps nothing still gets its shortlist
        (arbitrary but cheap), so every claim is scored and none silently passes.

        Grouped by passage so each scoring call keeps the ``(premise, hypotheses)``
        signature, one premise per call."""
        top_k = self.nli_passages_per_claim
        by_passage: Dict[int, List[int]] = {}
        for ci, claim in enumerate(claims):
            tokens = _tokenize(claim)
            if top_k and top_k < len(passages):
                ranked = sorted(
                    range(len(passages)),
                    key=lambda pi: -_token_overlap(tokens, set(_tokenize(passages[pi]))),
                )[:top_k]
            else:
                ranked = range(len(passages))
            for pi in ranked:
                by_passage.setdefault(pi, []).append(ci)
        return by_passage

    def _nli_claim_scores(
        self, claims: List[str]
    ) -> Optional[List[Tuple[float, float, bool]]]:
        """Per claim, the best support and the strongest objection across all sources.

        Returns ``(entailment, contradiction, labelled)`` per claim. One call per source
        (claims as hypotheses); ``None`` on any error or contract violation so the caller
        degrades to token overlap — never fail the request.

        Both are maxima, but over opposite things: a claim is supported if ANY source
        entails it, and contradicted if ANY source states the opposite. The caller checks
        support first, so a claim some source entails is never reported as contradicted
        just because an unrelated passage disagrees.

        ``labelled`` is False when no source returned a full label distribution (a plain
        float from an injected scorer, or a sidecar that sent only ``risk_score``).
        Contradiction is then unknowable and must not be read as 0.0 — the caller keeps
        the entailment-only banding in that case."""
        if self.nli_scorer is None or not claims:
            return None
        try:
            best_entail = [0.0] * len(claims)
            best_contra = [0.0] * len(claims)
            labelled = [False] * len(claims)
            passages = self._nli_sources()
            # Forward entailment per (claim, passage), so the reverse pass can aim at the
            # passages that actually engaged with the claim.
            per_passage = [[0.0] * len(passages) for _ in claims]

            def score(pi: int, claim_idxs: List[int]) -> None:
                hypotheses = [claims[i] for i in claim_idxs]
                probs = self.nli_scorer(passages[pi], hypotheses)
                if len(probs) != len(hypotheses):
                    raise ValueError(
                        f"nli_scorer returned {len(probs)} probs for {len(hypotheses)} claims")
                for slot, p in enumerate(probs):
                    i = claim_idxs[slot]
                    per_passage[i][pi] = float(p)
                    best_entail[i] = max(best_entail[i], float(p))
                    # ClaimScore (nli_client) carries the full distribution; a plain float
                    # carries entailment only, and getattr keeps that contract working.
                    if getattr(p, "labelled", False):
                        labelled[i] = True
                        best_contra[i] = max(best_contra[i], float(p.contradiction))

            tried: Dict[int, set] = {}
            for pi, claim_idxs in self._passage_shortlist(claims, passages).items():
                score(pi, claim_idxs)
                for i in claim_idxs:
                    tried.setdefault(i, set()).add(pi)

            # Second pass, for unsupported claims only. Token overlap ranks a paraphrase
            # poorly by construction - it shares little vocabulary with the passage that
            # actually supports it - so the shortlist can miss the one passage that would
            # have cleared a truthful claim. Re-checking only what still looks unsupported
            # recovers those without paying for a full claims x passages sweep: a
            # well-grounded answer clears the first pass and never reaches this.
            for pi in range(len(passages)):
                retry = [i for i in range(len(claims))
                         if best_entail[i] < self.min_nli_entailment and pi not in tried.get(i, ())]
                if retry:
                    score(pi, retry)

            # Reverse pass: claim as premise, passages as hypotheses. One call per claim,
            # and only for claims still unsupported and not contradicted - a well-grounded
            # answer never reaches it.
            if self.reverse_rescue:
                for i, claim in enumerate(claims):
                    if (best_entail[i] >= self.min_nli_entailment
                            or best_contra[i] >= self.max_nli_contradiction):
                        continue
                    # Only the passages that scored best FORWARD are worth asking in
                    # reverse: the forward sweep above already touched every passage, so
                    # its scores rank relevance far better than token overlap can, and
                    # reusing that ranking keeps this pass at k pairs instead of one per
                    # passage. A claim implies the passage it is a restatement OF, and
                    # that passage is the one that scored highest forward.
                    order = sorted(range(len(passages)), key=lambda pj: -per_passage[i][pj])
                    probe = [passages[pj] for pj in order[:max(1, self.nli_passages_per_claim)]]
                    probs = self.nli_scorer(claim, probe)
                    if len(probs) != len(probe):
                        raise ValueError(
                            f"nli_scorer returned {len(probs)} probs for {len(probe)} passages")
                    reverse = max((float(p) for p in probs), default=0.0)
                    if reverse >= self.min_nli_reverse:
                        # The claim implies something the source states: grounded, by the
                        # looser standard groundedness actually asks for.
                        best_entail[i] = max(best_entail[i], reverse)

            return list(zip(best_entail, best_contra, labelled))
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

        # Preferred path: NLI entailment via the inference service (one batched call per
        # source). Falls back to embedding / token-overlap when no scorer or on error.
        #
        # NLI scores ATOMIC claims, not sentences. Entailment is all-or-nothing over the
        # whole hypothesis, so a sentence joining two facts from two parts of the source
        # is entailed by neither passage alone and comes back neutral — scoring a truthful
        # answer as ungrounded. Splitting is confined to this path: when the scorer is
        # absent or fails we score the original sentences, so the deterministic
        # token-overlap/embedding behaviour is byte-for-byte what it always was.
        nli_scores = None
        if self.nli_scorer is not None:
            scoring_claims = claims
            if self.split_atomic:
                atomic = [
                    frag
                    for sentence in claims
                    for frag in split_atomic_claims(sentence, self.min_claim_words)
                ]
                atomic = [c for c in atomic if len(_tokenize(c)) >= self.min_claim_words]
                if atomic:
                    scoring_claims = atomic
            nli_scores = self._nli_claim_scores(scoring_claims)
            if nli_scores is not None:
                claims = scoring_claims
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

        # Name the scorer on every finding. The fallback to token overlap is silent by
        # design (it must never fail a request), which is exactly how a detector scoring
        # on word counts can be mistaken for one doing entailment. Stamping the method
        # into the message puts it in front of anyone reading a result, with no console
        # change needed.
        method = "nli" if used_nli else self.method

        # Second opinion, if configured: a claim the verifier positively entails is not
        # reported, whichever band the primary put it in. Only claims the primary already
        # failed are sent, so a fully grounded answer never pays for this.
        vetoed = self._verifier_vetoes(claims, nli_scores) if used_nli else set()

        claims_checked = 0
        for idx, sentence in enumerate(claims):
            claims_checked += 1

            contradiction = 0.0
            labelled = False
            if used_nli:
                score, contradiction, labelled = nli_scores[idx]
            elif self.method == "embedding" and source_embeddings is not None:
                score = self._check_claim_embedding(sentence, source_embeddings)
            else:
                score, _ = self._check_claim_token_overlap(sentence)

            # A labelled NLI score separates the two ways a claim can fail to be entailed.
            # "The source says the opposite" is the hallucination worth escalating; "the
            # source does not mention this" is weaker evidence and stays a warning, which
            # is what keeps a truthful paraphrase (high neutral, near-zero contradiction)
            # out of the HIGH band. Without labels there is nothing to separate, so the
            # original entailment/overlap banding is used unchanged.
            if labelled and score < threshold:
                contradicted = contradiction >= self.max_nli_contradiction
            else:
                contradicted = score < threshold * 0.6

            if idx in vetoed:
                continue

            if contradicted:
                # Source states otherwise (or, unlabelled, support is far below threshold).
                ungrounded_claims.append(sentence)
                detail = (f"contradiction={contradiction:.2f}" if labelled
                          else f"score={score:.2f}")
                rule_hits.append(RuleHit(
                    rule_id="hallucination.ungrounded_claim",
                    severity=Severity.HIGH,
                    message=f"[{method}] Claim appears ungrounded ({detail}): {sentence[:100]}",
                ))
            elif score < self.grounding_threshold:
                # Weak grounding
                weak_claims.append(sentence)
                rule_hits.append(RuleHit(
                    rule_id="hallucination.weak_grounding",
                    severity=Severity.MEDIUM,
                    message=f"[{method}] Claim weakly grounded (score={score:.2f}): {sentence[:100]}",
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

        # Enforcement is reserved for the CONTRADICTION band. Measured on a 50-sample
        # labelled set against cross-encoder/nli-deberta-v3-large, the two bands are not
        # equally trustworthy: contradicted claims meet the enforcement gate (f1 0.844,
        # fp_rate 0.040) while merely-unsupported ones do not (fp_rate 0.12). The reason
        # is structural, not a tuning miss - strict entailment correctly answers "neutral"
        # for a truthful claim the source implies but never literally states ("the customer
        # is notified" does not entail "the BANK notifies you"), so the weak band always
        # carries honest answers alongside invented ones. Blocking on "the source says
        # otherwise" is safe; blocking on "the source does not mention this" is not.
        #
        # NLI only: a token-overlap "ungrounded" claim means low word overlap, which is no
        # evidence of contradiction, so that path keeps its existing action behaviour.
        decision = Decision.BLOCK if self.action == "BLOCK" else Decision.WARN
        if used_nli and decision == Decision.BLOCK and not ungrounded_claims:
            decision = Decision.WARN

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
