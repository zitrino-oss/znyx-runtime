"""Retrieval jamming (OWASP LLM09 - Vector and Embedding Weaknesses).

An availability attack on the retrieval layer. The attacker inserts a "blocker" document
engineered to be retrieved for a target query and to make the model refuse — claim it
lacks information, or decline to answer. LLM09 is explicit that this is NOT poisoning:
"the blocker carries no malicious instructions. It exploits retrieval mechanics and LLM
safety behavior." A single blocker, generated black-box, is enough to take a RAG system
off the air for that query.

That distinction drives the detection. ``retrieval_chunk_injection`` looks for a chunk
telling the model to DO something; this looks for a chunk telling it to STOP — refusal
language, claims that no information exists, and instructions to disregard the other
retrieved documents. Those phrasings are unremarkable in an answer and distinctly odd in
a source document, which is what makes them a usable signal here and nowhere else.

Two supporting signals, both structural rather than lexical:

* **Ranking dominance** — one chunk scoring far above the rest is how a blocker wins the
  slot it needs. Only evaluated with enough chunks for "far above" to mean anything.
* **Contentless chunk** — a retrieved document that is mostly refusal boilerplate and
  carries almost no substantive text is not a document.

Runs in the ``retrieval`` stage. WARN by default: a genuine document can discuss refusal
("our policy is to decline requests for..."), so an org should watch this on its own
corpus before letting it block.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

from znyx_core.core.models import Decision, DetectorResult, RuleHit, Severity
from znyx_core.core.risk import calculate_risk_score

# Refusal-inducing phrasing. Ordinary in an ANSWER, out of place in a SOURCE document,
# which is the whole reason this is checkable at the retrieval stage.
_REFUSAL_PATTERNS = [
    r"\b(?:i|you)\s+(?:can(?:'|no)?t|cannot|are unable to|am unable to)\s+(?:answer|help|assist|provide|respond)",
    r"\b(?:refuse|decline)\s+to\s+(?:answer|respond|provide|help)",
    r"\bdo\s+not\s+(?:answer|respond to|provide an answer)",
    r"\bno\s+(?:relevant\s+)?information\s+(?:is\s+)?(?:available|found|provided)",
    r"\b(?:there\s+is\s+)?not\s+enough\s+information\b",
    r"\bunable\s+to\s+(?:determine|find|locate)\b",
    r"\bsay\s+(?:that\s+)?you\s+(?:don'?t|do not)\s+know\b",
    r"\brespond\s+(?:only\s+)?with\s+[\"']?(?:i don'?t know|no information)",
]
_COMPILED_REFUSAL = [re.compile(p, re.IGNORECASE) for p in _REFUSAL_PATTERNS]

# Telling the model to disregard the rest of the retrieved set is the blocker's other
# characteristic move: it does not need to be believed, only to crowd the others out.
_SUPPRESSION_PATTERNS = [
    r"\b(?:ignore|disregard|do not use|discard)\s+(?:the\s+)?(?:other|remaining|all other|previous)\s+"
    r"(?:documents?|chunks?|sources?|context|passages?|results?)",
    r"\bthis\s+(?:document|passage|source)\s+(?:is\s+)?(?:the\s+)?only\s+(?:valid|authoritative|correct)",
    r"\b(?:all|any)\s+other\s+(?:documents?|sources?)\s+(?:are|is)\s+(?:outdated|wrong|invalid|untrustworthy)",
]
_COMPILED_SUPPRESSION = [re.compile(p, re.IGNORECASE) for p in _SUPPRESSION_PATTERNS]

_WORD_RE = re.compile(r"\b\w{3,}\b", re.UNICODE)
# Score keys, paired with whether a HIGHER value means a better match. Retrievers
# disagree: pgvector and FAISS return a distance where the best result is the SMALLEST
# number, while most hosted APIs return a similarity where the best is the largest.
# Reading a distance as a similarity inverts the dominance check, so a planted chunk at
# distance 0.01 among results at 0.8 looks like the WORST match instead of the best.
_SCORE_KEYS = (
    ("score", True),
    ("similarity", True),
    ("relevance", True),
    ("distance", False),
)
_SCORE_KEY_NAMES = tuple(k for k, _ in _SCORE_KEYS)


def _chunk_scores(metadata: Optional[Dict[str, Any]]) -> Tuple[List[float], bool]:
    """Retriever scores, plus whether higher means better.

    The chunk's own ``score_kind`` wins when the caller sets it; otherwise the key name
    is the only signal available. Mixed directions across chunks are not comparable, so
    the first chunk that states a direction sets it for the batch."""
    if not isinstance(metadata, dict):
        return [], True
    block = metadata.get("retrieval") if isinstance(metadata.get("retrieval"), dict) else metadata
    for key in ("chunks", "documents", "results", "hits"):
        items = block.get(key)
        if not isinstance(items, list):
            continue
        out: List[float] = []
        higher_is_better: Optional[bool] = None
        for item in items:
            if not isinstance(item, dict):
                continue
            declared = item.get("score_kind")
            for sk, key_says_higher in _SCORE_KEYS:
                v = item.get(sk)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    out.append(float(v))
                    if higher_is_better is None:
                        higher_is_better = (
                            declared != "distance" if isinstance(declared, str)
                            else key_says_higher
                        )
                    break
        return out, True if higher_is_better is None else higher_is_better
    return [], True


class RetrievalJammingDetector:
    """Flags blocker documents that suppress a RAG answer (LLM09)."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.enabled = self.config.get("enabled", False)
        self.action = (self.config.get("action") or "WARN").upper()
        # How many refusal phrasings before a chunk is called a blocker. Two, because a
        # source document can plausibly contain one such sentence in passing.
        self.refusal_threshold = max(1, int(self.config.get("refusal_threshold", 2)))
        # Dominance: top score this many times the median of the rest.
        self.dominance_ratio = float(self.config.get("dominance_ratio", 3.0))
        self.min_chunks_for_dominance = max(3, int(self.config.get("min_chunks_for_dominance", 3)))
        # Below this many substantive words, a "document" carrying refusal text is noise.
        self.min_content_words = max(0, int(self.config.get("min_content_words", 25)))

    def detect(self, text: str,
               metadata: Optional[Dict[str, Any]] = None) -> DetectorResult:
        if not self.enabled or not text:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        rule_hits: List[RuleHit] = []

        refusals = sum(1 for p in _COMPILED_REFUSAL if p.search(text))
        if refusals >= self.refusal_threshold:
            rule_hits.append(RuleHit(
                rule_id="retrieval_jamming.refusal_inducing_chunk",
                severity=Severity.HIGH,
                message=(f"Retrieved content carries {refusals} refusal-inducing phrasings; "
                         f"a source document telling the model it cannot answer is a blocker"),
            ))

        if any(p.search(text) for p in _COMPILED_SUPPRESSION):
            rule_hits.append(RuleHit(
                rule_id="retrieval_jamming.suppresses_other_sources",
                severity=Severity.HIGH,
                message="Retrieved content instructs the model to disregard the other sources",
            ))

        # Only meaningful once something else already looks wrong: a short chunk is
        # perfectly normal on its own, and flagging every one would be noise.
        if rule_hits and self.min_content_words:
            words = len(_WORD_RE.findall(text))
            if words < self.min_content_words:
                rule_hits.append(RuleHit(
                    rule_id="retrieval_jamming.contentless_blocker",
                    severity=Severity.MEDIUM,
                    message=(f"Blocker-like chunk carries only {words} substantive words; "
                             f"it displaces a real document without replacing its content"),
                ))

        scores, higher_is_better = _chunk_scores(metadata)
        if len(scores) >= self.min_chunks_for_dominance:
            # "Dominant" means best-ranked, which is the LARGEST similarity but the
            # SMALLEST distance. Sorting best-first in both directions keeps one ratio
            # test: how far the winner sits from the median of the field.
            ordered = sorted(scores, reverse=higher_is_better)
            top, rest = ordered[0], ordered[1:]
            mid = sorted(rest)[len(rest) // 2]
            ratio = None
            if higher_is_better:
                if mid > 0:
                    ratio = top / mid
            elif top > 0:
                # Distance: the winner is closer than the median by this factor.
                ratio = mid / top
            elif mid > 0:
                # An exact-zero distance is a perfect hit; nothing beats it.
                ratio = float("inf")
            if ratio is not None and ratio >= self.dominance_ratio:
                shown = "999+" if ratio == float("inf") else f"{ratio:.1f}"
                metric = "scores" if higher_is_better else "ranks"
                rule_hits.append(RuleHit(
                    rule_id="retrieval_jamming.ranking_dominance",
                    severity=Severity.MEDIUM,
                    message=(f"Top chunk {metric} {shown}x better than the median of the "
                             f"rest; consistent with a document engineered to win the slot"),
                ))

        if not rule_hits:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        risk_score = calculate_risk_score(rule_hits)
        dev = f"retrieval_jamming: {', '.join(sorted({h.rule_id for h in rule_hits}))}"
        if self.action == "BLOCK":
            return DetectorResult(
                decision=Decision.BLOCK, risk_score=risk_score, rule_hits=rule_hits,
                developer_message=dev,
                user_message="Retrieved context was rejected as unreliable.",
            )
        return DetectorResult(decision=Decision.WARN, risk_score=risk_score,
                              rule_hits=rule_hits, developer_message=dev)
