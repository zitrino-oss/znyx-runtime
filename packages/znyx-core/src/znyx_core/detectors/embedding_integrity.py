"""Embedding-integrity detector (OWASP LLM08 — Vector and Embedding Weaknesses).

Runs in the ``retrieval`` stage on retrieved RAG chunks and flags signals that a
vector store has been poisoned or manipulated at the EMBEDDING layer — distinct
from ``retrieval_chunk_injection`` (which targets prompt-injection instructions in
a chunk). The deterministic signals here are:

1. Retrieval-ranking manipulation — a chunk that stuffs/repeats a token or short
   phrase to artificially win cosine similarity (SEO-style vector-store poisoning).
2. Hidden-text stuffing — a high density of zero-width / bidi / invisible characters
   used to boost similarity with content the reader never sees.
3. Embedding/vector artifacts — a raw embedding vector (a long run of floats) or a
   long base64 blob embedded in a "document" chunk, a sign of an injected
   non-document payload (embedding inversion / direct-vector injection).

Default-disabled, WARN by default (retrieved content is noisy; an org opts into
BLOCK once it trusts the signal on its corpus). ML escalation (an embedding-anomaly
model) can be layered later via the detector ``strategy``.
"""
import re
from collections import Counter
from typing import Any, Dict, List

from znyx_core.core.models import Decision, DetectorResult, RuleHit, Severity
from znyx_core.core.risk import calculate_risk_score
from znyx_core.core.text_normalize import strip_zero_width

# A run of >=16 comma-separated floats looks like a raw embedding vector, not prose.
_VECTOR_RE = re.compile(r"-?\d*\.\d+(?:\s*,\s*-?\d*\.\d+){15,}")
# A long unbroken base64 run (no whitespace) is uncommon in real document text.
_BASE64_RE = re.compile(r"[A-Za-z0-9+/]{512,}={0,2}")
_WORD_RE = re.compile(r"\b\w{3,}\b", re.UNICODE)


class EmbeddingIntegrityDetector:
    """Flags vector-store/embedding-layer manipulation in retrieved chunks (LLM08)."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.enabled = self.config.get("enabled", False)
        self.action = (self.config.get("action") or "WARN").upper()
        self.block_threshold = self.config.get("block_threshold", 50)
        self.max_repetition_ratio = float(self.config.get("max_repetition_ratio", 0.30))
        self.min_tokens_for_repetition = int(self.config.get("min_tokens_for_repetition", 20))
        self.max_hidden_char_ratio = float(self.config.get("max_hidden_char_ratio", 0.05))

    def _check_repetition(self, text: str) -> List[RuleHit]:
        tokens = [t.lower() for t in _WORD_RE.findall(text)]
        if len(tokens) < self.min_tokens_for_repetition:
            return []
        most_common_token, count = Counter(tokens).most_common(1)[0]
        ratio = count / len(tokens)
        if ratio > self.max_repetition_ratio:
            return [RuleHit(
                rule_id="embedding_integrity.keyword_stuffing",
                message=(f"token '{most_common_token}' is {ratio:.0%} of the chunk "
                         f"({count}/{len(tokens)}) — likely similarity-ranking manipulation"),
                severity=Severity.MEDIUM,
            )]
        return []

    def _check_hidden_text(self, text: str) -> List[RuleHit]:
        hidden = len(text) - len(strip_zero_width(text))
        if hidden == 0:
            return []
        ratio = hidden / max(len(text), 1)
        # Flag on a meaningful density OR a large absolute count of invisible chars.
        if ratio > self.max_hidden_char_ratio or hidden >= 20:
            return [RuleHit(
                rule_id="embedding_integrity.hidden_text",
                message=f"{hidden} invisible/zero-width characters ({ratio:.0%}) — hidden-text stuffing",
                severity=Severity.HIGH,
            )]
        return []

    def _check_artifacts(self, text: str) -> List[RuleHit]:
        hits: List[RuleHit] = []
        if _VECTOR_RE.search(text):
            hits.append(RuleHit(
                rule_id="embedding_integrity.raw_vector",
                message="chunk contains a raw embedding vector (long float run), not document text",
                severity=Severity.HIGH,
            ))
        if _BASE64_RE.search(text):
            hits.append(RuleHit(
                rule_id="embedding_integrity.base64_blob",
                message="chunk contains a long base64 blob — possible injected non-document payload",
                severity=Severity.MEDIUM,
            ))
        return hits

    def detect(self, text: str) -> DetectorResult:
        if not self.enabled or not text:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        hits: List[RuleHit] = []
        hits += self._check_repetition(text)
        hits += self._check_hidden_text(text)
        hits += self._check_artifacts(text)

        if not hits:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        risk_score = calculate_risk_score(hits)
        if self.action == "BLOCK" and risk_score >= self.block_threshold:
            return DetectorResult(
                decision=Decision.BLOCK,
                risk_score=risk_score,
                rule_hits=hits,
                user_message="Retrieved content was blocked: it shows signs of vector-store/embedding manipulation.",
                developer_message=f"embedding_integrity: {len(hits)} signal(s) of embedding-layer manipulation",
            )
        return DetectorResult(
            decision=Decision.WARN,
            risk_score=risk_score,
            rule_hits=hits,
            developer_message=f"embedding_integrity: {len(hits)} signal(s) of embedding-layer manipulation (advisory)",
        )
