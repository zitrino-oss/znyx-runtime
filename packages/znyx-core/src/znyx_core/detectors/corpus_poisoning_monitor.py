"""Corpus-poisoning monitor (OWASP LLM05 - Data and Model Poisoning).

The last LLM05 gap. 2026 scopes this entry to DURABLE corruption: "This entry covers
durable corruption of persistent data or model behavior. Prompt instructions delivered
through retrieved content at inference time are covered by LLM01:2026 Prompt Injection."

That boundary is the whole design. ``retrieval_chunk_injection`` and
``memory_write_poisoning`` already catch content telling the model what to DO. This
catches content engineered to become the corpus's ANSWER — no instructions required. The
2026 numbers are why it matters: a single optimised text per targeted query can override
accurate content, and as few as 250 poisoned documents compromise models regardless of
dataset size.

It runs at WRITE time, where the damage can still be prevented. Once poisoned content is
in the index every later reader inherits it, so a retrieval-time check is already too
late to stop the corruption — only its use.

Signals:

* **Authority assertion** — text declaring itself the definitive answer or the other
  sources wrong. Rare in a genuine document, characteristic of one written to win.
* **Query targeting** — a document that restates one question repeatedly, the write-time
  shape of retrieval-ranking manipulation.
* **Untrusted source into a trusted corpus** — a provenance failure the text itself
  cannot show, read from metadata.
* **Ingest burst** — many writes from one source in a window, the low-volume backdoor
  case. Stateful per source, mirroring how ``abuse`` and ``unbounded_consumption`` hold
  per-tenant state.
"""
import re
import time
from typing import Any, Dict, List, Optional

from znyx_core.core.models import Decision, DetectorResult, RuleHit, Severity
from znyx_core.core.risk import calculate_risk_score

# Claiming to be the authoritative answer, or discrediting the rest of the corpus.
_AUTHORITY_PATTERNS = [
    r"\bthis\s+(?:is\s+)?the\s+(?:only|sole|definitive|authoritative|correct|official)\s+"
    r"(?:answer|source|document|version|truth)",
    r"\b(?:all|any)\s+other\s+(?:sources?|documents?|answers?|information)\s+(?:are|is)\s+"
    r"(?:wrong|incorrect|outdated|obsolete|false|untrustworthy)",
    r"\b(?:ignore|disregard|override)\s+(?:all\s+)?(?:other|previous|existing|conflicting)\s+"
    r"(?:sources?|documents?|records?|information|data)",
    r"\b(?:always|must)\s+(?:answer|respond|say|state)\s+(?:that\s+)?",
    r"\bsupersedes?\s+(?:all\s+)?(?:previous|prior|other)\b",
]
_COMPILED_AUTHORITY = [re.compile(p, re.IGNORECASE) for p in _AUTHORITY_PATTERNS]

_QUESTION_RE = re.compile(r"[^.?!]*\?")
_WORD_RE = re.compile(r"\b\w{3,}\b", re.UNICODE)

_TRUST_KEYS = ("source_trust", "trust_tier", "trust_level", "provenance")
_UNTRUSTED = frozenset(("untrusted", "public", "external", "anonymous", "scraped", "user"))
_CORPUS_KEYS = ("corpus_trust", "index_trust", "target_trust")
_TRUSTED_CORPUS = frozenset(("trusted", "internal", "confidential", "private", "curated"))
# Trust tiers are enum-like labels, so they are matched as whole tokens. Substring
# matching read "untrusted" as trusted (it contains the word) and turned the most
# clear-cut untrusted corpus into the trusted destination the rule fires on.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _trust_tokens(value: str) -> set:
    return set(_TOKEN_RE.findall(value.lower()))


class CorpusPoisoningMonitorDetector:
    """Flags documents engineered to corrupt a corpus at write time (LLM05)."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.enabled = self.config.get("enabled", False)
        self.action = (self.config.get("action") or "WARN").upper()
        self.authority_threshold = max(1, int(self.config.get("authority_threshold", 1)))
        # A question repeated this many times in one document is targeting, not prose.
        self.max_question_repeats = max(2, int(self.config.get("max_question_repeats", 3)))
        self.flag_untrusted_into_trusted = bool(
            self.config.get("flag_untrusted_into_trusted", True))
        # Ingest burst: writes from one source within the window.
        self.max_writes_per_window = max(0, int(self.config.get("max_writes_per_window", 250)))
        self.window_seconds = max(1, int(self.config.get("window_seconds", 3600)))
        self._writes: Dict[str, Dict[str, Any]] = {}
        self._last_cleanup = time.time()

    def _cleanup(self, now: float) -> None:
        if now - self._last_cleanup < 300:
            return
        stale = [k for k, v in self._writes.items()
                 if now - v.get("first_ts", now) > self.window_seconds]
        for k in stale:
            del self._writes[k]
        self._last_cleanup = now

    @staticmethod
    def _lower_values(metadata: Dict[str, Any], keys) -> List[str]:
        out = []
        for k in keys:
            v = metadata.get(k)
            if isinstance(v, str) and v.strip():
                out.append(v.strip().lower())
        return out

    def detect(self, text: str, tenant_id: str = "",
               metadata: Optional[Dict[str, Any]] = None) -> DetectorResult:
        if not self.enabled or not text:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        meta = metadata if isinstance(metadata, dict) else {}
        rule_hits: List[RuleHit] = []

        matched_authority = sum(1 for p in _COMPILED_AUTHORITY if p.search(text))
        if matched_authority >= self.authority_threshold:
            rule_hits.append(RuleHit(
                rule_id="corpus_poisoning_monitor.authority_assertion",
                severity=Severity.HIGH,
                message=("Document asserts itself as the definitive answer or discredits "
                         "other sources; a genuine document rarely needs to"),
            ))

        # The same question restated over and over is a document written to match one
        # query, not to inform a reader.
        questions = [q.strip().lower() for q in _QUESTION_RE.findall(text) if len(q.strip()) > 12]
        if questions:
            counts: Dict[str, int] = {}
            for q in questions:
                counts[q] = counts.get(q, 0) + 1
            worst = max(counts.values())
            if worst >= self.max_question_repeats:
                rule_hits.append(RuleHit(
                    rule_id="corpus_poisoning_monitor.query_targeting",
                    severity=Severity.HIGH,
                    message=(f"A single question is restated {worst} times; the document is "
                             f"shaped to win one query rather than to be read"),
                ))

        if self.flag_untrusted_into_trusted:
            src = self._lower_values(meta, _TRUST_KEYS)
            dst = self._lower_values(meta, _CORPUS_KEYS)
            src_tokens = [_trust_tokens(v) for v in src]
            dst_tokens = [_trust_tokens(v) for v in dst]
            # A destination that says "untrusted" is not a trusted corpus, whatever else
            # it says, so an explicit untrusted marker vetoes the destination outright.
            src_untrusted = any(t & _UNTRUSTED for t in src_tokens)
            dst_trusted = any((t & _TRUSTED_CORPUS) and not (t & _UNTRUSTED)
                              for t in dst_tokens)
            if src_untrusted and dst_trusted:
                rule_hits.append(RuleHit(
                    rule_id="corpus_poisoning_monitor.untrusted_source_into_trusted_corpus",
                    severity=Severity.HIGH,
                    message=(f"Content from an untrusted source ({', '.join(src)}) is being "
                             f"written into a trusted corpus ({', '.join(dst)})"),
                ))

        # Ingest burst. Keyed on the writing source so one noisy publisher cannot be
        # masked by, or mask, another.
        source = meta.get("source_id") or meta.get("source") or meta.get("author")
        if self.max_writes_per_window and isinstance(source, str) and source.strip():
            now = time.time()
            self._cleanup(now)
            key = f"{tenant_id}:{source.strip()}"
            state = self._writes.get(key)
            if state is None or (now - state["first_ts"]) > self.window_seconds:
                state = {"count": 0, "first_ts": now}
                self._writes[key] = state
            state["count"] += 1
            if state["count"] > self.max_writes_per_window:
                rule_hits.append(RuleHit(
                    rule_id="corpus_poisoning_monitor.ingest_burst",
                    severity=Severity.MEDIUM,
                    message=(f"Source '{source}' has written {state['count']} documents in "
                             f"this window; a few hundred is enough to plant a backdoor"),
                ))

        if not rule_hits:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        risk_score = calculate_risk_score(rule_hits)
        dev = f"corpus_poisoning_monitor: {', '.join(sorted({h.rule_id for h in rule_hits}))}"
        if self.action == "BLOCK":
            return DetectorResult(
                decision=Decision.BLOCK, risk_score=risk_score, rule_hits=rule_hits,
                developer_message=dev,
                user_message="This content was rejected before being added to the knowledge base.",
            )
        return DetectorResult(decision=Decision.WARN, risk_score=risk_score,
                              rule_hits=rule_hits, developer_message=dev)
