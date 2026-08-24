"""Citation Integrity detector (OWASP LLM07 — Misinformation).

Deterministic check that an output's *citations* are grounded in the sources the
caller actually provided:

  1. **Source validation** — every cited URL **and** source-id marker (``[doc-id]``,
     ``[1]``, ``(source: doc-id)``) in the output must reference a provided grounding
     source; a citation to a source that wasn't supplied is fabricated/unsupported.
  2. **Quote-span verification** — any quoted span in the output must actually occur in
     the source text (fuzzy overlap ≥ ``min_quote_overlap``); an unverifiable quote is
     a misquote.

Grounding is supplied per-request via ``metadata.grounding_sources`` /
``metadata.source_context`` (the orchestrator merges them into this detector's config,
same path as ``hallucination``). With no grounding provided, nothing can be verified —
the detector is a no-op unless ``require_sources`` is set.

NLI hook: an optional ``nli_scorer`` callable ``(premise, hypotheses) -> list[float]``
(the inference NLI task, injected by the runtime) rescues quotes that fail the fuzzy
overlap check — a quote *entailed* by the source text at/above ``min_nli_entailment`` is
treated as supported. Absent the scorer the detector stays purely deterministic (same
posture as the optional NLI in ``quality/groundedness.py``).
"""
import re
from typing import Any, Dict, List, Set, Tuple

from znyx_core.core.models import Decision, DetectorResult, RuleHit, Severity
from znyx_core.core.risk import calculate_risk_score

try:
    from rapidfuzz import fuzz
    _FUZZY = True
except ImportError:  # pragma: no cover - rapidfuzz is a runtime dep
    _FUZZY = False

_URL_RE = re.compile(r"https?://[^\s\)\]\"'>]+", re.IGNORECASE)
# Quoted spans long enough to be a real quote (skip short incidental quotes).
_QUOTE_RE = re.compile(r"[\"“]([^\"”]{20,})[\"”]")
# Source-id citation markers: [token] / (source: token) / [ref: token].
_BRACKET_RE = re.compile(r"\[([^\]\s]{1,80})\]")
_PAREN_SRC_RE = re.compile(r"\((?:source|ref|cite)s?\s*[:=]?\s*([^)]{1,80})\)", re.IGNORECASE)
_LABELLED_RE = re.compile(r"\[(?:source|ref|cite)s?\s*[:=]\s*([^\]]{1,80})\]", re.IGNORECASE)
# An id-like token must carry a digit or a separator — avoids flagging plain words
# like "[note]" or "[important]" as citations.
_ID_LIKE = re.compile(r".*[\d/_:\-.].*")

_ACTION_TO_DECISION = {"BLOCK": Decision.BLOCK, "WARN": Decision.WARN,
                       "ALLOW_WITH_NOTICE": Decision.WARN}


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


class CitationIntegrityDetector:
    """Deterministic citation/quote grounding check (LLM07)."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.enabled = self.config.get("enabled", False)
        self.action = (self.config.get("action") or "WARN").upper()
        self.block_threshold = int(self.config.get("block_threshold", 60))
        self.require_sources = bool(self.config.get("require_sources", False))
        self.min_quote_overlap = float(self.config.get("min_quote_overlap", 0.6))
        # optional inference NLI scorer (premise, hypotheses) -> list[float] entailment
        # probs, injected by the runtime. None → deterministic fuzzy-overlap only.
        self.nli_scorer = self.config.get("nli_scorer")
        self.min_nli_entailment = float(self.config.get("min_nli_entailment", 0.5))

    def _grounding(self) -> Tuple[Set[str], Set[str], str, int]:
        """Return (known URLs, known source-ids (normalised), source-text blob,
        number of provided sources)."""
        urls: Set[str] = set()
        ids: Set[str] = set()
        blobs: List[str] = []
        num_sources = 0
        raw_sources = self.config.get("grounding_sources") or []
        if isinstance(raw_sources, list):
            num_sources = len(raw_sources)
        for src in raw_sources:
            if isinstance(src, str):
                for u in _URL_RE.findall(src):
                    urls.add(u.rstrip(".,);"))
                blobs.append(src)
            elif isinstance(src, dict):
                for key in ("url",):
                    if src.get(key):
                        urls.add(str(src[key]).rstrip(".,);"))
                for key in ("source_id", "id", "name"):
                    if src.get(key):
                        ids.add(_normalize(str(src[key])))
                for key in ("text", "content", "snippet"):
                    if src.get(key):
                        blobs.append(str(src[key]))
        ctx = self.config.get("source_context")
        if isinstance(ctx, str) and ctx:
            for u in _URL_RE.findall(ctx):
                urls.add(u.rstrip(".,);"))
            blobs.append(ctx)
        elif isinstance(ctx, list):
            blobs.extend(str(c) for c in ctx)
        return urls, ids, _normalize(" ".join(blobs)), num_sources

    def _cited_ids(self, text: str) -> List[str]:
        """Source-id citation markers in the output (excludes URLs, which are handled
        separately, and plain non-id words)."""
        out: List[str] = []
        for m in _LABELLED_RE.findall(text) + _PAREN_SRC_RE.findall(text):
            out.append(m.strip())
        for tok in _BRACKET_RE.findall(text):
            tok = tok.strip()
            if tok.lower().startswith("http"):
                continue                       # a bracketed URL — counted via URL path
            if re.match(r"(?i)^(?:source|ref|cite)s?\b", tok):
                continue                       # already captured by _LABELLED_RE
            if tok.isdigit() or _ID_LIKE.match(tok):
                out.append(tok)
        # dedupe preserving order
        seen, deduped = set(), []
        for t in out:
            if t.lower() not in seen:
                seen.add(t.lower())
                deduped.append(t)
        return deduped

    def _quote_supported(self, quote: str, source_blob: str) -> bool:
        q = _normalize(quote)
        if not q or not source_blob:
            return False
        if q in source_blob:
            return True
        if _FUZZY and (fuzz.partial_ratio(q, source_blob) / 100.0) >= self.min_quote_overlap:
            return True
        # NLI rescue: a quote the source text entails is supported even if the surface
        # forms differ (paraphrase). Failures degrade silently to the deterministic verdict.
        if self.nli_scorer is not None:
            try:
                probs = self.nli_scorer(source_blob, [quote])
                if probs and float(probs[0]) >= self.min_nli_entailment:
                    return True
            except Exception:  # noqa: BLE001 — never let the optional scorer fail the check
                pass
        return False

    def detect(self, text: str) -> DetectorResult:
        if not self.enabled or not text:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        cited_urls = [u.rstrip(".,);") for u in _URL_RE.findall(text)]
        cited_ids = self._cited_ids(text)
        quotes = _QUOTE_RE.findall(text)
        known_urls, known_ids, source_blob, num_sources = self._grounding()
        has_grounding = bool(known_urls or known_ids or source_blob or num_sources)

        rule_hits: List[RuleHit] = []

        # No grounding supplied: can't verify. Only flag if sources are required and the
        # output is making cited/quoted claims.
        if not has_grounding:
            if self.require_sources and (cited_urls or cited_ids or quotes):
                rule_hits.append(RuleHit(
                    rule_id="citation_integrity.missing_sources",
                    severity=Severity.MEDIUM,
                    message="Output cites sources but no grounding sources were provided to verify them",
                ))
            return self._result(rule_hits)

        # 1a. Unsupported URL citations.
        for url in dict.fromkeys(cited_urls):
            if url not in known_urls and _normalize(url) not in source_blob:
                rule_hits.append(RuleHit(
                    rule_id="citation_integrity.unsupported_citation",
                    severity=Severity.HIGH,
                    message=f"Cited source not present in provided grounding: {url}",
                ))

        # 1b. Unsupported source-id citations.
        for cid in cited_ids:
            norm = _normalize(cid)
            if cid.isdigit():
                supported = 1 <= int(cid) <= num_sources
            else:
                supported = (norm in known_ids or cid in known_urls or
                             (len(norm) >= 3 and norm in source_blob))
            if not supported:
                rule_hits.append(RuleHit(
                    rule_id="citation_integrity.unsupported_citation",
                    severity=Severity.HIGH,
                    message=f"Cited source id not present in provided grounding: {cid}",
                ))

        # 2. Unverifiable quotes — a quoted span not found in the source text.
        for quote in quotes:
            if not self._quote_supported(quote, source_blob):
                snippet = quote[:60] + ("…" if len(quote) > 60 else "")
                rule_hits.append(RuleHit(
                    rule_id="citation_integrity.unverified_quote",
                    severity=Severity.MEDIUM,
                    message=f"Quoted text not found in provided sources: \"{snippet}\"",
                ))

        return self._result(rule_hits)

    def _result(self, rule_hits: List[RuleHit]) -> DetectorResult:
        if not rule_hits:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)
        risk_score = calculate_risk_score(rule_hits)
        decision = _ACTION_TO_DECISION.get(self.action, Decision.WARN)
        dev_msg = "; ".join(sorted({h.message for h in rule_hits}))[:500]
        if decision == Decision.BLOCK and risk_score < self.block_threshold:
            decision = Decision.WARN
        if decision == Decision.BLOCK:
            return DetectorResult(
                decision=Decision.BLOCK, risk_score=risk_score, rule_hits=rule_hits,
                developer_message=dev_msg,
                user_message="This response cites sources that could not be verified.",
            )
        return DetectorResult(decision=Decision.WARN, risk_score=risk_score,
                              rule_hits=rule_hits, developer_message=dev_msg)
