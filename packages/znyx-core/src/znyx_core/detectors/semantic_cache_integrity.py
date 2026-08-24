"""Semantic-cache integrity (OWASP LLM09 - Vector and Embedding Weaknesses).

A semantic cache answers a new question with a stored answer when the two questions
embed close enough together. That is the same similarity search LLM09 is about, with two
differences that make it worse: the result is served INSTEAD of asking the model, so no
downstream control ever sees the substitution, and the "index" is other users' answers.

Three failures, and they are not the same failure:

* **Cross-tenant hit.** The entry was written by another tenant. Shared caches are the
  default in every library that ships one, because partitioning costs hit rate. This is
  a live leak, so like ``tenant_scope_assertion`` it blocks regardless of the configured
  action.
* **Collision.** Two questions embed close together and mean different things
  ("cancel my order" / "can I cancel my order?" is fine; "delete the staging database" /
  "delete the production database" is not). A loose threshold turns the cache into a
  wrong-answer generator, and the wrongness is invisible because the answer is fluent.
* **Poisoning.** An attacker who can write to the cache seeds an entry whose embedding
  sits near a valuable query. Every later asker gets the planted answer. The write is
  ordinary; the damage is durable, which is why this checks the entry's age and
  attribution rather than only its similarity.

Cache internals live in the caller's cache, so the evidence comes from metadata:

    metadata = {
        "semantic_cache": {
            "hit": True,
            "similarity": 0.86,
            "cached_query": "the question the stored answer was written for",
            "tenant_id": "acme",
            "partitioned": True,
            "entry_age_seconds": 900,
        }
    }
"""
import re
from typing import Any, Dict, List, Optional

from znyx_core.core.models import Decision, DetectorResult, RuleHit, Severity
from znyx_core.core.risk import calculate_risk_score

_CACHE_BLOCKS = ("semantic_cache", "cache", "llm_cache", "prompt_cache")
_HIT_KEYS = ("hit", "cache_hit", "was_hit", "served_from_cache")
_SIMILARITY_KEYS = ("similarity", "score", "cosine", "match_score")
_DISTANCE_KEYS = ("distance",)
_QUERY_KEYS = ("cached_query", "cached_prompt", "source_query", "key_text", "matched_query")
_TENANT_KEYS = ("tenant_id", "tenant", "org_id", "owner", "customer_id", "namespace")
_AGE_KEYS = ("entry_age_seconds", "age_seconds", "ttl_elapsed_seconds")
_PARTITION_KEYS = ("partitioned", "tenant_partitioned", "scoped", "namespaced")

_WORD_RE = re.compile(r"[a-z0-9']+")
# Words carrying no discriminating power, so their overlap must not reassure us that two
# questions mean the same thing.
_STOPWORDS = frozenset((
    "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "in", "on", "for",
    "and", "or", "my", "me", "i", "you", "it", "this", "that", "can", "do", "does",
    "how", "what", "when", "where", "why", "please", "with", "from", "at", "by",
))


def _cache_block(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    for key in _CACHE_BLOCKS:
        block = metadata.get(key)
        if isinstance(block, dict):
            return block
    return {}


def _first(block: Dict[str, Any], keys):
    for k in keys:
        v = block.get(k)
        if v is not None:
            return v
    return None


def _content_tokens(text: str) -> set:
    return {w for w in _WORD_RE.findall((text or "").lower()) if w not in _STOPWORDS}


def _overlap(a: str, b: str) -> Optional[float]:
    """Jaccard overlap of content words. None when either side has nothing to compare,
    because 0.0 would then read as total divergence rather than as no evidence."""
    ta, tb = _content_tokens(a), _content_tokens(b)
    if not ta or not tb:
        return None
    return len(ta & tb) / len(ta | tb)


class SemanticCacheIntegrityDetector:
    """Guards answers served from a semantic cache (LLM09)."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.enabled = self.config.get("enabled", False)
        # WARN by default: a cache that has run unchecked will have accumulated loose
        # hits, and blocking all of them on day one takes the application down rather
        # than the attack. A cross-tenant hit still blocks, below.
        self.action = (self.config.get("action") or "WARN").upper()
        # A cross-tenant answer is a leak in progress, not a warning sign.
        self.block_cross_tenant = bool(self.config.get("block_cross_tenant", True))
        # Below this similarity a hit is a guess. 0.9 is deliberately stricter than the
        # 0.8 most cache libraries default to: their default optimises hit rate.
        self.min_similarity = float(self.config.get("min_similarity", 0.90))
        # Embedding proximity with no shared content words is the collision signature.
        self.min_token_overlap = float(self.config.get("min_token_overlap", 0.30))
        self.require_partitioning = bool(self.config.get("require_partitioning", True))
        # 0 disables. A cache entry has no natural expiry, which is what makes a planted
        # one durable.
        self.max_entry_age_seconds = max(0, int(self.config.get("max_entry_age_seconds", 0)))

    def detect(self, text: str, tenant_id: str = "",
               metadata: Optional[Dict[str, Any]] = None) -> DetectorResult:
        if not self.enabled:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        block = _cache_block(metadata)
        if not block:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)
        hit = _first(block, _HIT_KEYS)
        # Nothing was served from the cache, so there is nothing to vouch for. An absent
        # flag alongside a similarity still counts: some caches populate the block only
        # on a hit in the first place.
        if hit is False or (hit is None and _first(block, _SIMILARITY_KEYS) is None):
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        rule_hits: List[RuleHit] = []
        cross_tenant = False

        entry_tenant = _first(block, _TENANT_KEYS)
        expected = (tenant_id or "").strip()
        if isinstance(entry_tenant, str) and entry_tenant.strip() and expected:
            if entry_tenant.strip() != expected:
                cross_tenant = True
                rule_hits.append(RuleHit(
                    rule_id="semantic_cache_integrity.cross_tenant_cache_hit",
                    severity=Severity.HIGH,
                    message=(f"Cached answer belongs to tenant '{entry_tenant}', not the "
                             f"requesting tenant '{expected}'"),
                ))
        elif self.require_partitioning:
            partitioned = _first(block, _PARTITION_KEYS)
            if partitioned is False or (partitioned is None and entry_tenant is None):
                rule_hits.append(RuleHit(
                    rule_id="semantic_cache_integrity.unpartitioned_cache",
                    severity=Severity.HIGH,
                    message=("Cache hit carries no tenant attribution and the cache does "
                             "not report partitioning; any tenant can be served any entry"),
                ))

        similarity = _first(block, _SIMILARITY_KEYS)
        if similarity is None:
            distance = _first(block, _DISTANCE_KEYS)
            if isinstance(distance, (int, float)) and not isinstance(distance, bool):
                # Cosine distance and cosine similarity are complements, which is the
                # conversion every cache library documents.
                similarity = 1.0 - float(distance)
        if isinstance(similarity, (int, float)) and not isinstance(similarity, bool):
            if float(similarity) < self.min_similarity:
                rule_hits.append(RuleHit(
                    rule_id="semantic_cache_integrity.low_similarity_hit",
                    severity=Severity.MEDIUM,
                    message=(f"Answer served on a {float(similarity):.2f} similarity, below "
                             f"the {self.min_similarity:.2f} floor; a loose threshold serves "
                             f"a different question's answer"),
                ))

        cached_query = _first(block, _QUERY_KEYS)
        if isinstance(cached_query, str) and cached_query.strip():
            overlap = _overlap(text, cached_query)
            if overlap is not None and overlap < self.min_token_overlap:
                rule_hits.append(RuleHit(
                    rule_id="semantic_cache_integrity.query_divergence",
                    severity=Severity.HIGH,
                    message=(f"Cached question shares {overlap:.0%} of its content words "
                             f"with this one; close in embedding space and different in "
                             f"meaning is what a cache collision looks like"),
                ))

        if self.max_entry_age_seconds:
            age = _first(block, _AGE_KEYS)
            if isinstance(age, (int, float)) and not isinstance(age, bool):
                if int(age) > self.max_entry_age_seconds:
                    rule_hits.append(RuleHit(
                        rule_id="semantic_cache_integrity.stale_cache_entry",
                        severity=Severity.MEDIUM,
                        message=(f"Cache entry is {int(age)}s old, past the "
                                 f"{self.max_entry_age_seconds}s limit; a planted entry "
                                 f"keeps paying out until something expires it"),
                    ))

        if not rule_hits:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        risk_score = calculate_risk_score(rule_hits)
        dev = f"semantic_cache_integrity: {', '.join(sorted({h.rule_id for h in rule_hits}))}"
        if self.action == "BLOCK" or (cross_tenant and self.block_cross_tenant):
            return DetectorResult(
                decision=Decision.BLOCK, risk_score=risk_score, rule_hits=rule_hits,
                developer_message=dev,
                user_message="A cached answer could not be confirmed as matching your request.",
            )
        return DetectorResult(decision=Decision.WARN, risk_score=risk_score,
                              rule_hits=rule_hits, developer_message=dev)
