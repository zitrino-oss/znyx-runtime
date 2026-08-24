"""Tenant-scope assertion (OWASP LLM09 - Vector and Embedding Weaknesses).

2026's LLM09 opens with cross-tenant leakage through shared similarity search: the
search runs across the WHOLE index and the tenant filter is applied afterwards, in
application code. The entry is blunt about why that fails — "the attack succeeds even
when every document is correctly tagged and every API call authenticated, because the
access-control decision happens after the embedding-space search has run".

This is the backstop for that. It cannot move the filter into the index query, which is
the actual fix (LLM09 mitigation #1), but it can refuse to let an unscoped or
cross-tenant result reach the model, and say so loudly enough that the real fix happens.

Three findings, in descending severity:

* **Cross-tenant chunk** — a retrieved chunk is tagged for a different tenant than the
  request. This is a live leak, not a warning sign, so it is the one finding that
  defaults to blocking regardless of the configured action.
* **Unattributed chunk** — a chunk carries no tenant tag at all. Nothing can prove it
  belongs to this caller, and "probably ours" is not an access-control decision.
* **Post-retrieval filtering** — the caller reports that scoping was applied after the
  search. The result may be correct today and still leak through result counts, score
  distributions, and timing, which is exactly the probing attack LLM09 describes.

Chunk attribution is read from request metadata, since the detector sees chunk TEXT and
the tenant tag lives beside it in the vector store:

    metadata = {
        "retrieval": {
            "scope_enforced_in_query": True,
            "chunks": [{"id": "doc-1", "tenant_id": "acme"}, ...],
        }
    }
"""
from typing import Any, Dict, List, Optional

from znyx_core.core.models import Decision, DetectorResult, RuleHit, Severity
from znyx_core.core.risk import calculate_risk_score

_RETRIEVAL_BLOCKS = ("retrieval", "rag", "retrieval_context")
_CHUNK_LISTS = ("chunks", "documents", "results", "hits", "sources")
_TENANT_KEYS = ("tenant_id", "tenant", "org_id", "owner", "customer_id", "namespace")
_SCOPE_FLAGS = ("scope_enforced_in_query", "scoped_query", "prefilter",
                "filter_in_query", "tenant_filter_in_query")


def _retrieval_block(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    for k in _RETRIEVAL_BLOCKS:
        v = metadata.get(k)
        if isinstance(v, dict):
            return v
    return metadata


def _chunks(block: Dict[str, Any]) -> List[Dict[str, Any]]:
    for k in _CHUNK_LISTS:
        v = block.get(k)
        if isinstance(v, list):
            return [c for c in v if isinstance(c, dict)]
    return []


def _tenant_of(chunk: Dict[str, Any]) -> Optional[str]:
    for k in _TENANT_KEYS:
        v = chunk.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        # Vector stores commonly nest the tag under a metadata dict.
        meta = chunk.get("metadata")
        if isinstance(meta, dict):
            mv = meta.get(k)
            if isinstance(mv, str) and mv.strip():
                return mv.strip()
    return None


class TenantScopeAssertionDetector:
    """Asserts retrieved chunks belong to the requesting tenant (LLM09)."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.enabled = self.config.get("enabled", False)
        # WARN by default so turning it on does not immediately break a deployment that
        # has not yet plumbed chunk attribution through. A confirmed cross-tenant chunk
        # blocks anyway — see below.
        self.action = (self.config.get("action") or "WARN").upper()
        # A cross-tenant chunk is a leak in progress. Allowing an org to downgrade that to
        # a warning would make the control decorative, so it is separately controlled and
        # defaults to blocking even when action is WARN.
        self.block_cross_tenant = bool(self.config.get("block_cross_tenant", True))
        self.require_tenant_tags = bool(self.config.get("require_tenant_tags", True))
        self.require_query_scoping = bool(self.config.get("require_query_scoping", True))

    def detect(self, text: str, tenant_id: str = "",
               metadata: Optional[Dict[str, Any]] = None) -> DetectorResult:
        if not self.enabled:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        block = _retrieval_block(metadata)
        chunks = _chunks(block)
        rule_hits: List[RuleHit] = []
        cross_tenant = False

        expected = (tenant_id or "").strip()
        untagged = 0
        for chunk in chunks:
            owner = _tenant_of(chunk)
            if owner is None:
                untagged += 1
                continue
            if expected and owner != expected:
                cross_tenant = True
                rule_hits.append(RuleHit(
                    rule_id="tenant_scope_assertion.cross_tenant_chunk",
                    severity=Severity.HIGH,
                    message=(f"Retrieved chunk belongs to tenant '{owner}', not the "
                             f"requesting tenant '{expected}'"),
                ))

        if self.require_tenant_tags and untagged:
            rule_hits.append(RuleHit(
                rule_id="tenant_scope_assertion.unattributed_chunk",
                severity=Severity.HIGH,
                message=(f"{untagged} retrieved chunk(s) carry no tenant attribution; "
                         f"ownership cannot be verified"),
            ))

        # Only meaningful when the caller told us something about scoping. Absent the
        # flag we do not guess — an unreported pipeline is not evidence of a bad one.
        if self.require_query_scoping and chunks:
            reported = [block.get(f) for f in _SCOPE_FLAGS if f in block]
            if reported and not any(bool(v) for v in reported):
                rule_hits.append(RuleHit(
                    rule_id="tenant_scope_assertion.post_retrieval_filtering",
                    severity=Severity.MEDIUM,
                    message=("Tenant scoping was applied after the similarity search; "
                             "result counts and timing can still leak other tenants' data"),
                ))

        if not rule_hits:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        risk_score = calculate_risk_score(rule_hits)
        dev = f"tenant_scope_assertion: {', '.join(sorted({h.rule_id for h in rule_hits}))}"
        blocking = self.action == "BLOCK" or (cross_tenant and self.block_cross_tenant)
        if blocking:
            return DetectorResult(
                decision=Decision.BLOCK, risk_score=risk_score, rule_hits=rule_hits,
                developer_message=dev,
                user_message="Retrieved context could not be confirmed as belonging to your account.",
            )
        return DetectorResult(decision=Decision.WARN, risk_score=risk_score,
                              rule_hits=rule_hits, developer_message=dev)
