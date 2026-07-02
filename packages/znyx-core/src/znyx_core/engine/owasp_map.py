"""OWASP-LLM-2025 coverage map (P0).

A static, honest mapping of ZNYX detectors to the OWASP LLM Top-10 (2025): which
detectors are a *dedicated* control for a category vs *adjacent* (partial) coverage,
the pipeline stage(s) and languages each covers, and — critically — where there is **no
dedicated control today** so the coverage endpoint can surface it as a gap rather than
claim blanket LLM01–LLM10 coverage.

``compute_coverage(enabled)`` joins an org's enabled detectors against this map and
returns a per-category status (full / partial / uncovered) + recommended controls.
Recommendations may name not-yet-implemented detectors (P1b/P2) — flagged via
``DETECTOR_META[...].implemented`` — so gaps point at the planned control.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class DetectorMeta:
    stages: List[str]
    languages: List[str]          # ["*"] = language-agnostic (pattern/structural)
    implemented: bool = True       # False = planned (P1b/P2), shown only as a recommendation


# Detector → where it runs + languages + whether it exists today. Planned controls
# (implemented=False) are referenced only as gap recommendations.
DETECTOR_META: Dict[str, DetectorMeta] = {
    # implemented (registered) detectors
    "jailbreak":                DetectorMeta(["input"], ["en"]),
    "exfiltration":             DetectorMeta(["input", "output"], ["en"]),
    "pii":                      DetectorMeta(["input", "output"], ["*"]),
    "secrets":                  DetectorMeta(["input", "output"], ["*"]),
    "sensitive_business_data":  DetectorMeta(["input", "output"], ["en"]),
    "compliance":               DetectorMeta(["input", "output"], ["en"]),
    "malicious_url":            DetectorMeta(["input", "output"], ["*"]),
    "code_safety":              DetectorMeta(["input", "output"], ["*"]),
    "structure":                DetectorMeta(["output"], ["*"]),
    "tools":                    DetectorMeta(["tool"], ["*"]),
    "system_prompt_leakage":    DetectorMeta(["output"], ["*"]),
    "hallucination":            DetectorMeta(["output"], ["en"]),
    "citation_integrity":       DetectorMeta(["output"], ["*"]),
    "numerical_consistency":    DetectorMeta(["output"], ["*"]),
    "document_metadata_leakage": DetectorMeta(["input", "output"], ["*"]),
    "abuse":                    DetectorMeta(["input", "output"], ["*"]),
    # registered detectors that are content/policy controls, not an OWASP-LLM category
    # (kept in the matrix so the coverage page lists every detector; categories=[]).
    "toxicity":                 DetectorMeta(["input", "output"], ["en"]),
    "bias":                     DetectorMeta(["input", "output"], ["en"]),
    "sentiment":                DetectorMeta(["input", "output"], ["en"]),
    "topic_restriction":        DetectorMeta(["input", "output"], ["en"]),
    "competitor":               DetectorMeta(["input", "output"], ["en"]),
    "gibberish":                DetectorMeta(["input"], ["*"]),
    "language":                 DetectorMeta(["input"], ["*"]),
    "copyright":                DetectorMeta(["output"], ["en"]),
    # P1b new-stage / lifecycle gap detectors — now implemented (registered + wired).
    "retrieval_chunk_injection": DetectorMeta(["retrieval"], ["en"]),
    "tool_output_injection":     DetectorMeta(["tool"], ["en"]),
    "memory_write_poisoning":    DetectorMeta(["memory_write"], ["en"]),
    "excessive_agency":          DetectorMeta(["agent_plan", "agent_loop"], ["*"]),
    "unbounded_consumption":     DetectorMeta(["agent_loop", "input", "output"], ["*"]),
    "mcp_manifest_scanner":      DetectorMeta(["tool_registration"], ["en"]),
    # LLM08 vector/embedding-store integrity (retrieval stage) — implemented.
    "embedding_integrity":       DetectorMeta(["retrieval"], ["*"]),
}


@dataclass(frozen=True)
class CategoryMap:
    id: str
    name: str
    dedicated: List[str] = field(default_factory=list)   # primary, IMPLEMENTED controls
    adjacent: List[str] = field(default_factory=list)    # partial/indirect coverage
    recommended: List[str] = field(default_factory=list)  # add these to close gaps (may be planned)
    note: Optional[str] = None                            # honesty note for structural gaps


# The 10 categories, mapped honestly. Where no dedicated control exists today
# (LLM03/LLM04/LLM08) ``dedicated`` is empty and ``note`` explains the gap.
OWASP_LLM_2025: List[CategoryMap] = [
    CategoryMap("LLM01", "Prompt Injection",
                dedicated=["jailbreak", "exfiltration", "retrieval_chunk_injection",
                           "tool_output_injection", "memory_write_poisoning"],
                note="Attempts to override the model's instructions, directly via "
                     "jailbreak and exfiltration prompts, or indirectly through poisoned "
                     "retrieved chunks, tool results, and memory writes. Enable all five "
                     "controls for full direct and indirect injection coverage."),
    CategoryMap("LLM02", "Sensitive Information Disclosure",
                dedicated=["pii", "secrets", "sensitive_business_data", "document_metadata_leakage"],
                adjacent=["compliance"],
                note="Leakage of personal data, credentials, confidential business "
                     "information, or document metadata in prompts or responses. PII, "
                     "secrets, sensitive business data, and metadata detectors redact or "
                     "block it before it leaves the boundary."),
    CategoryMap("LLM03", "Supply Chain",
                dedicated=["mcp_manifest_scanner"],
                note="Risks from third-party tools, plugins, and models pulled into the "
                     "app. The MCP manifest scanner inspects tool manifests at "
                     "registration time (a registration lifecycle hook, not per request), "
                     "so manifests must be routed through it when tools are registered. "
                     "Model and dependency provenance remains a broader gap."),
    CategoryMap("LLM04", "Data and Model Poisoning",
                adjacent=["retrieval_chunk_injection"],
                note="Tampering with training data, fine-tuning data, or retrieved "
                     "context to bias or corrupt model behaviour. There is no dedicated "
                     "control yet; retrieval-chunk injection gives adjacent coverage for "
                     "the RAG path, and a dedicated data poisoning monitor is planned."),
    CategoryMap("LLM05", "Improper Output Handling",
                dedicated=["structure", "code_safety", "malicious_url"],
                note="Unsafe model output passed downstream without validation, such as "
                     "executable code, malformed structure, or malicious links. Structure "
                     "enforcement, code-safety scanning, and malicious-URL detection "
                     "validate output before it is used."),
    CategoryMap("LLM06", "Excessive Agency",
                dedicated=["excessive_agency"],
                adjacent=["tools"],
                note="Agents taking over-broad, unauthorised, or destructive actions "
                     "through their tools or plans. The excessive-agency detector "
                     "risk-scores agent plans and loop step actions, and tool-invocation "
                     "governance adds adjacent coverage."),
    CategoryMap("LLM07", "System Prompt Leakage",
                dedicated=["system_prompt_leakage"],
                adjacent=["exfiltration"],
                note="Exposure of the system prompt or hidden instructions, which can "
                     "reveal secrets or enable further attacks. The system-prompt-leakage "
                     "detector blocks extraction attempts, and exfiltration detection adds "
                     "adjacent coverage."),
    CategoryMap("LLM08", "Vector and Embedding Weaknesses",
                dedicated=["embedding_integrity"],
                adjacent=["retrieval_chunk_injection"],
                note="Attacks on the embedding and vector-store layer of RAG systems, "
                     "such as poisoned embeddings, similarity-ranking manipulation, or "
                     "injected vector payloads. The embedding-integrity detector flags "
                     "these in retrieved chunks; retrieval-chunk injection adds adjacent "
                     "coverage."),
    CategoryMap("LLM09", "Misinformation",
                dedicated=["hallucination", "citation_integrity", "numerical_consistency"],
                note="Confident but false, unsupported, or fabricated model output. "
                     "Hallucination, citation-integrity, and numerical-consistency "
                     "detectors flag claims that are not grounded in the provided "
                     "context."),
    CategoryMap("LLM10", "Unbounded Consumption",
                dedicated=["unbounded_consumption"],
                adjacent=["abuse"],
                note="Runaway token, cost, or loop consumption, whether from abuse or "
                     "agent loops. The unbounded-consumption detector enforces token, "
                     "cost, and loop budgets, and abuse rate-limiting adds adjacent "
                     "coverage."),
]

_CATEGORY_BY_ID = {c.id: c for c in OWASP_LLM_2025}


def _impl(keys: Iterable[str]) -> List[str]:
    """Keep only keys that name a detector that exists today."""
    return [k for k in keys if DETECTOR_META.get(k) and DETECTOR_META[k].implemented]


# Partial credit (0..1) a category earns when only *adjacent* (indirect) controls
# are enabled. Adjacent controls give defence-in-depth but are not a dedicated
# control for the category, so they are capped well below a single dedicated one.
ADJACENT_CREDIT = 0.25

# Overall-score bands, highest threshold first. A score >= threshold gets that band.
# Tuned so "Strong" requires near-complete deployment of available controls and
# "Critical" flags an org that has barely any OWASP-mapped controls on.
SCORE_BANDS = [
    (85, "strong",   "Strong coverage"),
    (60, "moderate", "Moderate coverage"),
    (35, "limited",  "Limited coverage"),
    (0,  "critical", "Critical gaps"),
]


def score_band(score: int) -> tuple[str, str]:
    """(band_key, band_label) for a 0..100 score."""
    for threshold, key, label in SCORE_BANDS:
        if score >= threshold:
            return key, label
    return "critical", "Critical gaps"


def compute_coverage(enabled: Iterable[str]) -> List[dict]:
    """Per-category coverage given the set of enabled detector keys.

    Each category exposes a ``coverage_ratio`` (0..1) and ``score`` (0..100):
    the fraction of the category's *available dedicated controls* that are enabled.
    Status is derived from that fraction so it never overstates protection:

    * ``full``       — every available dedicated control for the category is enabled
    * ``partial``    — some (but not all) dedicated controls, or only adjacent ones
    * ``uncovered``  — a dedicated control exists but none is enabled
    * ``no_control`` — the platform has no dedicated control for this category yet
                       (a platform gap, not a misconfiguration the org can fix)

    Only IMPLEMENTED detectors count as live coverage; planned (not-yet-built)
    controls only ever appear under ``recommended``."""
    enabled_set = {k for k in (enabled or [])
                   if DETECTOR_META.get(k) and DETECTOR_META[k].implemented}
    out: List[dict] = []
    for cat in OWASP_LLM_2025:
        impl_dedicated = _impl(cat.dedicated)            # available dedicated controls
        dedicated_total = len(impl_dedicated)
        enabled_dedicated = [d for d in impl_dedicated if d in enabled_set]
        enabled_adjacent = [a for a in cat.adjacent if a in enabled_set]
        d_enabled = len(enabled_dedicated)

        if dedicated_total > 0:
            if d_enabled == dedicated_total:
                status, ratio = "full", 1.0
            elif d_enabled > 0:
                status, ratio = "partial", d_enabled / dedicated_total
            elif enabled_adjacent:
                status, ratio = "partial", ADJACENT_CREDIT
            else:
                status, ratio = "uncovered", 0.0
        else:  # no dedicated control available in the platform
            if enabled_adjacent:
                status, ratio = "partial", ADJACENT_CREDIT
            else:
                status, ratio = "no_control", 0.0

        # Recommend the dedicated controls not yet enabled, then the planned controls.
        recommended = [d for d in cat.dedicated if d not in enabled_set]
        recommended += [r for r in cat.recommended if r not in enabled_set and r not in recommended]

        out.append({
            "id": cat.id,
            "name": cat.name,
            "status": status,
            "has_dedicated_control": dedicated_total > 0,
            "dedicated_enabled": d_enabled,
            "dedicated_total": dedicated_total,
            "coverage_ratio": round(ratio, 3),
            "score": round(ratio * 100),
            "enabled_detectors": enabled_dedicated + enabled_adjacent,
            "recommended": recommended,
            "stages": sorted({s for d in (cat.dedicated + cat.adjacent)
                              for s in (DETECTOR_META.get(d).stages if DETECTOR_META.get(d) else [])}),
            "note": cat.note,
        })
    return out


def compute_score(categories: List[dict]) -> dict:
    """Overall OWASP-LLM coverage score (0..100) from per-category coverage.

    Equal-weighted mean of each category's ``coverage_ratio`` across all 10
    categories — transparent and defensible: the score is simply the share of
    available OWASP-mapped controls you have deployed, averaged over the Top-10.
    Categories with no native control yet still count (real exposure) but are
    listed separately so the gap is attributable to the platform, not the org."""
    n = len(categories) or 1
    score = round(100 * sum(c["coverage_ratio"] for c in categories) / n)
    band, band_label = score_band(score)
    return {
        "score": score,
        "band": band,
        "band_label": band_label,
        "categories_total": len(categories),
        "categories_full": sum(1 for c in categories if c["status"] == "full"),
        # Gaps the org can close by enabling an available control.
        "fixable_gap_ids": [c["id"] for c in categories
                            if c["dedicated_total"] > c["dedicated_enabled"]],
        # Gaps with no native control in the platform yet (not org-fixable).
        "no_control_ids": [c["id"] for c in categories if c["status"] == "no_control"],
    }


def detector_owasp_categories(detector_key: str) -> List[str]:
    """The OWASP category ids a given detector contributes to (dedicated or adjacent)."""
    return [c.id for c in OWASP_LLM_2025
            if detector_key in c.dedicated or detector_key in c.adjacent]
