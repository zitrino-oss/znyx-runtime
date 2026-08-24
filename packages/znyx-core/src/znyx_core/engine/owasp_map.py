"""OWASP-LLM-2026 coverage map.

A static, honest mapping of ZNYX detectors to the OWASP LLM Top-10 (2026): which
detectors are a *dedicated* control for a category vs *adjacent* (partial) coverage,
the pipeline stage(s) and languages each covers, and — critically — where there is **no
dedicated control today** so the coverage endpoint can surface it as a gap rather than
claim blanket LLM01–LLM10 coverage.

``compute_coverage(enabled)`` joins an org's enabled detectors against this map and
returns a per-category status (full / partial / uncovered) + recommended controls.
Recommendations may name not-yet-implemented detectors — flagged via
``DETECTOR_META[...].implemented`` — so gaps point at the planned control.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class DetectorMeta:
    stages: List[str]
    languages: List[str]          # ["*"] = language-agnostic (pattern/structural)
    implemented: bool = True       # False = planned, shown only as a recommendation


# Detector → where it runs + languages + whether it exists today. Planned controls
# (implemented=False) are referenced only as gap recommendations. Nothing is planned
# today: every control the 2026 edition requires is built.
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
    # new-stage / lifecycle gap detectors — now implemented (registered + wired).
    "retrieval_chunk_injection": DetectorMeta(["retrieval"], ["en"]),
    "tool_output_injection":     DetectorMeta(["tool"], ["en"]),
    "memory_write_poisoning":    DetectorMeta(["memory_write"], ["en"]),
    "excessive_agency":          DetectorMeta(["agent_plan", "agent_loop"], ["*"]),
    "unbounded_consumption":     DetectorMeta(["agent_loop", "input", "output"], ["*"]),
    "mcp_manifest_scanner":      DetectorMeta(["tool_registration"], ["en"]),
    # LLM09 vector/embedding-store integrity (retrieval stage) — implemented.
    "embedding_integrity":       DetectorMeta(["retrieval"], ["*"]),

    # ── 2026 controls (built) ─────────────────────────────────────────────────
    # The controls the 2026 edition added or widened. Each was registered here as
    # planned (implemented=False) first, so its category read "partial" with a named
    # next step rather than an empty gap list, and flipped to implemented as it
    # landed. All eight are built and registered now, which is what takes the
    # requirement ceiling to 100; the mechanism stays in place for the next edition.
    "multimodal_injection":  DetectorMeta(["input"], ["*"]),
    "reasoning_trace_disclosure":  DetectorMeta(["output"], ["*"]),
    "tool_permission_audit":     DetectorMeta(["tool_registration"], ["*"]),
    "human_approval_gate":       DetectorMeta(["agent_plan", "agent_loop"], ["*"]),
    "corpus_poisoning_monitor": DetectorMeta(["memory_write", "retrieval"], ["*"]),
    "tenant_scope_assertion":  DetectorMeta(["retrieval"], ["*"]),
    "retrieval_jamming":  DetectorMeta(["retrieval"], ["*"]),
    "semantic_cache_integrity":  DetectorMeta(["input", "output", "retrieval"], ["*"]),
    "output_control_char_sanitizer":  DetectorMeta(["output"], ["*"]),
}


@dataclass(frozen=True)
class LifecycleMeta:
    """A control that is NOT a request-time detector.

    Coverage could previously only count "a detector enabled in a policy", so real
    controls that run at build, publish, or load time scored zero and the map
    understated what ZNYX does — the mirror image of the overclaim it exists to stop.

    ``evidence`` states how a caller proves the control is actually in force for an
    org. A lifecycle control counts toward a category's requirements like any other,
    but it is only credited when that evidence is present: existing in the codebase
    is not the same as being active for this customer.
    """
    name: str
    evidence: str


LIFECYCLE_META: Dict[str, LifecycleMeta] = {
    "scorecard_gate": LifecycleMeta(
        "Publish blocked for a model-backed detector without a passing scorecard",
        evidence="Always in force: BundleService._enforce_scorecard_gates raises "
                 "PolicyValidationError on publish, so no org can opt out."),
    "model_provenance": LifecycleMeta(
        "Model artifacts signed and accompanied by an AIBOM at the promotion boundary",
        evidence="A signature verified against a transparency log, or an AIBOM recorded "
                 "for the artifact. Signing proves origin, not safety, so this is credit "
                 "for provenance only — it pairs with the scorecard gate, which is what "
                 "establishes the model behaves. Credited from a provenance record whose "
                 "signature verified against the org's key, or which carries an AIBOM, "
                 "recorded at the promotion boundary."),
    "inference_artifact_integrity": LifecycleMeta(
        "Chat templates, tokenizer configs, and adapters diffed against a recorded baseline",
        evidence="A provenance record carries an artifact baseline AND the environment's "
                 "latest report matches it on every behavioural file. 2026 names "
                 "inference-time artifacts explicitly: a chat template edited to append "
                 "an instruction to every turn is a persistent prompt injection that no "
                 "request-time control can see, because the template has already been "
                 "applied by the time a detector reads the request."),
    "model_artifact_digest_pinning": LifecycleMeta(
        "Pinned model artifacts verified by sha256 before they serve",
        evidence="A reported model pin carries a sha256. znyx_inference verify_pinned() "
                 "fails closed on mismatch and refuses implicit downloads — but it only "
                 "verifies when a digest is actually pinned, so credit requires evidence "
                 "that this org pins one."),
}


@dataclass(frozen=True)
class CategoryMap:
    id: str
    name: str
    dedicated: List[str] = field(default_factory=list)   # primary detector controls
    adjacent: List[str] = field(default_factory=list)    # partial/indirect coverage
    recommended: List[str] = field(default_factory=list)  # add these to close gaps (may be planned)
    lifecycle: List[str] = field(default_factory=list)   # build/publish/load-time controls
    note: Optional[str] = None                            # honesty note for structural gaps


# The 10 categories, mapped honestly, in the OWASP LLM Top-10 **2026** order.
# 2026 renumbered most of the list and re-scoped one entry: 2025's LLM07 System
# Prompt Leakage became the broader LLM08 Hidden Context Exposure, which covers any
# non-user-facing context (system prompt, tool/function schemas, retrieved policy
# text, behavioural rules), not just the system prompt. Where no dedicated control
# exists today ``dedicated`` is empty and ``note`` explains the gap.
OWASP_LLM_2026: List[CategoryMap] = [
    CategoryMap("LLM01", "Prompt Injection",
                dedicated=["jailbreak", "exfiltration", "retrieval_chunk_injection",
                           "tool_output_injection", "memory_write_poisoning",
                           "multimodal_injection"],
                note="Attempts to override the model's instructions, directly via "
                     "jailbreak and exfiltration prompts, or indirectly through poisoned "
                     "retrieved chunks, tool results, and memory writes. Enable all six "
                     "controls for full direct and indirect injection coverage. 2026 also "
                     "folds cross-modal injection (instructions hidden in an image or "
                     "audio track) into this category; the cross-modal detector governs "
                     "the OCR and transcript text the caller extracts, and reports any "
                     "attachment that reached the model with nothing extracted."),
    CategoryMap("LLM02", "Sensitive Information Disclosure",
                dedicated=["pii", "secrets", "sensitive_business_data", "document_metadata_leakage",
                           "reasoning_trace_disclosure"],
                adjacent=["compliance"],
                note="Leakage of personal data, credentials, confidential business "
                     "information, or document metadata in prompts or responses. PII, "
                     "secrets, sensitive business data, and metadata detectors redact or "
                     "block it before it leaves the boundary. 2026 widened the channel "
                     "list past the final answer, so the reasoning-trace detector treats "
                     "extended-thinking traces and tool-call arguments as outputs too."),
    CategoryMap("LLM03", "Excessive Agency",
                dedicated=["excessive_agency", "tool_permission_audit", "human_approval_gate"],
                adjacent=["tools"],
                note="Agents taking over-broad, unauthorised, or destructive actions "
                     "through their tools or plans. All three of LLM03's root causes are "
                     "covered: excessive functionality and permissions by the "
                     "registration-time tool-permission audit, excessive autonomy by the "
                     "human-approval gate, and the runtime action itself by the "
                     "excessive-agency detector. 2026 moved this from LLM06 to third, the "
                     "list's most consequential promotion."),
    CategoryMap("LLM04", "Supply Chain",
                dedicated=["mcp_manifest_scanner"],
                lifecycle=["model_artifact_digest_pinning", "scorecard_gate", "model_provenance"],
                note="Risks from third-party tools, plugins, and models pulled into the "
                     "app. The MCP manifest scanner inspects tool manifests at "
                     "registration time (a registration lifecycle hook, not per request), "
                     "so manifests must be routed through it when tools are registered. "
                     "Model provenance is enforced at the promotion boundary: an Ed25519 "
                     "signature bound to the exact model, revision, and digest, plus an "
                     "AIBOM, both checked when a model is pinned into a scope. Signing "
                     "proves origin and not safety, so it pairs with the scorecard gate "
                     "rather than replacing it. Dependency provenance for the "
                     "application's own supply chain stays outside ZNYX's reach."),
    CategoryMap("LLM05", "Data and Model Poisoning",
                dedicated=["corpus_poisoning_monitor"],
                lifecycle=["model_artifact_digest_pinning",
                           "inference_artifact_integrity"],
                adjacent=["retrieval_chunk_injection"],
                note="Tampering with training data, fine-tuning data, or retrieved "
                     "context to bias or corrupt model behaviour; 2026 also absorbs "
                     "fine-tuning subversion and scopes the entry to durable corruption "
                     "rather than inference-time instructions, which belong to LLM01. The "
                     "corpus-poisoning monitor screens documents at write time, where the "
                     "corruption can still be prevented; retrieval-chunk injection gives "
                     "adjacent coverage on the read path. Artifact-integrity diffing "
                     "covers the inference-time artifacts 2026 calls out by name, the "
                     "chat template, tokenizer config, and LoRA adapters that change "
                     "behaviour without changing a weight. Training-data provenance "
                     "remains outside the runtime's reach."),
    CategoryMap("LLM06", "Unbounded Consumption",
                dedicated=["unbounded_consumption"],
                adjacent=["abuse"],
                note="Runaway token, cost, or loop consumption, whether from abuse or "
                     "agent loops. The unbounded-consumption detector enforces token, "
                     "cost, and loop budgets, and abuse rate-limiting adds adjacent "
                     "coverage. The detector also caps extended-thinking tokens and "
                     "detects agent loops that spin on one state, the two surfaces 2026 "
                     "added. 2026 raised this four places, from LLM10."),
    CategoryMap("LLM07", "Misinformation",
                dedicated=["hallucination", "citation_integrity", "numerical_consistency"],
                note="Confident but false, unsupported, or fabricated model output. "
                     "Hallucination, citation-integrity, and numerical-consistency "
                     "detectors flag claims that are not grounded in the provided "
                     "context."),
    CategoryMap("LLM08", "Hidden Context Exposure",
                dedicated=["system_prompt_leakage"],
                adjacent=["exfiltration"],
                note="Exposure of hidden, non-user-facing context: the system prompt, "
                     "developer instructions, tool and function schemas, retrieved policy "
                     "text, and behavioural rules. The system-prompt-leakage detector "
                     "now covers all three: registered system prompts, registered "
                     "policy text, and the tool/function schemas a request carries, which "
                     "are fingerprinted per call because they are generated rather than "
                     "registered. 2026 widened this category beyond the system prompt (it "
                     "was LLM07 System Prompt Leakage in 2025)."),
    CategoryMap("LLM09", "Vector and Embedding Weaknesses",
                dedicated=["embedding_integrity", "tenant_scope_assertion",
                           "retrieval_jamming", "semantic_cache_integrity"],
                adjacent=["retrieval_chunk_injection"],
                note="Attacks on the embedding and vector-store layer of RAG systems, "
                     "such as poisoned embeddings, similarity-ranking manipulation, "
                     "injected vector payloads, cross-tenant leakage through a shared "
                     "index, and blocker documents planted to make the model refuse. "
                     "Embedding integrity flags manipulated chunks, tenant-scope "
                     "assertion refuses chunks that cannot be shown to belong to the "
                     "caller, retrieval jamming catches the availability attack, and "
                     "semantic-cache integrity guards the cache that answers without "
                     "asking the model at all; retrieval-chunk injection adds adjacent "
                     "coverage."),
    CategoryMap("LLM10", "Improper Output Handling",
                dedicated=["structure", "code_safety", "malicious_url",
                           "output_control_char_sanitizer"],
                note="Unsafe model output passed downstream without validation, such as "
                     "executable code, malformed structure, malicious links, or terminal "
                     "control sequences. Structure enforcement, code-safety scanning, and "
                     "malicious-URL detection validate output before it is used, and the "
                     "control-character sanitiser neutralises ANSI escapes before output "
                     "reaches a terminal, log viewer, or IDE pane. 2026 widened this to "
                     "insecure assistant-generated code at scale and to interpreting "
                     "sinks, and dropped it from LLM05 to last."),
]

_CATEGORY_BY_ID = {c.id: c for c in OWASP_LLM_2026}


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


def compute_coverage(
    enabled: Iterable[str],
    lifecycle_active: Iterable[str] = (),
) -> List[dict]:
    """Per-category coverage given the set of enabled detector keys.

    Each category exposes a ``coverage_ratio`` (0..1) and ``score`` (0..100):
    the fraction of the category's *required* controls that are enabled. Required
    means every dedicated control the category needs, whether or not ZNYX has built
    it — so a category cannot read "full" while a control the 2026 edition asks for
    is still on the roadmap. Status:

    * ``full``       — every required control exists and is enabled
    * ``partial``    — some required controls are enabled, or only adjacent ones
    * ``uncovered``  — built controls exist for this category but none is enabled
    * ``no_control`` — nothing is built for this category yet

    Two different gaps are reported separately, because they have different owners.
    ``dedicated_enabled``/``dedicated_total`` count only BUILT controls: that is the
    part an org can close by changing policy. ``unbuilt`` names the required controls
    the platform has not shipped, and ``blocked_by_platform`` marks a category where
    the org has already enabled everything available and the remaining gap is ours.

    Only IMPLEMENTED detectors ever count as live coverage."""
    enabled_set = {k for k in (enabled or [])
                   if DETECTOR_META.get(k) and DETECTOR_META[k].implemented}
    # Only recognised lifecycle keys count, and only when the caller supplies evidence.
    lifecycle_set = {k for k in (lifecycle_active or []) if k in LIFECYCLE_META}
    out: List[dict] = []
    for cat in OWASP_LLM_2026:
        impl_dedicated = _impl(cat.dedicated)            # BUILT dedicated controls
        unbuilt = [d for d in cat.dedicated if d not in impl_dedicated]
        # Lifecycle controls exist in the platform, so they sit alongside built detectors
        # in the denominator and count as coverage only when evidenced as active.
        lifecycle_on = [l for l in cat.lifecycle if l in lifecycle_set]
        dedicated_total = len(impl_dedicated) + len(cat.lifecycle)
        required_total = len(cat.dedicated) + len(cat.lifecycle)
        enabled_dedicated = [d for d in impl_dedicated if d in enabled_set]
        enabled_adjacent = [a for a in cat.adjacent if a in enabled_set]
        d_enabled = len(enabled_dedicated) + len(lifecycle_on)
        # Everything the org could turn on is on, and the shortfall is ours to build.
        blocked_by_platform = bool(unbuilt) and d_enabled == dedicated_total

        if required_total > 0:
            if d_enabled == required_total:              # only reachable with no unbuilt
                status, ratio = "full", 1.0
            elif d_enabled > 0:
                status, ratio = "partial", d_enabled / required_total
            elif enabled_adjacent:
                status, ratio = "partial", ADJACENT_CREDIT
            elif dedicated_total > 0:
                status, ratio = "uncovered", 0.0
            else:                                        # nothing built to enable
                status, ratio = "no_control", 0.0
        else:
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
            "required_total": required_total,
            "unbuilt": unbuilt,
            "lifecycle_controls": list(cat.lifecycle),
            "lifecycle_active": lifecycle_on,
            "blocked_by_platform": blocked_by_platform,
            "coverage_ratio": round(ratio, 3),
            "score": round(ratio * 100),
            "enabled_detectors": enabled_dedicated + enabled_adjacent + lifecycle_on,
            "recommended": recommended,
            "stages": sorted({s for d in (cat.dedicated + cat.adjacent)
                              for s in (DETECTOR_META.get(d).stages if DETECTOR_META.get(d) else [])}),
            "note": cat.note,
        })
    return out


def compute_score(categories: List[dict]) -> dict:
    """Overall OWASP-LLM coverage from per-category coverage.

    TWO numbers, because there are two different questions and one number cannot
    answer both honestly:

    * ``score`` — requirement coverage. The share of what the 2026 edition ASKS FOR
      that is actually running, averaged equally over the Top-10. Controls ZNYX has
      not built count against it, so this is the number that answers "am I covered
      against the Top 10?" and the number that must not overclaim. Its ceiling sits
      below 100 while any required control is unbuilt.

    * ``deployment_score`` — the share of BUILT controls the org has enabled. The
      actionable number: it reaches 100 when an operator has turned on everything
      available to them, and it never penalises a customer for our roadmap.

    Reporting only the first would grade every customer down for gaps they cannot
    close; reporting only the second is the overclaim this map exists to avoid."""
    n = len(categories) or 1
    score = round(100 * sum(c["coverage_ratio"] for c in categories) / n)
    # Deployment: per category, the fraction of built controls enabled. Categories
    # with nothing built are excluded — an org cannot deploy what does not exist.
    deployable = [c for c in categories if c["dedicated_total"] > 0]
    deployment_score = round(
        100 * sum(c["dedicated_enabled"] / c["dedicated_total"] for c in deployable)
        / (len(deployable) or 1)
    )
    band, band_label = score_band(score)
    dep_band, dep_band_label = score_band(deployment_score)
    return {
        "score": score,
        "band": band,
        "band_label": band_label,
        "deployment_score": deployment_score,
        "deployment_band": dep_band,
        "deployment_band_label": dep_band_label,
        "categories_total": len(categories),
        "categories_full": sum(1 for c in categories if c["status"] == "full"),
        # Gaps the org can close today by enabling a control that already exists.
        "fixable_gap_ids": [c["id"] for c in categories
                            if c["dedicated_total"] > c["dedicated_enabled"]],
        # Gaps with no native control in the platform yet (not org-fixable).
        "no_control_ids": [c["id"] for c in categories if c["status"] == "no_control"],
        # Categories where the org has enabled everything available and the remaining
        # shortfall is a control ZNYX has not built. Reported apart from fixable gaps so
        # a customer is never asked to fix something only we can.
        "platform_gap_ids": [c["id"] for c in categories if c.get("blocked_by_platform")],
    }


def detector_owasp_categories(detector_key: str) -> List[str]:
    """The OWASP category ids a given detector contributes to (dedicated or adjacent)."""
    return [c.id for c in OWASP_LLM_2026
            if detector_key in c.dedicated or detector_key in c.adjacent]
