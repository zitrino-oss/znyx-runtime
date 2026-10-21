"""The optimisation catalogue (plan section 7): technique ids, stages and profiles.

Shared by the runtime (which plans and applies techniques), the control plane (which
validates a key's choices and reports the ledger) and the console (which lists them),
so the three never disagree about what a technique id means.

Stages: A on by default (never changes answers); B opt-in per app (predictable
effect); C opt-in after a shadow test (can change answers). ``built`` marks the
techniques this release implements; the others are accepted in a key's settings and
take effect once a runtime that has them is deployed.

Opting in (decided 2026-10-21): a key with no profile of its own runs stage A only; its
profile's stage B and C techniques apply once a profile is chosen for it, or a
technique is switched on for it by name. Techniques marked ``explicit_only`` are never
switched on by a profile: the key has to name them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Tuple


@dataclass(frozen=True)
class Technique:
    id: str
    number: int                  # the plan's technique number
    stage: str                   # "A" | "B" | "C"
    title: str
    built: bool = False          # implemented in this runtime
    explicit_only: bool = False  # never switched on by a profile, only by name
    caution: str = ""            # what switching it on gives up, in one sentence


_CATALOGUE = (
    Technique("prompt_cache", 1, "A", "Prompt-cache alignment", True),
    Technique("coalesce", 2, "A", "Request coalescing", True),
    Technique("dedupe", 3, "A", "Deduplication", True),
    Technique("minify", 4, "A", "Format minification", True),
    Technique("exact_cache", 5, "B", "Exact response cache"),
    Technique("output_caps", 6, "B", "Output caps and runaway-loop stop"),
    Technique("budget_degrade", 7, "B", "Budget-aware degradation"),
    Technique("batch", 8, "B", "Batch submission"),
    Technique("tool_pruning", 9, "C", "Tool pruning"),
    Technique("semantic_cache", 10, "C", "Semantic cache"),
    Technique("relevance_filter", 11, "C", "Relevance filtering"),
    Technique("summarise_history", 12, "C", "History summarisation"),
    Technique("compress", 13, "C", "Prompt compression"),
    Technique("right_size", 14, "C", "Model right-sizing"),
    Technique("structured_output", 15, "B", "Structured output"),
    Technique("repair", 16, "B", "Validate and repair"),
    Technique("grounded_retry", 17, "B", "Groundedness retry"),
    Technique("computed_claims", 18, "B", "Computable claims"),
    Technique("business_rules", 19, "B", "Business-rule checks"),
    Technique("context_order", 20, "C", "Context ordering"),
    Technique("escalate", 21, "B", "Escalate on failure"),
    Technique("voting", 22, "B", "Self-consistency voting"),
    Technique("tuning", 23, "B", "Continuous tuning"),
    Technique("pii_tokenise", 30, "B", "PII tokenise and restore"),
    Technique("mark_untrusted", 31, "C", "Marking untrusted content"),
    Technique("risk_adaptive", 32, "B", "Risk-adaptive checking"),
    Technique("parallel_input", 40, "B", "Parallel input check", explicit_only=True,
              caution="The provider receives the request before the input checks finish; the answer is "
                      "held back until they pass and the call is cancelled if they fail."),
)
TECHNIQUES: Dict[str, Technique] = {t.id: t for t in _CATALOGUE}
STAGE_C = frozenset(t.id for t in _CATALOGUE if t.stage == "C")


@dataclass(frozen=True)
class Profile:
    name: str
    adds: Tuple[str, ...]
    keeps_off: FrozenSet[str]


# Section 7 of the plan. Every profile keeps the security floor (techniques 24 to 29 and
# whatever the policy requires); stage A is on in all of them.
PROFILES: Dict[str, Profile] = {p.name: p for p in (
    Profile("balanced", ("exact_cache", "output_caps", "structured_output"), STAGE_C),
    Profile("cost", ("budget_degrade", "batch", "semantic_cache", "relevance_filter", "summarise_history",
                     "compress", "right_size"), frozenset({"voting"})),
    Profile("accuracy", ("repair", "grounded_retry", "computed_claims", "business_rules", "escalate", "voting"),
            frozenset({"summarise_history", "compress", "right_size"})),
    # Parallel input check (40) is explicit-only, so this profile never adds it.
    Profile("latency", ("exact_cache", "semantic_cache", "right_size"),
            frozenset({"batch", "voting"})),
    Profile("security", ("pii_tokenise", "mark_untrusted", "risk_adaptive"),
            frozenset({"semantic_cache", "parallel_input"})),
)}
DEFAULT_PROFILE = "balanced"


@dataclass(frozen=True)
class KeyOptimisation:
    """A key's own choices on top of its profile (``optimisation:`` in its config).
    ``disable`` always wins; stage C also needs ``approved`` (a reviewed shadow test).
    Ids this runtime does not know are ignored (they may come from a newer console)."""
    enable: Tuple[str, ...] = ()
    disable: Tuple[str, ...] = ()
    approved: Tuple[str, ...] = ()


def active_techniques(profile: str, opts: KeyOptimisation, chosen: bool = True) -> List[str]:
    """The built techniques a key's calls use, in catalogue order. ``chosen``: the key has
    a profile of its own (without one it runs stage A and what it names)."""
    prof = PROFILES.get(profile) or PROFILES[DEFAULT_PROFILE]
    adds = set(prof.adds) if chosen and profile in PROFILES else set()
    out = []
    for t in _CATALOGUE:
        if not t.built or t.id in opts.disable:
            continue
        from_profile = t.id in adds and not t.explicit_only
        on = t.stage == "A" or from_profile or t.id in opts.enable
        if t.id in prof.keeps_off and t.id not in opts.enable:
            on = False
        if t.stage == "C" and t.id not in opts.approved:
            on = False
        if on:
            out.append(t.id)
    return out




def catalogue() -> List[Dict[str, object]]:
    """The techniques as plain data (for an API or a console)."""
    return [{"id": t.id, "number": t.number, "stage": t.stage, "title": t.title, "built": t.built,
             "explicit_only": t.explicit_only, "caution": t.caution} for t in _CATALOGUE]


def profiles() -> List[Dict[str, object]]:
    return [{"name": p.name, "adds": list(p.adds), "keeps_off": sorted(p.keeps_off)} for p in PROFILES.values()]
