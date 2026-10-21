"""The optimisation layer (plan section 7): techniques that make a gateway call cheaper,
faster or more accurate, chosen per call by a planner from the key's profile and
opt-ins, with each technique's effect measured for the savings ledger.

The security floor holds by construction, not by configuration. Techniques receive
the request or the answer, never the policy or the evaluator, and the planner has no
operation that skips or weakens a check:

- request-side techniques run before the input guard, so it reads exactly what is
  sent;
- forward strategies (coalescing now; caches and right-sizing later) run after the
  input guard, so nothing unchecked reaches a provider;
- answer-side techniques run before the output guard, and every caller's answer is
  checked again, a shared or cached one included.

Stages: A on by default (never changes answers); B opt-in per app (predictable
effect); C opt-in after a shadow test (can change answers).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from znyx_core.optimisation import (  # noqa: F401 - the catalogue (shared with the control plane)
    CALLER_MAY_SKIP, DEFAULT_PROFILE, PROFILES, STAGE_C, TECHNIQUES, KeyOptimisation, Profile, Technique,
    active_techniques,
)
from znyx_runtime.gateway import formats as fmts
from znyx_runtime.gateway.optimise import stage_a, stage_b, tokens, verify
from znyx_runtime.gateway.optimise.cache import DEFAULT_TTL_SECONDS, ResponseCache
from znyx_runtime.gateway.optimise.coalesce import Coalescer


@dataclass
class Effect:
    """What one technique did on one call: the ledger entry."""
    technique: str
    stage: str
    saved_input_tokens: int = 0
    saved_usd: Optional[float] = None
    latency_ms: Optional[int] = None     # time added (positive) or saved (negative)
    verified: Optional[str] = None       # "pass" | "fail" when the result was checked
    estimated: bool = False              # the saving is the gateway's own count
    basis: str = "tokens"                # how it is priced: tokens | cache | call

    def entry(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"technique": self.technique, "stage": self.stage,
                               "saved_input_tokens": int(self.saved_input_tokens),
                               "estimated": self.estimated}
        if self.saved_usd is not None:
            out["saved_usd"] = round(self.saved_usd, 8)
        if self.latency_ms is not None:
            out["latency_ms"] = int(self.latency_ms)
        if self.verified:
            out["verified"] = self.verified
        return out


class Plan:
    """The techniques one call uses, and what they did."""

    def __init__(self, profile: Optional[str], active: List[str], family: str, model: Optional[str],
                 opts: Optional[KeyOptimisation] = None):
        self.profile = profile
        self.active = frozenset(active)
        self.family = family
        self.model = model
        self.opts = opts or KeyOptimisation()
        self.caller = ""                                   # whose state this call counts in
        self.session: Optional[str] = None                 # the session or caller-named trace
        self.near_budget = False
        # Verification (stage B): the policy's output contract and grounding settings, handed
        # in by the pipeline; the extra calls retries and votes made (model, usage, cost).
        self.contract: Optional[verify.Contract] = None
        self.grounding_config: Dict[str, Any] = {}
        self.sources: List[Dict[str, str]] = []
        self.extra_calls: List[Tuple[str, fmts.Usage, Optional[float]]] = []
        self.answer_changed = False
        self.escalated_from: Optional[str] = None
        # Security and latency (stage B): the call's PII placeholders, and whether its
        # session is checked more strictly.
        self.vault = stage_b.PiiVault()
        self.elevated = False
        self.effects: Dict[str, Effect] = {}
        self.cache_shape: Optional[stage_a.CacheShape] = None
        self.shared_usage: Optional[fmts.Usage] = None    # a coalesced call: the leader's

    def on(self, technique: str) -> bool:
        return technique in self.active

    def effect(self, technique: str, **values: Any) -> Effect:
        current = self.effects.get(technique)
        if current is None:
            current = self.effects[technique] = Effect(technique, TECHNIQUES[technique].stage, **values)
            return current
        current.saved_input_tokens += int(values.get("saved_input_tokens") or 0)
        return current

    # -- the call's steps ----------------------------------------------------------

    def before_guard(self, fmt, body: Dict[str, Any], record, check_roles: Tuple[str, ...]) -> None:
        """Request-side techniques. The input guard runs next, on the changed request."""
        def count(text: str) -> int:
            return tokens.count(text, self.family, self.model)

        for technique, run in (("minify", lambda f, b: stage_a.minify(f, b, count)),
                               ("dedupe", lambda f, b: stage_a.dedupe(f, b, check_roles, count))):
            if not self.on(technique):
                continue
            t0 = time.perf_counter()
            changed, saved_tokens = run(fmt, body)
            if changed:
                record.note(technique, changed)
                effect = self.effect(technique, saved_input_tokens=max(0, saved_tokens), estimated=True)
                effect.latency_ms = int((time.perf_counter() - t0) * 1000)

    def before_send(self, fmt, provider, body: Dict[str, Any], record, seen: stage_a.SeenSet,
                    caller: str) -> None:
        """Changes that alter no text, after the input guard: answer caps, the contract's
        schema for the provider, cache markers."""
        self._cap(record.api, provider, body, record)
        if self.on("structured_output") and verify.request_schema(provider, body, self.contract, record.model):
            record.note("schema_requested")
        shape = self.cache_shape
        if not self.on("prompt_cache") or shape is None:
            return
        if provider.format == "anthropic":
            if not stage_a.anthropic_host(provider):
                return
            if shape.app_managed:
                record.note("prompt_cache_app_managed")
                return
            added = stage_a.anthropic_markers(body, shape, seen, caller, record.model or "")
            if added:
                record.note("prompt_cache", added)
                self.effect("prompt_cache", basis="cache")
        elif stage_a.openai_cache_key(body, shape, provider, caller):
            # OpenAI caches on its own: the key only improves the hit rate, and how much of
            # the discount it earned cannot be told, so the ledger claims none of it.
            record.note("prompt_cache_key")

    def _cap(self, api: str, provider, body: Dict[str, Any], record) -> None:
        caps = []
        cap = self.opts.setting("max_output_tokens") if self.on("output_caps") else None
        if cap:
            caps.append(("output_caps", cap))
        if self.on("budget_degrade") and self.near_budget:
            caps.append(("budget_degrade", stage_b.degrade_cap(self.opts.setting("max_output_tokens"),
                                                               self.opts.setting("degrade_max_output_tokens"))))
        if not caps:
            return
        technique, tightest = min(caps, key=lambda c: c[1])
        if stage_b.apply_output_cap(api, provider, body, tightest):
            record.note("output_capped" if technique == "output_caps" else "budget_degraded")
            self.effect(technique)

    @staticmethod
    def deterministic(body: Dict[str, Any]) -> bool:
        """Settings under which the same request is expected to get the same answer: not
        streamed, not stored, temperature 0, one choice."""
        if body.get("stream") is True or body.get("store") is True:
            return False
        temperature = body.get("temperature")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or temperature != 0:
            return False
        n = body.get("n")
        return n is None or (isinstance(n, int) and not isinstance(n, bool) and n == 1)

    def coalescing_allowed(self, body: Dict[str, Any]) -> bool:
        # A caller whose answer may be checked and asked again does not share a call.
        return self.on("coalesce") and self.deterministic(body) and not self.verifies()

    def wants_contract(self) -> bool:
        return any(self.on(t) for t in ("structured_output", "repair", "business_rules"))

    def verifies(self) -> bool:
        """Whether the answer will be checked: a contract the key's techniques use, computed
        claims, or grounding sources for the groundedness retry (escalation acts only on
        what these find)."""
        if self.contract is not None and self.wants_contract():
            return True
        return self.on("computed_claims") or (self.on("grounded_retry") and bool(self.sources))

    def vote_count(self, body: Dict[str, Any], headers: Mapping[str, str]) -> int:
        """Answers to take for a call the app marks as high-stakes (technique 22)."""
        if not self.on("voting") or body.get("stream") is True:
            return 0
        if str(headers.get("x-znyx-stakes") or "").strip().lower() != "high":
            return 0
        return int(self.opts.setting("vote_samples") or 3)

    def caching_allowed(self, body: Dict[str, Any]) -> bool:
        return self.on("exact_cache") and self.deterministic(body)

    def cache_ttl(self) -> int:
        return int(self.opts.setting("cache_ttl_seconds") or DEFAULT_TTL_SECONDS)

    def loop_limits(self) -> Optional[Tuple[Optional[int], Optional[int], int]]:
        """(max calls, max tokens, window seconds) of the runaway-loop stop, when it applies."""
        if not self.on("output_caps") or not self.session:
            return None
        calls, tokens_ = self.opts.setting("loop_max_calls"), self.opts.setting("loop_max_tokens")
        if not calls and not tokens_:
            return None
        return calls, tokens_, int(self.opts.setting("loop_window_seconds") or stage_b.DEFAULT_LOOP_WINDOW_SECONDS)

    # -- the ledger ----------------------------------------------------------------

    def settle(self, record, prices) -> None:
        """Price each technique's effect once the call's usage is known, and write the
        ledger fields on the record. Nothing is saved on a call that was never sent."""
        record.profile = self.profile
        if not self.effects:
            return
        billed = record.input_tokens is not None
        price = prices.lookup(record.response_model or record.model) if billed else None
        total: Optional[float] = 0.0 if price is not None else None
        entries = []
        for effect in self.effects.values():
            if not billed:
                effect.saved_input_tokens, effect.saved_usd = 0, 0.0
            elif effect.basis == "cache":
                usage = self.shared_usage or fmts.Usage(record.input_tokens, record.output_tokens,
                                                         record.cached_input_tokens, record.cache_write_tokens)
                effect.saved_input_tokens = usage.cached_input_tokens or 0
                effect.saved_usd = stage_a.cache_saving(usage, price) if price is not None else None
            elif effect.basis == "tokens":
                effect.saved_usd = (effect.saved_input_tokens * price["input_per_mtok"] / 1_000_000
                                    if price is not None else None)
            if total is not None and effect.saved_usd is not None:
                total += effect.saved_usd
            entries.append(effect.entry())
        record.optimisations = entries
        if not billed:
            record.saved_usd = 0.0
        elif total is not None and record.cost_usd is not None:
            # Without the call's own cost a saving cannot be set against anything.
            record.saved_usd = round(total, 8)
        if record.saved_usd is not None and record.cost_usd is not None:
            record.baseline_cost_usd = round(max(0.0, record.cost_usd + record.saved_usd), 8)


class Optimiser:
    """Plans each call and holds what techniques share between calls in this process
    (in-flight calls for coalescing, prefixes seen for the prompt cache, stored answers)."""

    def __init__(self) -> None:
        self.coalescer = Coalescer()
        self.seen = stage_a.SeenSet()            # prefixes seen recently (prompt cache)
        self.cache = ResponseCache()             # stored answers (exact cache)

    def plan(self, key, api: str, provider, body: Dict[str, Any], headers: Mapping[str, str],
             caller: str, model: Optional[str]) -> Plan:
        profile = key.profile if key.profile in PROFILES else DEFAULT_PROFILE
        opts = getattr(key, "optimisation", None) or KeyOptimisation()
        off = str(headers.get("x-znyx-optimise") or "").strip().lower() in ("off", "none", "false", "0")
        chosen = bool(getattr(key, "profile_chosen", False))
        # The ledger names a profile only when the key has one (else it runs stage A only).
        active = active_techniques(profile, opts, chosen)
        if off:
            active = [t for t in active if t not in CALLER_MAY_SKIP]
        plan = Plan(profile if chosen else None, active, provider.format, model, opts)
        plan.caller = caller
        if plan.on("prompt_cache"):
            plan.cache_shape = stage_a.CacheShape(api, body, caller, model or "")
        return plan
