"""Detector orchestrator - runs detectors in the correct order and collects results.

This replaces the long conditional chain that used to live inside
``GuardrailsEvaluator.evaluate()``.  The orchestrator is stateless itself;
detector caching is delegated to the ``DetectorRegistry``.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from znyx_core.core.models import Decision, DetectorResult, DetectorTimingResult, EvaluationRequest, Stage
from znyx_core.detectors.plugin import PluginRegistry
from znyx_core.engine.backends import build_strategy
from znyx_core.engine.detector_registry import DetectorRegistry
from znyx_core.engine.egress import prepare_and_audit_egress
from znyx_core.engine.escalation import run_with_strategy
from znyx_core.engine.scorecard_gate import resolve_gated_action
from znyx_core.middleware.otel import create_detector_span

logger = logging.getLogger(__name__)

# Valid per-request stages — the dispatcher rejects anything else.
_VALID_STAGES = frozenset(s.value for s in Stage)


@dataclass
class OrchestrationResult:
    """Collects detector results and the (possibly transformed) text."""
    results: List[DetectorResult] = field(default_factory=list)
    current_text: str = ""
    early_block: bool = False
    detector_timings: List[DetectorTimingResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Ordered detector specs.  Each tuple is:
#   (policy_key, default_enabled, context_filter, can_transform)
#
# context_filter: None      => runs in every dispatched stage (input/output and, when
#                              explicitly dispatched, the new stages)
#                 "output"   => runs only in the output stage
#                 (a, b, ...) => a tuple/list runs only in any of those stages (
#                              multi-stage detectors, e.g. agent_plan + tool)
# can_transform:  if True, a REDACT/TRANSFORM result updates current_text
# ---------------------------------------------------------------------------
# Policy key aliases — maps user-facing names to internal registry keys.
# e.g. a policy with "tool_governance" is treated identically to "tools".
_POLICY_KEY_ALIASES: dict = {
    "tool_governance":    "tools",
    "schema_enforcement": "structure",
}

_DETECTOR_PIPELINE: List[tuple] = [
    # 0. Abuse (stateful) — scoped to input/output so its per-request rate-limit / flood
    # buckets are consumed once per user turn, NOT once per agentic sub-stage call
    # (retrieval + N agent-steps + memory-write would otherwise burn many request tokens).
    ("abuse",             True,  ("input", "output"), False),
    # 1. Secrets (hard block)
    ("secrets",           True,  None,     False),
    # 2. Exfiltration
    ("exfiltration",      True,  None,     False),
    # 2a. Gibberish
    ("gibberish",         False, None,     False),
    # 2b. Language
    ("language",          False, None,     False),
    # 3. Topic restriction
    ("topic_restriction", True,  None,     False),
    # 4. Toxicity
    ("toxicity",          True,  None,     False),
    # 4a. Bias
    ("bias",              False, None,     False),
    # 4b. Sentiment
    ("sentiment",         False, None,     False),
    # 4c. Compliance (can transform)
    ("compliance",        False, None,     True),
    # 5. Competitor (can transform)
    ("competitor",        True,  None,     True),
    # 6. Jailbreak (stateful)
    ("jailbreak",         True,  None,     False),
    # 7. PII (can redact)
    ("pii",               True,  None,     True),
    # 7a. URL / code detectors run both directions; copyright/hallucination output-only
    ("malicious_url",     False, None,     True),
    ("copyright",         False, "output", False),
    ("code_safety",       False, None,     False),
    ("hallucination",     False, "output", False),
    # 7b. deterministic gap detectors (LLM02 / LLM07 / LLM08).
    ("sensitive_business_data", False, None,     True),
    ("citation_integrity",      False, "output", False),
    ("system_prompt_leakage",   False, "output", False),
    ("numerical_consistency",   False, "output", False),
    ("document_metadata_leakage", False, None,   False),
    # 7c. new-stage gap detectors (LLM01 / LLM03 / LLM06). Each is scoped to its
    # stage(s) via the ctx_filter (a tuple = runs in any of those stages). They are
    # default-disabled and only fire on their stage's evaluate endpoint / benchmark
    # dispatch. mcp_manifest_scanner and tool_permission_audit are tool_registration
    # HOOKS, not in this pipeline.
    ("retrieval_chunk_injection", False, "retrieval",                       False),
    ("embedding_integrity",       False, "retrieval",                       False),
    ("tenant_scope_assertion",    False, "retrieval",                       False),
    ("retrieval_jamming",         False, "retrieval",                       False),
    ("tool_output_injection",     False, "tool",                            False),
    # excessive_agency risk-scores a PLAN (agent_plan) or a live step's action
    # (agent_loop) — both reachable via their evaluate endpoints. It is NOT scoped to
    # "tool" (the tool stage carries tool-RESULT text, not an action plan).
    ("excessive_agency",          False, ("agent_plan", "agent_loop"),      False),
    # Same stages as excessive_agency: it scores what the action DOES, this asks
    # whether a human agreed to it. Both must see a plan before it runs and each
    # live step as it is taken.
    ("human_approval_gate",       False, ("agent_plan", "agent_loop"),      False),
    ("unbounded_consumption",     False, ("agent_loop", "input", "output"), False),
    ("memory_write_poisoning",    False, "memory_write",                    False),
    # Durable corpus corruption (LLM05) as distinct from injected instructions
    # (LLM01): also exposed as an ingest hook for the corpus write path.
    ("corpus_poisoning_monitor",  False, ("memory_write", "retrieval"),     False),
    ("reasoning_trace_disclosure", False, "output",                        False),
    # Output-side counterpart to the input normaliser: stops model output rewriting a
    # terminal, where the input path stops an attacker hiding from the detectors.
    ("output_control_char_sanitizer", False, "output",                      False),
    ("multimodal_injection",      False, "input",                           False),
    # A semantic cache is consulted on the way IN and can also be reported alongside the
    # answer, so it runs in both directions; the detector no-ops without a cache block.
    ("semantic_cache_integrity",  False, ("input", "output", "retrieval"),  False),
    # 8. Structure (output only, never blocks early)
    ("structure",         False, "output", False),
    # 9. Tool governance (text must be JSON: {"tool_name": ..., "arguments": ...}).
    # Scoped to input/output: the tool INVOCATION is governed by a direct call in
    # evaluate_tool(), so `tools` must not also fire on the tool-RESULT pipeline pass
    # (context="tool"), where the text is a result, not an invocation.
    ("tools",             False, ("input", "output"), False),
]


def get_pipeline_defaults() -> dict:
    """Return {policy_key: {"enabled": True}} for every detector whose
    default_enabled flag is True. Used to seed the first published bundle
    on project creation — not for runtime evaluation."""
    return {
        key: {"enabled": True}
        for key, default_enabled, _, _ in _DETECTOR_PIPELINE
        if default_enabled
    }


class DetectorOrchestrator:
    """Runs detectors from a registry in a fixed order, respecting early-block."""

    def __init__(self, registry: DetectorRegistry,
                 plugin_registry: Optional[PluginRegistry] = None,
                 egress_sink=None, scorecard_public_key: Optional[str] = None):
        self.registry = registry
        self.plugin_registry = plugin_registry
        # optional callback(EgressEvent) for boundary-crossing calls. The runtime
        # wires a durable sink; the control plane writes egress_events rows;
        # None = no emission (default — zero behaviour change for current callers).
        self.egress_sink = egress_sink
        # Tamper-evident scorecard stamps (console-less tier): when an Ed25519 public key
        # is configured (arg or ZNYX_SCORECARD_PUBLIC_KEY), a stamp that PERMITS enforcement
        # must carry a valid signature or it's forced to fail-closed (WARN). None → trust mode
        # (unchanged): the bool is honoured as-is, as managed bundles rely on the bundle sig.
        from znyx_core.engine.scorecard_stamp import resolve_verification_key
        self._scorecard_public_key = resolve_verification_key(scorecard_public_key)

    # -- public helpers for callers that want split input/output phases ------

    def run_input_detectors(
        self, text: str, policy: Dict[str, Any],
        request: "EvaluationRequest", *, judge_ctx=None,
    ) -> OrchestrationResult:
        return self._run(text, policy, request, context="input", judge_ctx=judge_ctx)

    def run_output_detectors(
        self, text: str, policy: Dict[str, Any],
        request: "EvaluationRequest", *, judge_ctx=None,
    ) -> OrchestrationResult:
        return self._run(text, policy, request, context="output", judge_ctx=judge_ctx)

    def run_detectors(
        self, text: str, policy: Dict[str, Any],
        request: "EvaluationRequest", context: str = "input", *, judge_ctx=None,
    ) -> OrchestrationResult:
        """Generalized stage dispatch.

        Runs the detector pipeline filtered to ``context``, where ``context`` is any
        per-request stage (input / output / retrieval / tool / agent_plan /
        agent_loop / memory_write). Detectors are selected by their per-policy
        ``stages`` list or the pipeline's default ``ctx_filter``. This replaces the
        old input-vs-output branch so a new stage is routed correctly rather than
        being mistreated as output.
        """
        return self._run(text, policy, request, context=context, judge_ctx=judge_ctx)

    # -- internal pipeline runner -------------------------------------------

    def _run(
        self, text: str, policy: Dict[str, Any],
        request: "EvaluationRequest", context: str, *, judge_ctx=None,
    ) -> OrchestrationResult:
        # Dispatcher boundary: the stage must be a known Stage value. Reject
        # arbitrary strings so an unknown stage can't silently run the default
        # (ctx_filter=None) detectors. Callers pass a fixed Stage; new-stage endpoints will
        # bind typed stage requests to these same values.
        if context not in _VALID_STAGES:
            raise ValueError(
                f"Unknown evaluation stage '{context}'. Valid stages: {sorted(_VALID_STAGES)}"
            )
        orch = OrchestrationResult(current_text=text)

        # Also collect any alias keys from the policy (e.g. "tool_governance" → "tools")
        alias_overrides: Dict[str, Any] = {}
        for alias, canonical in _POLICY_KEY_ALIASES.items():
            if alias in policy and canonical not in policy:
                alias_overrides[canonical] = policy[alias]

        effective_policy = {**policy, **alias_overrides}

        for policy_key, default_enabled, ctx_filter, can_transform in _DETECTOR_PIPELINE:
            config = effective_policy.get(policy_key, {})
            if not config.get('enabled', False):
                continue

            # Per-policy stages override: if the config specifies stages, use that
            # list to decide whether this detector runs in the current context.
            # If stages is absent/None, fall back to the pipeline's ctx_filter default.
            policy_stages = config.get('stages')
            if policy_stages:
                if context not in policy_stages:
                    continue
            elif ctx_filter is not None:
                # ctx_filter may be a single stage string or a tuple/list of stages.
                allowed = (ctx_filter,) if isinstance(ctx_filter, str) else tuple(ctx_filter)
                if context not in allowed:
                    continue

            # Grounding-aware detectors get the request's source_context /
            # grounding_sources merged into their config (hallucination + citation_integrity).
            if policy_key in ("hallucination", "citation_integrity"):
                config = self._enrich_hallucination_config(config, request)

            detector = self.registry.get_or_create(policy_key, config)

            # NLI groundedness wiring: build the per-request NLI scorer from the detector's
            # `nli` config block (same factory + egress gate as the quality scorer) and set
            # it on the instance. The callable can't live in `config` (the registry digests
            # json.dumps(config)), so it's an instance attribute set after creation - safe
            # across concurrent requests because the registry returns a FRESH instance for
            # these two detectors on every call (they are request-scoped, never cached).
            if policy_key in ("hallucination", "citation_integrity"):
                detector.nli_scorer = self._build_nli_scorer(config, effective_policy, request)
            t0 = time.perf_counter()
            with create_detector_span(policy_key, stage=context) as span:
                result = self._invoke_detector(policy_key, detector, orch.current_text, request, context)
                if span:
                    span.set_attribute("detector.decision", result.decision.value if result.decision else "ALLOW")
                    span.set_attribute("detector.risk_score", result.risk_score)
            # if the detector's policy carries a model-backed `strategy`, escalate
            # (deterministic → ml → llm) from the deterministic result. No strategy →
            # deterministic-only path unchanged (zero behaviour change for current policies).
            strategy = build_strategy(config, runtime_policy=effective_policy.get("runtime_policy"),
                                      policy=effective_policy)
            if strategy is not None:
                # when a composition root injected a judge context AND this strategy
                # escalates to a judge mode, build the multi-judge consensus caller (audit +
                # deny-of-wallet budget wired in). Otherwise judge_caller stays None and the
                # escalation uses the generic backend transport — zero behaviour change.
                judge_caller = None
                if judge_ctx is not None and any(
                    m in ("local_llm", "remote_llm", "remote_api") for m in strategy.order
                ):
                    from znyx_core.engine.judge_runtime import build_escalation_judge_caller
                    judge_caller = build_escalation_judge_caller(
                        policy_key, config, judge_ctx, request)
                result = run_with_strategy(
                    policy_key, result, strategy, orch.current_text,
                    request=request, egress_sink=self.egress_sink,
                    judge_caller=judge_caller,
                )
            # runtime action resolution: a model-backed detector whose enforcement
            # gate didn't pass (stamped into the policy at publish via `_scorecard_gate`)
            # has its BLOCK/REDACT downgraded to WARN. Defence in depth — the publish-time
            # blocker is primary; this protects DB-less runtimes honouring a stamped bundle.
            result = self._verify_and_apply_scorecard_gate(policy_key, config, result)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            orch.results.append(result)

            transformed = False
            if result.decision == Decision.BLOCK:
                orch.early_block = True
                # Continue running remaining detectors for full observability.
                # The final decision will still be BLOCK (aggregator picks worst),
                # but all detector results will be reported to the caller.
            elif (can_transform and context in ("input", "output")
                  and result.decision in (Decision.REDACT, Decision.TRANSFORM)
                  and result.sanitized_text):
                # Only rewrite text on the input/output stages. On the new stages
                # (retrieval / agent_plan / agent_loop / memory_write) a legacy
                # transform-capable detector (e.g. pii REDACT) must NOT silently mutate
                # the retrieved chunk / plan / memory value — it still flags (WARN/BLOCK),
                # but the content the agent sees is left intact for the caller to handle.
                orch.current_text = result.sanitized_text
                transformed = True

            orch.detector_timings.append(DetectorTimingResult(
                detector_name=policy_key,
                decision=result.decision.value if result.decision else None,
                risk_score=result.risk_score,
                latency_ms=elapsed_ms,
                rule_hits=result.rule_hits,
                transformed=transformed,
                # carry the model-backed enrichment + per-layer attempts into the
                # persisted timing record so the trace renders them without re-eval.
                confidence=result.confidence,
                calibrated_score=result.calibrated_score,
                label_scores=result.label_scores,
                model_version=result.model_version,
                execution_mode=result.execution_mode,
                fallback_path=result.fallback_path,
                external_egress=result.external_egress,
                threshold=result.threshold,
                layer_results=result.layer_results,
            ))

        # Standalone model-backed / vendor detectors: an enabled policy key that is NOT a
        # built-in pipeline detector but carries a model-backed `strategy` + `backends` — e.g. an
        # installed remote_api vendor moderation detector. The vendor IS the detector, so the
        # deterministic base is a pass-through ALLOW and the strategy runs the configured backend
        # (egress-gated + scorecard-gated, same path as pipeline detectors). Without this, such a
        # key was silently ignored by the fixed pipeline.
        _pipeline_keys = {k for k, *_ in _DETECTOR_PIPELINE}
        _reserved = {"runtime_policy", "custom_detectors", "output_contract", "quality_scoring",
                     "_multilingual", "_scorecard_gate"}
        for std_key, std_config in list(effective_policy.items()):
            if (std_key in _pipeline_keys or std_key in _POLICY_KEY_ALIASES or std_key in _reserved
                    or not isinstance(std_config, dict) or not std_config.get("enabled")):
                continue
            std_strategy = build_strategy(
                std_config, runtime_policy=effective_policy.get("runtime_policy"),
                policy=effective_policy)
            if std_strategy is None:
                continue  # no model-backed strategy → nothing to run (a bare {enabled} is inert by design)
            std_stages = std_config.get("stages") or ("input", "output")
            if context not in std_stages:
                continue
            t0 = time.perf_counter()
            judge_caller = None
            if judge_ctx is not None and any(
                m in ("local_llm", "remote_llm", "remote_api") for m in std_strategy.order
            ):
                from znyx_core.engine.judge_runtime import build_escalation_judge_caller
                judge_caller = build_escalation_judge_caller(std_key, std_config, judge_ctx, request)
            result = run_with_strategy(
                std_key, DetectorResult(decision=Decision.ALLOW, risk_score=0),
                std_strategy, orch.current_text, request=request,
                egress_sink=self.egress_sink, judge_caller=judge_caller)
            result = self._verify_and_apply_scorecard_gate(std_key, std_config, result)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            orch.results.append(result)
            if result.decision == Decision.BLOCK:
                orch.early_block = True
            orch.detector_timings.append(DetectorTimingResult(
                detector_name=std_key,
                decision=result.decision.value if result.decision else None,
                risk_score=result.risk_score, latency_ms=elapsed_ms, rule_hits=result.rule_hits,
                transformed=False, confidence=result.confidence,
                calibrated_score=result.calibrated_score, label_scores=result.label_scores,
                model_version=result.model_version, execution_mode=result.execution_mode,
                fallback_path=result.fallback_path, external_egress=result.external_egress,
                threshold=result.threshold, layer_results=result.layer_results))

        # Custom detectors (plugin system) - run after built-ins
        self._run_custom_detectors(effective_policy, orch, request, context)

        return orch

    # -- detector invocation helpers ----------------------------------------

    def _invoke_detector(
        self, policy_key: str, detector: Any, text: str,
        request: "EvaluationRequest", context: str,
    ) -> DetectorResult:
        """Call the detector's ``detect`` method with the right signature."""
        if policy_key == "abuse":
            user_id = request.metadata.get('user_id') if request.metadata else None
            return detector.detect(
                text,
                tenant_id=request.tenant_id,
                app_id=request.app_id,
                user_id=user_id,
                context=context,
            )
        if policy_key == "jailbreak":
            # tenant/app scope the conversation history so two tenants reusing
            # the same conversation id never share escalation state.
            conversation_id = request.metadata.get('conversation_id') if request.metadata else None
            return detector.detect(text, conversation_id=conversation_id,
                                   tenant_id=request.tenant_id, app_id=request.app_id)
        if policy_key == "unbounded_consumption":
            # Stateful (LLM06): needs scope + a per-identity key (session > user) for
            # per-session budget accounting and the budget signals in metadata
            # (tokens/cost/agent_step). user_id is the fallback so two users without a
            # session id don't share one budget bucket.
            session_id = getattr(request, "session_id", None)
            user_id = None
            if request.metadata:
                session_id = session_id or request.metadata.get("session_id") or request.metadata.get("conversation_id")
                user_id = request.metadata.get("user_id")
            return detector.detect(
                text,
                tenant_id=request.tenant_id,
                app_id=request.app_id,
                session_id=session_id,
                context=context,
                metadata=request.metadata,
                user_id=user_id,
            )
        if policy_key == "corpus_poisoning_monitor":
            return detector.detect(text, tenant_id=request.tenant_id,
                                   metadata=request.metadata)
        if policy_key in ("tenant_scope_assertion", "semantic_cache_integrity"):
            # Both compare the REQUESTING tenant against the tenant recorded on what was
            # served: retrieved chunks in one case, a cache entry in the other.
            return detector.detect(text, tenant_id=request.tenant_id,
                                   metadata=request.metadata)
        if policy_key == "reasoning_trace_disclosure":
            # Also needs the ENVIRONMENT: log-probability exposure is gated by where the
            # response is going, not banned outright, since the same detail is a
            # legitimate debugging tool on a development endpoint.
            return detector.detect(text, metadata=request.metadata, env=request.env)
        if policy_key in ("retrieval_jamming", "output_control_char_sanitizer",
                          "multimodal_injection"):
            # Each reads its evidence (scores, traces, attachments) from metadata.
            return detector.detect(text, metadata=request.metadata)
        if policy_key == "human_approval_gate":
            # Approval evidence (approver identity, ticket) travels in metadata, not text.
            return detector.detect(text, metadata=request.metadata)
        if policy_key == "system_prompt_leakage":
            # Needs the request's tool/function schemas so live schema disclosure is caught
            # without prior registration (LLM08 covers all hidden context, not just prompts).
            return detector.detect(text, metadata=request.metadata)
        if policy_key == "tools":
            import json as _json
            try:
                tool_data = _json.loads(text)
                tool_name = tool_data.get("tool_name", "")
                tool_args = tool_data.get("arguments", tool_data.get("args", {}))
            except (ValueError, TypeError, AttributeError):
                tool_name = text
                tool_args = {}
            return detector.detect(tool_name, tool_args)
        return detector.detect(text)

    def _verify_and_apply_scorecard_gate(self, detector_id: str, config: Dict[str, Any],
                                         result: "DetectorResult") -> "DetectorResult":
        """Instance wrapper that adds tamper-evident stamp verification (console-less tier)
        on top of the pure ``_apply_scorecard_gate`` decision logic.

        When a verification key is configured, a stamp that PERMITS enforcement
        (``enforcement_passed: true``) is honoured only if its Ed25519 signature is valid for
        this detector + model_version + validated_at; otherwise the stamp is neutralised
        (forced to not-passed) so the gate downgrades BLOCK/REDACT to WARN. With no key
        configured this is a pass-through (trust mode), so behaviour is unchanged for managed
        bundles (protected by the bundle signature) and for existing unsigned YAML."""
        if self._scorecard_public_key and isinstance(config, dict):
            gate = config.get("_scorecard_gate")
            if isinstance(gate, dict) and gate.get("enforcement_passed") is True:
                from znyx_core.engine.scorecard_stamp import verify_stamp
                if not verify_stamp(detector_id, gate, self._scorecard_public_key):
                    logger.warning(
                        "scorecard stamp for '%s' permits enforcement but its signature is "
                        "missing/invalid; failing closed to advisory WARN", detector_id)
                    # Shallow copy so we don't mutate the shared policy dict.
                    config = {**config, "_scorecard_gate": {**gate, "enforcement_passed": False}}
        return self._apply_scorecard_gate(config, result)

    @staticmethod
    def _apply_scorecard_gate(config: Dict[str, Any], result: "DetectorResult") -> "DetectorResult":
        """Advisory-by-default for MODEL-BACKED / judge detectors: a BLOCK/REDACT only
        enforces when the publish-stamped ``_scorecard_gate`` says ``enforcement_passed is
        True``. A non-passing stamp downgrades to WARN — and so does the ABSENCE of a stamp
        on a model-backed/judge detector (fail-closed: an unstamped policy reaching the
        runtime via a non-bundle / validate=False / tampered path must not let a judge or
        ML detector BLOCK without a passing scorecard). Purely deterministic detectors have
        no scorecard and are never gated."""
        if not isinstance(config, dict) or result.decision not in (Decision.BLOCK, Decision.REDACT):
            return result
        gate = config.get("_scorecard_gate")
        if gate is not None:
            # Explicit stamp: only a dict stamp with enforcement_passed literally True
            # permits enforcement. A string "false", 0, None, or a tampered NON-DICT stamp
            # must fail CLOSED to WARN — never raise (that would crash the pipeline rather
            # than downgrade).
            enforcement_passed = isinstance(gate, dict) and gate.get("enforcement_passed") is True
            _, downgraded = resolve_gated_action(result.decision.value, enforcement_passed)
            if not downgraded:
                return result
            note = " [scorecard enforcement gate not met → downgraded to WARN]"
        else:
            # No stamp: deterministic detectors are unaffected; a model-backed/judge
            # detector fails closed to advisory.
            from znyx_core.engine.scorecard_gate import is_model_backed
            if not is_model_backed(config):
                return result
            note = " [unstamped model-backed detector → advisory WARN until an enforcement-tier scorecard is stamped]"
        return result.model_copy(update={
            "decision": Decision.WARN,
            "developer_message": ((result.developer_message or "") + note).strip(),
        })

    @staticmethod
    def _enrich_hallucination_config(config: Dict[str, Any], request: "EvaluationRequest") -> Dict[str, Any]:
        if request.metadata:
            if 'source_context' in request.metadata:
                config = {**config, 'source_context': request.metadata['source_context']}
            if 'grounding_sources' in request.metadata:
                config = {**config, 'grounding_sources': request.metadata['grounding_sources']}
        return config

    def _build_nli_scorer(self, config: Dict[str, Any], effective_policy: Dict[str, Any],
                          request: "EvaluationRequest"):
        """Build the inference NLI scorer ``(premise, hypotheses) -> list[float]`` from a
        detector's ``nli`` config block, routed through the egress gate — the same factory
        the quality scorer uses. None when the detector has no enabled ``nli`` block (the
        detector then keeps its deterministic token-overlap / fuzzy-overlap path)."""
        from znyx_core.engine.quality.nli_client import nli_scorer_from_config
        return nli_scorer_from_config(
            config,
            runtime_policy=effective_policy.get("runtime_policy"),
            policy=effective_policy,
            egress_sink=self.egress_sink,
            request=request,
        )

    def _run_custom_detectors(self, policy: Dict[str, Any], orch: OrchestrationResult,
                              request: "EvaluationRequest" = None,
                              context: str = "input") -> None:
        custom_detectors = policy.get('custom_detectors', [])
        if not custom_detectors or not self.plugin_registry:
            return
        for custom_cfg in custom_detectors:
            detector_name = custom_cfg.get('name', '')
            detector_config = custom_cfg.get('config', {})
            # Stage filter: a custom detector runs only in its declared `stages`, or — when
            # none are declared — the default input/output stages. This stops a custom
            # webhook/remote detector from receiving (and egressing) retrieval chunks,
            # agent plans, or memory writes just because the new stages now exist.
            cfg_stages = custom_cfg.get('stages')
            allowed_stages = cfg_stages if cfg_stages else ("input", "output")
            if context not in allowed_stages:
                continue
            try:
                # a custom detector that declares an egress URL POSTs the text to
                # an external endpoint — a boundary-crossing egress. Route it through
                # the SAME gate as model backends (no_external_calls / allowlist /
                # residency / redact + fail-closed audit). A denial skips the detector
                # (it simply doesn't run); a missing/failed audit fails closed (no
                # un-audited crossing). Covers BOTH declarative egress plugins: the
                # `webhook` type (config 'url') and the `remote` type (config
                # 'endpoint_url') — keyed on either so a sibling type can't slip past.
                call_text = orch.current_text
                egressed = False
                egress_url = None
                if isinstance(detector_config, dict):
                    egress_url = detector_config.get('url') or detector_config.get('endpoint_url')
                if egress_url:
                    rp = policy.get('runtime_policy') or {}
                    rbe = detector_config.get('redact_before_egress')
                    prep = prepare_and_audit_egress(
                        "remote_api", orch.current_text,
                        endpoint_url=egress_url,
                        region=detector_config.get('region'),
                        in_boundary=None,
                        no_external_calls=bool(rp.get('no_external_calls')),
                        egress_allowlist=detector_config.get('egress_allowlist'),
                        allowed_regions=rp.get('allowed_regions'),
                        redact_pii=rbe.get('pii', True) if isinstance(rbe, dict) else False,
                        redact_secrets=rbe.get('secrets', True) if isinstance(rbe, dict) else False,
                        detector_key=f"custom:{detector_name}",
                        request=request,
                        egress_sink=self.egress_sink,
                        pii_config=policy.get('pii') if isinstance(policy.get('pii'), dict) else None,
                        secrets_config=policy.get('secrets') if isinstance(policy.get('secrets'), dict) else None,
                    )
                    if not prep.proceed:
                        logger.info("custom detector '%s' egress denied (%s) — skipped",
                                    detector_name, prep.reason)
                        continue
                    call_text = prep.call_text
                    egressed = prep.decision.is_egress

                t0 = time.perf_counter()
                custom_result = self.plugin_registry.detect(
                    detector_name, call_text, detector_config
                )
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                # the plugin itself rarely sets external_egress, so reflect the
                # boundary crossing the gate just performed onto both the result and
                # the trace timing — otherwise the audited egress shows external_egress
                # =False end-to-end.
                if egressed:
                    custom_result.external_egress = True
                # Record whenever the result carries ANY signal — a non-ALLOW
                # decision, rule hits, a positive risk score, model-backed
                # layer_results, OR a boundary crossing (so a no-hit egress still
                # leaves a visible timing row for trace/UI consumers). A no-hit BLOCK
                # (or a remote/ML plugin returning scores without rule hits) must still
                # count toward the decision and be persisted with its full contract;
                # previously a rule_hits-only gate silently dropped it.
                meaningful = (
                    (custom_result.decision is not None and custom_result.decision != Decision.ALLOW)
                    or bool(custom_result.rule_hits)
                    or custom_result.risk_score > 0
                    or bool(custom_result.layer_results)
                    or egressed
                )
                if meaningful:
                    orch.results.append(custom_result)
                    orch.detector_timings.append(DetectorTimingResult(
                        detector_name=f"custom:{detector_name}",
                        decision=custom_result.decision.value if custom_result.decision else None,
                        risk_score=custom_result.risk_score,
                        latency_ms=elapsed_ms,
                        rule_hits=custom_result.rule_hits,
                        transformed=False,
                        # preserve the model-backed contract for custom/remote plugins.
                        confidence=custom_result.confidence,
                        calibrated_score=custom_result.calibrated_score,
                        label_scores=custom_result.label_scores,
                        model_version=custom_result.model_version,
                        execution_mode=custom_result.execution_mode,
                        fallback_path=custom_result.fallback_path,
                        external_egress=custom_result.external_egress,
                        threshold=custom_result.threshold,
                        layer_results=custom_result.layer_results,
                    ))
                    if custom_result.decision == Decision.BLOCK:
                        orch.early_block = True
                        break
            except Exception as e:
                logger.error(f"Custom detector '{detector_name}' failed: {e}")
