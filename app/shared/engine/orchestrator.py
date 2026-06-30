"""Detector orchestrator - runs detectors in the correct order and collects results.

This replaces the long conditional chain that used to live inside
``GuardrailsEvaluator.evaluate()``.  The orchestrator is stateless itself;
detector caching is delegated to the ``DetectorRegistry``.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.shared.core.models import Decision, DetectorResult, DetectorTimingResult, EvaluationRequest
from app.shared.detectors.plugin import PluginRegistry
from app.shared.engine.detector_registry import DetectorRegistry
from app.shared.middleware.otel import create_detector_span

logger = logging.getLogger(__name__)


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
# context_filter: None  => runs in both "input" and "output"
#                 "output" => runs only in output context
# can_transform:  if True, a REDACT/TRANSFORM result updates current_text
# ---------------------------------------------------------------------------
# Policy key aliases — maps user-facing names to internal registry keys.
# e.g. a policy with "tool_governance" is treated identically to "tools".
_POLICY_KEY_ALIASES: dict = {
    "tool_governance":    "tools",
    "schema_enforcement": "structure",
}

_DETECTOR_PIPELINE: List[tuple] = [
    # 0. Abuse (stateful)
    ("abuse",             True,  None,     False),
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
    # 8. Structure (output only, never blocks early)
    ("structure",         False, "output", False),
    # 9. Tool governance (text must be JSON: {"tool_name": ..., "arguments": ...})
    ("tools",             False, None,     False),
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
                 plugin_registry: Optional[PluginRegistry] = None):
        self.registry = registry
        self.plugin_registry = plugin_registry

    # -- public helpers for callers that want split input/output phases ------

    def run_input_detectors(
        self, text: str, policy: Dict[str, Any],
        request: "EvaluationRequest",
    ) -> OrchestrationResult:
        return self._run(text, policy, request, context="input")

    def run_output_detectors(
        self, text: str, policy: Dict[str, Any],
        request: "EvaluationRequest",
    ) -> OrchestrationResult:
        return self._run(text, policy, request, context="output")

    # -- internal pipeline runner -------------------------------------------

    def _run(
        self, text: str, policy: Dict[str, Any],
        request: "EvaluationRequest", context: str,
    ) -> OrchestrationResult:
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
            elif ctx_filter is not None and ctx_filter != context:
                continue

            # Special-case: hallucination needs metadata merged into config
            if policy_key == "hallucination":
                config = self._enrich_hallucination_config(config, request)

            detector = self.registry.get_or_create(policy_key, config)
            t0 = time.perf_counter()
            with create_detector_span(policy_key) as span:
                result = self._invoke_detector(policy_key, detector, orch.current_text, request, context)
                if span:
                    span.set_attribute("detector.decision", result.decision.value if result.decision else "ALLOW")
                    span.set_attribute("detector.risk_score", result.risk_score)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            orch.results.append(result)

            transformed = False
            if result.decision == Decision.BLOCK:
                orch.early_block = True
                # Continue running remaining detectors for full observability.
                # The final decision will still be BLOCK (aggregator picks worst),
                # but all detector results will be reported to the caller.
            elif can_transform and result.decision in (Decision.REDACT, Decision.TRANSFORM) and result.sanitized_text:
                orch.current_text = result.sanitized_text
                transformed = True

            orch.detector_timings.append(DetectorTimingResult(
                detector_name=policy_key,
                decision=result.decision.value if result.decision else None,
                risk_score=result.risk_score,
                latency_ms=elapsed_ms,
                rule_hits=result.rule_hits,
                transformed=transformed,
            ))

        # Custom detectors (plugin system) - run after built-ins
        self._run_custom_detectors(policy, orch)

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
            conversation_id = request.metadata.get('conversation_id') if request.metadata else None
            return detector.detect(text, conversation_id=conversation_id)
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

    @staticmethod
    def _enrich_hallucination_config(config: Dict[str, Any], request: "EvaluationRequest") -> Dict[str, Any]:
        if request.metadata:
            if 'source_context' in request.metadata:
                config = {**config, 'source_context': request.metadata['source_context']}
            if 'grounding_sources' in request.metadata:
                config = {**config, 'grounding_sources': request.metadata['grounding_sources']}
        return config

    def _run_custom_detectors(self, policy: Dict[str, Any], orch: OrchestrationResult) -> None:
        custom_detectors = policy.get('custom_detectors', [])
        if not custom_detectors or not self.plugin_registry:
            return
        for custom_cfg in custom_detectors:
            detector_name = custom_cfg.get('name', '')
            detector_config = custom_cfg.get('config', {})
            try:
                t0 = time.perf_counter()
                custom_result = self.plugin_registry.detect(
                    detector_name, orch.current_text, detector_config
                )
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                if custom_result.rule_hits:
                    orch.results.append(custom_result)
                    orch.detector_timings.append(DetectorTimingResult(
                        detector_name=f"custom:{detector_name}",
                        decision=custom_result.decision.value if custom_result.decision else None,
                        risk_score=custom_result.risk_score,
                        latency_ms=elapsed_ms,
                        rule_hits=custom_result.rule_hits,
                        transformed=False,
                    ))
                    if custom_result.decision == Decision.BLOCK:
                        orch.early_block = True
                        break
            except Exception as e:
                logger.error(f"Custom detector '{detector_name}' failed: {e}")
