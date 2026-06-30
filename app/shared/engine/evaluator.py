import json
import logging
import uuid
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timezone

from app.shared.core.models import EvaluationRequest, EvaluationResponse, DetectorResult, Decision, ToolEvaluationRequest, DetectorTimingResult
from app.shared.core.decision import DecisionAggregator
from app.shared.detectors.plugin import PluginRegistry
from app.shared.detectors.tools import ToolGovernanceDetector
from app.shared.engine.detector_registry import DetectorRegistry, default_registry
from app.shared.engine.orchestrator import DetectorOrchestrator
from app.shared.engine.quality_scorer import QualityScorer
from app.shared.engine.remediation import RemediationHandler

logger = logging.getLogger(__name__)

# Optional metrics collector - graceful no-op if not available
try:
    from app.shared.middleware.metrics import MetricsCollector
    _metrics: Optional["MetricsCollector"] = MetricsCollector()
except Exception:
    _metrics = None


class GuardrailsEvaluator:
    """Main evaluator that runs all detectors and aggregates results.

    Stateful detectors (abuse, jailbreak) are kept as long-lived instances
    so that rate-limit buckets and conversation history survive across requests.

    The evaluator is policy-agnostic: callers must resolve the policy dict
    before calling evaluate(). For convenience, a policy_resolver can be
    provided for YAML-mode backward compatibility.
    """

    def __init__(self, policy_resolver=None, log_redacted_text: bool = False,
                 on_evaluation: Optional[Callable] = None,
                 plugin_registry: Optional[PluginRegistry] = None,
                 registry: Optional[DetectorRegistry] = None):
        self.policy_resolver = policy_resolver
        self.log_redacted_text = log_redacted_text
        self.on_evaluation = on_evaluation  # telemetry callback
        self.plugin_registry = plugin_registry

        self._registry = registry or default_registry
        self._orchestrator = DetectorOrchestrator(
            self._registry, plugin_registry=self.plugin_registry
        )
        self._remediation = RemediationHandler()

    async def evaluate(
        self,
        request: EvaluationRequest,
        context: str = "input",
        policy: Optional[Dict[str, Any]] = None,
        db=None
    ) -> EvaluationResponse:
        """
        Evaluate text against all guardrails.

        Args:
            request: The evaluation request
            context: "input" or "output"
            policy: Pre-resolved policy dict. If None, resolved via policy_resolver or DB.
            db: Optional DB session for legacy DB-mode resolution
        """
        start_time = datetime.now(timezone.utc)

        # Resolve policy: prefer explicit > DB > YAML resolver
        policy = await self._resolve_policy(request, policy, db)
        if isinstance(policy, EvaluationResponse):
            return policy  # error response from failed resolution

        policy_version = policy.get('policy_version', 'unknown')

        # Run detector pipeline
        if context == "input":
            orch = self._orchestrator.run_input_detectors(request.text, policy, request)
        else:
            orch = self._orchestrator.run_output_detectors(request.text, policy, request)

        # Aggregate all results
        final_result = DecisionAggregator.aggregate(orch.results)

        # Preserve the chained sanitized text (each detector builds on previous)
        if orch.current_text != request.text:
            final_result.sanitized_text = orch.current_text

        response = self._build_response(request, final_result, policy_version)

        # Always emit telemetry/metrics - including early BLOCK decisions
        latency_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

        # Enrich response with observability data
        response.latency_ms = latency_ms
        response.trace_id = request.trace_id or str(uuid.uuid4())
        response.session_id = getattr(request, "session_id", None)
        response.span_id = getattr(request, "span_id", None)
        response.detector_results = orch.detector_timings

        # Quality scoring - output context only, informational (never blocks)
        quality_config = policy.get("quality_scoring", {})
        if context == "output" and quality_config.get("enabled", False):
            scorer = QualityScorer(quality_config)
            original_input = (request.metadata or {}).get("original_input", "")
            response.quality = scorer.score(
                input_text=original_input,
                output_text=request.text,
                metadata=request.metadata,
            )

        # Apply on_fail remediation if configured
        response = self._remediation.apply(response, policy, orch.results)

        self._log_evaluation(request, response, policy_version)
        self._emit_telemetry(request, response, latency_ms, context)
        self._record_metrics(context, response, latency_ms / 1000.0)
        return response

    async def evaluate_tool(self, request: ToolEvaluationRequest,
                            policy: Optional[Dict[str, Any]] = None,
                            db=None) -> EvaluationResponse:
        """Evaluate tool invocation against governance policies."""
        start_time = datetime.now(timezone.utc)

        if policy is None:
            if db is not None:
                from app.shared.policy.resolver import DBPolicyResolver
                resolver = DBPolicyResolver(db, cache_ttl=60)
                policy = await resolver.resolve(
                    tenant_id=request.tenant_id,
                    app_id=request.app_id,
                    agent_id=request.agent_id,
                    env=request.env
                )
            elif self.policy_resolver:
                policy = self.policy_resolver.resolve(
                    tenant_id=request.tenant_id,
                    app_id=request.app_id,
                    agent_id=request.agent_id,
                    env=request.env
                )
            else:
                raise ValueError("No policy provided and no resolver configured")

        policy_version = policy.get('policy_version', 'unknown')
        results = []

        tools_config = policy.get('tools', {})
        if tools_config.get('enabled', True):
            tool_detector = self._registry.get_or_create("tools", tools_config)
            tool_result = tool_detector.detect(request.tool_name, request.tool_args)
            results.append(tool_result)

        final_result = DecisionAggregator.aggregate(results)

        response = EvaluationResponse(
            request_id=request.request_id,
            decision=final_result.decision or Decision.ALLOW,
            risk_score=final_result.risk_score,
            policy_version=policy_version,
            rule_hits=final_result.rule_hits,
            sanitized_text=None,
            sanitized_tool_args=None,
            user_message=final_result.user_message,
            developer_message=final_result.developer_message
        )

        latency_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
        response.latency_ms = latency_ms
        response.trace_id = getattr(request, "trace_id", None) or str(uuid.uuid4())
        response.session_id = getattr(request, "session_id", None)
        response.span_id = getattr(request, "span_id", None)

        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request.request_id,
            "tenant_id": request.tenant_id,
            "app_id": request.app_id,
            "agent_id": request.agent_id,
            "env": request.env,
            "tool_name": request.tool_name,
            "decision": response.decision.value,
            "risk_score": response.risk_score,
            "policy_version": policy_version,
            "rule_hit_ids": [hit.rule_id for hit in response.rule_hits],
        }
        logger.info(json.dumps(log_data))
        self._emit_telemetry(request, response, latency_ms)
        self._record_metrics("tool", response, latency_ms / 1000.0)

        return response

    # -- private helpers ----------------------------------------------------

    async def _resolve_policy(self, request, policy, db):
        """Resolve policy dict, returning an EvaluationResponse on failure."""
        if policy is not None:
            return policy
        try:
            if db is not None:
                from app.shared.policy.resolver import DBPolicyResolver
                resolver = DBPolicyResolver(db, cache_ttl=60)
                return await resolver.resolve(
                    tenant_id=request.tenant_id,
                    app_id=request.app_id,
                    agent_id=request.agent_id,
                    env=request.env
                )
            elif self.policy_resolver:
                return self.policy_resolver.resolve(
                    tenant_id=request.tenant_id,
                    app_id=request.app_id,
                    agent_id=request.agent_id,
                    env=request.env
                )
            else:
                raise ValueError("No policy provided and no resolver configured")
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Policy resolution failed: {e}")
            # Fail closed: block if policy can't be resolved
            return EvaluationResponse(
                request_id=request.request_id,
                decision=Decision.BLOCK,
                risk_score=100,
                policy_version="error",
                rule_hits=[],
                user_message="Request blocked: policy resolution failed.",
                developer_message=f"Policy resolution error: {e}",
            )

    def _build_response(
        self,
        request: EvaluationRequest,
        result: DetectorResult,
        policy_version: str
    ) -> EvaluationResponse:
        return EvaluationResponse(
            request_id=request.request_id,
            decision=result.decision or Decision.ALLOW,
            risk_score=result.risk_score,
            policy_version=policy_version,
            rule_hits=result.rule_hits,
            sanitized_text=result.sanitized_text,
            user_message=result.user_message,
            developer_message=result.developer_message
        )

    def _log_evaluation(
        self,
        request: EvaluationRequest,
        response: EvaluationResponse,
        policy_version: str
    ) -> None:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request.request_id,
            "tenant_id": request.tenant_id,
            "app_id": request.app_id,
            "agent_id": request.agent_id,
            "env": request.env,
            "decision": response.decision.value,
            "risk_score": response.risk_score,
            "policy_version": policy_version,
            "rule_hit_ids": [hit.rule_id for hit in response.rule_hits],
        }
        if self.log_redacted_text and response.sanitized_text:
            log_data["sanitized_text"] = response.sanitized_text
        logger.info(json.dumps(log_data))

    def _emit_telemetry(self, request, response: EvaluationResponse, latency_ms: int, context: str = "input"):
        """Fire telemetry callback if configured."""
        if self.on_evaluation:
            try:
                self.on_evaluation({
                    "request_id": request.request_id,
                    "tenant_id": request.tenant_id,
                    "app_id": request.app_id,
                    "agent_id": request.agent_id,
                    "env": request.env,
                    "context": context,
                    "decision": response.decision.value,
                    "risk_score": response.risk_score,
                    "policy_version": getattr(response, "policy_version", None),
                    "rule_hit_ids": [h.rule_id for h in response.rule_hits],
                    "latency_ms": latency_ms,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "trace_id": getattr(response, "trace_id", None),
                    "session_id": getattr(request, "session_id", None),
                    "span_id": getattr(request, "span_id", None),
                    "detector_results": [
                        dt.model_dump() for dt in (response.detector_results or [])
                    ],
                    "quality_scores": (
                        response.quality.model_dump() if response.quality else None
                    ),
                })
            except Exception:
                logger.debug("Telemetry callback failed", exc_info=True)

    def _record_metrics(self, context: str, response: EvaluationResponse,
                        duration_seconds: float) -> None:
        """Record evaluation metrics to the Prometheus collector (best-effort).

        Emits four metric families:

        - ``guardrails_evaluations_total{context, decision}`` — counter
        - ``guardrails_evaluation_duration_seconds{context}`` — histogram
        - ``guardrails_detector_hits_total{detector_name, action}`` — counter
        - ``guardrails_detector_latency_seconds{detector_name}`` — histogram
          (per-detector p99 latency is the most useful debug signal when
          one detector is dragging the eval loop down)
        - ``guardrails_risk_score{context}`` — histogram, useful for
          decision-threshold tuning over time
        """
        if _metrics is None:
            return
        try:
            decision = response.decision.value if response.decision else "ALLOW"
            _metrics.counter_inc(
                "guardrails_evaluations_total",
                labels={"context": context, "decision": decision},
            )
            _metrics.histogram_observe(
                "guardrails_evaluation_duration_seconds",
                duration_seconds,
                labels={"context": context},
            )
            if response.risk_score is not None:
                _metrics.histogram_observe(
                    "guardrails_risk_score",
                    float(response.risk_score),
                    labels={"context": context},
                )
            for hit in response.rule_hits:
                _metrics.counter_inc(
                    "guardrails_detector_hits_total",
                    labels={"detector_name": hit.rule_id, "action": decision},
                )
            # Per-detector latency from the orchestrator results, if present.
            for dr in (response.detector_results or []):
                if getattr(dr, "latency_ms", None) is None:
                    continue
                _metrics.histogram_observe(
                    "guardrails_detector_latency_seconds",
                    dr.latency_ms / 1000.0,
                    labels={"detector_name": dr.detector_name or "unknown"},
                )
        except Exception:
            logger.debug("Metrics recording failed", exc_info=True)
