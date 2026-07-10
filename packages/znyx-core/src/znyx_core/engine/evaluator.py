import asyncio
import functools
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timezone

from znyx_core.core.models import EvaluationRequest, EvaluationResponse, DetectorResult, Decision, ToolEvaluationRequest, Stage
from znyx_core.core.decision import DecisionAggregator
from znyx_core.detectors.plugin import PluginRegistry
from znyx_core.engine.detector_registry import DetectorRegistry, default_registry
from znyx_core.engine.orchestrator import DetectorOrchestrator
from znyx_core.engine.quality_scorer import QualityScorer
from znyx_core.engine.remediation import RemediationHandler

logger = logging.getLogger(__name__)

# Single-worker executor for the synchronous detector pipeline. Running it OFF
# the event loop keeps the loop free for I/O / other requests; running it on a
# SINGLE worker SERIALIZES detector execution. Serialization is required because
# the stateful detectors (abuse rate-limit buckets, jailbreak conversation
# history) are long-lived shared instances with NO internal locking — a
# multi-worker pool would race (corrupt the buckets / mutate an OrderedDict mid
# iteration). The GIL means >1 worker wouldn't speed up the pure-Python regex
# work anyway, so a single worker costs nothing and is safe. ALL run_detectors
# callers must go through this executor so two never run concurrently.
_DETECTOR_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="detector")

# Optional metrics collector - graceful no-op if not available
try:
    from znyx_core.middleware.metrics import MetricsCollector
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
                 registry: Optional[DetectorRegistry] = None,
                 egress_sink: Optional[Callable] = None):
        self.policy_resolver = policy_resolver
        self.log_redacted_text = log_redacted_text
        self.on_evaluation = on_evaluation  # telemetry callback
        self.plugin_registry = plugin_registry

        self._registry = registry or default_registry
        # pass-through egress audit sink to the orchestrator (escalation engine) AND
        # the quality NLI scorer, so every boundary-crossing call is gated + audited.
        self._egress_sink = egress_sink
        self._orchestrator = DetectorOrchestrator(
            self._registry, plugin_registry=self.plugin_registry, egress_sink=egress_sink
        )
        self._remediation = RemediationHandler()

    async def _run_detectors(self, *args, **kwargs):
        """Run the synchronous orchestrator off the event loop on the single
        shared detector worker (see _DETECTOR_EXECUTOR): serialized + race-free.
        In the worker thread there is no running loop, so RemoteDetector/judge
        take their asyncio.run path."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _DETECTOR_EXECUTOR,
            functools.partial(self._orchestrator.run_detectors, *args, **kwargs),
        )

    async def evaluate(
        self,
        request: EvaluationRequest,
        context: str = "input",
        policy: Optional[Dict[str, Any]] = None,
        db=None,
        *,
        judge_ctx=None,
    ) -> EvaluationResponse:
        """
        Evaluate text against all guardrails.

        Args:
            request: The evaluation request
            context: "input" or "output"
            policy: Pre-resolved policy dict. If None, resolved via policy_resolver or DB.
            db: Optional DB session for legacy DB-mode resolution
            judge_ctx: Optional JudgeExecutionContext. When a composition root injects
                it, escalation judges run multi-judge consensus and quality judges record
                audit + honour deny-of-wallet budgets. None = judges use the plain transport
                with no CP enforcement (zero behaviour change for current callers).
        """
        start_time = datetime.now(timezone.utc)

        # Resolve policy: prefer explicit > DB > YAML resolver
        policy = await self._resolve_policy(request, policy, db)
        if isinstance(policy, EvaluationResponse):
            return policy  # error response from failed resolution

        policy_version = policy.get('policy_version', 'unknown')

        # Run detector pipeline. Generalized stage dispatch: route by the
        # actual stage (input/output/retrieval/tool/agent_plan/agent_loop/memory_write)
        # rather than the old input-vs-output branch, so a new stage's detectors are
        # selected correctly instead of the stage being treated as output.
        # Offload the synchronous detector pipeline to the single shared worker
        # (frees the event loop without racing the long-lived stateful detectors).
        orch = await self._run_detectors(
            request.text, policy, request,
            context=context, judge_ctx=judge_ctx,
        )

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
            # Optional NLI-backed groundedness via the inference sidecar; None when
            # the quality config doesn't opt in → deterministic token-overlap fallback.
            # The scorer routes through the egress gate (no_external_calls / allowlist
            # / residency / redact + fail-closed audit) via the same sink as detectors.
            from znyx_core.engine.quality.nli_client import nli_scorer_from_config
            nli_scorer = nli_scorer_from_config(
                quality_config,
                runtime_policy=policy.get("runtime_policy"),
                policy=policy,
                egress_sink=self._egress_sink,
                request=request,
            )
            # LLM-judge evaluators (opt-in via judge_mode + a `judge` config block).
            # Each routes through the same egress sink; rubrics default to the built-in
            # set (a control-plane caller can pass the org's registered rubrics). Falls back
            # to the deterministic scorer per-metric when a judge call is denied/unavailable.
            judge_evaluators = None
            if quality_config.get("judge_mode") and isinstance(quality_config.get("judge"), dict):
                from znyx_core.engine.quality.judge_evaluator import build_judge_evaluators
                # an injected judge_ctx supplies the org's registered rubrics, the
                # deny-of-wallet budget check, the audit sink, and (runtime/test) the
                # provider caller. Absent it, the evaluators run with built-in rubrics and
                # no CP enforcement (unchanged).
                _jc = judge_ctx
                judge_evaluators = build_judge_evaluators(
                    quality_config["judge"], egress_sink=self._egress_sink, request=request,
                    runtime_policy=policy.get("runtime_policy"),
                    rubrics=(_jc.rubrics if _jc else None),
                    rubric_versions=(_jc.rubric_versions if _jc else None),
                    caller=(_jc.provider_caller if _jc else None),
                    audit_sink=(_jc.audit_sink if _jc else None),
                    budget_check=(_jc.budget_check if _jc else None),
                )
            scorer = QualityScorer(quality_config, nli_scorer=nli_scorer,
                                   judge_evaluators=judge_evaluators)
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
                            db=None, *, judge_ctx=None) -> EvaluationResponse:
        """Evaluate tool invocation against governance policies."""
        start_time = datetime.now(timezone.utc)

        if policy is None:
            if db is not None:
                from znyx_core.policy.db_resolver_registry import get_db_policy_resolver
                resolver = get_db_policy_resolver(db, cache_ttl=60)
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
        detector_timings = []

        tools_config = policy.get('tools', {})
        if tools_config.get('enabled', True):
            tool_detector = self._registry.get_or_create("tools", tools_config)
            tool_result = tool_detector.detect(request.tool_name, request.tool_args)
            results.append(tool_result)

        # (LLM01): if the caller supplied the tool RESULT text, scan it through the
        # generalized pipeline at context="tool" (tool_output_injection + any enabled
        # content detectors), so it gets the SAME treatment as every other stage — per-
        # detector timings, strategy escalation, and scorecard-gate handling — rather than
        # an ad-hoc one-off detector call. `tools` governance is scoped to input/output so
        # it does not re-run here.
        tool_result_text = getattr(request, "tool_result", None)
        if tool_result_text:
            stage_req = EvaluationRequest(
                request_id=request.request_id, tenant_id=request.tenant_id,
                app_id=request.app_id, agent_id=request.agent_id, env=request.env,
                text=tool_result_text, metadata=request.metadata,
                trace_id=getattr(request, "trace_id", None),
                session_id=getattr(request, "session_id", None),
                span_id=getattr(request, "span_id", None),
            )
            orch = await self._run_detectors(tool_result_text, policy, stage_req,
                                             context="tool", judge_ctx=judge_ctx)
            results.extend(orch.results)
            detector_timings = orch.detector_timings

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
        response.detector_results = detector_timings

        # Apply on_fail remediation (e.g. tool_output_injection.on_fail: ask_human),
        # consistent with the input/output evaluate path.
        response = self._remediation.apply(response, policy, results)

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

    async def evaluate_stage(self, scoped, stage: str,
                             policy: Optional[Dict[str, Any]] = None,
                             db=None, *, judge_ctx=None) -> EvaluationResponse:
        """Evaluate a typed per-stage request (new stages: retrieval / agent_plan /
        agent_loop / memory_write).

        Flattens the typed ``_ScopedRequest`` into a text ``EvaluationRequest`` — folding
        the stage-specific structured payload into metadata so detectors that need more
        than flat text (e.g. ``unbounded_consumption`` budget signals) can read it — then
        dispatches through the generalized stage pipeline."""
        if stage not in {s.value for s in Stage}:
            raise ValueError(f"Unknown evaluation stage '{stage}'")
        eval_request = self._to_eval_request(scoped, stage)
        return await self.evaluate(eval_request, context=stage, policy=policy, db=db,
                                   judge_ctx=judge_ctx)

    @staticmethod
    def _to_eval_request(scoped, stage: str) -> EvaluationRequest:
        from znyx_core.core.models import (
            AgentPlanEvaluationRequest, AgentStepEvaluationRequest,
            MemoryWriteEvaluationRequest, RetrievalEvaluationRequest,
        )
        metadata = dict(scoped.metadata or {})
        if isinstance(scoped, AgentStepEvaluationRequest):
            metadata["agent_step"] = {
                **(metadata.get("agent_step") or {}),
                "action": scoped.action,
                "iteration": scoped.iteration,
                "max_iterations": scoped.max_iterations,
            }
        elif isinstance(scoped, RetrievalEvaluationRequest):
            metadata["retrieval_chunks"] = [c.model_dump() for c in scoped.chunks]
        elif isinstance(scoped, AgentPlanEvaluationRequest):
            metadata["agent_plan"] = scoped.plan
        elif isinstance(scoped, MemoryWriteEvaluationRequest) and scoped.memory_key is not None:
            metadata["memory_key"] = scoped.memory_key
        return EvaluationRequest(
            request_id=scoped.request_id,
            tenant_id=scoped.tenant_id,
            app_id=scoped.app_id,
            agent_id=scoped.agent_id,
            env=scoped.env,
            text=scoped.to_evaluation_text(),
            metadata=metadata or None,
            trace_id=scoped.trace_id,
            session_id=scoped.session_id,
            span_id=scoped.span_id,
        )

    # -- private helpers ----------------------------------------------------

    async def resolve_policy(self, request, db=None):
        """Public: resolve a request's policy, or return a fail-closed BLOCK
        EvaluationResponse if resolution fails. Lets a composition root (CP eval route)
        resolve the policy ONCE — to detect judge usage and materialise judge secrets —
        then pass the same dict back into ``evaluate(policy=...)`` (no double resolution)."""
        return await self._resolve_policy(request, None, db)

    async def _resolve_policy(self, request, policy, db):
        """Resolve policy dict, returning an EvaluationResponse on failure."""
        if policy is not None:
            return policy
        try:
            if db is not None:
                from znyx_core.policy.db_resolver_registry import get_db_policy_resolver
                resolver = get_db_policy_resolver(db, cache_ttl=60)
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
