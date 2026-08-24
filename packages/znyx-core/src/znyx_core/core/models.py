from enum import Enum
from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field, field_validator


class Decision(str, Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    REDACT = "REDACT"
    WARN = "WARN"
    TRANSFORM = "TRANSFORM"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ExecutionMode(str, Enum):
    """The six detector execution modes. Defined here (the dependency-free
    core models module) so the policy schema, the backend, and LayerResult all share
    one definition."""
    local_deterministic = "local_deterministic"
    local_ml = "local_ml"
    local_embedding = "local_embedding"
    local_llm = "local_llm"
    remote_llm = "remote_llm"
    remote_api = "remote_api"


_EXECUTION_MODE_VALUES = frozenset(m.value for m in ExecutionMode)


def _ensure_unit_range_scores(v: Optional[Dict[str, float]], field: str = "label_scores"):
    """Validate that every score in a label→prob mapping is within [0,1]. Shared by
    DetectorResult / LayerResult / DetectorTimingResult so trace artifacts can't persist
    out-of-range probabilities."""
    if v is not None:
        for label, score in v.items():
            if not (0.0 <= score <= 1.0):
                raise ValueError(f"{field}[{label!r}]={score} out of range [0,1]")
    return v


def _ensure_valid_execution_mode(v: Optional[str]) -> Optional[str]:
    """Validate an execution_mode string against ExecutionMode. Shared by the
    per-layer LayerResult and the final selected-mode scalar on DetectorResult /
    DetectorTimingResult (which the backend persists into the trace)."""
    if v is not None and v not in _EXECUTION_MODE_VALUES:
        raise ValueError(f"execution_mode {v!r} is not a valid ExecutionMode")
    return v


class RuleHit(BaseModel):
    rule_id: str = Field(..., description="Identifier for the rule that was triggered")
    severity: Severity = Field(..., description="Severity level of the rule violation")
    message: str = Field(..., description="Developer-facing message explaining the rule hit")


class LayerResult(BaseModel):
    """One execution-mode attempt within a single detector.

    The escalation appends one entry per backend it runs (deterministic → ml → llm)
    and flags the ``selected`` (final) one. Persisting every attempt is what makes the
    per-layer decision-divergence metric and the trace layer-comparison UI
    computable directly from a stored trace, with no re-evaluation. All fields optional
    so a deterministic-only detector simply never produces a LayerResult.
    """
    execution_mode: str = Field(..., description="local_deterministic|local_ml|local_embedding|local_llm|remote_llm|remote_api")
    decision: Optional[str] = None
    native_score: Optional[float] = Field(default=None, description="The backend's raw score (scale varies by mode)")
    normalized_score: Optional[float] = Field(default=None, description="native_score mapped onto the common 0..100 risk scale")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    calibrated_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    label_scores: Optional[Dict[str, float]] = None
    category: Optional[str] = None
    threshold: Optional[float] = None
    model_id: Optional[str] = None
    model_version: Optional[str] = None
    rubric_version: Optional[str] = None
    latency_ms: Optional[int] = None
    token_count: Optional[int] = None
    cost_usd: Optional[float] = None
    rationale: Optional[str] = None
    evidence_spans: Optional[List[Dict[str, Any]]] = None
    external_egress: bool = False
    fallback_reason: Optional[str] = None
    selected: bool = False

    @field_validator("execution_mode")
    @classmethod
    def _valid_execution_mode(cls, v: str) -> str:
        return _ensure_valid_execution_mode(v)

    @field_validator("decision")
    @classmethod
    def _valid_decision(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in {d.value for d in Decision}:
            raise ValueError(f"decision {v!r} is not a valid Decision")
        return v

    @field_validator("label_scores")
    @classmethod
    def _label_scores_in_unit_range(cls, v):
        return _ensure_unit_range_scores(v)


class EvaluationRequest(BaseModel):
    request_id: str = Field(..., description="Unique identifier for this request")
    tenant_id: str = Field(..., description="Tenant identifier")
    app_id: str = Field(..., description="Application identifier")
    agent_id: str = Field(default="default", description="Agent identifier")
    env: str = Field(default="prod", description="Environment (prod, staging, dev)")
    text: str = Field(..., description="Text to evaluate")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata")
    trace_id: Optional[str] = Field(default=None, description="Distributed trace ID for correlation")
    session_id: Optional[str] = Field(default=None, description="Session/conversation ID for grouping")
    span_id: Optional[str] = Field(default=None, description="Span ID within a trace")


class ToolEvaluationRequest(BaseModel):
    """Request model for tool governance evaluation"""
    request_id: str = Field(..., description="Unique identifier for this request")
    tenant_id: str = Field(..., description="Tenant identifier")
    app_id: str = Field(..., description="Application identifier")
    agent_id: str = Field(default="default", description="Agent identifier")
    env: str = Field(default="prod", description="Environment (prod, staging, dev)")
    tool_name: str = Field(..., description="Name of the tool being invoked")
    tool_args: Dict[str, Any] = Field(..., description="Tool arguments (arbitrary JSON)")
    tool_result: Optional[str] = Field(
        default=None,
        description="Optional tool-result text re-entering context. When present, it is "
                    "scanned for prompt injection (tool_output_injection, LLM01).",
    )
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata")
    trace_id: Optional[str] = Field(default=None, description="Distributed trace ID for correlation")
    session_id: Optional[str] = Field(default=None, description="Session/conversation ID for grouping")
    span_id: Optional[str] = Field(default=None, description="Span ID within a trace")


class Stage(str, Enum):
    """Per-request evaluation stages. The orchestrator filters detectors by
    the active stage; new stages route through the generalized dispatcher instead of
    being treated as output."""
    input = "input"
    output = "output"
    retrieval = "retrieval"          # retrieved RAG chunks before they enter context
    tool = "tool"                    # tool-result text before it re-enters context
    agent_plan = "agent_plan"        # a pre-execution multi-step plan
    agent_loop = "agent_loop"        # a single agent-loop iteration (budgets/depth)
    memory_write = "memory_write"    # text being written to agent memory


class HookStage(str, Enum):
    """Lifecycle hooks — evaluated at registration/publish time, NOT per message."""
    tool_registration = "tool_registration"   # MCP/manifest scan at tool registration
    policy_publish = "policy_publish"          # contradiction analysis at publish


class _ScopedRequest(BaseModel):
    """Common scope/trace fields shared by the per-stage request models."""
    request_id: str
    tenant_id: str
    app_id: str
    agent_id: str = "default"
    env: str = "prod"
    metadata: Optional[Dict[str, Any]] = None
    trace_id: Optional[str] = None
    session_id: Optional[str] = None
    span_id: Optional[str] = None

    def to_evaluation_text(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError


class RetrievalChunk(BaseModel):
    content: str = Field(..., description="The retrieved chunk text")
    source_id: Optional[str] = Field(default=None, description="Identifier of the source document")
    score: Optional[float] = Field(default=None, description="Retriever relevance score")
    # Ownership. ``tenant_scope_assertion`` (LLM09) cannot prove a chunk belongs to the
    # caller without it, and an untagged chunk is a finding rather than an assumption,
    # so this is a first-class field and not something buried in free metadata.
    tenant_id: Optional[str] = Field(
        default=None, description="Tenant the chunk belongs to, as recorded in the index")
    # Whether ``score`` counts up (similarity) or down (distance). Retrievers disagree,
    # and reading a distance as a similarity inverts every ranking check built on it.
    score_kind: Optional[Literal["similarity", "distance"]] = Field(
        default=None,
        description="Whether a higher score is better ('similarity') or worse ('distance')")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Vector-store metadata carried alongside the chunk")


class RetrievalEvaluationRequest(_ScopedRequest):
    """Retrieval-stage request: the RAG chunks about to enter the model context."""
    chunks: List[RetrievalChunk] = Field(default_factory=list)
    # Reported by the caller: was the tenant filter part of the index query, or applied
    # to the results afterwards? Left None when the caller does not know, because an
    # unreported pipeline is not evidence of a bad one.
    scope_enforced_in_query: Optional[bool] = Field(
        default=None,
        description="True when tenant scoping was applied inside the index query")

    def to_evaluation_text(self) -> str:
        return "\n\n".join(c.content for c in self.chunks)


class AgentPlanEvaluationRequest(_ScopedRequest):
    """agent_plan-stage request: a proposed multi-step plan (free JSON)."""
    plan: Any = Field(..., description="The proposed plan (list of steps / structured JSON)")

    def to_evaluation_text(self) -> str:
        import json as _json
        return _json.dumps(self.plan, ensure_ascii=False, default=str)


class AgentStepEvaluationRequest(_ScopedRequest):
    """agent_loop-stage request: one iteration of an agent loop (budget/depth checks)."""
    action: str = Field(default="", description="The action/tool the agent intends to take this step")
    iteration: int = Field(default=0, ge=0)
    max_iterations: Optional[int] = Field(default=None, ge=0)

    def to_evaluation_text(self) -> str:
        return self.action


class MemoryWriteEvaluationRequest(_ScopedRequest):
    """memory_write-stage request: text being persisted to agent memory."""
    memory_key: Optional[str] = None
    memory_value: str = Field(..., description="The value being written to memory")

    def to_evaluation_text(self) -> str:
        return self.memory_value


class DetectorTimingResult(BaseModel):
    """Per-detector timing and result for observability traces."""
    detector_name: str
    decision: Optional[str] = None
    risk_score: int = 0
    latency_ms: int = 0
    rule_hits: List[RuleHit] = Field(default_factory=list)
    transformed: bool = False
    # Model-backed enrichment — final/aggregated scalars + the per-layer attempts.
    # All additive with safe defaults so existing SDK consumers are unaffected; the
    # waterfall/layer-comparison UI renders from layer_results without re-evaluation.
    confidence: Optional[float] = None
    calibrated_score: Optional[float] = None
    label_scores: Optional[Dict[str, float]] = None
    model_version: Optional[str] = None
    execution_mode: Optional[str] = None
    fallback_path: Optional[str] = None
    external_egress: bool = False
    threshold: Optional[float] = None
    layer_results: List[LayerResult] = Field(default_factory=list)

    @field_validator("label_scores")
    @classmethod
    def _label_scores_in_unit_range(cls, v):
        return _ensure_unit_range_scores(v)

    @field_validator("execution_mode")
    @classmethod
    def _valid_execution_mode(cls, v):
        return _ensure_valid_execution_mode(v)


class QualityScore(BaseModel):
    """A single quality dimension score."""
    metric: str
    score: float = Field(..., ge=0.0, le=1.0)
    details: str = ""
    sub_scores: Optional[Dict[str, float]] = None
    # per-claim grounding evidence (NLI-backed groundedness) — [{claim, source_id, support}]
    evidence_spans: Optional[List[Dict[str, Any]]] = None
    # optional LLM-judge evaluator fields (additive/back-compat; None for deterministic scorers).
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    label: Optional[str] = None
    rationale: Optional[str] = None
    judge_model: Optional[str] = None
    rubric_version: Optional[str] = None
    latency_ms: Optional[int] = None


class JudgeVerdict(BaseModel):
    """Structured output of an LLM-judge DETECTOR call.

    A judge returns this for detector-style evaluation (block/allow with a risk score and
    rationale). All fields beyond ``decision`` are optional so a terse judge reply still
    parses. ``decision`` is validated against the canonical Decision set; an unknown value
    is rejected (the caller then treats the reply as malformed and falls back)."""
    decision: Optional[str] = None
    risk_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    category: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    rationale: Optional[str] = None
    evidence_spans: Optional[List[Dict[str, Any]]] = None

    @field_validator("decision")
    @classmethod
    def _valid_decision(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in {d.value for d in Decision}:
            raise ValueError(f"decision {v!r} is not a valid Decision")
        return v


class EvaluatorVerdict(BaseModel):
    """Structured output of an LLM-judge EVALUATOR call — a quality
    metric score with rationale + judge provenance. Maps onto a QualityScore."""
    metric: str
    score: float = Field(..., ge=0.0, le=1.0)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    label: Optional[str] = None
    rationale: Optional[str] = None
    evidence_spans: Optional[List[Dict[str, Any]]] = None
    judge_model: Optional[str] = None
    rubric_version: Optional[str] = None
    latency_ms: Optional[int] = None


class QualityReport(BaseModel):
    """Aggregate quality report across all scored dimensions."""
    scores: List[QualityScore] = Field(default_factory=list)
    overall_score: float = 0.0
    evaluated_at: str = ""

    def get_score(self, metric: str) -> Optional[float]:
        for s in self.scores:
            if s.metric == metric:
                return s.score
        return None


class FieldError(BaseModel):
    """Field-level validation error from structured output contracts."""
    path: str = Field(..., description="JSON path to the field (e.g., /user/name)")
    message: str = Field(..., description="Error description")
    expected: Optional[str] = Field(default=None, description="Expected value/type")
    actual: Optional[str] = Field(default=None, description="Actual value/type")


class RemediationAction(str, Enum):
    """Actions to take when a guardrail detector fires."""
    NOOP = "noop"             # Do nothing (default passthrough)
    REASK = "reask"           # Ask the LLM to try again
    FIX = "fix"               # Apply an automatic fix (e.g., strip offending text)
    FILTER_FIELD = "filter_field"  # Remove specific fields from output
    REFRAIN = "refrain"       # Return empty / canned response instead
    EXCEPTION = "exception"   # Raise an exception to the caller
    CUSTOM = "custom"         # Delegate to a named custom handler
    ASK_HUMAN = "ask_human"   # Queue for human review before proceeding


class RemediationResult(BaseModel):
    """Outcome of a remediation action."""
    action: RemediationAction
    applied: bool = False
    reask_prompt: Optional[str] = None
    fixed_text: Optional[str] = None
    filtered_fields: List[str] = Field(default_factory=list)
    refrain_message: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 0
    max_attempts: int = 1


class EvaluationResponse(BaseModel):
    request_id: str
    decision: Decision
    risk_score: int = Field(..., ge=0, le=100, description="Risk score from 0-100")
    policy_version: str
    rule_hits: List[RuleHit] = Field(default_factory=list)
    sanitized_text: Optional[str] = Field(default=None, description="Sanitized text if REDACT/TRANSFORM")
    sanitized_tool_args: Optional[Dict[str, Any]] = Field(default=None, description="Sanitized tool args (for tool evaluation)")
    user_message: Optional[str] = Field(default=None, description="Safe message to show end-user when blocked")
    developer_message: Optional[str] = Field(default=None, description="Developer-facing explanation")
    latency_ms: Optional[int] = Field(default=None, description="Total evaluation latency in milliseconds")
    trace_id: Optional[str] = Field(default=None, description="Trace ID for distributed tracing correlation")
    session_id: Optional[str] = Field(default=None, description="Session/conversation ID echoed from request")
    span_id: Optional[str] = Field(default=None, description="Span ID within a trace echoed from request")
    detector_results: List[DetectorTimingResult] = Field(default_factory=list, description="Per-detector timing breakdown")
    quality: Optional[QualityReport] = Field(default=None, description="Response quality scores (output context only)")
    field_errors: List[FieldError] = Field(default_factory=list, description="Field-level errors from structured output validation")
    remediation: Optional[RemediationResult] = Field(default=None, description="Remediation action applied after detector decision")
    pending_review_id: Optional[str] = Field(default=None, description="Human review queue ID if ask_human remediation was triggered")


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"


class DetectorResult(BaseModel):
    """Internal model for detector results"""
    rule_hits: List[RuleHit] = Field(default_factory=list)
    risk_score: int = Field(default=0, ge=0, le=100)
    decision: Optional[Decision] = None
    sanitized_text: Optional[str] = None
    user_message: Optional[str] = None
    developer_message: Optional[str] = None
    field_errors: List[FieldError] = Field(default_factory=list)
    # Model-backed detection contract. All optional with safe defaults so
    # the 19 deterministic built-ins and every SDK keep working untouched; populated
    # by the RemoteDetector when an inference service / judge returns them.
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Model confidence 0..1 in the decision")
    calibrated_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Calibrated probability 0..1 (post-calibration)")
    label_scores: Optional[Dict[str, float]] = Field(default=None, description="Per-label probabilities (0..1) from a classifier")
    model_version: Optional[str] = Field(default=None, description="model_id@revision that produced this result")
    # Final/aggregated fields + per-layer attempts. The escalation populates
    # execution_mode/fallback_path/external_egress and appends layer_results.
    execution_mode: Optional[str] = Field(default=None, description="The mode that produced the selected result")
    fallback_path: Optional[str] = Field(default=None, description="Why a fallback fired (e.g. no_external_calls, residency_denied, timeout)")
    external_egress: bool = Field(default=False, description="True if any content left the trust boundary for this detector")
    threshold: Optional[float] = Field(default=None, description="Decision threshold applied to the selected layer")
    calibration_dataset_id: Optional[str] = Field(default=None, description="Dataset id the calibration was fit on")
    layer_results: List[LayerResult] = Field(default_factory=list, description="One entry per execution-mode attempt (escalation)")

    @field_validator("label_scores")
    @classmethod
    def _label_scores_in_unit_range(cls, v):
        return _ensure_unit_range_scores(v)

    @field_validator("execution_mode")
    @classmethod
    def _valid_execution_mode(cls, v):
        return _ensure_valid_execution_mode(v)
