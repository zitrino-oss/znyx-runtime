from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


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


class RuleHit(BaseModel):
    rule_id: str = Field(..., description="Identifier for the rule that was triggered")
    severity: Severity = Field(..., description="Severity level of the rule violation")
    message: str = Field(..., description="Developer-facing message explaining the rule hit")


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
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata")
    trace_id: Optional[str] = Field(default=None, description="Distributed trace ID for correlation")
    session_id: Optional[str] = Field(default=None, description="Session/conversation ID for grouping")
    span_id: Optional[str] = Field(default=None, description="Span ID within a trace")


class DetectorTimingResult(BaseModel):
    """Per-detector timing and result for observability traces."""
    detector_name: str
    decision: Optional[str] = None
    risk_score: int = 0
    latency_ms: int = 0
    rule_hits: List[RuleHit] = Field(default_factory=list)
    transformed: bool = False


class QualityScore(BaseModel):
    """A single quality dimension score."""
    metric: str
    score: float = Field(..., ge=0.0, le=1.0)
    details: str = ""
    sub_scores: Optional[Dict[str, float]] = None


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
