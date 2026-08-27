"""
Policy schema validation using Pydantic.

Validates policy configuration dicts at load time. Logs warnings for
invalid configs but never crashes -- callers always get a PolicySchema
back (possibly with defaults filled in).
"""
import logging
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from znyx_core.core.models import ExecutionMode, Stage

logger = logging.getLogger(__name__)


# ── Per-detector strategy/egress config ──────────────────────────────
# These are TYPED (and the nested blocks forbid unknown keys) so a malformed
# strategy/egress config is a strict-validation BLOCKER instead of being silently
# accepted under the base DetectorConfig's extra="allow". ExecutionMode is the
# shared taxonomy from core.models.

_FALLBACK_VALUES = Literal["fail_open", "fail_closed", "fallback_to_deterministic"]


class EscalateWhen(BaseModel):
    """Predicates deciding whether to escalate to the next backend in `order`."""
    model_config = ConfigDict(extra="forbid")
    deterministic_score_between: Optional[List[int]] = None
    ml_confidence_below: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("deterministic_score_between")
    @classmethod
    def _valid_band(cls, v: Optional[List[int]]) -> Optional[List[int]]:
        if v is None:
            return v
        if len(v) != 2:
            raise ValueError("deterministic_score_between must be a [low, high] pair")
        low, high = v
        if not (0 <= low <= 100 and 0 <= high <= 100):
            raise ValueError("deterministic_score_between values must be within 0..100")
        if low > high:
            raise ValueError("deterministic_score_between must have low <= high")
        return v


class StrategyConfig(BaseModel):
    """Per-detector multi-mode execution strategy (the backend resolves it)."""
    model_config = ConfigDict(extra="forbid")
    order: Optional[List[ExecutionMode]] = None
    escalate_when: Optional[EscalateWhen] = None
    fallback: Optional[_FALLBACK_VALUES] = None
    timeout_ms: Optional[int] = Field(default=None, ge=0)
    # Additive mode: the ML layer AUGMENTS the deterministic result (worst-of decision +
    # merged rule_hits + max risk) instead of REPLACING it. For detectors whose ML layer
    # catches things the deterministic layer can't (pii_ner unstructured PII, language)
    # while the deterministic decision (e.g. regex PII redaction) must never be lost.
    additive: Optional[bool] = None

    @field_validator("order")
    @classmethod
    def _order_nonempty(cls, v: Optional[List[ExecutionMode]]) -> Optional[List[ExecutionMode]]:
        if v is not None and len(v) == 0:
            raise ValueError("strategy.order must be a non-empty list if provided")
        return v


class RedactBeforeEgress(BaseModel):
    """What to scrub before any boundary-crossing call."""
    model_config = ConfigDict(extra="forbid")
    pii: bool = True
    secrets: bool = True
    custom: Optional[List[str]] = None


class BackendModeConfig(BaseModel):
    """Per-execution-mode backend config: how the escalation path reaches a
    given mode's model/judge/vendor. endpoint_url is where the (extended) RemoteDetector
    transport posts; model pins feed the inference model registry; auth_value is a
    secret:// sentinel, never a raw key."""
    model_config = ConfigDict(extra="forbid")
    endpoint_url: Optional[str] = None
    model_id: Optional[str] = None
    revision: Optional[str] = None
    sha256: Optional[str] = None
    task: Optional[str] = None
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    provider: Optional[str] = None         # remote_api adapter (openai_moderation, …)
    judge_id: Optional[str] = None         # local_llm / remote_llm judge id
    model: Optional[str] = None            # judge model name (e.g. gpt-4o, llama3.2)
    members: Optional[int] = Field(default=None, ge=1)  # consensus member count (K)
    method: Optional[str] = None           # voting method: majority | weighted
    judge: Optional[bool] = None           # marks backend as a judge call
    timeout_ms: Optional[int] = Field(default=None, ge=0)
    auth_type: Optional[str] = None
    auth_value: Optional[str] = None       # secret://<id> sentinel
    region: Optional[str] = None           # destination region for residency checks
    # egress semantics: a co-located/in-boundary sidecar is NOT egress. Defaults
    # True for local_* sidecar modes (self-host posture); set False for a hosted/
    # network inference endpoint. Ignored for remote_llm/remote_api (always egress).
    in_boundary: Optional[bool] = None
    # remote_api vendor-adapter settings: the action when the vendor flags, plus a
    # free-form provider_config the adapter consumes (Azure api-version/severity,
    # Bedrock guardrail id/version + AWS secret/session credentials).
    action: Optional[str] = None
    provider_config: Optional[Dict[str, Any]] = None
    # local_llm / remote_llm judge backend: the judge flag + model + multi-judge consensus
    # (members/method). These are read by build_escalation_judge_caller (build_strategy
    # itself drops them), so they must validate at publish time.
    judge: Optional[bool] = None
    model: Optional[str] = None
    members: Optional[int] = Field(default=None, ge=1)
    method: Optional[Literal["majority", "weighted"]] = None
    rubric_name: Optional[str] = None      # reference into the judge rubric registry


class DetectorBackendsConfig(BaseModel):
    """Per-mode backend blocks, one optional sub-block per ExecutionMode. The
    strategy's `order` selects which of these run; this block says how to reach each."""
    model_config = ConfigDict(extra="forbid")
    local_deterministic: Optional[BackendModeConfig] = None
    local_ml: Optional[BackendModeConfig] = None
    local_embedding: Optional[BackendModeConfig] = None
    local_llm: Optional[BackendModeConfig] = None
    remote_llm: Optional[BackendModeConfig] = None
    remote_api: Optional[BackendModeConfig] = None


class RuntimePolicyConfig(BaseModel):
    """Top-level runtime/egress policy (lint now; enforced at egress).

    extra="allow" so this can grow without breaking forward-compat, and so it is
    accepted by strict validation today rather than flagged as an unknown key."""
    model_config = ConfigDict(extra="allow")
    no_external_calls: bool = False
    max_model_detector_latency_ms: Optional[int] = Field(default=None, ge=0)
    allowed_regions: Optional[List[str]] = None
    data_residency: Optional[str] = None


# ── Detector config models ──────────────────────────────────────────────────

class DetectorConfig(BaseModel):
    """Base detector configuration shared by all detectors."""
    model_config = ConfigDict(extra="allow")
    enabled: bool = False
    action: Optional[str] = None
    # Restrict this detector to specific pipeline stages. None = use orchestrator
    # default. Accepts the full per-request stage taxonomy: input, output,
    # retrieval, tool, agent_plan, agent_loop, memory_write.
    stages: Optional[List[Stage]] = None
    # Model-backed execution strategy + egress controls. Typed so a
    # malformed block (e.g. strategy.timeout_ms="bad") is rejected at publish time
    # rather than silently kept by extra="allow".
    strategy: Optional[StrategyConfig] = None
    backends: Optional[DetectorBackendsConfig] = None
    timeout_ms: Optional[int] = Field(default=None, ge=0)
    fallback: Optional[_FALLBACK_VALUES] = None
    egress_allowlist: Optional[List[str]] = None
    redact_before_egress: Optional[RedactBeforeEgress] = None

    @field_validator("stages")
    @classmethod
    def stages_must_be_nonempty(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None and len(v) == 0:
            raise ValueError("stages must be a non-empty list if provided")
        return v

    @field_validator("egress_allowlist")
    @classmethod
    def _egress_hosts_nonblank(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            for host in v:
                if not isinstance(host, str) or not host.strip():
                    raise ValueError("egress_allowlist entries must be non-empty host strings")
        return v


class SecretsConfig(DetectorConfig):
    exceptions: Optional[List[str]] = None
    detect_base64: bool = True


class PIIConfig(DetectorConfig):
    action: str = "REDACT"
    redaction_strategy: str = "full"
    skip_private_ips: bool = True
    types: Optional[Dict[str, dict]] = None


class JailbreakConfig(DetectorConfig):
    threshold: int = 60
    detect_encoding: bool = True
    track_conversation: bool = False


class ToxicityConfig(DetectorConfig):
    action: str = "WARN"
    custom_terms: Optional[Dict[str, str]] = None
    detect_evasion: bool = True
    context_aware: bool = True


class TopicRestrictionConfig(DetectorConfig):
    action: str = "BLOCK"
    blocked_topics: Optional[List[str]] = None
    allowed_topics: Optional[List[str]] = None
    use_synonyms: bool = True
    context_aware: bool = True


class CompetitorConfig(DetectorConfig):
    action: str = "WARN"
    competitors: Optional[List[str]] = None
    fuzzy_matching: bool = True
    fuzzy_threshold: int = 85
    allowlist_contexts: Optional[List[str]] = None
    competitor_aliases: Optional[Dict[str, List[str]]] = None


class ExfiltrationConfig(DetectorConfig):
    block_threshold: int = 40


class SensitiveBusinessDataConfig(DetectorConfig):
    """(LLM02): confidential business-data dictionaries + allowlist."""
    action: str = "WARN"
    categories: Optional[Dict[str, List[str]]] = None
    allowlist: Optional[List[str]] = None
    block_threshold: int = 60
    severity: str = "medium"


class CitationIntegrityConfig(DetectorConfig):
    """(LLM07): validate cited URLs/source-ids + quote spans against grounding."""
    action: Literal["BLOCK", "WARN"] = "WARN"          # only BLOCK/WARN are honoured
    block_threshold: int = Field(default=60, ge=0, le=100)
    require_sources: bool = False
    min_quote_overlap: float = Field(default=0.6, ge=0.0, le=1.0)


class SystemPromptFingerprintEntry(BaseModel):
    """One registered fingerprint delivered into the resolved policy: keyed shingle
    hashes + the shingle size they were built at (≥8, the dictionary-attack floor)."""
    model_config = ConfigDict(extra="allow")
    hashes: List[str] = Field(min_length=1)
    min_shingle_tokens: int = Field(default=8, ge=8)
    # What kind of hidden context this is. 2026's LLM08 spans more than the system
    # prompt, so a hit can name what leaked instead of always saying "system prompt".
    kind: Literal["system_prompt", "tool_schema", "policy_text"] = "system_prompt"


class SystemPromptLeakageConfig(DetectorConfig):
    """(LLM08): match output against registered hidden-context fingerprints by keyed
    shingle-hash overlap (hash-only; no raw text), and against the tool/function schemas
    the request itself carried. ``fingerprint_key`` + the keyed hash sets are delivered
    into the resolved policy from the fingerprint registry."""
    action: Literal["BLOCK", "WARN"] = "BLOCK"         # leaked context is blocked/warned, never redacted
    fingerprint_key: Optional[str] = None              # per-org HMAC pepper (hex)
    fingerprints: Optional[List[SystemPromptFingerprintEntry]] = None
    match_threshold: int = Field(default=2, ge=1)
    # Fingerprint the request's own tool schemas per call. Tool definitions are generated
    # per request rather than registered once, so this is what actually covers them.
    check_tool_schemas: bool = True

    @field_validator("fingerprint_key")
    @classmethod
    def _valid_hex_key(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if len(v) % 2 != 0 or len(v) < 32:   # ≥16 bytes of pepper
            raise ValueError("fingerprint_key must be a hex string of ≥16 bytes (≥32 hex chars)")
        try:
            bytes.fromhex(v)
        except ValueError:
            raise ValueError("fingerprint_key must be a valid hex string")
        return v


class ToolsConfig(DetectorConfig):
    allowed: Optional[List[str]] = None
    denied: Optional[List[str]] = None
    schemas: Optional[Dict[str, dict]] = None
    domain_allowlist: Optional[List[str]] = None
    max_arg_size: int = 10000


class StructureConfig(DetectorConfig):
    expected_format: Optional[str] = None
    json_schema: Optional[dict] = None
    xml_root: Optional[str] = None
    schema_name: Optional[str] = None  # Named schema from control plane


class OutputContractConfig(BaseModel):
    """Structured output contract - validates LLM output against a JSON schema."""
    model_config = ConfigDict(extra="allow")
    enabled: bool = False
    schema_name: Optional[str] = None
    schema_version: Optional[int] = None
    json_schema: Optional[dict] = None
    on_fail: Optional[str] = None  # remediation action: "fix", "reask", "block"
    field_level_errors: bool = True


class QualityScoringConfig(BaseModel):
    """Response quality scoring configuration."""
    model_config = ConfigDict(extra="allow")
    enabled: bool = False
    metrics: Optional[List[str]] = None
    weights: Optional[Dict[str, float]] = None
    thresholds: Optional[Dict[str, float]] = None


class CustomDetectorRef(BaseModel):
    """Reference to a custom detector (remote, regex, webhook, or catalog)."""
    model_config = ConfigDict(extra="allow")
    name: str
    type: Optional[str] = None  # "remote", "regex", "webhook", "catalog"
    config: Optional[Dict[str, Any]] = None
    on_fail: Optional[str] = None
    # Stages this custom detector runs in. None = the default input/output stages only,
    # so a webhook/remote custom detector doesn't start receiving retrieval chunks, agent
    # plans, or memory writes (and egressing them) just because the new stages exist.
    # Opt a custom detector into a new stage explicitly by listing it here.
    stages: Optional[List[Stage]] = None

    @field_validator("stages")
    @classmethod
    def _stages_nonempty(cls, v: Optional[List[Stage]]) -> Optional[List[Stage]]:
        # An empty list is treated as "absent" by the orchestrator (runs on default
        # input/output), so reject it at publish time rather than silently — consistent
        # with DetectorConfig.stages.
        if v is not None and len(v) == 0:
            raise ValueError("stages must be a non-empty list if provided")
        return v


class AbuseConfig(DetectorConfig):
    max_chars_input: int = 100000
    max_chars_output: int = 100000
    max_tool_args_size: int = 50000
    rate_limit_per_minute: int = 60
    prompt_flood_threshold: int = 3
    prompt_flood_window: int = 60


class GibberishConfig(DetectorConfig):
    action: str = "BLOCK"
    entropy_threshold: float = 5.5
    max_special_char_ratio: float = 0.4
    detect_invisible_chars: bool = True
    detect_token_stuffing: bool = True
    repetition_threshold: int = 5


class LanguageConfig(DetectorConfig):
    action: str = "BLOCK"
    allowed_languages: Optional[List[str]] = None
    blocked_languages: Optional[List[str]] = None
    detect_mixed: bool = False
    min_text_length: int = 20


class BiasConfig(DetectorConfig):
    action: str = "WARN"
    protected_attributes: Optional[List[str]] = None
    sensitivity: str = "medium"
    industry_preset: str = "general"


class SentimentConfig(DetectorConfig):
    action: str = "WARN"
    blocked_tones: Optional[List[str]] = None
    min_sentiment_score: float = -0.3


class ComplianceConfig(DetectorConfig):
    action: str = "WARN"
    industry: str = "general"
    ai_disclosure: bool = False
    ai_disclosure_text: Optional[str] = None
    prohibited_claims: Optional[List[str]] = None
    required_disclaimers: Optional[List[str]] = None


class MaliciousURLConfig(DetectorConfig):
    action: str = "WARN"
    block_ip_urls: bool = True
    block_shorteners: bool = True
    check_data_uris: bool = True
    domain_blocklist: Optional[List[str]] = None
    domain_allowlist: Optional[List[str]] = None
    max_subdomain_depth: int = 3


class CopyrightConfig(DetectorConfig):
    action: str = "WARN"
    verbatim_threshold: int = 8
    check_code_licenses: bool = True
    code_license_blocklist: Optional[List[str]] = None
    check_lyrics: bool = True
    check_books: bool = True


class CodeSafetyConfig(DetectorConfig):
    action: str = "WARN"
    languages: Optional[List[str]] = None
    block_unsafe_functions: bool = True
    check_sql_injection: bool = True
    check_xss: bool = True
    check_command_injection: bool = True
    check_path_traversal: bool = True


class HallucinationConfig(DetectorConfig):
    action: str = "WARN"
    method: str = "token_overlap"
    grounding_threshold: float = 0.5
    min_claim_words: int = 3


# ── new-stage / lifecycle gap detectors (OWASP LLM01 / LLM03 / LLM06 / LLM04) ──

class RetrievalChunkInjectionConfig(DetectorConfig):
    """(LLM01): scan retrieved RAG chunks for injection markers (``retrieval`` stage)."""
    action: Literal["BLOCK", "WARN"] = "WARN"
    block_threshold: int = Field(default=50, ge=0, le=100)


class ToolOutputInjectionConfig(DetectorConfig):
    """(LLM01): scan tool-result text re-entering context (``tool`` stage)."""
    action: Literal["BLOCK", "WARN"] = "WARN"
    block_threshold: int = Field(default=50, ge=0, le=100)


class EmbeddingIntegrityConfig(DetectorConfig):
    """LLM09: vector-store/embedding manipulation in retrieved chunks (``retrieval`` stage)."""
    action: Literal["BLOCK", "WARN"] = "WARN"
    block_threshold: int = Field(default=50, ge=0, le=100)
    max_repetition_ratio: float = Field(default=0.30, ge=0.0, le=1.0)
    min_tokens_for_repetition: int = Field(default=20, ge=1)
    max_hidden_char_ratio: float = Field(default=0.05, ge=0.0, le=1.0)


class UnboundedConsumptionConfig(DetectorConfig):
    """(LLM06): per-session token/cost budgets, thinking-token caps, agent-loop depth
    caps, and state-hash loop detection."""
    action: Literal["BLOCK", "WARN"] = "BLOCK"
    max_session_tokens: int = Field(default=200_000, ge=0)
    max_session_cost_usd: float = Field(default=10.0, ge=0)
    max_iterations: int = Field(default=50, ge=1)
    max_tool_depth: int = Field(default=25, ge=1)
    session_window_seconds: int = Field(default=3600, ge=1)
    # Extended-thinking exhaustion: a short prompt driving a long reasoning run passes
    # every input-size check, so it needs budgets of its own.
    max_thinking_tokens_per_request: int = Field(default=32_000, ge=0)
    max_session_thinking_tokens: int = Field(default=200_000, ge=0)
    # Thinking tokens per input token. 0 disables the ratio check.
    max_thinking_to_input_ratio: float = Field(default=50.0, ge=0)
    # Floor before the ratio applies, so a tiny prompt cannot trip it on its own.
    ratio_min_thinking_tokens: int = Field(default=4_000, ge=0)
    # Agent loop spinning on one state: repeats allowed before it is called a loop.
    max_repeated_states: int = Field(default=3, ge=2)
    max_tracked_states: int = Field(default=64, ge=1)


class SemanticCacheIntegrityConfig(DetectorConfig):
    """LLM09: answers served from a semantic cache (cross-tenant hits, collisions,
    planted entries)."""
    block_cross_tenant: bool = True
    min_similarity: float = Field(default=0.90, ge=0.0, le=1.0)
    min_token_overlap: float = Field(default=0.30, ge=0.0, le=1.0)
    require_partitioning: bool = True
    max_entry_age_seconds: int = Field(default=0, ge=0)


class CorpusPoisoningMonitorConfig(DetectorConfig):
    """(LLM05): screen documents entering a corpus for durable poisoning."""
    action: Literal["BLOCK", "WARN"] = "WARN"
    authority_threshold: int = Field(default=1, ge=1)
    max_question_repeats: int = Field(default=3, ge=2)
    flag_untrusted_into_trusted: bool = True
    max_writes_per_window: int = Field(default=250, ge=0)
    window_seconds: int = Field(default=3600, ge=1)


class TenantScopeAssertionConfig(DetectorConfig):
    """(LLM09): assert retrieved chunks belong to the requesting tenant."""
    action: Literal["BLOCK", "WARN"] = "WARN"
    # A confirmed cross-tenant chunk is a leak in progress, so it blocks even in WARN
    # mode unless an operator deliberately turns this off.
    block_cross_tenant: bool = True
    require_tenant_tags: bool = True
    require_query_scoping: bool = True


class RetrievalJammingConfig(DetectorConfig):
    """(LLM09): detect blocker documents that suppress a RAG answer."""
    action: Literal["BLOCK", "WARN"] = "WARN"
    refusal_threshold: int = Field(default=2, ge=1)
    dominance_ratio: float = Field(default=3.0, ge=1.0)
    min_chunks_for_dominance: int = Field(default=3, ge=3)
    min_content_words: int = Field(default=25, ge=0)


class ReasoningTraceDisclosureConfig(DetectorConfig):
    """(LLM02): treat reasoning traces and tool arguments as output channels."""
    action: Literal["BLOCK", "WARN"] = "WARN"
    require_trace_scanned: bool = True
    flag_tool_arguments: bool = True
    min_trace_chars: int = Field(default=40, ge=0)


class OutputControlCharSanitizerConfig(DetectorConfig):
    """(LLM10): neutralise terminal control sequences in model output."""
    # REDACT by default: the answer is usable once neutralised, so withholding it costs
    # more than it saves.
    action: Literal["BLOCK", "WARN", "REDACT"] = "REDACT"
    allow_color_codes: bool = True


class MultimodalInjectionConfig(DetectorConfig):
    """(LLM01): govern caller-extracted OCR/transcript text, and flag unscanned media."""
    action: Literal["BLOCK", "WARN"] = "BLOCK"
    # Report media that arrived with nothing extracted. A control that stays silent about
    # what it could not inspect makes the gap invisible.
    flag_unextracted_media: bool = True
    match_threshold: int = Field(default=1, ge=1)


class ToolPermissionAuditConfig(DetectorConfig):
    """(LLM03): audit tool/function declarations for excessive functionality and
    permissions at registration time (``tool_registration`` hook)."""
    # WARN by default: a registration finding is design feedback, and blocking a tool the
    # app already depends on breaks the app rather than the attack.
    action: Literal["BLOCK", "WARN"] = "WARN"
    flag_open_ended: bool = True
    flag_destructive_bundling: bool = True
    flag_wildcard_scopes: bool = True
    flag_unconstrained_params: bool = True
    # Tool names the operator has reviewed and sanctioned.
    allowed_tools: Optional[List[str]] = None


class HumanApprovalGateConfig(DetectorConfig):
    """(LLM03): require human approval before irreversible or high-impact agent actions
    (``agent_plan`` / ``agent_loop``)."""
    action: Literal["BLOCK", "WARN"] = "BLOCK"
    # Action categories that need a human. Shares excessive_agency's taxonomy; mutating
    # actions are excluded by default because gating everything causes approval fatigue.
    gated_categories: Optional[List[str]] = None
    # Reject an approval that names no approver: it cannot be audited.
    require_approver_identity: bool = True


class ExcessiveAgencyConfig(DetectorConfig):
    """(LLM03): risk-score agent plans / agent-loop step actions (``agent_plan``/``agent_loop``)."""
    action: Literal["BLOCK", "WARN"] = "WARN"
    # 50 so a single HIGH-severity action reaches BLOCK when action=BLOCK (matches the detector).
    block_threshold: int = Field(default=50, ge=0, le=100)
    warn_threshold: int = Field(default=30, ge=0, le=100)
    high_risk_actions: Optional[List[str]] = None


class MemoryWritePoisoningConfig(DetectorConfig):
    """(LLM01): scan agent memory writes for persistent injection (``memory_write``)."""
    action: Literal["BLOCK", "WARN"] = "BLOCK"
    block_threshold: int = Field(default=50, ge=0, le=100)


class McpManifestScannerConfig(DetectorConfig):
    """(LLM01/LLM04): scan tool/MCP manifests at registration (``tool_registration`` hook)."""
    action: Literal["BLOCK", "WARN"] = "WARN"
    block_threshold: int = Field(default=50, ge=0, le=100)
    allowed_domains: Optional[List[str]] = None
    dangerous_permissions: Optional[List[str]] = None


class NumericalConsistencyConfig(DetectorConfig):
    """Deferred backlog: deterministic arithmetic-equation consistency (output)."""
    action: Literal["BLOCK", "WARN"] = "WARN"
    block_threshold: int = Field(default=60, ge=0, le=100)


class DocumentMetadataLeakageConfig(DetectorConfig):
    """Deferred backlog: deterministic document-artifact / hidden-text leakage."""
    action: Literal["BLOCK", "WARN"] = "WARN"
    block_threshold: int = Field(default=60, ge=0, le=100)


# ── Top-level policy schema ────────────────────────────────────────────────

class PolicySchema(BaseModel):
    """Schema for a single policy scope (default, tenant, app, agent, env)."""
    model_config = ConfigDict(extra="allow")
    secrets: Optional[SecretsConfig] = None
    pii: Optional[PIIConfig] = None
    jailbreak: Optional[JailbreakConfig] = None
    toxicity: Optional[ToxicityConfig] = None
    topic_restriction: Optional[TopicRestrictionConfig] = None
    competitor: Optional[CompetitorConfig] = None
    exfiltration: Optional[ExfiltrationConfig] = None
    tools: Optional[ToolsConfig] = None
    structure: Optional[StructureConfig] = None
    abuse: Optional[AbuseConfig] = None
    gibberish: Optional[GibberishConfig] = None
    language: Optional[LanguageConfig] = None
    bias: Optional[BiasConfig] = None
    sentiment: Optional[SentimentConfig] = None
    compliance: Optional[ComplianceConfig] = None
    malicious_url: Optional[MaliciousURLConfig] = None
    copyright: Optional[CopyrightConfig] = None
    code_safety: Optional[CodeSafetyConfig] = None
    hallucination: Optional[HallucinationConfig] = None
    # deterministic gap detectors (OWASP LLM02 / LLM07 / LLM08)
    sensitive_business_data: Optional[SensitiveBusinessDataConfig] = None
    citation_integrity: Optional[CitationIntegrityConfig] = None
    system_prompt_leakage: Optional[SystemPromptLeakageConfig] = None
    # new-stage / lifecycle gap detectors (OWASP LLM01 / LLM03 / LLM06 / LLM04)
    retrieval_chunk_injection: Optional[RetrievalChunkInjectionConfig] = None
    embedding_integrity: Optional[EmbeddingIntegrityConfig] = None
    tool_output_injection: Optional[ToolOutputInjectionConfig] = None
    unbounded_consumption: Optional[UnboundedConsumptionConfig] = None
    excessive_agency: Optional[ExcessiveAgencyConfig] = None
    memory_write_poisoning: Optional[MemoryWritePoisoningConfig] = None
    mcp_manifest_scanner: Optional[McpManifestScannerConfig] = None
    tool_permission_audit: Optional[ToolPermissionAuditConfig] = None
    human_approval_gate: Optional[HumanApprovalGateConfig] = None
    tenant_scope_assertion: Optional[TenantScopeAssertionConfig] = None
    retrieval_jamming: Optional[RetrievalJammingConfig] = None
    reasoning_trace_disclosure: Optional[ReasoningTraceDisclosureConfig] = None
    output_control_char_sanitizer: Optional[OutputControlCharSanitizerConfig] = None
    multimodal_injection: Optional[MultimodalInjectionConfig] = None
    corpus_poisoning_monitor: Optional[CorpusPoisoningMonitorConfig] = None
    semantic_cache_integrity: Optional[SemanticCacheIntegrityConfig] = None
    numerical_consistency: Optional[NumericalConsistencyConfig] = None
    document_metadata_leakage: Optional[DocumentMetadataLeakageConfig] = None
    # Structured output contract
    output_contract: Optional[OutputContractConfig] = None
    # Response quality scoring
    quality_scoring: Optional[QualityScoringConfig] = None
    # Custom detector references (remote, regex, webhook, catalog)
    custom_detectors: Optional[List[CustomDetectorRef]] = None
    # Per-detector on_fail remediation fallback
    on_fail: Optional[str] = None
    # Top-level runtime/egress policy (lint + enforcement)
    runtime_policy: Optional[RuntimePolicyConfig] = None


def validate_policy(policy_dict: dict) -> PolicySchema:
    """Validate a policy configuration dict and return a PolicySchema.

    Logs warnings for validation errors but never raises -- the caller
    always gets a usable PolicySchema back (with defaults for invalid fields).

    Args:
        policy_dict: Raw policy dict (e.g. from YAML).

    Returns:
        A validated PolicySchema instance.
    """
    try:
        return PolicySchema(**policy_dict)
    except ValidationError as exc:
        logger.warning(
            "Policy validation errors (using defaults for invalid fields): %s",
            exc.errors(),
        )
        # Build a best-effort schema by dropping keys that fail validation
        safe_dict = {}
        for key, value in policy_dict.items():
            try:
                PolicySchema(**{key: value})
                safe_dict[key] = value
            except ValidationError:
                logger.warning("Dropping invalid policy key '%s'", key)
        try:
            return PolicySchema(**safe_dict)
        except ValidationError:
            logger.warning("Could not build PolicySchema at all; returning empty defaults")
            return PolicySchema()


# ── Strict validation: editor warnings vs publish blockers ───────────

# Top-level keys that are not detector slots but legitimately appear in stored /
# resolved policies (injected by the resolver, used as detector aliases, or
# carried as metadata). PolicySchema uses extra="allow", so these are accepted —
# they must NOT be reported as unknown-key warnings.
_KNOWN_EXTRA_POLICY_KEYS = frozenset({
    "policy_version",       # stamped onto the resolved policy by the resolver
    "schema_enforcement",   # alias for the structure / output-contract detector
    "tool_governance",      # alias for the tools detector
    "_multilingual",        # multilingual overlay metadata
    "metadata", "description", "name", "version",
})


class PolicyValidationIssue(BaseModel):
    """A single policy-validation finding (a blocker or a warning)."""
    code: str               # machine code, e.g. "string_type", "value_error", "unknown_key"
    loc: str                # dotted path, e.g. "pii.action" ("" = top level)
    message: str            # human-readable explanation


class PolicyValidationResult(BaseModel):
    """Outcome of strict policy validation."""
    valid: bool                                                       # False iff blockers present
    blockers: List[PolicyValidationIssue] = Field(default_factory=list)
    warnings: List[PolicyValidationIssue] = Field(default_factory=list)


class PolicyValidationError(ValueError):
    """Raised when a policy fails strict (upsert/publish) validation.

    Carries the full :class:`PolicyValidationResult` so callers can surface the
    blocker/warning lists (e.g. as an HTTP 422 body).
    """

    def __init__(self, result: "PolicyValidationResult") -> None:
        self.result = result
        summary = "; ".join(f"{b.loc or '(root)'}: {b.message}" for b in result.blockers)
        super().__init__(f"Policy has {len(result.blockers)} blocking error(s): {summary}")


def _loc_to_str(loc: tuple) -> str:
    return ".".join(str(p) for p in loc)


def validate_policy_strict(policy_dict: dict) -> PolicyValidationResult:
    """Strictly validate a policy dict, separating fatal blockers from advisory warnings.

    Unlike :func:`validate_policy` -- which silently drops invalid keys so the
    forward-compatible runtime YAML load never crashes -- this surfaces every
    problem so a policy can be *rejected* at upsert/publish time:

    * **blockers** are typed fields that fail schema validation (wrong types,
      an empty ``stages`` list, a malformed detector block). Today these are
      silently dropped, so an invalid setting can vanish from an active bundle
      with no signal. They make ``valid`` False and must block the write.
    * **warnings** are unrecognised top-level keys. ``PolicySchema`` uses
      ``extra="allow"``, so these are accepted and kept (some keys are
      legitimately dynamic) -- but unknown ones are worth flagging in the editor.
    """
    blockers: List[PolicyValidationIssue] = []
    warnings: List[PolicyValidationIssue] = []

    if not isinstance(policy_dict, dict):
        blockers.append(PolicyValidationIssue(
            code="dict_type", loc="",
            message="Policy must be a JSON object.",
        ))
        return PolicyValidationResult(valid=False, blockers=blockers, warnings=warnings)

    try:
        PolicySchema.model_validate(policy_dict)
    except ValidationError as exc:
        for err in exc.errors():
            blockers.append(PolicyValidationIssue(
                code=str(err.get("type", "validation_error")),
                loc=_loc_to_str(err.get("loc", ())),
                message=str(err.get("msg", "invalid value")),
            ))

    known = set(PolicySchema.model_fields.keys()) | _KNOWN_EXTRA_POLICY_KEYS
    for key in policy_dict:
        if key not in known:
            warnings.append(PolicyValidationIssue(
                code="unknown_key", loc=str(key),
                message=(
                    f"Unrecognised policy key '{key}' -- accepted, but it is not a known "
                    "detector or setting and will be ignored by the engine."
                ),
            ))

    return PolicyValidationResult(valid=not blockers, blockers=blockers, warnings=warnings)


def assert_policy_publishable(policy_dict: dict) -> PolicyValidationResult:
    """Run strict validation; raise :class:`PolicyValidationError` if there are blockers.

    Returns the :class:`PolicyValidationResult` (carrying any warnings) when the
    policy is publishable, so callers can still log/surface warnings.
    """
    result = validate_policy_strict(policy_dict)
    if not result.valid:
        raise PolicyValidationError(result)
    return result
