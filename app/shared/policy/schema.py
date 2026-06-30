"""
Policy schema validation using Pydantic.

Validates policy configuration dicts at load time. Logs warnings for
invalid configs but never crashes -- callers always get a PolicySchema
back (possibly with defaults filled in).
"""
import logging
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)


# ── Detector config models ──────────────────────────────────────────────────

class DetectorConfig(BaseModel):
    """Base detector configuration shared by all detectors."""
    model_config = ConfigDict(extra="allow")
    enabled: bool = False
    action: Optional[str] = None
    # Restrict this detector to specific pipeline stages. None = use orchestrator default.
    stages: Optional[List[Literal["input", "output"]]] = None

    @field_validator("stages")
    @classmethod
    def stages_must_be_nonempty(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None and len(v) == 0:
            raise ValueError("stages must be a non-empty list if provided")
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
    # Structured output contract
    output_contract: Optional[OutputContractConfig] = None
    # Response quality scoring
    quality_scoring: Optional[QualityScoringConfig] = None
    # Custom detector references (remote, regex, webhook, catalog)
    custom_detectors: Optional[List[CustomDetectorRef]] = None
    # Per-detector on_fail remediation fallback
    on_fail: Optional[str] = None


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
