"""Detector registry with instance caching.

Each detector is registered by its policy-key name and lazily instantiated
the first time it is requested.  Instances are cached by a hash of their
config dict so they are recreated only when the policy changes.
"""

import json
import logging
from typing import Any, Dict, Type

from znyx_core.detectors.pii import PIIDetector
from znyx_core.detectors.jailbreak import JailbreakDetector
from znyx_core.detectors.toxicity import ToxicityDetector
from znyx_core.detectors.competitor import CompetitorDetector
from znyx_core.detectors.topic import TopicDetector
from znyx_core.detectors.secrets import SecretsDetector
from znyx_core.detectors.exfiltration import ExfiltrationDetector
from znyx_core.detectors.structure import StructureDetector
from znyx_core.detectors.abuse import AbuseDetector
from znyx_core.detectors.gibberish import GibberishDetector
from znyx_core.detectors.language import LanguageDetector
from znyx_core.detectors.bias import BiasDetector
from znyx_core.detectors.sentiment import SentimentDetector
from znyx_core.detectors.compliance import ComplianceDetector
from znyx_core.detectors.malicious_url import MaliciousURLDetector
from znyx_core.detectors.copyright import CopyrightDetector
from znyx_core.detectors.code_safety import CodeSafetyDetector
from znyx_core.detectors.hallucination import HallucinationDetector
from znyx_core.detectors.tools import ToolGovernanceDetector
from znyx_core.detectors.sensitive_business_data import SensitiveBusinessDataDetector
from znyx_core.detectors.citation_integrity import CitationIntegrityDetector
from znyx_core.detectors.system_prompt_leakage import SystemPromptLeakageDetector
from znyx_core.detectors.retrieval_chunk_injection import RetrievalChunkInjectionDetector
from znyx_core.detectors.embedding_integrity import EmbeddingIntegrityDetector
from znyx_core.detectors.tool_output_injection import ToolOutputInjectionDetector
from znyx_core.detectors.unbounded_consumption import UnboundedConsumptionDetector
from znyx_core.detectors.excessive_agency import ExcessiveAgencyDetector
from znyx_core.detectors.memory_write_poisoning import MemoryWritePoisoningDetector
from znyx_core.detectors.mcp_manifest_scanner import McpManifestScannerDetector
from znyx_core.detectors.tool_permission_audit import ToolPermissionAuditDetector
from znyx_core.detectors.human_approval_gate import HumanApprovalGateDetector
from znyx_core.detectors.tenant_scope_assertion import TenantScopeAssertionDetector
from znyx_core.detectors.retrieval_jamming import RetrievalJammingDetector
from znyx_core.detectors.reasoning_trace_disclosure import ReasoningTraceDisclosureDetector
from znyx_core.detectors.output_control_char_sanitizer import OutputControlCharSanitizerDetector
from znyx_core.detectors.multimodal_injection import MultimodalInjectionDetector
from znyx_core.detectors.corpus_poisoning_monitor import CorpusPoisoningMonitorDetector
from znyx_core.detectors.semantic_cache_integrity import SemanticCacheIntegrityDetector
from znyx_core.detectors.numerical_consistency import NumericalConsistencyDetector
from znyx_core.detectors.document_metadata_leakage import DocumentMetadataLeakageDetector

logger = logging.getLogger(__name__)


class DetectorRegistry:
    """Registry that maps detector names to classes and caches instances."""

    def __init__(self):
        # name -> detector class
        self._classes: Dict[str, Type] = {}
        # (name, config_hash) -> detector instance
        self._instances: Dict[str, Any] = {}
        self._config_hashes: Dict[str, int] = {}

    def register(self, name: str, detector_class: Type) -> None:
        """Register a detector class under the given policy-key name."""
        self._classes[name] = detector_class

    def get_or_create(self, name: str, config: Dict[str, Any]) -> Any:
        """Return a cached detector instance, recreating only if config changed."""
        config_hash = hash(json.dumps(config, sort_keys=True))
        if name in self._instances and self._config_hashes.get(name) == config_hash:
            return self._instances[name]
        cls = self._classes.get(name)
        if cls is None:
            raise KeyError(f"No detector registered under '{name}'")
        instance = cls(config)
        self._instances[name] = instance
        self._config_hashes[name] = config_hash
        return instance

    def has(self, name: str) -> bool:
        return name in self._classes


def _build_default_registry() -> DetectorRegistry:
    """Build a registry pre-populated with all built-in detectors."""
    registry = DetectorRegistry()
    # Input-and-output detectors (run in both contexts)
    registry.register("abuse", AbuseDetector)
    registry.register("secrets", SecretsDetector)
    registry.register("exfiltration", ExfiltrationDetector)
    registry.register("gibberish", GibberishDetector)
    registry.register("language", LanguageDetector)
    registry.register("topic_restriction", TopicDetector)
    registry.register("toxicity", ToxicityDetector)
    registry.register("bias", BiasDetector)
    registry.register("sentiment", SentimentDetector)
    registry.register("compliance", ComplianceDetector)
    registry.register("competitor", CompetitorDetector)
    registry.register("jailbreak", JailbreakDetector)
    registry.register("pii", PIIDetector)
    # Output-only detectors
    registry.register("malicious_url", MaliciousURLDetector)
    registry.register("copyright", CopyrightDetector)
    registry.register("code_safety", CodeSafetyDetector)
    registry.register("hallucination", HallucinationDetector)
    registry.register("sensitive_business_data", SensitiveBusinessDataDetector)
    registry.register("citation_integrity", CitationIntegrityDetector)
    registry.register("system_prompt_leakage", SystemPromptLeakageDetector)
    # new-stage detectors (retrieval / tool / agent_loop / agent_plan / memory_write)
    registry.register("retrieval_chunk_injection", RetrievalChunkInjectionDetector)
    registry.register("embedding_integrity", EmbeddingIntegrityDetector)
    registry.register("tool_output_injection", ToolOutputInjectionDetector)
    registry.register("unbounded_consumption", UnboundedConsumptionDetector)
    registry.register("excessive_agency", ExcessiveAgencyDetector)
    registry.register("memory_write_poisoning", MemoryWritePoisoningDetector)
    # lifecycle hook (tool_registration) — registered for the hook path, NOT in
    # _DETECTOR_PIPELINE (it does not run per request).
    registry.register("mcp_manifest_scanner", McpManifestScannerDetector)
    registry.register("tool_permission_audit", ToolPermissionAuditDetector)
    registry.register("human_approval_gate", HumanApprovalGateDetector)
    registry.register("tenant_scope_assertion", TenantScopeAssertionDetector)
    registry.register("retrieval_jamming", RetrievalJammingDetector)
    registry.register("reasoning_trace_disclosure", ReasoningTraceDisclosureDetector)
    registry.register("output_control_char_sanitizer", OutputControlCharSanitizerDetector)
    registry.register("multimodal_injection", MultimodalInjectionDetector)
    registry.register("corpus_poisoning_monitor", CorpusPoisoningMonitorDetector)
    registry.register("semantic_cache_integrity", SemanticCacheIntegrityDetector)
    registry.register("numerical_consistency", NumericalConsistencyDetector)
    registry.register("document_metadata_leakage", DocumentMetadataLeakageDetector)
    registry.register("structure", StructureDetector)
    # Tool governance
    registry.register("tools", ToolGovernanceDetector)
    return registry


# Module-level singleton with all 19 built-in detectors registered.
default_registry = _build_default_registry()
