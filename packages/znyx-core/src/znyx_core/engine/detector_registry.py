"""Detector registry with instance caching.

Each detector is registered by its policy-key name and lazily instantiated
the first time it is requested.  Instances are cached per (name, config
digest) in a bounded LRU, so policies with different configs (two tenants,
or one tenant's two apps) each keep their own long-lived instance - and the
state a stateful detector accumulates (rate buckets, conversation history)
survives policy alternation instead of being reset on every switch.
"""

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple, Type

from znyx_core.config.tunables import DETECTOR_INSTANCE_CACHE_SIZE

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


def _config_digest(config: Dict[str, Any]) -> str:
    """Stable content digest of a JSON-serializable config dict.

    sha256 rather than the salted builtin hash() so two distinct configs can
    never silently collide into one shared (stateful) instance.
    """
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


# Detectors that must NOT be cached: the orchestrator enriches their config
# per request (source_context / grounding_sources merged in) and sets a
# per-request nli_scorer instance attribute after creation. Caching them would
# add one LRU entry per unique grounding AND share that per-request attribute
# across concurrent requests, so they get a fresh instance on every call.
_REQUEST_SCOPED = frozenset({"hallucination", "citation_integrity"})


class DetectorRegistry:
    """Registry that maps detector names to classes and caches instances."""

    def __init__(self, max_instances: Optional[int] = None):
        # name -> detector class
        self._classes: Dict[str, Type] = {}
        # (name, config digest) -> detector instance, in LRU order
        self._instances: "OrderedDict[Tuple[str, str], Any]" = OrderedDict()
        self._lock = threading.RLock()
        self._max_instances = max(16, max_instances or DETECTOR_INSTANCE_CACHE_SIZE)

    def register(self, name: str, detector_class: Type) -> None:
        """Register a detector class under the given policy-key name."""
        self._classes[name] = detector_class

    def get_or_create(self, name: str, config: Dict[str, Any]) -> Any:
        """Return the cached detector instance for (name, config), creating it
        on first use and evicting the least-recently-used entry when full."""
        cls = self._classes.get(name)
        if cls is None:
            raise KeyError(f"No detector registered under '{name}'")
        if name in _REQUEST_SCOPED:
            return cls(config)
        key = (name, _config_digest(config))
        with self._lock:
            instance = self._instances.get(key)
            if instance is not None:
                self._instances.move_to_end(key)
                return instance
            # Construct under the lock on purpose: init is config parsing and
            # regex compilation, and holding the lock guarantees a stateful
            # detector is never built twice for the same key - two live copies
            # would fork its rate buckets / history.
            instance = cls(config)
            self._instances[key] = instance
            while len(self._instances) > self._max_instances:
                self._instances.popitem(last=False)
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
