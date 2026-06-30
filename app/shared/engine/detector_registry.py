"""Detector registry with instance caching.

Each detector is registered by its policy-key name and lazily instantiated
the first time it is requested.  Instances are cached by a hash of their
config dict so they are recreated only when the policy changes.
"""

import json
import logging
from typing import Any, Dict, Optional, Type

from app.shared.detectors.pii import PIIDetector
from app.shared.detectors.jailbreak import JailbreakDetector
from app.shared.detectors.toxicity import ToxicityDetector
from app.shared.detectors.competitor import CompetitorDetector
from app.shared.detectors.topic import TopicDetector
from app.shared.detectors.secrets import SecretsDetector
from app.shared.detectors.exfiltration import ExfiltrationDetector
from app.shared.detectors.structure import StructureDetector
from app.shared.detectors.abuse import AbuseDetector
from app.shared.detectors.gibberish import GibberishDetector
from app.shared.detectors.language import LanguageDetector
from app.shared.detectors.bias import BiasDetector
from app.shared.detectors.sentiment import SentimentDetector
from app.shared.detectors.compliance import ComplianceDetector
from app.shared.detectors.malicious_url import MaliciousURLDetector
from app.shared.detectors.copyright import CopyrightDetector
from app.shared.detectors.code_safety import CodeSafetyDetector
from app.shared.detectors.hallucination import HallucinationDetector
from app.shared.detectors.tools import ToolGovernanceDetector

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
    registry.register("structure", StructureDetector)
    # Tool governance
    registry.register("tools", ToolGovernanceDetector)
    return registry


# Module-level singleton with all 19 built-in detectors registered.
default_registry = _build_default_registry()
