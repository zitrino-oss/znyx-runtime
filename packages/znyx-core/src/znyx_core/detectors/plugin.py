"""
Custom Detector Plugin System.

Allows users to register custom detectors that run alongside built-in ones.
Plugins can be:
  1. Python classes loaded from a directory (file-based plugins)
  2. HTTP endpoints (webhook-based plugins)
  3. Regex-based rules (config-only, no code)

All custom detectors must produce a DetectorResult.
"""
import importlib.util
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

import httpx

from znyx_core.core.models import DetectorResult, RuleHit, Severity, Decision
from znyx_core.detectors.remote import RemoteDetector as _RemoteDetectorImpl
from znyx_core.net_guard import EgressTarget, UnsafeEgressURL, resolve_egress_target

logger = logging.getLogger(__name__)


class BaseCustomDetector:
    """Base class that all file-based custom detectors must extend."""

    name: str = "custom_detector"
    version: str = "1.0.0"

    def detect(self, text: str, config: Dict[str, Any]) -> DetectorResult:
        """
        Evaluate text and return a DetectorResult.

        Args:
            text: The input text to evaluate.
            config: Detector-specific configuration from the policy.

        Returns:
            DetectorResult with rule_hits, risk_score, decision, etc.
        """
        raise NotImplementedError("Custom detectors must implement detect()")


class RegexDetector(BaseCustomDetector):
    """Built-in detector that runs user-defined regex rules."""

    name = "regex"

    def detect(self, text: str, config: Dict[str, Any]) -> DetectorResult:
        rules = config.get("rules", [])
        rule_hits = []
        max_score = 0

        for rule in rules:
            pattern = rule.get("pattern", "")
            rule_id = rule.get("id", f"regex_{pattern[:20]}")
            severity = Severity(rule.get("severity", "medium"))
            score = rule.get("score", 50)
            message = rule.get("message", f"Matched regex pattern: {pattern}")

            try:
                if re.search(pattern, text, re.IGNORECASE):
                    rule_hits.append(RuleHit(
                        rule_id=rule_id,
                        severity=severity,
                        message=message,
                    ))
                    max_score = max(max_score, score)
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")

        decision = None
        if rule_hits:
            action = config.get("action", "WARN")
            decision = Decision(action)

        return DetectorResult(
            rule_hits=rule_hits,
            risk_score=max_score,
            decision=decision,
        )


class WebhookDetector(BaseCustomDetector):
    """Detector that calls an external HTTP endpoint."""

    name = "webhook"

    def detect(self, text: str, config: Dict[str, Any]) -> DetectorResult:
        # SSRF guard: the webhook URL is customer-supplied, so use the strict posture
        # (block private/loopback/link-local/metadata). A webhook POSTs the text to an
        # external endpoint — never let a misconfigured/hostile URL turn that into an
        # internal-network or cloud-metadata probe. Fail SAFE: skip the call on a bad
        # URL rather than fail-open into an unguarded request.
        url = config.get("url", "")
        # Resolve+validate DNS ONCE and connect to the pinned IP below, so a
        # rebinding attacker can't flip a public name to an internal/metadata IP
        # between this check and the request.
        try:
            target = resolve_egress_target(url, allow_private=False)
        except UnsafeEgressURL as e:
            logger.error("Webhook detector blocked unsafe URL (%s): %s", url, e)
            return DetectorResult()

        # Sync wrapper - async version below
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an async context - can't use asyncio.run
                # Use a sync httpx client instead
                return self._detect_sync(text, config, target)
            return asyncio.run(self._detect_async(text, config, target))
        except RuntimeError:
            return self._detect_sync(text, config, target)

    def _detect_sync(self, text: str, config: Dict[str, Any], target: EgressTarget) -> DetectorResult:
        url = config.get("url", "")
        timeout = config.get("timeout", 5)
        headers = config.get("headers", {})

        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    target.connect_url,
                    json={"text": text, "config": config},
                    headers={**headers, "Host": target.host_header},
                    extensions={"sni_hostname": target.sni_hostname},
                )
                if response.status_code == 200:
                    return self._parse_response(response.json())
        except Exception as e:
            logger.error(f"Webhook detector error ({url}): {e}")

        # Fail open - return no hits
        return DetectorResult()

    async def _detect_async(self, text: str, config: Dict[str, Any], target: EgressTarget) -> DetectorResult:
        url = config.get("url", "")
        timeout = config.get("timeout", 5)
        headers = config.get("headers", {})

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    target.connect_url,
                    json={"text": text, "config": config},
                    headers={**headers, "Host": target.host_header},
                    extensions={"sni_hostname": target.sni_hostname},
                )
                if response.status_code == 200:
                    return self._parse_response(response.json())
        except Exception as e:
            logger.error(f"Webhook detector error ({url}): {e}")

        return DetectorResult()

    @staticmethod
    def _parse_response(data: dict) -> DetectorResult:
        rule_hits = []
        for hit in data.get("rule_hits", []):
            rule_hits.append(RuleHit(
                rule_id=hit.get("rule_id", "webhook_hit"),
                severity=Severity(hit.get("severity", "medium")),
                message=hit.get("message", "Webhook detector match"),
            ))

        decision = None
        if data.get("decision"):
            decision = Decision(data["decision"])

        return DetectorResult(
            rule_hits=rule_hits,
            risk_score=data.get("risk_score", 0),
            decision=decision,
            sanitized_text=data.get("sanitized_text"),
            user_message=data.get("user_message"),
        )


class RemoteDetectorPlugin(BaseCustomDetector):
    """Built-in plugin that wraps RemoteDetector for the plugin registry.

    Config must include ``endpoint_url``. Supports all RemoteDetector options:
    auth_type, auth_header, auth_value, fail_open, circuit breaker settings, etc.
    """

    name = "remote"
    version = "1.0.0"

    def detect(self, text: str, config: Dict[str, Any]) -> DetectorResult:
        detector = _RemoteDetectorImpl(config)
        return detector.detect(text)


class PluginRegistry:
    """
    Registry for custom detector plugins.

    Loads plugins from:
    1. A configured plugin directory (GUARDRAILS_PLUGIN_DIR)
    2. Programmatic registration via register()
    """

    def __init__(self):
        self._detectors: Dict[str, BaseCustomDetector] = {
            "regex": RegexDetector(),
            "webhook": WebhookDetector(),
            "remote": RemoteDetectorPlugin(),
        }

    def register(self, name: str, detector: BaseCustomDetector) -> None:
        """Register a custom detector instance."""
        self._detectors[name] = detector
        logger.info(f"Registered custom detector: {name} (v{detector.version})")

    def unregister(self, name: str) -> bool:
        """Unregister a custom detector."""
        if name in ("regex", "webhook", "remote"):
            return False  # Can't unregister built-in plugins
        return self._detectors.pop(name, None) is not None

    def get(self, name: str) -> Optional[BaseCustomDetector]:
        return self._detectors.get(name)

    def list_detectors(self) -> List[dict]:
        return [
            {"name": name, "version": d.version, "type": type(d).__name__}
            for name, d in self._detectors.items()
        ]

    def load_plugins_from_directory(self, plugin_dir: str) -> int:
        """
        Load all .py files from a directory as detector plugins.
        Each file must define a class that extends BaseCustomDetector.
        Returns the number of plugins loaded.
        """
        path = Path(plugin_dir)
        if not path.is_dir():
            logger.warning(f"Plugin directory not found: {plugin_dir}")
            return 0

        loaded = 0
        for py_file in path.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            try:
                module_name = f"guardrails_plugin_{py_file.stem}"
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if not spec or not spec.loader:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                # Find detector classes in the module
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, BaseCustomDetector)
                        and attr is not BaseCustomDetector
                    ):
                        instance = attr()
                        self.register(instance.name, instance)
                        loaded += 1
                        logger.info(f"Loaded plugin detector '{instance.name}' from {py_file.name}")

            except Exception as e:
                logger.error(f"Failed to load plugin from {py_file}: {e}")

        return loaded

    def detect(self, name: str, text: str, config: Dict[str, Any]) -> DetectorResult:
        """Run a named custom detector. Returns empty result if not found."""
        detector = self.get(name)
        if not detector:
            logger.warning(f"Custom detector not found: {name}")
            return DetectorResult()

        try:
            return detector.detect(text, config)
        except Exception as e:
            logger.error(f"Custom detector '{name}' error: {e}")
            return DetectorResult()


# Global plugin registry
plugin_registry = PluginRegistry()


def init_plugins():
    """Initialize plugins from the configured directory."""
    plugin_dir = os.getenv("GUARDRAILS_PLUGIN_DIR", "")
    if plugin_dir:
        count = plugin_registry.load_plugins_from_directory(plugin_dir)
        logger.info(f"Loaded {count} custom detector plugins from {plugin_dir}")
