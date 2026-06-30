"""Remediation handler - executes on_fail actions after detector decisions.

Each detector section in a policy can specify an ``on_fail`` action and
optional ``on_fail_config``.  The RemediationHandler inspects the aggregated
evaluation result and, if a non-ALLOW decision was reached, applies the
configured remediation strategy.

Supported actions:
    noop       - do nothing (default)
    reask      - return a reask prompt so the caller can retry the LLM
    fix        - strip the offending text (uses sanitized_text or regex)
    filter_field - remove named JSON fields from structured output
    refrain    - replace the output with a canned "I can't help with that" message
    exception  - signal the caller to raise an error
    custom     - delegate to a user-supplied handler name (future extension)
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional

from app.shared.core.models import (
    Decision,
    DetectorResult,
    EvaluationResponse,
    RemediationAction,
    RemediationResult,
)

logger = logging.getLogger(__name__)

# Default canned responses
_DEFAULT_REFRAIN = "I'm unable to provide that response."
_DEFAULT_REASK = "Your previous response was flagged. Please rephrase your answer."


class RemediationHandler:
    """Stateless handler that applies on_fail remediation to an evaluation result.

    Custom handlers can be registered via ``register_custom_handler()``
    and referenced in policies as ``on_fail: custom`` with
    ``on_fail_config: {handler: "handler_name"}``.
    """

    # Class-level registry of custom handler functions
    _custom_handlers: Dict[str, Any] = {}

    @classmethod
    def register_custom_handler(cls, name: str, handler: Any) -> None:
        """Register a custom remediation handler function.

        Args:
            name: Handler name referenced in ``on_fail_config.handler``.
            handler: Callable ``(config: dict, response: EvaluationResponse) -> RemediationResult``.
        """
        cls._custom_handlers[name] = handler
        logger.info(f"Registered custom remediation handler: {name}")

    @classmethod
    def unregister_custom_handler(cls, name: str) -> None:
        """Remove a registered custom handler."""
        cls._custom_handlers.pop(name, None)

    @classmethod
    def list_custom_handlers(cls) -> List[str]:
        """Return names of all registered custom handlers."""
        return list(cls._custom_handlers.keys())

    def apply(
        self,
        response: EvaluationResponse,
        policy: Dict[str, Any],
        detector_results: Optional[List[DetectorResult]] = None,
    ) -> EvaluationResponse:
        """Apply remediation based on policy on_fail settings.

        Looks at each detector section's ``on_fail`` and applies the **first**
        matching action whose detector actually triggered.  If no on_fail is
        configured, the response passes through unchanged.
        """
        if response.decision == Decision.ALLOW:
            return response

        # Collect on_fail configs from all detector sections that fired
        action, config = self._resolve_action(response, policy)
        if action is None or action == RemediationAction.NOOP:
            return response

        result = self._execute(action, config, response)
        response.remediation = result

        # Side-effects: modify response based on action outcome
        if result.applied:
            if action == RemediationAction.FIX and result.fixed_text is not None:
                response.sanitized_text = result.fixed_text
                response.decision = Decision.TRANSFORM
            elif action == RemediationAction.FILTER_FIELD and result.filtered_fields:
                response.sanitized_text = self._filter_fields(
                    response.sanitized_text or "", result.filtered_fields,
                )
                response.decision = Decision.TRANSFORM
            elif action == RemediationAction.REFRAIN:
                response.sanitized_text = result.refrain_message or _DEFAULT_REFRAIN
                response.decision = Decision.BLOCK
                response.user_message = result.refrain_message or _DEFAULT_REFRAIN
            elif action == RemediationAction.REASK:
                response.developer_message = result.reask_prompt or _DEFAULT_REASK
            elif action == RemediationAction.ASK_HUMAN:
                response.decision = Decision.BLOCK
                response.user_message = result.refrain_message
                # Extract review_id from the reask_prompt encoding
                if result.reask_prompt and "review_id:" in result.reask_prompt:
                    parts = result.reask_prompt.split(":")
                    idx = parts.index("review_id") if "review_id" in parts else -1
                    if idx >= 0 and idx + 1 < len(parts):
                        response.pending_review_id = parts[idx + 1]

        return response

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_action(
        self, response: EvaluationResponse, policy: Dict[str, Any],
    ) -> tuple:
        """Find the first on_fail action from the detector that triggered."""
        # Check which detectors fired via rule_hits
        fired_detectors = set()
        for hit in response.rule_hits:
            # rule_id format is typically "detector_name" or "detector_name:sub"
            base = hit.rule_id.split(":")[0] if hit.rule_id else ""
            fired_detectors.add(base)

        # Walk policy sections looking for on_fail on fired detectors
        for key, section in policy.items():
            if not isinstance(section, dict):
                continue
            on_fail = section.get("on_fail")
            if not on_fail:
                continue
            # Check if this detector key fired
            if key in fired_detectors:
                try:
                    action = RemediationAction(on_fail)
                except ValueError:
                    logger.warning(f"Unknown on_fail action '{on_fail}' for detector '{key}'")
                    continue
                return action, section.get("on_fail_config", {})

        # Global fallback
        global_on_fail = policy.get("on_fail")
        if global_on_fail:
            try:
                action = RemediationAction(global_on_fail)
                return action, policy.get("on_fail_config", {})
            except ValueError:
                pass

        return None, {}

    def _execute(
        self, action: RemediationAction, config: Dict[str, Any],
        response: EvaluationResponse,
    ) -> RemediationResult:
        """Execute a single remediation action."""
        handler = {
            RemediationAction.REASK: self._do_reask,
            RemediationAction.FIX: self._do_fix,
            RemediationAction.FILTER_FIELD: self._do_filter_field,
            RemediationAction.REFRAIN: self._do_refrain,
            RemediationAction.EXCEPTION: self._do_exception,
            RemediationAction.CUSTOM: self._do_custom,
            RemediationAction.ASK_HUMAN: self._do_ask_human,
        }.get(action)

        if handler is None:
            return RemediationResult(action=action, applied=False)

        return handler(config, response)

    def _do_reask(self, config: Dict[str, Any], response: EvaluationResponse) -> RemediationResult:
        prompt = config.get("prompt", _DEFAULT_REASK)
        max_attempts = config.get("max_attempts", 3)
        # Include the rule hits in the reask prompt for context
        if config.get("include_violations", True) and response.rule_hits:
            violations = "; ".join(h.message for h in response.rule_hits)
            prompt = f"{prompt}\n\nViolations: {violations}"
        return RemediationResult(
            action=RemediationAction.REASK,
            applied=True,
            reask_prompt=prompt,
            max_attempts=max_attempts,
        )

    def _do_fix(self, config: Dict[str, Any], response: EvaluationResponse) -> RemediationResult:
        text = response.sanitized_text or ""
        # Use patterns from config to strip offending content
        patterns = config.get("strip_patterns", [])
        replacement = config.get("replacement", "[REMOVED]")
        fixed = text
        for pattern in patterns:
            try:
                fixed = re.sub(pattern, replacement, fixed)
            except re.error:
                logger.warning(f"Invalid fix pattern: {pattern}")
        # If no patterns, use sanitized_text as-is (already fixed by detector)
        if not patterns and response.sanitized_text:
            fixed = response.sanitized_text
        return RemediationResult(
            action=RemediationAction.FIX,
            applied=True,
            fixed_text=fixed,
        )

    def _do_filter_field(self, config: Dict[str, Any], response: EvaluationResponse) -> RemediationResult:
        fields = config.get("fields", [])
        return RemediationResult(
            action=RemediationAction.FILTER_FIELD,
            applied=bool(fields),
            filtered_fields=fields,
        )

    def _do_refrain(self, config: Dict[str, Any], response: EvaluationResponse) -> RemediationResult:
        message = config.get("message", _DEFAULT_REFRAIN)
        return RemediationResult(
            action=RemediationAction.REFRAIN,
            applied=True,
            refrain_message=message,
        )

    def _do_exception(self, config: Dict[str, Any], response: EvaluationResponse) -> RemediationResult:
        error_msg = config.get("message", "Guardrail violation detected")
        return RemediationResult(
            action=RemediationAction.EXCEPTION,
            applied=True,
            error=error_msg,
        )

    def _do_custom(self, config: Dict[str, Any], response: EvaluationResponse) -> RemediationResult:
        """Execute a user-supplied custom remediation handler.

        The handler is looked up by name in a registry of callables.  Each
        handler receives ``(config, response)`` and must return a
        ``RemediationResult``.

        Register handlers via ``RemediationHandler.register_custom_handler()``.
        """
        handler_name = config.get("handler")
        if not handler_name:
            return RemediationResult(
                action=RemediationAction.CUSTOM, applied=False,
                error="No handler specified in on_fail_config",
            )

        handler_fn = self._custom_handlers.get(handler_name)
        if handler_fn is None:
            logger.warning(f"Custom handler '{handler_name}' not registered")
            return RemediationResult(
                action=RemediationAction.CUSTOM, applied=False,
                error=f"Custom handler '{handler_name}' not registered",
            )

        try:
            result = handler_fn(config, response)
            if not isinstance(result, RemediationResult):
                # Allow handlers to return a dict for convenience
                result = RemediationResult(
                    action=RemediationAction.CUSTOM,
                    applied=True,
                    fixed_text=result.get("fixed_text") if isinstance(result, dict) else None,
                    refrain_message=result.get("refrain_message") if isinstance(result, dict) else None,
                )
            logger.info(f"Custom handler '{handler_name}' executed successfully")
            return result
        except Exception as e:
            logger.exception(f"Custom handler '{handler_name}' failed: {e}")
            return RemediationResult(
                action=RemediationAction.CUSTOM, applied=False,
                error=f"Handler error: {e}",
            )

    def _do_ask_human(self, config: Dict[str, Any], response: EvaluationResponse) -> RemediationResult:
        """Queue the evaluation for human review.

        The response is held in a PENDING state until a human reviewer
        approves or rejects it via the review queue API.
        """
        queue_name = config.get("queue", "default")
        timeout_minutes = config.get("timeout_minutes", 60)
        message = config.get("message", "This content has been flagged for human review.")

        # Generate a review ID that the caller can poll
        import uuid
        review_id = str(uuid.uuid4())

        return RemediationResult(
            action=RemediationAction.ASK_HUMAN,
            applied=True,
            refrain_message=message,
            reask_prompt=f"review_queue:{queue_name}:review_id:{review_id}:timeout:{timeout_minutes}",
        )

    @staticmethod
    def _filter_fields(text: str, fields: List[str]) -> str:
        """Remove named fields from JSON text."""
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                for field in fields:
                    data.pop(field, None)
                return json.dumps(data)
        except (json.JSONDecodeError, TypeError):
            pass
        return text
