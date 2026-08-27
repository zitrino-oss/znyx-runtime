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
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from znyx_core.core.models import (
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

# Same "~/.znyx/<name>.spool" convention as znyx_runtime.audit_sink._DEFAULT_SPOOL /
# znyx_runtime.judge_audit_sink._DEFAULT_JUDGE_SPOOL, so a single-host deploy has the
# runtime and the control plane agree on a path with zero explicit wiring.
_DEFAULT_ASK_HUMAN_SPOOL = Path.home() / ".znyx" / "ask-human.spool"


class AskHumanSpool:
    """Durable append-only JSON-lines spool of ask_human review requests.

    The runtime is deliberately DB-free (see the module docstring), so
    ``RemediationHandler._do_ask_human`` cannot enqueue directly into a control
    plane's review queue. It durably spools the request here instead; a
    control plane drains the spool and inserts each item into its own review
    queue on the other end.

    Mirrors ``znyx_runtime.audit_sink.SpoolAuditSink`` / ``znyx_runtime.
    judge_audit_sink.JudgeAuditSpool`` exactly: one JSON object per line, an
    append + flush + fsync before returning, and a lock guarding concurrent
    writers. Reimplemented here (rather than imported) because znyx_core does not
    -- and must not -- depend on znyx_runtime.
    """

    def __init__(self, spool_path: Optional[str] = None):
        self.path = Path(spool_path) if spool_path else _DEFAULT_ASK_HUMAN_SPOOL
        self._lock = threading.Lock()

    def _append_sync(self, line: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def record(self, event: Dict[str, Any]) -> None:
        """Durably append one ask_human event. Best-effort: a write failure is
        logged and swallowed -- the BLOCK decision the caller already returned
        does not depend on the review actually reaching the queue, so remediation
        must never raise here."""
        line = json.dumps(event, separators=(",", ":"), sort_keys=True, default=str)
        try:
            with self._lock:
                self._append_sync(line)
        except Exception as exc:  # noqa: BLE001 - never fail the response over a spool write
            logger.warning("ask_human spool write failed (review not queued): %s", exc)

    def read_all(self) -> List[Dict[str, Any]]:
        """Every spooled event (used by the CP drainer / tests). A corrupt/partial
        line is skipped so one bad record can't abort the drain."""
        if not self.path.exists():
            return []
        out: List[Dict[str, Any]] = []
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                logger.warning("skipping unparseable ask_human spool line: %s", exc)
        return out


class RemediationHandler:
    """Stateless handler that applies on_fail remediation to an evaluation result.

    Custom handlers can be registered via ``register_custom_handler()``
    and referenced in policies as ``on_fail: custom`` with
    ``on_fail_config: {handler: "handler_name"}``.
    """

    # Class-level registry of custom handler functions
    _custom_handlers: Dict[str, Any] = {}

    def __init__(self, ask_human_spool_path: Optional[str] = None):
        """
        Args:
            ask_human_spool_path: Override for the ask_human durable spool file.
                None uses ``ZNYX_ASK_HUMAN_SPOOL`` if set, else ``~/.znyx/ask-human.spool``.
        """
        self._ask_human_spool = AskHumanSpool(
            ask_human_spool_path or os.getenv("ZNYX_ASK_HUMAN_SPOOL") or None
        )

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
        request: Optional[Any] = None,
    ) -> EvaluationResponse:
        """Apply remediation based on policy on_fail settings.

        Looks at each detector section's ``on_fail`` and applies the **first**
        matching action whose detector actually triggered.  If no on_fail is
        configured, the response passes through unchanged.

        Args:
            request: The originating evaluation request (``EvaluationRequest`` /
                ``ToolEvaluationRequest``), if the caller has one. Optional and
                unused by every action except ``ask_human``, which reads
                ``tenant_id``/``app_id``/``env``/``text`` off it (via ``getattr``, so
                any request-shaped object works) to durably record the review with
                enough context for a human reviewer. Omitting it still queues the
                review; it is just recorded with no org scope or input text.
        """
        if response.decision == Decision.ALLOW:
            return response

        # Collect on_fail configs from all detector sections that fired
        action, config = self._resolve_action(response, policy)
        if action is None or action == RemediationAction.NOOP:
            return response

        result = self._execute(action, config, response, request)
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
            # rule_id is "<detector>.<sub>" (most detectors) or "<detector>:<sub>" — take
            # the first segment under EITHER separator so per-detector on_fail resolves for
            # dot-namespaced detectors (e.g. "excessive_agency.destructive_action"), not
            # only the colon form.
            base = re.split(r"[:.]", hit.rule_id, maxsplit=1)[0] if hit.rule_id else ""
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
        response: EvaluationResponse, request: Optional[Any] = None,
    ) -> RemediationResult:
        """Execute a single remediation action."""
        # ask_human is dispatched separately: it is the one action that reads the
        # originating request (for org scope + input text), so it needs an extra
        # argument the uniform dict-dispatch below doesn't pass to the others.
        if action == RemediationAction.ASK_HUMAN:
            return self._do_ask_human(config, response, request)

        handler = {
            RemediationAction.REASK: self._do_reask,
            RemediationAction.FIX: self._do_fix,
            RemediationAction.FILTER_FIELD: self._do_filter_field,
            RemediationAction.REFRAIN: self._do_refrain,
            RemediationAction.EXCEPTION: self._do_exception,
            RemediationAction.CUSTOM: self._do_custom,
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

    def _do_ask_human(
        self, config: Dict[str, Any], response: EvaluationResponse,
        request: Optional[Any] = None,
    ) -> RemediationResult:
        """Queue the evaluation for human review.

        The response is held in a PENDING state until a human reviewer approves or
        rejects it via the review queue API. The runtime has no DB (see module
        docstring), so the review request is durably spooled here rather than
        inserted directly; a control plane drains the spool into its own review
        queue, the same spool-then-drain transport as the egress and judge audit
        trails.
        """
        queue_name = config.get("queue", "default")
        timeout_minutes = config.get("timeout_minutes", 60)
        message = config.get("message", "This content has been flagged for human review.")

        # Generate a review ID that the caller can poll
        import uuid
        review_id = str(uuid.uuid4())

        self._ask_human_spool.record({
            "review_id": review_id,
            "queue_name": queue_name,
            "timeout_minutes": timeout_minutes,
            "message": message,
            # org/tenant scope, mirroring znyx_core.engine.egress's
            # `org_scope=getattr(request, "tenant_id", None)` -- None when the caller
            # didn't pass a request, or it isn't set on it.
            "org_scope": getattr(request, "tenant_id", None) if request is not None else None,
            "app_id": getattr(request, "app_id", None) if request is not None else None,
            "env": getattr(request, "env", None) if request is not None else None,
            "request_id": response.request_id,
            "trace_id": response.trace_id,
            # The flagged text lives on the request, not the response -- fall back to
            # sanitized_text (e.g. tool-arg evaluation) when there is no request.
            "input_text": (
                getattr(request, "text", None) if request is not None else response.sanitized_text
            ),
            "decision": getattr(response.decision, "value", str(response.decision)),
            "risk_score": response.risk_score,
            "rule_hits": [h.model_dump() for h in (response.rule_hits or [])],
            "occurred_at": datetime.now(timezone.utc).isoformat(),
        })

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
