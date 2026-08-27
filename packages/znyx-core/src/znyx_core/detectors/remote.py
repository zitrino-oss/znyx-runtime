"""Remote detector - call external inference endpoints with retry, backoff, and circuit breaker.

Replaces the simple ``WebhookDetector`` with a production-grade remote
detector that supports:
  - Configurable auth (API key header, bearer token)
  - Exponential backoff retry
  - Circuit breaker (open after N failures, half-open after cooldown)
  - Configurable input/output field mapping
  - Health-check endpoint
"""
import asyncio
import logging
import re
import time
from enum import Enum
from typing import Any, Dict, Optional

import httpx

from znyx_core.core.models import Decision, DetectorResult, RuleHit, Severity
from znyx_core.net_guard import resolve_egress_target, UnsafeEgressURL

logger = logging.getLogger(__name__)


def _slug(value: Any) -> str:
    """Lowercase, non-alphanumerics collapsed to ``_`` — safe as a rule_id segment."""
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _severity_for(risk_score: int) -> Severity:
    """Mirror the LOW/MEDIUM/HIGH risk bands the deterministic detectors use, so an
    ML-layer hit sorts alongside them instead of always reporting MEDIUM."""
    if risk_score >= 80:
        return Severity.HIGH
    if risk_score >= 40:
        return Severity.MEDIUM
    return Severity.LOW


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Simple circuit breaker with configurable thresholds."""

    def __init__(self, failure_threshold: int = 5, cooldown_seconds: float = 30.0):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: float = 0.0

    def record_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

    def allow_request(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.cooldown_seconds:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        # HALF_OPEN: allow one request to test
        return True


class MalformedRemoteResponse(ValueError):
    """Response payload that violates the detector response contract.

    Treated like a transport error: the attempt counts as a circuit-breaker
    failure and the request resolves through the configured fail_open /
    fail_closed policy, never silently as ALLOW.
    """


class RemoteDetector:
    """Detector that delegates to a remote HTTP endpoint.

    Config keys:
        endpoint_url (str):      Required. The inference endpoint URL.
        auth_type (str):         "api_key" | "bearer" | "none" (default "none")
        auth_header (str):       Header name for api_key auth (default "X-API-Key")
        auth_value (str):        The API key or bearer token value
        input_field (str):       JSON field name to send text in (default "text")
        output_decision_field:   JSON path for decision in response (default "decision")
        output_score_field:      JSON path for risk_score (default "risk_score")
        output_message_field:    JSON path for message (default "message")
        output_confidence_field: JSON path for confidence (default "confidence")
        output_calibrated_field: JSON path for calibrated_score (default "calibrated_score")
        output_label_scores_field: JSON path for label_scores dict (default "label_scores")
        output_model_version_field: JSON path for model_version (default "model_version")
        task (str):              Optional inference task name added to the request payload
        model_id (str):          Optional pinned model id added to the request payload
        model_revision (str):    Optional pinned revision added to the request payload
        timeout_seconds (float): Per-attempt request timeout (default 10.0)
        total_deadline_seconds (float): Optional wall-clock budget across ALL retries
                                 (honours policy timeout_ms). None = today's behaviour
                                 (per-attempt timeout with full retries).
        max_retries (int):       Max retry attempts (default 2)
        retry_backoff (float):   Base backoff seconds (default 0.5)
        circuit_failure_threshold (int): Failures before circuit opens (default 5)
        circuit_cooldown (float): Seconds before half-open (default 30.0)
        health_url (str):        Optional health-check endpoint
        fail_open (bool):        If True, ALLOW on error; if False, BLOCK (default True)
    """

    def __init__(self, config: Dict[str, Any]):
        self.endpoint_url = config.get("endpoint_url", "")
        self.auth_type = config.get("auth_type", "none")
        self.auth_header = config.get("auth_header", "X-API-Key")
        self.auth_value = config.get("auth_value", "")
        self.input_field = config.get("input_field", "text")
        self.output_decision_field = config.get("output_decision_field", "decision")
        self.output_score_field = config.get("output_score_field", "risk_score")
        self.output_message_field = config.get("output_message_field", "message")
        # model-backed confidence contract field paths.
        self.output_confidence_field = config.get("output_confidence_field", "confidence")
        self.output_calibrated_field = config.get("output_calibrated_field", "calibrated_score")
        self.output_label_scores_field = config.get("output_label_scores_field", "label_scores")
        self.output_model_version_field = config.get("output_model_version_field", "model_version")
        # richer request payload (inference task + pinned model).
        self.task = config.get("task")
        self.model_id = config.get("model_id")
        self.model_revision = config.get("model_revision") or config.get("revision")
        self.timeout = config.get("timeout_seconds", 10.0)
        # total wall-clock deadline across retries (None = per-attempt only).
        self.total_deadline = config.get("total_deadline_seconds")
        self.max_retries = config.get("max_retries", 2)
        self.retry_backoff = config.get("retry_backoff", 0.5)
        self.fail_open = config.get("fail_open", True)

        self._circuit = CircuitBreaker(
            failure_threshold=config.get("circuit_failure_threshold", 5),
            cooldown_seconds=config.get("circuit_cooldown", 30.0),
        )
        self._health_url = config.get("health_url")
        # Per-request runner params (e.g. allowed_languages for language detection).
        self._params = config.get("params")

    def _build_payload(self, text: str) -> Dict[str, Any]:
        """Build the request body. Keeps the back-compat ``{input_field: text}``
        shape and adds the task/model pins only when configured."""
        payload: Dict[str, Any] = {self.input_field: text}
        if self.task:
            payload["task"] = self.task
        if self.model_id:
            payload["model_id"] = self.model_id
            if self.model_revision:
                payload["revision"] = self.model_revision
        if self._params:
            payload["params"] = self._params
        return payload

    def _build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.auth_type == "api_key" and self.auth_value:
            headers[self.auth_header] = self.auth_value
        elif self.auth_type == "bearer" and self.auth_value:
            headers["Authorization"] = f"Bearer {self.auth_value}"
        return headers

    def detect(self, text: str) -> DetectorResult:
        """Synchronous wrapper for the async detection call.

        When called from the synchronous orchestrator pipeline, we run
        the async method via asyncio.  If already inside an event loop
        (FastAPI), this is invoked via ``asyncio.run`` in a thread.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're in an async context - run sync via new event loop in thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self._detect_async(text))
                # Cap the thread wait by the total deadline when one is set.
                wait = (self.total_deadline + 5) if self.total_deadline else (self.timeout + 5)
                return future.result(timeout=wait)
        else:
            return asyncio.run(self._detect_async(text))

    async def detect_async(self, text: str) -> DetectorResult:
        """Native async entrypoint — no thread hop.

        Preferred from already-async callers (e.g. the escalation path), which
        can reuse the running event loop instead of paying the thread-per-call cost
        of the sync ``detect()`` wrapper.
        """
        return await self._detect_async(text)

    async def _detect_async(self, text: str) -> DetectorResult:
        """Call the remote endpoint with retry, circuit breaker, and an optional
        total wall-clock deadline across all attempts."""
        if not self.endpoint_url:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        # Total budget across retries. None → per-attempt timeout only.
        deadline = (time.monotonic() + self.total_deadline) if self.total_deadline else None

        # SSRF guard: never let a configured endpoint reach the cloud-metadata
        # service / link-local range (credential theft). Private/internal IPs
        # are permitted because remote detectors may be self-hosted on an
        # internal network. Resolve+validate DNS ONCE here and connect to the
        # pinned IP below, so a rebinding attacker can't flip the name to a
        # metadata IP between this check and the connection. Resolution is
        # blocking, so run it off the event loop.
        try:
            target = await asyncio.to_thread(
                resolve_egress_target, self.endpoint_url, allow_private=True
            )
        except UnsafeEgressURL as e:
            logger.warning("Remote detector endpoint blocked by SSRF guard: %s", e)
            return self._fail_result(f"Endpoint blocked: {e}")

        if not self._circuit.allow_request():
            logger.warning("Remote detector circuit is open - failing open/closed per config")
            return self._fail_result("Circuit breaker open")

        payload = self._build_payload(text)
        headers = self._build_headers()

        last_error = None
        for attempt in range(self.max_retries + 1):
            # Honour the total deadline: shrink the per-attempt timeout to the
            # remaining budget, and bail before starting an attempt with none left.
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    last_error = last_error or TimeoutError("total deadline exceeded")
                    break
                attempt_timeout = min(self.timeout, remaining)
            else:
                attempt_timeout = self.timeout

            try:
                async with httpx.AsyncClient(timeout=attempt_timeout) as client:
                    resp = await client.post(
                        target.connect_url,
                        json=payload,
                        headers={**headers, "Host": target.host_header},
                        extensions={"sni_hostname": target.sni_hostname},
                    )
                    resp.raise_for_status()
                    data = resp.json()

                # Validate the contract BEFORE recording success: a malformed
                # payload (raised below) is a failure like any transport error,
                # so it retries, counts against the circuit, and resolves via
                # the fail_open/fail_closed policy.
                result = self._parse_response(data)
                self._circuit.record_success()
                return result

            except Exception as e:
                last_error = e
                self._circuit.record_failure()
                if attempt < self.max_retries:
                    wait = self.retry_backoff * (2 ** attempt)
                    # Don't sleep past the total deadline.
                    if deadline is not None:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        wait = min(wait, remaining)
                    logger.debug(f"Remote detector retry {attempt + 1} after {wait}s: {e}")
                    await asyncio.sleep(wait)

        logger.error(f"Remote detector failed after {self.max_retries + 1} attempts: {last_error}")
        return self._fail_result(str(last_error))

    def _parse_response(self, data: Any) -> DetectorResult:
        """Parse the remote endpoint response into a DetectorResult.

        Raises MalformedRemoteResponse when the payload is not a JSON object
        or its decision field is missing or not a valid Decision value. That
        must not default to ALLOW: a broken (or tampered-with) endpoint would
        silently bypass a fail-closed configuration.
        """
        if not isinstance(data, dict):
            raise MalformedRemoteResponse(
                f"expected a JSON object, got {type(data).__name__}"
            )

        decision_str = self._get_nested(data, self.output_decision_field)
        if decision_str is None:
            raise MalformedRemoteResponse(
                f"response missing decision field '{self.output_decision_field}'"
            )
        try:
            decision = Decision(str(decision_str).upper())
        except ValueError:
            raise MalformedRemoteResponse(
                f"invalid decision value {decision_str!r}"
            ) from None

        score = self._get_nested(data, self.output_score_field)
        message = self._get_nested(data, self.output_message_field)

        # A malformed score alongside a valid decision clamps rather than
        # fails - the decision is the load-bearing field of the contract.
        try:
            risk_score = int(score) if score is not None else 0
        except (TypeError, ValueError):
            risk_score = 0
        risk_score = min(100, max(0, risk_score))

        label_scores = self._coerce_label_scores(
            self._get_nested(data, self.output_label_scores_field))

        # A non-ALLOW decision ALWAYS yields a rule hit. This previously also required a
        # non-empty ``message``, so a sidecar response without that field would BLOCK the
        # request while contributing nothing to ``rule_hit_ids`` — leaving the audit trail
        # to attribute the block to whichever deterministic detector happened to hit.
        rule_hits = []
        if decision != Decision.ALLOW:
            top_label = self._top_label(label_scores)
            rule_hits.append(RuleHit(
                rule_id=self._rule_id(_slug(top_label) if top_label else "detected"),
                severity=_severity_for(risk_score),
                message=str(message) if message else (
                    f"{self.task or 'remote'} model returned "
                    f"{decision.value} (risk={risk_score})"
                ),
            ))

        return DetectorResult(
            decision=decision,
            risk_score=risk_score,
            rule_hits=rule_hits,
            developer_message=str(message) if message else None,
            # confidence contract (all optional; absent fields stay None).
            confidence=self._coerce_float(self._get_nested(data, self.output_confidence_field)),
            calibrated_score=self._coerce_float(self._get_nested(data, self.output_calibrated_field)),
            label_scores=label_scores,
            model_version=(
                str(mv) if (mv := self._get_nested(data, self.output_model_version_field)) is not None else None
            ),
        )

    def _rule_id(self, suffix: str) -> str:
        """Task-scoped rule id, so an ML-layer hit is attributable in the audit trail.

        ``prompt_injection.injection`` / ``toxicity.toxic`` group cleanly in analytics and
        tell an operator which model fired. Falls back to the old generic ``remote_detector``
        prefix only when no task is configured (a bare custom webhook backend).
        """
        return f"{_slug(self.task) if self.task else 'remote_detector'}.{suffix}"

    @staticmethod
    def _top_label(label_scores: Optional[Dict[str, float]]) -> Optional[str]:
        """Highest-scoring label, used to name the rule hit. None when the backend
        reported no label_scores (the minimal 3-field contract)."""
        if not label_scores:
            return None
        return max(label_scores.items(), key=lambda kv: kv[1])[0]

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        # Used for confidence / calibrated_score, which DetectorResult constrains to
        # [0,1]. Clamp so a misbehaving inference service returning e.g. 1.2 or -0.3
        # doesn't raise a ValidationError and abort the whole parse.
        if value is None:
            return None
        try:
            return min(1.0, max(0.0, float(value)))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_label_scores(value: Any) -> Optional[Dict[str, float]]:
        if not isinstance(value, dict):
            return None
        out: Dict[str, float] = {}
        for k, v in value.items():
            try:
                # label scores are probabilities — clamp to [0,1] (matches the
                # DetectorResult contract) so an out-of-range value from a misbehaving
                # service doesn't abort the parse.
                out[str(k)] = min(1.0, max(0.0, float(v)))
            except (TypeError, ValueError):
                continue
        return out or None

    def _fail_result(self, error: str) -> DetectorResult:
        if self.fail_open:
            return DetectorResult(
                decision=Decision.ALLOW, risk_score=0,
                developer_message=f"Remote detector unavailable (fail-open): {error}",
            )
        return DetectorResult(
            decision=Decision.BLOCK, risk_score=100,
            rule_hits=[RuleHit(
                rule_id=self._rule_id("unavailable"),
                severity=Severity.HIGH,
                message=f"Remote detector unavailable (fail-closed): {error}",
            )],
            developer_message=f"Remote detector error: {error}",
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the remote endpoint."""
        url = self._health_url or self.endpoint_url
        if not url:
            return {"healthy": False, "error": "No endpoint configured"}
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url, headers=self._build_headers())
                return {
                    "healthy": resp.status_code < 400,
                    "status_code": resp.status_code,
                    "circuit_state": self._circuit.state.value,
                }
        except Exception as e:
            return {"healthy": False, "error": str(e), "circuit_state": self._circuit.state.value}

    @property
    def circuit_state(self) -> str:
        return self._circuit.state.value

    @staticmethod
    def _get_nested(data: Dict[str, Any], path: str) -> Any:
        """Get a value from nested dict using dot-separated path."""
        if not path:
            return None
        parts = path.split(".")
        current = data
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current
