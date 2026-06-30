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
import time
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx

from app.shared.core.models import Decision, DetectorResult, RuleHit, Severity
from app.shared.net_guard import assert_safe_egress_url, UnsafeEgressURL

logger = logging.getLogger(__name__)


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
        timeout_seconds (float): Request timeout (default 10.0)
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
        self.timeout = config.get("timeout_seconds", 10.0)
        self.max_retries = config.get("max_retries", 2)
        self.retry_backoff = config.get("retry_backoff", 0.5)
        self.fail_open = config.get("fail_open", True)

        self._circuit = CircuitBreaker(
            failure_threshold=config.get("circuit_failure_threshold", 5),
            cooldown_seconds=config.get("circuit_cooldown", 30.0),
        )
        self._health_url = config.get("health_url")

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
                return future.result(timeout=self.timeout + 5)
        else:
            return asyncio.run(self._detect_async(text))

    async def _detect_async(self, text: str) -> DetectorResult:
        """Call the remote endpoint with retry and circuit breaker."""
        if not self.endpoint_url:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        # SSRF guard: never let a configured endpoint reach the cloud-metadata
        # service / link-local range (credential theft). Private/internal IPs
        # are permitted because remote detectors may be self-hosted on an
        # internal network. The DNS resolution is blocking, so run it off the
        # event loop.
        try:
            await asyncio.to_thread(assert_safe_egress_url, self.endpoint_url, allow_private=True)
        except UnsafeEgressURL as e:
            logger.warning("Remote detector endpoint blocked by SSRF guard: %s", e)
            return self._fail_result(f"Endpoint blocked: {e}")

        if not self._circuit.allow_request():
            logger.warning("Remote detector circuit is open - failing open/closed per config")
            return self._fail_result("Circuit breaker open")

        payload = {self.input_field: text}
        headers = self._build_headers()

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(
                        self.endpoint_url, json=payload, headers=headers,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                self._circuit.record_success()
                return self._parse_response(data)

            except Exception as e:
                last_error = e
                self._circuit.record_failure()
                if attempt < self.max_retries:
                    wait = self.retry_backoff * (2 ** attempt)
                    logger.debug(f"Remote detector retry {attempt + 1} after {wait}s: {e}")
                    await asyncio.sleep(wait)

        logger.error(f"Remote detector failed after {self.max_retries + 1} attempts: {last_error}")
        return self._fail_result(str(last_error))

    def _parse_response(self, data: Dict[str, Any]) -> DetectorResult:
        """Parse the remote endpoint response into a DetectorResult."""
        decision_str = self._get_nested(data, self.output_decision_field)
        score = self._get_nested(data, self.output_score_field)
        message = self._get_nested(data, self.output_message_field)

        try:
            decision = Decision(str(decision_str).upper()) if decision_str else Decision.ALLOW
        except ValueError:
            decision = Decision.ALLOW

        risk_score = int(score) if score is not None else 0
        rule_hits = []
        if decision != Decision.ALLOW and message:
            rule_hits.append(RuleHit(
                rule_id="remote_detector",
                severity=Severity.MEDIUM,
                message=str(message),
            ))

        return DetectorResult(
            decision=decision,
            risk_score=min(100, max(0, risk_score)),
            rule_hits=rule_hits,
            developer_message=str(message) if message else None,
        )

    def _fail_result(self, error: str) -> DetectorResult:
        if self.fail_open:
            return DetectorResult(
                decision=Decision.ALLOW, risk_score=0,
                developer_message=f"Remote detector unavailable (fail-open): {error}",
            )
        return DetectorResult(
            decision=Decision.BLOCK, risk_score=100,
            rule_hits=[RuleHit(
                rule_id="remote_detector",
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
