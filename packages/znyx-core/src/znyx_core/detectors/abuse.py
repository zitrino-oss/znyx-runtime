"""
Abuse Controls Detector

Implements:
- Payload size limits
- Rate limiting (token bucket in the shared state store)
- Prompt flood detection (repeat detection in the shared state store)

Rate and flood state lives in ``znyx_core.state_store`` so replicas pointed at
the same Redis enforce one limit between them and a restart does not reset it.
With the default in-process store the behaviour is per process, as before.
"""
import hashlib
import logging
from typing import List, Dict, Any, Optional
from znyx_core.core.models import DetectorResult, RuleHit, Severity, Decision
from znyx_core.state_store import StateUnavailable, get_state_store

logger = logging.getLogger(__name__)


class AbuseDetector:
    """Detects abuse patterns: payload limits, rate limiting, prompt flooding"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize abuse detector.

        Args:
            config: Configuration dict with keys:
                - enabled: bool (default: True)
                - max_chars_input: int (default: 100000)
                - max_chars_output: int (default: 100000)
                - max_tool_args_size: int (default: 50000)
                - rate_limit_per_minute: int (default: 60)
                - prompt_flood_threshold: int (default: 3) - same hash within 60s
                - prompt_flood_window: int (default: 60) - seconds
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        self.max_chars_input = config.get('max_chars_input', 100000)
        self.max_chars_output = config.get('max_chars_output', 100000)
        self.max_tool_args_size = config.get('max_tool_args_size', 50000)
        self.rate_limit_per_minute = config.get('rate_limit_per_minute', 60)
        self.prompt_flood_threshold = config.get('prompt_flood_threshold', 3)
        self.prompt_flood_window = config.get('prompt_flood_window', 60)
        self.repetitive_content_threshold = config.get('repetitive_content_threshold', 10)

    def _make_key(self, tenant_id: str, app_id: str, user_id: Optional[str] = None) -> str:
        """Create rate limit key from identifiers"""
        if user_id:
            return f"{tenant_id}:{app_id}:{user_id}"
        return f"{tenant_id}:{app_id}"

    def check_payload_size(self, text: str, context: str = "input") -> Optional[RuleHit]:
        """
        Check if payload size exceeds limits.

        Args:
            text: Text to check
            context: "input"|"output"|"tool_args"

        Returns:
            RuleHit if too large, None if OK
        """
        size = len(text)

        limit_map = {
            "input": self.max_chars_input,
            "output": self.max_chars_output,
            "tool_args": self.max_tool_args_size
        }

        limit = limit_map.get(context, self.max_chars_input)

        if size > limit:
            return RuleHit(
                rule_id=f"abuse.payload_too_large_{context}",
                severity=Severity.MEDIUM,
                message=f"Payload size ({size} chars) exceeds limit ({limit} chars)"
            )

        return None

    def check_rate_limit(
        self,
        tenant_id: str,
        app_id: str,
        user_id: Optional[str] = None
    ) -> Optional[RuleHit]:
        """
        Check if rate limit exceeded.

        Args:
            tenant_id: Tenant identifier
            app_id: Application identifier
            user_id: Optional user identifier

        Returns:
            RuleHit if rate limited, None if OK
        """
        identity = self._make_key(tenant_id, app_id, user_id)
        # The limit is part of the key: a bucket's rate and capacity are fixed when
        # it is created, so two policies with different limits must not share one.
        key = f"abuse:rl:{self.rate_limit_per_minute}:{identity}"

        try:
            allowed = get_state_store().consume_token(
                key,
                rate_per_second=self.rate_limit_per_minute / 60.0,
                capacity=self.rate_limit_per_minute,
            )
        except StateUnavailable as exc:
            # Shared store down with ZNYX_STATE_FAIL_MODE=closed: fail closed.
            logger.warning("Abuse rate limit failing closed, shared state unavailable: %s", exc)
            return RuleHit(
                rule_id="abuse.rate_limited",
                severity=Severity.MEDIUM,
                message="Rate limit could not be verified (shared state store unavailable); request denied"
            )

        if not allowed:
            return RuleHit(
                rule_id="abuse.rate_limited",
                severity=Severity.MEDIUM,
                message=f"Rate limit exceeded: {self.rate_limit_per_minute} requests per minute"
            )

        return None

    def check_prompt_flood(
        self,
        text: str,
        tenant_id: str,
        app_id: str,
        user_id: Optional[str] = None
    ) -> Optional[RuleHit]:
        """
        Check for prompt flooding (repeated identical requests).

        Args:
            text: Text to check
            tenant_id: Tenant identifier
            app_id: Application identifier
            user_id: Optional user identifier

        Returns:
            RuleHit if flood detected, None if OK
        """
        identity = self._make_key(tenant_id, app_id, user_id)

        # Only a hash of the text goes into the key, never the text itself.
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        # Window and threshold are part of the key so policies that differ in
        # either never share a counter.
        key = (f"abuse:flood:{self.prompt_flood_window}:{self.prompt_flood_threshold}:"
               f"{identity}:{text_hash}")

        # Fixed-window approximation of "same hash N times within W seconds": the
        # counter starts on the first sighting and resets W seconds later, rather
        # than sliding with each request.
        try:
            total = get_state_store().incr(key, {"n": 1}, ttl_seconds=self.prompt_flood_window)
        except StateUnavailable as exc:
            # detect() always runs check_rate_limit first, which fails closed on
            # the same outage, so the request is already denied there. Do not add
            # a second, misleading flood hit.
            logger.warning("Prompt flood check skipped, shared state unavailable: %s", exc)
            return None

        # Occurrences before this request (incr already counted it).
        count = int(total.get("n", 1)) - 1

        # Check if threshold exceeded
        if count >= self.prompt_flood_threshold:
            return RuleHit(
                rule_id="abuse.prompt_flood",
                severity=Severity.MEDIUM,
                message=f"Prompt flooding detected: {count} identical requests within {self.prompt_flood_window}s"
            )

        return None

    def check_repetitive_content(self, text: str) -> Optional[RuleHit]:
        """
        Check for highly repetitive content within a single request.

        Splits text into sentences/lines and flags if any phrase appears
        more than repetitive_content_threshold times.
        """
        import re as _re
        # Split on sentence-ending punctuation or newlines
        parts = [p.strip() for p in _re.split(r'[.!?\n]+', text) if p.strip()]
        if not parts:
            return None

        counts: Dict[str, int] = {}
        for part in parts:
            key = part.lower()
            counts[key] = counts.get(key, 0) + 1

        max_count = max(counts.values())
        if max_count >= self.repetitive_content_threshold:
            repeated = next(p for p, c in counts.items() if c == max_count)
            return RuleHit(
                rule_id="abuse.repetitive_content",
                severity=Severity.MEDIUM,
                message=(
                    f"Repetitive content detected: phrase repeated {max_count} times "
                    f"(threshold: {self.repetitive_content_threshold}): '{repeated[:80]}'"
                ),
            )
        return None

    def detect(
        self,
        text: str,
        tenant_id: str,
        app_id: str,
        user_id: Optional[str] = None,
        context: str = "input"
    ) -> DetectorResult:
        """
        Run all abuse checks.

        Args:
            text: Text to check
            tenant_id: Tenant identifier
            app_id: Application identifier
            user_id: Optional user identifier
            context: "input"|"output"|"tool_args"

        Returns:
            DetectorResult with BLOCK if abuse detected
        """
        if not self.enabled:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        rule_hits: List[RuleHit] = []

        # 1. Check payload size
        payload_hit = self.check_payload_size(text, context)
        if payload_hit:
            rule_hits.append(payload_hit)

        # 2. Check rate limit
        rate_hit = self.check_rate_limit(tenant_id, app_id, user_id)
        if rate_hit:
            rule_hits.append(rate_hit)

        # 3. Check prompt flood
        flood_hit = self.check_prompt_flood(text, tenant_id, app_id, user_id)
        if flood_hit:
            rule_hits.append(flood_hit)

        # 4. Check for repetitive content within this single request
        repetitive_hit = self.check_repetitive_content(text)
        if repetitive_hit:
            rule_hits.append(repetitive_hit)

        # No abuse detected
        if not rule_hits:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        # Calculate risk score
        risk_score = min(len(rule_hits) * 30, 100)

        # Block abuse
        return DetectorResult(
            decision=Decision.BLOCK,
            risk_score=risk_score,
            rule_hits=rule_hits,
            user_message="Request blocked due to abuse detection. Please try again later.",
            developer_message=f"Abuse detected: {', '.join([hit.rule_id for hit in rule_hits])}"
        )
