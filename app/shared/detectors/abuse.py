"""
Abuse Controls Detector

Implements:
- Payload size limits
- Rate limiting (in-memory token bucket for R2)
- Prompt flood detection (repeat detection)
"""
import time
import hashlib
from typing import List, Dict, Any, Optional
from collections import defaultdict, deque
from app.shared.core.models import DetectorResult, RuleHit, Severity, Decision


class TokenBucket:
    """Simple in-memory token bucket for rate limiting"""

    def __init__(self, rate: int, capacity: int):
        """
        Initialize token bucket.

        Args:
            rate: Tokens per second
            capacity: Maximum tokens
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens consumed, False if insufficient
        """
        now = time.time()
        elapsed = now - self.last_update

        # Refill tokens based on elapsed time (round to avoid float drift)
        self.tokens = min(self.capacity, round(self.tokens + elapsed * self.rate, 6))
        self.last_update = now

        # Try to consume
        if self.tokens >= tokens:
            self.tokens = round(self.tokens - tokens, 6)
            return True
        return False


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

        # In-memory storage (resets on restart; use Redis for multi-instance deployments)
        # key -> TokenBucket
        self.rate_limiters: Dict[str, TokenBucket] = {}

        # key -> deque of (hash, timestamp)
        self.prompt_hashes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))

        # Cleanup tracking
        self.last_cleanup = time.time()

    def _make_key(self, tenant_id: str, app_id: str, user_id: Optional[str] = None) -> str:
        """Create rate limit key from identifiers"""
        if user_id:
            return f"{tenant_id}:{app_id}:{user_id}"
        return f"{tenant_id}:{app_id}"

    def _cleanup_old_entries(self):
        """Periodically cleanup old entries (simple memory management)"""
        now = time.time()

        # Cleanup every 5 minutes
        if now - self.last_cleanup < 300:
            return

        # Remove old prompt hashes
        for key in list(self.prompt_hashes.keys()):
            hashes = self.prompt_hashes[key]
            # Remove entries older than window
            while hashes and (now - hashes[0][1]) > self.prompt_flood_window:
                hashes.popleft()

            # Remove key if empty
            if not hashes:
                del self.prompt_hashes[key]

        self.last_cleanup = now

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
        key = self._make_key(tenant_id, app_id, user_id)

        # Get or create token bucket
        if key not in self.rate_limiters:
            # Convert per-minute to per-second rate
            rate_per_second = self.rate_limit_per_minute / 60.0
            self.rate_limiters[key] = TokenBucket(
                rate=rate_per_second,
                capacity=self.rate_limit_per_minute
            )

        bucket = self.rate_limiters[key]

        # Try to consume 1 token
        if not bucket.consume(1):
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
        key = self._make_key(tenant_id, app_id, user_id)

        # Hash the text
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        now = time.time()

        # Get hash history for this key
        hashes = self.prompt_hashes[key]

        # Remove old entries
        while hashes and (now - hashes[0][1]) > self.prompt_flood_window:
            hashes.popleft()

        # Count occurrences of this hash
        count = sum(1 for h, _ in hashes if h == text_hash)

        # Add current hash
        hashes.append((text_hash, now))

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

        # Periodic cleanup
        self._cleanup_old_entries()

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
