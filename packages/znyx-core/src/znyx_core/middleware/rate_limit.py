"""
Global API Rate Limiting Middleware

Rate limiter for FastAPI/Starlette. Limits requests by client IP and
optionally by API key. When a Redis URL is configured (ZNYX_REDIS_URL or
REDIS_URL) counting happens in Redis so the limit is shared across
replicas; otherwise an in-memory sliding window is used (per-pod limits).
The redis package is an optional dependency - if it is missing or Redis
becomes unreachable, the middleware logs the degradation and falls back
to the in-memory limiter. Configurable via environment variables.
"""
import hashlib
import os
import time
import asyncio
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# Endpoints that bypass rate limiting
BYPASS_PATHS = {"/healthz", "/readyz"}

# How often to run cleanup (seconds)
CLEANUP_INTERVAL = 60


class SlidingWindowCounter:
    """Thread-safe sliding window rate limiter using asyncio locks."""

    def __init__(self, requests_per_minute: int, burst: int):
        self.requests_per_minute = requests_per_minute
        self.burst = burst
        self.window = 60.0  # 1 minute window
        # key -> list of request timestamps
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def is_allowed(self, key: str) -> Tuple[bool, int, int, float]:
        """
        Check if a request is allowed for the given key.

        Returns:
            (allowed, limit, remaining, reset_at)
        """
        now = time.time()
        window_start = now - self.window
        limit = self.requests_per_minute + self.burst

        async with self._lock:
            # Prune expired timestamps
            timestamps = self._requests[key]
            self._requests[key] = [t for t in timestamps if t > window_start]
            timestamps = self._requests[key]

            remaining = max(0, limit - len(timestamps))
            # Reset time is when the oldest request in the window expires
            reset_at = (timestamps[0] + self.window) if timestamps else (now + self.window)

            if len(timestamps) >= limit:
                return False, limit, 0, reset_at

            self._requests[key].append(now)
            remaining = max(0, limit - len(self._requests[key]))
            return True, limit, remaining, now + self.window

    async def cleanup(self):
        """Remove stale entries that have no requests in the current window."""
        now = time.time()
        window_start = now - self.window

        async with self._lock:
            stale_keys = [
                key for key, timestamps in self._requests.items()
                if not timestamps or all(t <= window_start for t in timestamps)
            ]
            for key in stale_keys:
                del self._requests[key]

            if stale_keys:
                logger.debug(f"Rate limiter cleanup: removed {len(stale_keys)} stale entries")


class RedisFixedWindowCounter:
    """Fixed-window rate limiter backed by Redis, shared across replicas.

    Uses INCR plus EXPIRE on a key scoped to the current minute window.
    The window id is part of the key, so a lost EXPIRE can only leak a
    small stale key - it never corrupts counting.
    """

    def __init__(self, client, requests_per_minute: int, burst: int,
                 prefix: str = "znyx:ratelimit"):
        self._client = client
        self.requests_per_minute = requests_per_minute
        self.burst = burst
        self.window = 60.0  # 1 minute window
        self.prefix = prefix

    async def is_allowed(self, key: str) -> Tuple[bool, int, int, float]:
        """
        Check if a request is allowed for the given key.

        Returns:
            (allowed, limit, remaining, reset_at)
        """
        now = time.time()
        window_id = int(now // self.window)
        reset_at = (window_id + 1) * self.window
        limit = self.requests_per_minute + self.burst

        redis_key = f"{self.prefix}:{key}:{window_id}"
        count = await self._client.incr(redis_key)
        if count == 1:
            await self._client.expire(redis_key, int(self.window * 2))

        remaining = max(0, limit - count)
        return count <= limit, limit, remaining, reset_at


def _create_redis_client(url: str):
    """Build an async Redis client, or None when unavailable.

    The redis package is an optional dependency - the runtime must not
    require it, so it is imported lazily and failure means in-memory
    fallback rather than a crash.
    """
    try:
        import redis.asyncio as aioredis
    except ImportError:
        logger.warning(
            "A Redis URL is configured for rate limiting but the 'redis' "
            "package is not installed; falling back to in-memory per-pod "
            "limits. Install the redis package to enable shared limits."
        )
        return None
    try:
        return aioredis.from_url(url)
    except Exception as e:
        logger.warning(
            "Could not create Redis client for rate limiting (%s); "
            "falling back to in-memory per-pod limits", e,
        )
        return None


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI/Starlette middleware for global API rate limiting.

    Rate limits by client IP address. If an X-API-Key header is present,
    the API key is also rate-limited independently.

    Configuration via environment variables:
        RATE_LIMIT_REQUESTS_PER_MINUTE  - max requests per minute (default: 60)
        RATE_LIMIT_BURST                - extra burst allowance (default: 10)
        RATE_LIMIT_ENABLED              - enable/disable (default: true)
        ZNYX_REDIS_URL / REDIS_URL      - Redis backend for replica-shared limits
        TRUSTED_PROXY_DEPTH             - trusted reverse proxies in front (default: 0)
    """

    def __init__(self, app, enabled=None, requests_per_minute=None, burst=None,
                 redis_client=None):
        super().__init__(app)
        self.enabled = (
            enabled if enabled is not None
            else os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
        )
        from znyx_core.config.tunables import (
            RATE_LIMIT_REQUESTS_PER_MINUTE,
            RATE_LIMIT_BURST,
        )

        self.requests_per_minute = (
            requests_per_minute if requests_per_minute is not None
            else RATE_LIMIT_REQUESTS_PER_MINUTE
        )
        self.burst = (
            burst if burst is not None
            else RATE_LIMIT_BURST
        )

        # Production safety gate: multi-pod deploys without Redis get per-pod
        # limits only. That silently halves whatever limit an operator
        # configured (and can allow 10× a distributed attacker through). We
        # fail-fast at startup so misconfigured deploys can't swallow load.
        from znyx_core.utils.env import is_production
        _is_prod = is_production()
        _redis_url = os.getenv("ZNYX_REDIS_URL") or os.getenv("REDIS_URL")
        _allow_in_memory = os.getenv("RATE_LIMIT_ALLOW_IN_MEMORY", "false").lower() in (
            "1", "true", "yes",
        )
        if self.enabled and _is_prod and not _redis_url and not _allow_in_memory:
            raise RuntimeError(
                "In-memory rate limiter rejected in production. Set "
                "ZNYX_REDIS_URL for a Redis-backed distributed limiter, or "
                "set RATE_LIMIT_ALLOW_IN_MEMORY=true if you understand the "
                "risk (per-pod limits only, no cross-pod coordination)."
            )

        # How many trusted reverse proxies sit in front of the app.
        # 0 (default) = ignore X-Forwarded-For entirely (use connection IP).
        # 1 = one proxy (e.g. ALB/nginx); take the last IP in the chain.
        self._trusted_proxy_depth = int(
            os.getenv("TRUSTED_PROXY_DEPTH", "0")
        )

        # In-memory limiters always exist: they are the primary backend when
        # no Redis is configured and the fallback when Redis is unreachable.
        self.ip_limiter = SlidingWindowCounter(self.requests_per_minute, self.burst)
        self.key_limiter = SlidingWindowCounter(self.requests_per_minute, self.burst)

        # Redis-backed shared limiter (optional). A client can be injected
        # directly (tests, custom pooling); otherwise it is built lazily
        # from the configured URL.
        self._redis_limiter = None
        self._redis_degraded = False
        if self.enabled:
            if redis_client is None and _redis_url:
                redis_client = _create_redis_client(_redis_url)
            if redis_client is not None:
                self._redis_limiter = RedisFixedWindowCounter(
                    redis_client, self.requests_per_minute, self.burst,
                )

        self._cleanup_task = None
        if self.enabled:
            backend = "Redis" if self._redis_limiter is not None else "in-memory"
            logger.info(
                "Rate limiting enabled (%s backend): %s/min + %s burst",
                backend, self.requests_per_minute, self.burst,
            )

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP, respecting trusted proxy depth.

        When ``TRUSTED_PROXY_DEPTH=0`` (default) the header is ignored
        entirely - only the TCP-level peer address is used.  This is
        the safest option when there is no reverse proxy.

        When ``TRUSTED_PROXY_DEPTH=N`` (e.g. 1 for a single ALB/nginx),
        we take the Nth entry from the **right** of the
        ``X-Forwarded-For`` chain, which is the one injected by the
        outermost trusted proxy.
        """
        if self._trusted_proxy_depth > 0:
            forwarded_for = request.headers.get("x-forwarded-for")
            if forwarded_for:
                parts = [p.strip() for p in forwarded_for.split(",")]
                # Index from the right: depth=1 → parts[-1] (added by proxy)
                idx = -self._trusted_proxy_depth
                if abs(idx) <= len(parts):
                    return parts[idx]
        return request.client.host if request.client else "unknown"

    async def _check(self, fallback_limiter: SlidingWindowCounter,
                     key: str) -> Tuple[bool, int, int, float]:
        """Run the shared Redis limiter when configured, else the in-memory one.

        Redis failures degrade to per-pod in-memory limits so the API stays
        up; the degradation is logged once and again on recovery.
        """
        if self._redis_limiter is not None:
            try:
                result = await self._redis_limiter.is_allowed(key)
                if self._redis_degraded:
                    self._redis_degraded = False
                    logger.info(
                        "Redis rate limiter recovered; shared limits restored"
                    )
                return result
            except Exception as e:
                if not self._redis_degraded:
                    self._redis_degraded = True
                    logger.warning(
                        "Redis rate limiter error (%s); falling back to "
                        "in-memory per-pod limits", e,
                    )
        return await fallback_limiter.is_allowed(key)

    async def _ensure_cleanup_task(self):
        """Start the periodic cleanup task if not already running."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

    async def _periodic_cleanup(self):
        """Periodically clean up stale rate limit entries."""
        while True:
            try:
                await asyncio.sleep(CLEANUP_INTERVAL)
                await self.ip_limiter.cleanup()
                await self.key_limiter.cleanup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Rate limiter cleanup error: {e}")

    async def dispatch(self, request: Request, call_next):
        # Skip if disabled
        if not self.enabled:
            return await call_next(request)

        # Skip health check endpoints
        if request.url.path in BYPASS_PATHS:
            return await call_next(request)

        # Ensure cleanup task is running
        await self._ensure_cleanup_task()

        # Check IP-based rate limit
        client_ip = self._get_client_ip(request)
        ip_key = f"ip:{client_ip}"
        allowed, limit, remaining, reset_at = await self._check(self.ip_limiter, ip_key)

        if not allowed:
            retry_after = max(1, int(reset_at - time.time()))
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please retry later."},
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(reset_at)),
                },
            )

        # Check API-key-based rate limit (if header present)
        api_key = request.headers.get("x-api-key")
        if api_key:
            ak_key = f"key:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"
            ak_allowed, ak_limit, ak_remaining, ak_reset = await self._check(self.key_limiter, ak_key)
            if not ak_allowed:
                retry_after = max(1, int(ak_reset - time.time()))
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Too many requests. Please retry later."},
                    headers={
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(ak_limit),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int(ak_reset)),
                    },
                )
            # Use the more restrictive remaining count
            remaining = min(remaining, ak_remaining)
            limit = min(limit, ak_limit)
            reset_at = min(reset_at, ak_reset)

        # Process the request
        response = await call_next(request)

        # Add rate limit headers to successful responses
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(reset_at))

        return response
