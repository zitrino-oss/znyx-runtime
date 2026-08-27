"""Rate-limit middleware basics: enforcement, headers, bypass paths, and the
production safety gate."""
import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from znyx_core.middleware.rate_limit import RateLimitMiddleware, SlidingWindowCounter


def _make_app(**middleware_kwargs):
    app = FastAPI()

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    app.add_middleware(RateLimitMiddleware, **middleware_kwargs)
    return app


class TestSlidingWindowCounter:
    def test_allows_up_to_limit_then_denies(self):
        async def scenario():
            counter = SlidingWindowCounter(requests_per_minute=2, burst=1)
            outcomes = [await counter.is_allowed("k") for _ in range(4)]
            return outcomes

        outcomes = asyncio.run(scenario())
        assert [allowed for allowed, *_ in outcomes] == [True, True, True, False]
        allowed, limit, remaining, reset_at = outcomes[-1]
        assert limit == 3 and remaining == 0 and reset_at > 0

    def test_keys_are_independent(self):
        async def scenario():
            counter = SlidingWindowCounter(requests_per_minute=1, burst=0)
            await counter.is_allowed("a")
            denied_a = (await counter.is_allowed("a"))[0]
            allowed_b = (await counter.is_allowed("b"))[0]
            return denied_a, allowed_b

        denied_a, allowed_b = asyncio.run(scenario())
        assert denied_a is False and allowed_b is True


class TestMiddleware:
    def test_enforces_limit_with_headers(self):
        client = TestClient(_make_app(enabled=True, requests_per_minute=3, burst=0))
        for expected_remaining in (2, 1, 0):
            response = client.get("/ping")
            assert response.status_code == 200
            assert response.headers["X-RateLimit-Limit"] == "3"
            assert response.headers["X-RateLimit-Remaining"] == str(expected_remaining)

        denied = client.get("/ping")
        assert denied.status_code == 429
        assert denied.headers["X-RateLimit-Remaining"] == "0"
        assert int(denied.headers["Retry-After"]) >= 1
        assert "Too many requests" in denied.json()["detail"]

    def test_health_endpoints_bypass_the_limit(self):
        client = TestClient(_make_app(enabled=True, requests_per_minute=1, burst=0))
        for _ in range(5):
            assert client.get("/healthz").status_code == 200

    def test_disabled_middleware_passes_everything(self):
        client = TestClient(_make_app(enabled=False, requests_per_minute=1, burst=0))
        for _ in range(5):
            response = client.get("/ping")
            assert response.status_code == 200
            assert "X-RateLimit-Limit" not in response.headers

    def test_api_key_gets_its_own_bucket(self):
        # A keyed request consumes both the IP and the key bucket; the
        # response reports the more restrictive remaining count.
        client = TestClient(_make_app(enabled=True, requests_per_minute=5, burst=0))
        first = client.get("/ping", headers={"X-API-Key": "abc"})
        second = client.get("/ping", headers={"X-API-Key": "abc"})
        assert first.status_code == second.status_code == 200
        assert int(second.headers["X-RateLimit-Remaining"]) < int(
            first.headers["X-RateLimit-Remaining"])

    def test_burst_extends_the_limit(self):
        client = TestClient(_make_app(enabled=True, requests_per_minute=1, burst=2))
        statuses = [client.get("/ping").status_code for _ in range(4)]
        assert statuses == [200, 200, 200, 429]


class TestProductionGate:
    def test_in_memory_limiter_rejected_in_production(self, monkeypatch):
        # In production without Redis the per-pod in-memory limiter silently
        # weakens configured limits, so construction must fail fast.
        for var in ("ZNYX_ENV", "ZNYX_ENVIRONMENT", "ENVIRONMENT", "GUARDRAILS_ENVIRONMENT"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv("ZNYX_REDIS_URL", raising=False)
        monkeypatch.delenv("REDIS_URL", raising=False)
        monkeypatch.delenv("RATE_LIMIT_ALLOW_IN_MEMORY", raising=False)
        app = _make_app(enabled=True, requests_per_minute=3, burst=0)
        # The middleware stack is built lazily, so the gate fires on first use.
        with pytest.raises(RuntimeError, match="rejected in production"):
            with TestClient(app) as client:
                client.get("/ping")

    def test_explicit_opt_in_allows_in_memory_in_production(self, monkeypatch):
        for var in ("ZNYX_ENV", "ZNYX_ENVIRONMENT", "ENVIRONMENT", "GUARDRAILS_ENVIRONMENT"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("RATE_LIMIT_ALLOW_IN_MEMORY", "true")
        client = TestClient(_make_app(enabled=True, requests_per_minute=3, burst=0))
        assert client.get("/ping").status_code == 200
