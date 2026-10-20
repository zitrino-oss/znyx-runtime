"""AbuseDetector keeps its rate-limit and flood state in the shared state store,
so separate detector instances (replicas) enforce one limit between them."""
import os

import pytest

from znyx_core.core.models import Decision
from znyx_core.detectors.abuse import AbuseDetector
from znyx_core.state_store import (
    RedisStateStore,
    StateUnavailable,
    make_redis_client,
    set_state_store,
)


def _rate_limited(result) -> bool:
    return any(h.rule_id == "abuse.rate_limited" for h in result.rule_hits)


def _flooded(result) -> bool:
    return any(h.rule_id == "abuse.prompt_flood" for h in result.rule_hits)


def _allowed_count(detectors, requests, tenant="t1", app="a1", user=None):
    """Send `requests` distinct prompts round-robin across `detectors`; count allowed."""
    allowed = 0
    for i in range(requests):
        det = detectors[i % len(detectors)]
        result = det.detect(f"distinct prompt {i}", tenant, app, user_id=user)
        if not _rate_limited(result):
            allowed += 1
    return allowed


def _cfg(**overrides):
    cfg = {"enabled": True, "rate_limit_per_minute": 5, "prompt_flood_threshold": 3}
    cfg.update(overrides)
    return cfg


def test_two_replicas_share_one_rate_limit():
    replica_a = AbuseDetector(_cfg())
    replica_b = AbuseDetector(_cfg())
    assert _allowed_count([replica_a, replica_b], 20) == 5


def test_different_identities_are_independent():
    det = AbuseDetector(_cfg())
    assert _allowed_count([det], 10, tenant="t1", app="a1") == 5
    # Other tenant, other app, and other user each get a full bucket.
    assert _allowed_count([det], 10, tenant="t2", app="a1") == 5
    assert _allowed_count([det], 10, tenant="t1", app="a2") == 5
    assert _allowed_count([det], 10, tenant="t1", app="a1", user="u1") == 5
    assert _allowed_count([det], 10, tenant="t1", app="a1", user="u2") == 5


def test_different_limits_do_not_share_a_bucket():
    small = AbuseDetector(_cfg(rate_limit_per_minute=2))
    large = AbuseDetector(_cfg(rate_limit_per_minute=8))
    assert _allowed_count([small], 10) == 2
    # Same identity, but the larger policy has its own, untouched bucket.
    assert _allowed_count([large], 20) == 8


def test_flood_flags_repeated_prompt_past_threshold():
    det = AbuseDetector(_cfg(rate_limit_per_minute=1000, prompt_flood_threshold=3))
    results = [det.detect("same prompt", "t1", "a1") for _ in range(6)]
    # Same rule as before: the 4th identical request is the first flagged.
    assert [_flooded(r) for r in results] == [False, False, False, True, True, True]
    assert results[3].decision == Decision.BLOCK


def test_flood_counts_are_shared_across_replicas():
    a = AbuseDetector(_cfg(rate_limit_per_minute=1000))
    b = AbuseDetector(_cfg(rate_limit_per_minute=1000))
    flags = [d.detect("same prompt", "t1", "a1") for d in (a, b, a, b)]
    assert [_flooded(r) for r in flags] == [False, False, False, True]


def test_flood_ignores_different_prompts_and_identities():
    det = AbuseDetector(_cfg(rate_limit_per_minute=1000, prompt_flood_threshold=3))
    for i in range(10):
        assert not _flooded(det.detect(f"prompt number {i}", "t1", "a1"))
    # The same text from other identities does not accumulate against t1.
    for _ in range(3):
        det.detect("shared text", "t1", "a1")
    assert not _flooded(det.detect("shared text", "t2", "a1"))
    assert not _flooded(det.detect("shared text", "t1", "a1", user_id="u1"))


def test_flood_keys_hold_a_hash_not_raw_text():
    seen = []

    class Recording:
        def consume_token(self, key, rate_per_second, capacity, cost=1.0):
            seen.append(key)
            return True

        def incr(self, key, amounts, ttl_seconds):
            seen.append(key)
            return {"n": 1.0}

    set_state_store(Recording())
    AbuseDetector(_cfg()).detect("super secret prompt text", "t1", "a1", user_id="u1")
    assert seen and all("super secret" not in k for k in seen)
    assert any(k.startswith("abuse:rl:5:t1:a1:u1") for k in seen)
    assert any(k.startswith("abuse:flood:60:3:t1:a1:u1:") for k in seen)


class _DownStore:
    """A store in fail-closed mode with the backend unreachable."""

    def consume_token(self, key, rate_per_second, capacity, cost=1.0):
        raise StateUnavailable("down")

    def incr(self, key, amounts, ttl_seconds):
        raise StateUnavailable("down")


def test_state_unavailable_fails_closed():
    set_state_store(_DownStore())
    det = AbuseDetector(_cfg())

    hit = det.check_rate_limit("t1", "a1")
    assert hit is not None and hit.rule_id == "abuse.rate_limited"
    assert "unavailable" in hit.message

    result = det.detect("hello", "t1", "a1")
    assert result.decision == Decision.BLOCK
    assert _rate_limited(result)
    # The outage is reported once, not as a bogus flood.
    assert not _flooded(result)


@pytest.mark.skipif(not os.environ.get("ZNYX_TEST_REDIS_URL"),
                    reason="ZNYX_TEST_REDIS_URL not set")
def test_two_replicas_share_one_rate_limit_on_redis():
    client = make_redis_client(os.environ["ZNYX_TEST_REDIS_URL"])
    client.flushdb()
    try:
        set_state_store(RedisStateStore(client))
        replica_a = AbuseDetector(_cfg())
        replica_b = AbuseDetector(_cfg())
        assert _allowed_count([replica_a, replica_b], 20) == 5

        # Flood counting is shared through Redis as well.
        flood_a = AbuseDetector(_cfg(rate_limit_per_minute=1000))
        flood_b = AbuseDetector(_cfg(rate_limit_per_minute=1000))
        flags = [d.detect("same prompt", "t9", "a9") for d in (flood_a, flood_b, flood_a, flood_b)]
        assert [_flooded(r) for r in flags] == [False, False, False, True]
    finally:
        client.flushdb()
