"""Shared state store: the in-memory backend everywhere, and the Redis backend
against a real Redis when ZNYX_TEST_REDIS_URL points at one (skipped otherwise)."""
import os
import threading
import uuid

import pytest

from znyx_core.state_store import (
    MemoryStateStore, RedisStateStore, SqliteStateStore, StateUnavailable, get_state_store,
    set_state_store,
)

REDIS_URL = os.getenv("ZNYX_TEST_REDIS_URL")


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _redis_store(**kw):
    import redis
    client = redis.Redis.from_url(REDIS_URL, socket_timeout=1)
    client.flushdb()
    return RedisStateStore(client, **kw)


def _stores():
    out = [pytest.param("memory", id="memory"), pytest.param("sqlite", id="sqlite")]
    out.append(pytest.param("redis", id="redis",
                            marks=pytest.mark.skipif(not REDIS_URL, reason="ZNYX_TEST_REDIS_URL not set")))
    return out


@pytest.fixture(params=_stores())
def store(request, tmp_path):
    if request.param == "memory":
        return MemoryStateStore()
    if request.param == "sqlite":
        return SqliteStateStore(str(tmp_path / "state.sqlite3"))
    return _redis_store()


def _key():
    return f"test:{uuid.uuid4().hex}"


def test_token_bucket_allows_capacity_then_refuses(store):
    key = _key()
    assert all(store.consume_token(key, rate_per_second=0.001, capacity=3) for _ in range(3))
    assert not store.consume_token(key, rate_per_second=0.001, capacity=3)


def test_counters_add_up_and_return_totals(store):
    key = _key()
    store.incr(key, {"tokens": 100, "cost": 0.25}, ttl_seconds=60)
    totals = store.incr(key, {"tokens": 50, "cost": 0.5}, ttl_seconds=60)
    assert totals["tokens"] == 150
    assert totals["cost"] == pytest.approx(0.75)
    assert store.get(key)["tokens"] == 150


def test_capped_list_keeps_the_newest(store):
    key = _key()
    for i in range(7):
        items = store.push_capped(key, str(i), max_len=3, ttl_seconds=60)
    assert items == ["4", "5", "6"]


def test_delete(store):
    key = _key()
    store.incr(key, {"n": 1}, ttl_seconds=60)
    store.delete(key)
    assert store.get(key) == {}


def test_reservation_is_atomic_across_counters_and_can_be_settled(store):
    key = _key()
    assert store.reserve(key, {"tokens": 70, "cost": 0.5}, {"tokens": 100, "cost": 1.0}, 60)
    assert not store.reserve(key, {"tokens": 40, "cost": 0.1}, {"tokens": 100, "cost": 1.0}, 60)
    assert store.get(key) == {"tokens": 70, "cost": 0.5}
    assert not store.reserve(key, {"tokens": 1, "cost": 0.6}, {"tokens": 100, "cost": 1.0}, 60)
    store.incr(key, {"tokens": -50, "cost": -0.4}, 60)
    assert store.reserve(key, {"tokens": 80, "cost": 0.1}, {"tokens": 100, "cost": 1.0}, 60)
    assert store.get(key)["tokens"] == 100


def test_concurrent_reservations_never_over_admit(store):
    from concurrent.futures import ThreadPoolExecutor
    key = _key()
    with ThreadPoolExecutor(12) as pool:
        results = list(pool.map(lambda _: store.reserve(key, {"tokens": 10}, {"tokens": 50}, 60), range(30)))
    assert sum(results) == 5 and store.get(key)["tokens"] == 50


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True])
def test_invalid_reservations_are_refused_before_reaching_the_store(store, value):
    with pytest.raises(ValueError):
        store.reserve(_key(), {"tokens": value}, {"tokens": 100}, 60)


def test_memory_window_resets_after_ttl():
    clock = FakeClock()
    store = MemoryStateStore(clock=clock)
    store.incr("k", {"n": 5}, ttl_seconds=10)
    clock.t += 5
    assert store.incr("k", {"n": 1}, ttl_seconds=10)["n"] == 6   # same window
    clock.t += 6
    assert store.incr("k", {"n": 1}, ttl_seconds=10)["n"] == 1   # a new window


def test_memory_token_bucket_refills():
    clock = FakeClock()
    store = MemoryStateStore(clock=clock)
    assert store.consume_token("b", rate_per_second=1.0, capacity=1)
    assert not store.consume_token("b", rate_per_second=1.0, capacity=1)
    clock.t += 1.0
    assert store.consume_token("b", rate_per_second=1.0, capacity=1)


def test_memory_store_is_bounded():
    store = MemoryStateStore(max_keys=100)
    for i in range(1000):
        store.incr(f"k{i}", {"n": 1}, ttl_seconds=60)
    assert len(store._data) == 100


def test_memory_store_is_thread_safe():
    store = MemoryStateStore()

    def work():
        for _ in range(500):
            store.incr("shared", {"n": 1}, ttl_seconds=60)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.get("shared")["n"] == 4000


@pytest.mark.skipif(not REDIS_URL, reason="ZNYX_TEST_REDIS_URL not set")
def test_replicas_sharing_redis_enforce_one_limit():
    """Three replicas (three stores, one Redis) together allow exactly the capacity."""
    import redis
    client = redis.Redis.from_url(REDIS_URL)
    client.flushdb()
    replicas = [RedisStateStore(redis.Redis.from_url(REDIS_URL)) for _ in range(3)]
    key = _key()
    allowed = sum(r.consume_token(key, rate_per_second=0.001, capacity=10)
                  for _ in range(10) for r in replicas)
    assert allowed == 10


@pytest.mark.skipif(not REDIS_URL, reason="ZNYX_TEST_REDIS_URL not set")
def test_redis_window_expires_on_the_server():
    import time
    store = _redis_store()
    key = _key()
    store.incr(key, {"n": 1}, ttl_seconds=0.3)
    time.sleep(0.5)
    assert store.incr(key, {"n": 1}, ttl_seconds=0.3)["n"] == 1


def _unreachable_client():
    from znyx_core.state_store import make_redis_client
    return make_redis_client("redis://127.0.0.1:1/0", timeout=0.2)


def test_an_outage_costs_one_timeout_not_one_per_call():
    import time
    pytest.importorskip("redis")
    store = RedisStateStore(_unreachable_client(), fail_mode="local")
    key = _key()
    t0 = time.perf_counter()
    for _ in range(50):
        store.consume_token(key, rate_per_second=100, capacity=1000)
    assert time.perf_counter() - t0 < 1.0


def test_unreachable_redis_degrades_to_local_limits():
    pytest.importorskip("redis")
    store = RedisStateStore(_unreachable_client(), fail_mode="local")
    key = _key()
    assert all(store.consume_token(key, rate_per_second=0.001, capacity=2) for _ in range(2))
    assert not store.consume_token(key, rate_per_second=0.001, capacity=2)


def test_unreachable_redis_fails_closed_when_asked():
    pytest.importorskip("redis")
    store = RedisStateStore(_unreachable_client(), fail_mode="closed")
    with pytest.raises(StateUnavailable):
        store.consume_token(_key(), rate_per_second=1, capacity=1)


def test_process_store_defaults_to_memory(monkeypatch):
    monkeypatch.delenv("ZNYX_STATE_BACKEND", raising=False)
    set_state_store(None)
    try:
        assert isinstance(get_state_store(), MemoryStateStore)
    finally:
        set_state_store(None)


def test_redis_backend_requires_a_url(monkeypatch):
    monkeypatch.setenv("ZNYX_STATE_BACKEND", "redis")
    monkeypatch.delenv("ZNYX_REDIS_URL", raising=False)
    set_state_store(None)
    try:
        with pytest.raises(RuntimeError):
            get_state_store()
    finally:
        set_state_store(None)


# -- values and lists (Phase 5: approvals, MCP sessions and tool pins) ----------------------

def test_values_are_stored_read_and_kept_first_on_request(store):
    key = _key()
    assert store.get_value(key) is None
    assert store.set_value(key, '{"a": 1}', ttl_seconds=60)
    assert store.get_value(key) == '{"a": 1}'
    assert not store.set_value(key, "second", ttl_seconds=60, only_if_absent=True)   # the first writer wins
    assert store.get_value(key) == '{"a": 1}'
    assert store.set_value(key, "replaced", ttl_seconds=60)
    assert store.get_value(key) == "replaced"
    store.delete(key)
    assert store.get_value(key) is None and store.set_value(key, "again", 60, only_if_absent=True)


def test_lists_are_read_without_changing_them(store):
    key = _key()
    assert store.get_list(key) == []
    store.push_capped(key, "a", max_len=2, ttl_seconds=60)
    store.push_capped(key, "b", max_len=2, ttl_seconds=60)
    store.push_capped(key, "c", max_len=2, ttl_seconds=60)
    assert store.get_list(key) == ["b", "c"] and store.get_list(key) == ["b", "c"]


def test_a_value_expires():
    clock = FakeClock()
    mem = MemoryStateStore(clock=clock)
    mem.set_value("k", "v", ttl_seconds=10)
    clock.t += 11
    assert mem.get_value("k") is None and mem.set_value("k", "w", 10, only_if_absent=True)
