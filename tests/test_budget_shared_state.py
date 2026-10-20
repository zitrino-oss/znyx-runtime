"""Session budgets and loop history live in the shared state store, not the detector.

Covers UnboundedConsumptionDetector and the corpus-poisoning ingest-burst counter:
two detector instances (standing in for two replicas) sharing one store enforce one
budget, windows reset, and a store that cannot be reached fails closed.
"""
import os

import pytest

from znyx_core.core.models import Decision
from znyx_core.detectors.corpus_poisoning_monitor import CorpusPoisoningMonitorDetector
from znyx_core.detectors.unbounded_consumption import UnboundedConsumptionDetector
from znyx_core.state_store import (
    MemoryStateStore, RedisStateStore, StateUnavailable, get_state_store,
    make_redis_client, set_state_store,
)

TOKEN_RULE = "unbounded_consumption.token_budget_exceeded"
LOOP_RULE = "unbounded_consumption.agent_loop_state_repeat"
DOWN_RULE = "unbounded_consumption.state_unavailable"


def _uc(**cfg):
    return UnboundedConsumptionDetector({"enabled": True, "max_session_tokens": 100, **cfg})


def _ids(result):
    return {h.rule_id for h in result.rule_hits}


def _call(det, session="s1", tokens=60, tenant="t", app="a", **kw):
    return det.detect("hi", tenant, app, session_id=session,
                      metadata={"tokens": tokens}, **kw)


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class DownStore:
    """A store in fail-closed mode with Redis unreachable."""

    def _down(self, *a, **k):
        raise StateUnavailable("shared state store unavailable")

    incr = get = push_capped = delete = consume_token = _down


# (a) one budget across instances ---------------------------------------------

def test_two_instances_share_one_session_budget():
    a, b = _uc(), _uc()
    assert _call(a).decision == Decision.ALLOW           # 60 of 100
    r = _call(b)                                         # 120 total, other instance
    assert r.decision == Decision.BLOCK
    assert TOKEN_RULE in _ids(r)


def test_cost_and_thinking_totals_are_shared_too():
    a = _uc(max_session_cost_usd=1.0, max_session_thinking_tokens=100)
    b = _uc(max_session_cost_usd=1.0, max_session_thinking_tokens=100)
    md = {"cost_usd": 0.6, "thinking_tokens": 60}
    assert a.detect("x", "t", "a", session_id="s", metadata=md).decision == Decision.ALLOW
    r = b.detect("x", "t", "a", session_id="s", metadata=md)
    assert {"unbounded_consumption.cost_budget_exceeded",
            "unbounded_consumption.thinking_token_budget_exceeded"} <= _ids(r)


def test_user_identity_accumulates_when_there_is_no_session():
    a, b = _uc(), _uc()
    a.detect("x", "t", "a", user_id="u1", metadata={"tokens": 60})
    r = b.detect("x", "t", "a", user_id="u1", metadata={"tokens": 60})
    assert TOKEN_RULE in _ids(r)


# (b) independence --------------------------------------------------------------

def test_different_sessions_tenants_and_apps_are_independent():
    d = _uc()
    _call(d, session="s1")
    assert _call(d, session="s2").decision == Decision.ALLOW
    assert _call(d, session="s1", tenant="other").decision == Decision.ALLOW
    assert _call(d, session="s1", app="other").decision == Decision.ALLOW
    assert TOKEN_RULE in _ids(_call(d, session="s1"))


def test_different_windows_never_share_counters():
    short, long_ = _uc(session_window_seconds=60), _uc(session_window_seconds=3600)
    _call(short)
    assert _call(long_).decision == Decision.ALLOW       # its own counter, 60 of 100


# (c) no identity ---------------------------------------------------------------

def test_no_identity_means_no_cumulative_accounting():
    d = _uc()
    for _ in range(5):
        r = d.detect("hi", "t", "a", metadata={"tokens": 60})
        assert r.decision == Decision.ALLOW
    assert d.detect("hi", "t", "a", metadata={"tokens": 150}).decision == Decision.BLOCK
    # Nothing was written to the store for an anonymous caller.
    assert not get_state_store()._data


# (d) window reset --------------------------------------------------------------

def test_window_resets_after_the_session_window():
    clock = FakeClock()
    set_state_store(MemoryStateStore(clock=clock))
    d = _uc(session_window_seconds=100)
    assert _call(d).decision == Decision.ALLOW
    assert _call(d).decision == Decision.BLOCK
    clock.now += 101
    assert _call(d).decision == Decision.ALLOW           # fresh window: 60 of 100
    assert _call(d).decision == Decision.BLOCK


def test_loop_history_expires_with_the_window():
    clock = FakeClock()
    set_state_store(MemoryStateStore(clock=clock))
    d = _uc(session_window_seconds=100, max_repeated_states=3)
    step = lambda: d.detect("x", "t", "a", session_id="s",
                            metadata={"agent_step": {"state": "same"}})
    step(); step()
    clock.now += 101
    assert step().decision == Decision.ALLOW             # count restarted at 1


# (e) loop detection across instances -------------------------------------------

def test_repeated_state_is_detected_across_instances():
    dets = [_uc(max_repeated_states=3) for _ in range(3)]
    results = [d.detect("x", "t", "a", session_id="s",
                        metadata={"agent_step": {"state": "spin"}}) for d in dets]
    assert [r.decision for r in results[:2]] == [Decision.ALLOW, Decision.ALLOW]
    assert results[2].decision == Decision.BLOCK and LOOP_RULE in _ids(results[2])


def test_progressing_loop_is_not_flagged_and_history_is_bounded():
    d = _uc(max_repeated_states=3, max_tracked_states=8)
    for i in range(50):
        r = d.detect("x", "t", "a", session_id="s",
                     metadata={"agent_step": {"state": f"step-{i}"}})
        assert r.decision == Decision.ALLOW
    keys = [k for k in get_state_store()._data if "states" in k]
    assert len(keys) == 1 and len(get_state_store()._data[keys[0]]["items"]) == 8


def test_spinning_after_a_full_history_is_still_caught():
    d = _uc(max_repeated_states=3, max_tracked_states=4)
    for i in range(10):
        d.detect("x", "t", "a", session_id="s", metadata={"agent_step": {"state": f"u{i}"}})
    for _ in range(2):
        d.detect("x", "t", "a", session_id="s", metadata={"agent_step": {"state": "spin"}})
    r = d.detect("x", "t", "a", session_id="s", metadata={"agent_step": {"state": "spin"}})
    assert LOOP_RULE in _ids(r)


def test_loop_state_keys_hold_a_hash_not_the_raw_state():
    d = _uc()
    d.detect("x", "t", "a", session_id="s",
             metadata={"agent_step": {"state": "SECRET-RAW-STATE-TEXT"}})
    blob = repr(get_state_store()._data)
    assert "SECRET-RAW-STATE-TEXT" not in blob
    assert all(k.startswith("budget:") for k in get_state_store()._data)


# (f) fail closed ---------------------------------------------------------------

def test_unavailable_store_fails_closed_on_the_budget():
    set_state_store(DownStore())
    r = _call(_uc(), tokens=1)
    assert r.decision == Decision.BLOCK
    hit = next(h for h in r.rule_hits if h.rule_id == DOWN_RULE)
    assert "unavailable" in hit.message and "over budget" in hit.message


def test_unavailable_store_in_warn_mode_warns():
    set_state_store(DownStore())
    assert _call(_uc(action="WARN"), tokens=1).decision == Decision.WARN


def test_unavailable_store_does_not_affect_anonymous_requests():
    set_state_store(DownStore())
    assert _uc().detect("hi", "t", "a", metadata={"tokens": 1}).decision == Decision.ALLOW


def test_unavailable_store_fails_closed_for_loop_history_only():
    class HistoryDown(MemoryStateStore):
        def push_capped(self, *a, **k):
            raise StateUnavailable("down")
    set_state_store(HistoryDown())
    r = _uc().detect("x", "t", "a", session_id="s", metadata={"agent_step": {"state": "s1"}})
    assert r.decision == Decision.BLOCK and DOWN_RULE in _ids(r)


def test_per_request_caps_still_apply_without_a_store():
    set_state_store(DownStore())
    r = _uc(max_thinking_tokens_per_request=10).detect(
        "x", "t", "a", session_id="s", metadata={"thinking_tokens": 50})
    assert "unbounded_consumption.thinking_tokens_per_request_exceeded" in _ids(r)


# corpus-poisoning ingest burst --------------------------------------------------

def _corpus(**cfg):
    return CorpusPoisoningMonitorDetector(
        {"enabled": True, "max_writes_per_window": 2, "window_seconds": 100, **cfg})


def _write(det, source="feed", tenant="t"):
    return det.detect("An ordinary document.", tenant_id=tenant, metadata={"source_id": source})


def test_ingest_burst_is_shared_across_instances():
    a, b = _corpus(), _corpus()
    _write(a); _write(b)
    r = _write(a)
    assert "corpus_poisoning_monitor.ingest_burst" in _ids(r)


def test_ingest_burst_is_per_source_and_tenant_and_resets():
    clock = FakeClock()
    set_state_store(MemoryStateStore(clock=clock))
    d = _corpus()
    for _ in range(3):
        _write(d)
    assert _write(d, source="other").decision == Decision.ALLOW
    assert _write(d, tenant="t2").decision == Decision.ALLOW
    clock.now += 101
    assert _write(d).decision == Decision.ALLOW


def test_ingest_burst_keys_do_not_contain_the_source():
    _write(_corpus(), source="alice@example.com")
    assert "alice@example.com" not in repr(get_state_store()._data)


def test_ingest_burst_fails_closed_when_the_store_is_unavailable():
    set_state_store(DownStore())
    r = _write(_corpus())
    assert "corpus_poisoning_monitor.state_unavailable" in _ids(r)
    assert r.decision == Decision.WARN                   # the detector's default action
    assert _write(_corpus(action="BLOCK")).decision == Decision.BLOCK


# (g) Redis ---------------------------------------------------------------------

REDIS_URL = os.environ.get("ZNYX_TEST_REDIS_URL")


@pytest.mark.skipif(not REDIS_URL, reason="ZNYX_TEST_REDIS_URL not set")
class TestRedisBacked:
    @pytest.fixture(autouse=True)
    def redis_store(self):
        client = make_redis_client(REDIS_URL)
        client.flushdb()
        set_state_store(RedisStateStore(client))
        yield client
        client.flushdb()

    def test_two_instances_share_one_session_budget(self):
        a, b = _uc(), _uc()
        assert _call(a).decision == Decision.ALLOW
        r = _call(b)
        assert r.decision == Decision.BLOCK and TOKEN_RULE in _ids(r)

    def test_message_reports_integer_tokens(self):
        _call(_uc())
        r = _call(_uc())
        msg = next(h.message for h in r.rule_hits if h.rule_id == TOKEN_RULE)
        assert msg == "Token budget exceeded: 120 > 100"

    def test_repeated_state_across_instances(self):
        results = [_uc(max_repeated_states=3).detect(
            "x", "t", "a", session_id="s", metadata={"agent_step": {"state": "spin"}})
            for _ in range(3)]
        assert LOOP_RULE in _ids(results[2]) and LOOP_RULE not in _ids(results[1])

    def test_keys_are_namespaced_and_expire(self, redis_store):
        _call(_uc(session_window_seconds=300))
        keys = [k.decode() for k in redis_store.keys("znyx:*")]
        assert keys == ["znyx:budget:uc:300:t:a:s:s1"]
        assert 0 < redis_store.ttl(keys[0]) <= 300

    def test_ingest_burst_shared(self):
        a, b = _corpus(), _corpus()
        _write(a); _write(b)
        assert "corpus_poisoning_monitor.ingest_burst" in _ids(_write(a))
