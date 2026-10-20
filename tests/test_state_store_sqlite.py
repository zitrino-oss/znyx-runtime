"""The on-disk state store that the worker processes of one `znyx serve` share."""
import multiprocessing
import os

import pytest

from znyx_core.state_store import SqliteStateStore, StateUnavailable, get_state_store, set_state_store


class FakeClock:
    def __init__(self):
        self.t = 1_700_000_000.0

    def __call__(self):
        return self.t


def _consume(path, key, n, out):
    store = SqliteStateStore(path)
    out.put(sum(store.consume_token(key, rate_per_second=0.0001, capacity=50) for _ in range(n)))


def _add(path, key, n):
    store = SqliteStateStore(path)
    for _ in range(n):
        store.incr(key, {"tokens": 1}, ttl_seconds=600)


def test_processes_sharing_the_file_enforce_one_limit(tmp_path):
    """Four processes drawing on one 50-token bucket get 50 between them, not 200."""
    path = str(tmp_path / "state.sqlite3")
    ctx = multiprocessing.get_context("spawn")
    out = ctx.Queue()
    procs = [ctx.Process(target=_consume, args=(path, "rl:app", 40, out)) for _ in range(4)]
    for p in procs:
        p.start()
    allowed = sum(out.get(timeout=60) for _ in procs)
    for p in procs:
        p.join(30)
    assert allowed == 50


def test_concurrent_counters_lose_no_updates(tmp_path):
    path = str(tmp_path / "state.sqlite3")
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_add, args=(path, "budget:s1", 100)) for _ in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
    assert SqliteStateStore(path).get("budget:s1")["tokens"] == 400


def test_window_expires_and_restarts_from_zero(tmp_path):
    clock = FakeClock()
    store = SqliteStateStore(str(tmp_path / "s.sqlite3"), clock=clock)
    store.incr("k", {"n": 5}, ttl_seconds=10)
    clock.t += 5
    assert store.incr("k", {"n": 1}, ttl_seconds=10)["n"] == 6      # same window
    clock.t += 6
    assert store.incr("k", {"n": 1}, ttl_seconds=10)["n"] == 1      # new window
    assert store.get("missing") == {}


def test_state_survives_a_restart(tmp_path):
    path = str(tmp_path / "s.sqlite3")
    SqliteStateStore(path).incr("session:42", {"tokens": 1500}, ttl_seconds=3600)
    assert SqliteStateStore(path).get("session:42")["tokens"] == 1500


def test_expired_rows_are_pruned(tmp_path):
    clock = FakeClock()
    store = SqliteStateStore(str(tmp_path / "s.sqlite3"), clock=clock)
    store.PRUNE_EVERY = 10
    for i in range(9):
        store.incr(f"k{i}", {"n": 1}, ttl_seconds=1)
    clock.t += 5
    store.incr("fresh", {"n": 1}, ttl_seconds=60)                     # 10th write prunes
    rows = store._conn().execute("SELECT COUNT(*) FROM kv").fetchone()[0]
    assert rows == 1


def test_the_file_is_private(tmp_path):
    path = tmp_path / "s.sqlite3"
    SqliteStateStore(str(path))
    assert oct(path.stat().st_mode & 0o777) == "0o600"


def _broken_store(tmp_path, **kw):
    store = SqliteStateStore(str(tmp_path / "s.sqlite3"), **kw)
    store._conn().close()                                             # any further use fails
    return store


def test_an_unusable_file_falls_back_to_local_limits(tmp_path):
    store = _broken_store(tmp_path)
    assert store.consume_token("k", rate_per_second=0.001, capacity=1)
    assert not store.consume_token("k", rate_per_second=0.001, capacity=1)   # still limited


def test_an_unusable_file_fails_closed_when_asked(tmp_path):
    store = _broken_store(tmp_path, fail_mode="closed")
    with pytest.raises(StateUnavailable):
        store.incr("k", {"n": 1}, ttl_seconds=60)


def test_selected_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("ZNYX_STATE_BACKEND", "sqlite")
    monkeypatch.setenv("ZNYX_STATE_SQLITE_PATH", str(tmp_path / "x" / "state.sqlite3"))
    set_state_store(None)
    try:
        assert isinstance(get_state_store(), SqliteStateStore)
    finally:
        set_state_store(None)


def test_sqlite_backend_requires_a_path(monkeypatch):
    monkeypatch.setenv("ZNYX_STATE_BACKEND", "sqlite")
    monkeypatch.delenv("ZNYX_STATE_SQLITE_PATH", raising=False)
    set_state_store(None)
    try:
        with pytest.raises(RuntimeError, match="ZNYX_STATE_SQLITE_PATH"):
            get_state_store()
    finally:
        set_state_store(None)
