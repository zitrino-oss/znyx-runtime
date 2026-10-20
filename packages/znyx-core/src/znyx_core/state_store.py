"""Shared state for detectors that count across requests.

Rate limits, flood detection, session token and cost budgets, multi-turn jailbreak
history and LLM-judge spend all keep state between requests. Held in each process,
N replicas behind a load balancer enforce limits N times looser, and a restart
resets them. This module gives those detectors one small interface with two
backends:

* ``MemoryStateStore`` (default): in-process, bounded, thread-safe. Exactly the old
  per-process behaviour, so a single runtime needs nothing new.
* ``RedisStateStore``: shared by every replica pointed at the same Redis, so three
  replicas enforce the same limits as one. Each operation is one atomic Lua script.
* ``SqliteStateStore``: shared by every process on one host through one file (the
  worker processes of one ``znyx serve``). Each operation is one transaction. No
  server and no extra dependency; not for replicas on different hosts.

Select with ``ZNYX_STATE_BACKEND=memory|sqlite|redis`` (``ZNYX_STATE_SQLITE_PATH``,
``ZNYX_REDIS_URL``). When Redis
is unreachable the Redis store falls back to a local memory store for that call
(``ZNYX_STATE_FAIL_MODE=local``, the default), so limits stay enforced per replica
rather than switching off; ``ZNYX_STATE_FAIL_MODE=closed`` raises instead, and
callers fail closed.

All keys are namespaced (``znyx:``). Values are caller-chosen; detectors must not
put raw prompt text in keys or values (use a hash).
"""
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)

_PREFIX = "znyx:"


class StateUnavailable(RuntimeError):
    """The shared store could not be reached and the fail mode is ``closed``."""


class StateStore(Protocol):
    def consume_token(self, key: str, rate_per_second: float, capacity: float,
                      cost: float = 1.0) -> bool:
        """Token bucket: take ``cost`` tokens if available. True when allowed."""

    def incr(self, key: str, amounts: Dict[str, float], ttl_seconds: float) -> Dict[str, float]:
        """Add ``amounts`` to the named counters under ``key`` and return the new
        totals. The window starts when the key is created and lasts ``ttl_seconds``;
        after it the counters start again from zero (a fixed window)."""

    def get(self, key: str) -> Dict[str, float]:
        """The current counters under ``key`` (empty when none or expired)."""

    def push_capped(self, key: str, item: str, max_len: int, ttl_seconds: float) -> List[str]:
        """Append ``item`` to the list under ``key``, keep the newest ``max_len``,
        refresh its lifetime, and return the list oldest first."""

    def delete(self, key: str) -> None:
        """Forget ``key``."""


PINNED_PREFIX = "pin:"


class MemoryStateStore:
    """In-process store. Bounded (least recently used keys are evicted past
    ``max_keys``) and thread-safe.

    Keys starting with ``pin:`` are never evicted, only expired: they hold state that
    must not be pushed out by other traffic (a spent budget would otherwise reset when
    enough sessions or cache entries arrive). Their callers keep them bounded (one per
    configured key and day, say)."""

    def __init__(self, max_keys: int = 100_000, clock=time.monotonic):
        self._data: "OrderedDict[str, dict]" = OrderedDict()
        self._lock = threading.Lock()
        self._max_keys = max_keys
        self._clock = clock

    def _entry(self, key: str, now: float, create: Optional[dict] = None) -> Optional[dict]:
        entry = self._data.get(key)
        if entry is not None and entry.get("expires") is not None and now >= entry["expires"]:
            del self._data[key]
            entry = None
        if entry is None and create is not None:
            entry = create
            self._data[key] = entry
            self._evict()
        if entry is not None:
            self._data.move_to_end(key)
        return entry

    def _evict(self) -> None:
        skipped = 0          # pinned keys moved out of the way; all of them: nothing to evict
        while len(self._data) > self._max_keys and skipped < len(self._data):
            oldest = next(iter(self._data))
            if oldest.startswith(PINNED_PREFIX):
                self._data.move_to_end(oldest)
                skipped += 1
            else:
                del self._data[oldest]

    def consume_token(self, key, rate_per_second, capacity, cost=1.0):
        now = self._clock()
        with self._lock:
            entry = self._entry(key, now, create={"tokens": float(capacity), "ts": now, "expires": None})
            elapsed = max(0.0, now - entry["ts"])
            entry["tokens"] = min(float(capacity), entry["tokens"] + elapsed * rate_per_second)
            entry["ts"] = now
            # Forget a full, idle bucket after it would have refilled anyway.
            entry["expires"] = now + (capacity / rate_per_second if rate_per_second > 0 else 3600) + 60
            if entry["tokens"] >= cost:
                entry["tokens"] -= cost
                return True
            return False

    def incr(self, key, amounts, ttl_seconds):
        now = self._clock()
        with self._lock:
            entry = self._entry(key, now, create={"fields": {}, "expires": now + ttl_seconds})
            fields = entry.setdefault("fields", {})
            for name, amount in amounts.items():
                fields[name] = fields.get(name, 0.0) + float(amount)
            return dict(fields)

    def get(self, key):
        now = self._clock()
        with self._lock:
            entry = self._entry(key, now)
            return dict(entry.get("fields", {})) if entry else {}

    def push_capped(self, key, item, max_len, ttl_seconds):
        now = self._clock()
        with self._lock:
            entry = self._entry(key, now, create={"items": [], "expires": None})
            items = entry.setdefault("items", [])
            items.append(item)
            if len(items) > max_len:
                del items[:len(items) - max_len]
            entry["expires"] = now + ttl_seconds
            return list(items)

    def delete(self, key):
        with self._lock:
            self._data.pop(key, None)


# Lua scripts: each operation is atomic on the Redis server.
_TOKEN_BUCKET = """
local key = KEYS[1]
local rate = tonumber(ARGV[1])
local capacity = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local now = tonumber(ARGV[4])
local state = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(state[1])
local ts = tonumber(state[2])
if tokens == nil then tokens = capacity; ts = now end
local elapsed = now - ts
if elapsed < 0 then elapsed = 0 end
tokens = math.min(capacity, tokens + elapsed * rate)
local allowed = 0
if tokens >= cost then tokens = tokens - cost; allowed = 1 end
redis.call('HSET', key, 'tokens', tostring(tokens), 'ts', tostring(now))
local ttl = 3600
if rate > 0 then ttl = math.ceil(capacity / rate) + 60 end
redis.call('EXPIRE', key, ttl)
return allowed
"""

_INCR = """
local key = KEYS[1]
local ttl = tonumber(ARGV[1])
local created = redis.call('EXISTS', key) == 0
for i = 2, #ARGV, 2 do
  redis.call('HINCRBYFLOAT', key, ARGV[i], ARGV[i + 1])
end
if created then redis.call('PEXPIRE', key, math.ceil(ttl * 1000)) end
return redis.call('HGETALL', key)
"""

_PUSH_CAPPED = """
local key = KEYS[1]
local max_len = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])
redis.call('RPUSH', key, ARGV[1])
redis.call('LTRIM', key, -max_len, -1)
redis.call('PEXPIRE', key, math.ceil(ttl * 1000))
return redis.call('LRANGE', key, 0, -1)
"""


class RedisStateStore:
    """Shared store on Redis. ``client`` is a redis-py ``Redis`` (or compatible)
    object; detectors run in worker threads, so the synchronous client is right."""

    # After a failure, skip Redis for this long before trying again, so an outage
    # costs one connection timeout per interval rather than one per request.
    DOWN_SECONDS = 5.0

    def __init__(self, client, fail_mode: str = "local", fallback: Optional[MemoryStateStore] = None):
        self._client = client
        self._fail_mode = fail_mode
        self._fallback = fallback or MemoryStateStore()
        self._scripts = {
            name: client.register_script(src)
            for name, src in (("bucket", _TOKEN_BUCKET), ("incr", _INCR), ("push", _PUSH_CAPPED))
        }
        self._warned = False
        self._down_until = 0.0

    def _available(self) -> bool:
        return time.monotonic() >= self._down_until

    def _skip(self, op: str):
        """Redis is known to be down: fail closed, or use the local store."""
        if self._fail_mode == "closed":
            raise StateUnavailable(f"shared state store unavailable during {op}")
        return self._fallback

    def _degraded(self, op: str, exc: Exception):
        self._down_until = time.monotonic() + self.DOWN_SECONDS
        if self._fail_mode == "closed":
            raise StateUnavailable(f"shared state store unavailable during {op}: {exc}") from exc
        if not self._warned:
            logger.warning("Shared state store unavailable (%s); enforcing limits per replica "
                           "until it returns: %s", op, exc)
            self._warned = True
        return self._fallback

    def consume_token(self, key, rate_per_second, capacity, cost=1.0):
        if not self._available():
            return self._skip("consume_token").consume_token(key, rate_per_second, capacity, cost)
        try:
            allowed = self._scripts["bucket"](
                keys=[_PREFIX + key],
                args=[rate_per_second, capacity, cost, time.time()])
            self._warned = False
            return bool(int(allowed))
        except StateUnavailable:
            raise
        except Exception as exc:
            return self._degraded("consume_token", exc).consume_token(key, rate_per_second, capacity, cost)

    def incr(self, key, amounts, ttl_seconds):
        args: List = [ttl_seconds]
        for name, amount in amounts.items():
            args += [name, float(amount)]
        if not self._available():
            return self._skip("incr").incr(key, amounts, ttl_seconds)
        try:
            flat = self._scripts["incr"](keys=[_PREFIX + key], args=args)
            self._warned = False
            return _pairs_to_dict(flat)
        except StateUnavailable:
            raise
        except Exception as exc:
            return self._degraded("incr", exc).incr(key, amounts, ttl_seconds)

    def get(self, key):
        if not self._available():
            return self._skip("get").get(key)
        try:
            return {_s(k): float(_s(v)) for k, v in self._client.hgetall(_PREFIX + key).items()}
        except Exception as exc:
            return self._degraded("get", exc).get(key)

    def push_capped(self, key, item, max_len, ttl_seconds):
        if not self._available():
            return self._skip("push_capped").push_capped(key, item, max_len, ttl_seconds)
        try:
            items = self._scripts["push"](keys=[_PREFIX + key], args=[item, max_len, ttl_seconds])
            self._warned = False
            return [_s(i) for i in items]
        except StateUnavailable:
            raise
        except Exception as exc:
            return self._degraded("push_capped", exc).push_capped(key, item, max_len, ttl_seconds)

    def delete(self, key):
        if not self._available():
            self._fallback.delete(key)
            return
        try:
            self._client.delete(_PREFIX + key)
        except Exception as exc:
            self._degraded("delete", exc).delete(key)


class SqliteStateStore:
    """Shared store in one SQLite file, for processes on one host.

    Same semantics as ``MemoryStateStore``, with wall-clock expiry so the state
    survives a restart. Each operation takes the write lock up front
    (``BEGIN IMMEDIATE``), so concurrent processes serialise rather than race. A
    failure (disk full, lock held past the timeout) is handled like a Redis outage:
    this call uses a local memory store (fail mode ``local``) or raises (``closed``).
    """

    DOWN_SECONDS = 5.0
    PRUNE_EVERY = 500

    def __init__(self, path: str, fail_mode: str = "local", timeout: float = 2.0,
                 fallback: Optional[MemoryStateStore] = None, clock=time.time):
        import sqlite3

        self._sqlite3 = sqlite3
        self.path = path
        self._fail_mode = fail_mode
        self._timeout = timeout
        self._fallback = fallback or MemoryStateStore()
        self._clock = clock
        self._local = threading.local()
        self._down_until = 0.0
        self._warned = False
        self._writes = 0
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._setup()
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

    def _setup(self) -> None:
        """Create the table and switch the file to WAL (readers never wait for the
        writer). Several workers start at once, and changing the journal mode fails
        immediately, without waiting, while another process holds a lock; the mode is
        stored in the file, so it is retried here briefly and never per connection.
        If setup still fails, operations degrade through the fail mode like any other
        store failure rather than crashing the worker."""
        deadline = time.monotonic() + max(self._timeout, 1.0) * 2
        while True:
            try:
                conn = self._conn()
                conn.execute("CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, "
                             "value TEXT NOT NULL, expires REAL)")
                if str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() != "wal":
                    conn.execute("PRAGMA journal_mode=WAL")
                return
            except self._sqlite3.OperationalError as exc:
                if time.monotonic() >= deadline:
                    logger.warning("Shared state file setup incomplete (%s); continuing", exc)
                    return
                time.sleep(0.02)

    def _conn(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._sqlite3.connect(self.path, timeout=self._timeout, isolation_level=None,
                                         check_same_thread=False)
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def _txn(self, op: str, fn):
        """Run ``fn(conn, now)`` in one write transaction; on failure fall back or raise."""
        if time.monotonic() < self._down_until:
            if self._fail_mode == "closed":
                raise StateUnavailable(f"shared state store unavailable during {op}")
            return None
        try:
            conn = self._conn()
            conn.execute("BEGIN IMMEDIATE")
            try:
                result = fn(conn, self._clock())
                self._writes += 1
                if self._writes % self.PRUNE_EVERY == 0:
                    conn.execute("DELETE FROM kv WHERE expires IS NOT NULL AND expires <= ?",
                                 (self._clock(),))
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            self._warned = False
            return result
        except self._sqlite3.Error as exc:
            self._down_until = time.monotonic() + self.DOWN_SECONDS
            if self._fail_mode == "closed":
                raise StateUnavailable(f"shared state store unavailable during {op}: {exc}") from exc
            if not self._warned:
                logger.warning("Shared state file unavailable (%s); enforcing limits per process "
                               "until it returns: %s", op, exc)
                self._warned = True
            return None

    @staticmethod
    def _decode(raw) -> Optional[dict]:
        """A stored value, or None when it is not what this store writes (a damaged or
        foreign row): treated as absent, so the next write replaces it rather than every
        request for that identity failing until the row expires."""
        try:
            value = json.loads(raw)
        except (TypeError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    @classmethod
    def _load(cls, conn, key: str, now: float) -> Optional[dict]:
        row = conn.execute("SELECT value, expires FROM kv WHERE key = ?", (_PREFIX + key,)).fetchone()
        if row is None or (row[1] is not None and now >= row[1]):
            return None
        return cls._decode(row[0])

    @staticmethod
    def _save(conn, key: str, value: dict, expires: Optional[float]) -> None:
        conn.execute("INSERT OR REPLACE INTO kv (key, value, expires) VALUES (?, ?, ?)",
                     (_PREFIX + key, json.dumps(value, separators=(",", ":")), expires))

    def consume_token(self, key, rate_per_second, capacity, cost=1.0):
        def op(conn, now):
            entry = self._load(conn, key, now)
            if not (entry and isinstance(entry.get("tokens"), (int, float))
                    and isinstance(entry.get("ts"), (int, float))):
                entry = {"tokens": float(capacity), "ts": now}
            elapsed = max(0.0, now - entry["ts"])
            tokens = min(float(capacity), entry["tokens"] + elapsed * rate_per_second)
            allowed = tokens >= cost
            if allowed:
                tokens -= cost
            expires = now + (capacity / rate_per_second if rate_per_second > 0 else 3600) + 60
            self._save(conn, key, {"tokens": tokens, "ts": now}, expires)
            return allowed
        result = self._txn("consume_token", op)
        if result is None:
            return self._fallback.consume_token(key, rate_per_second, capacity, cost)
        return result

    def incr(self, key, amounts, ttl_seconds):
        def op(conn, now):
            row = conn.execute("SELECT value, expires FROM kv WHERE key = ?", (_PREFIX + key,)).fetchone()
            decoded = None if row is None else self._decode(row[0])
            if decoded is None or (row[1] is not None and now >= row[1]):
                fields, expires = {}, now + ttl_seconds
            else:
                fields = {k: v for k, v in decoded.items() if isinstance(v, (int, float))}
                expires = row[1]
            for name, amount in amounts.items():
                fields[name] = fields.get(name, 0.0) + float(amount)
            self._save(conn, key, fields, expires)
            return dict(fields)
        result = self._txn("incr", op)
        if result is None:
            return self._fallback.incr(key, amounts, ttl_seconds)
        return result

    def get(self, key):
        result = self._txn("get", lambda conn, now: self._load(conn, key, now) or {})
        if result is None:
            return self._fallback.get(key)
        return {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}

    def push_capped(self, key, item, max_len, ttl_seconds):
        def op(conn, now):
            entry = self._load(conn, key, now)
            items = list(entry["items"]) if entry and isinstance(entry.get("items"), list) else []
            items.append(item)
            if len(items) > max_len:
                del items[:len(items) - max_len]
            self._save(conn, key, {"items": items}, now + ttl_seconds)
            return items
        result = self._txn("push_capped", op)
        if result is None:
            return self._fallback.push_capped(key, item, max_len, ttl_seconds)
        return result

    def delete(self, key):
        def op(conn, now):
            conn.execute("DELETE FROM kv WHERE key = ?", (_PREFIX + key,))
            return True
        if self._txn("delete", op) is None:
            self._fallback.delete(key)


def _s(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _pairs_to_dict(flat) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for i in range(0, len(flat) - 1, 2):
        out[_s(flat[i])] = float(_s(flat[i + 1]))
    return out


_store: Optional[StateStore] = None
_store_lock = threading.Lock()


def get_state_store() -> StateStore:
    """The process-wide store, built from the environment on first use."""
    global _store
    with _store_lock:
        if _store is None:
            _store = _from_env()
        return _store


def set_state_store(store: Optional[StateStore]) -> None:
    """Replace the process-wide store (tests, or an explicit composition root).
    ``None`` rebuilds it from the environment on next use."""
    global _store
    with _store_lock:
        _store = store


def make_redis_client(url: str, timeout: float = 0.5):
    """A Redis client for the state store: short timeouts and no automatic retries.
    redis-py retries with backoff by default, which would stall every guarded
    request for seconds while Redis is down; the store handles outages itself."""
    import redis  # optional dependency, only needed for the shared backend
    from redis.backoff import NoBackoff
    from redis.retry import Retry
    return redis.Redis.from_url(url, socket_timeout=timeout, socket_connect_timeout=timeout,
                                retry=Retry(NoBackoff(), 0))


def _from_env() -> StateStore:
    backend = os.getenv("ZNYX_STATE_BACKEND", "memory").strip().lower()
    if backend == "redis":
        url = os.getenv("ZNYX_REDIS_URL", "").strip()
        if not url:
            raise RuntimeError("ZNYX_STATE_BACKEND=redis requires ZNYX_REDIS_URL")
        client = make_redis_client(url)
        fail_mode = os.getenv("ZNYX_STATE_FAIL_MODE", "local").strip().lower()
        logger.info("Shared state store: redis (fail mode %s)", fail_mode)
        return RedisStateStore(client, fail_mode=fail_mode)
    if backend == "sqlite":
        path = os.getenv("ZNYX_STATE_SQLITE_PATH", "").strip()
        if not path:
            raise RuntimeError("ZNYX_STATE_BACKEND=sqlite requires ZNYX_STATE_SQLITE_PATH")
        fail_mode = os.getenv("ZNYX_STATE_FAIL_MODE", "local").strip().lower()
        logger.info("Shared state store: sqlite %s (fail mode %s)", path, fail_mode)
        return SqliteStateStore(path, fail_mode=fail_mode)
    if backend != "memory":
        raise RuntimeError(f"Unknown ZNYX_STATE_BACKEND '{backend}' (memory, sqlite or redis)")
    return MemoryStateStore()


def json_item(obj) -> str:
    """Compact JSON for push_capped items."""
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)
