"""Content-hash decision cache: exact request identity -> sha256 -> LRU of
recent results. Dependency-free.

A cached decision is only valid for the exact request that produced it, so the key
binds every input that can change the result: the raw UTF-8 text (byte-exact - no case
folding or whitespace collapsing, so "US" and "us" are distinct entries), the task, the
served model_version, the runner kind + configuration fingerprint, and any per-request
params. Keys carry a task prefix so a model reload can invalidate one task's entries.
"""
from __future__ import annotations

import json
import threading
from collections import OrderedDict
from hashlib import sha256
from typing import Any, Dict, Optional


def spec_fingerprint(spec: dict) -> str:
    """Stable digest of a runner spec (kind, model pin, threshold, options). Folding it
    into the cache key means any reconfiguration moves to a fresh key space on its own."""
    canon = json.dumps(spec or {}, sort_keys=True, separators=(",", ":"), default=repr)
    return sha256(canon.encode("utf-8")).hexdigest()[:16]


def content_key(model_version: str, text: str, *, task: str = "", scope: str = "",
                params: Optional[Dict[str, Any]] = None) -> str:
    """Exact cache identity for one scored item.

    ``scope`` is the serving identity beyond the pin - runner kind + spec fingerprint
    (see ``spec_fingerprint``). Fields are length-prefixed before hashing so adjacent
    values cannot collide by concatenation, and the text is hashed byte-exact:
    normalizing it (lowercase + whitespace collapse) made distinct inputs such as
    "US"/"us" or "May"/"may" share one cached decision.
    """
    param_part = ""
    if params:
        param_part = json.dumps(params, sort_keys=True, separators=(",", ":"), default=repr)
    h = sha256()
    for part in (task, model_version or "", scope, param_part, text or ""):
        b = part.encode("utf-8")
        h.update(len(b).to_bytes(4, "big"))
        h.update(b)
    return f"{task}:{h.hexdigest()}"


class ContentHashCache:
    """Thread-safe LRU keyed by ``content_key``. Tracks hit/miss counters
    for the observable cache-hit metric the acceptance test asserts."""

    def __init__(self, maxsize: int = 4096):
        self.maxsize = max(1, maxsize)
        self._store: "OrderedDict[str, object]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[object]:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self.hits += 1
                return self._store[key]
            self.misses += 1
            return None

    def put(self, key: str, value: object) -> None:
        with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self.maxsize:
                self._store.popitem(last=False)

    def invalidate_task(self, task: str) -> int:
        """Drop every entry cached under ``task``. Called when the task's model is
        reloaded: a reload can swap weights without changing the pin (a re-install),
        which silently invalidates every decision cached against it."""
        prefix = f"{task}:"
        with self._lock:
            stale = [k for k in self._store if k.startswith(prefix)]
            for k in stale:
                del self._store[k]
            return len(stale)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": self.size,
            "maxsize": self.maxsize,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }
