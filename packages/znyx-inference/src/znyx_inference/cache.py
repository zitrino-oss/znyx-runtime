"""Content-hash decision cache: normalize → sha256 → LRU of recent
results. Two identical normalized inputs to the same model hit the cache. Dependency-free.
"""
from __future__ import annotations

import re
import threading
from collections import OrderedDict
from hashlib import sha256
from typing import Optional

_WS = re.compile(r"\s+")


def content_key(model_version: str, text: str) -> str:
    """Stable cache key: model_version + sha256 of the normalized text. Normalization
    (lowercase + whitespace-collapse) means trivially-different inputs still hit."""
    norm = _WS.sub(" ", (text or "").strip().lower())
    digest = sha256(norm.encode("utf-8")).hexdigest()
    return f"{model_version}:{digest}"


class ContentHashCache:
    """Thread-safe LRU keyed by (model_version, content-hash). Tracks hit/miss counters
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
