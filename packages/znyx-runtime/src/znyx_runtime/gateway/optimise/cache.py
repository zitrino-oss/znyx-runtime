"""Exact response cache (technique 5): the same deterministic request again, within a
short time, is answered from the provider's stored answer.

Keyed per caller (the gateway key), provider URL, the headers sent upstream, the request
body after the input guard and the policy version (technique 29: a cache is never shared
across keys or policy versions). The stored answer is the provider's, before the output
guard, so every hit is checked again under the policy in force; only answers that passed
the output guard unchanged are stored. Kept in the worker process and bounded by size:
answers never go to the shared state store, and a restart empties it.
"""
from __future__ import annotations

import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional

DEFAULT_TTL_SECONDS = 300
MAX_ENTRY_BYTES = 1_000_000
DEFAULT_MAX_BYTES = 64 * 1024 * 1024


@dataclass
class CachedAnswer:
    content: bytes                 # the provider's response body, as received
    headers: Dict[str, str]        # the provider headers the gateway passes back
    upstream_ms: Optional[int]     # how long the provider took to produce it
    stored: float                  # time.monotonic() when stored
    hits: int = 0


class ResponseCache:
    def __init__(self, max_bytes: Optional[int] = None):
        if max_bytes is None:
            try:
                max_bytes = int(float(os.environ.get("ZNYX_GATEWAY_CACHE_MB", "64")) * 1024 * 1024)
            except ValueError:
                max_bytes = DEFAULT_MAX_BYTES
        self._max_bytes = max(0, max_bytes)
        self._entries: "OrderedDict[str, CachedAnswer]" = OrderedDict()
        self._bytes = 0

    def get(self, key: str, ttl_seconds: float) -> Optional[CachedAnswer]:
        entry = self._entries.get(key)
        if entry is None:
            return None
        if time.monotonic() - entry.stored >= ttl_seconds:
            self._drop(key)
            return None
        entry.hits += 1
        self._entries.move_to_end(key)
        return entry

    def put(self, key: str, content: bytes, headers: Dict[str, str], upstream_ms: Optional[int]) -> bool:
        size = len(content)
        if size > MAX_ENTRY_BYTES or size > self._max_bytes:
            return False
        if key in self._entries:
            self._drop(key)
        self._entries[key] = CachedAnswer(content, dict(headers), upstream_ms, time.monotonic())
        self._bytes += size
        while self._bytes > self._max_bytes and self._entries:
            self._drop(next(iter(self._entries)))
        return True

    def _drop(self, key: str) -> None:
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._bytes -= len(entry.content)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def size_bytes(self) -> int:
        return self._bytes
