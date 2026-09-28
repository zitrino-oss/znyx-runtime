"""Tracks which SDKs are calling this runtime, from their request headers.

Why this exists: an operator asking "what SDK version are my applications on?"
could not be answered before. Anonymous install telemetry records SDK versions
but carries no org - by design, it is anonymous - so it can never be attributed
to a customer. The runtime, however, sits in the customer's own network and sees
every SDK call, and it already reports to the control plane. So the SDK tells the
runtime who it is (``X-Znyx-Sdk`` / ``X-Znyx-Sdk-Version``), and the runtime
forwards that with its existing heartbeat.

The values are UNTRUSTED. They come from whatever called the runtime, so they are
length-capped and charset-filtered here, and nothing downstream may derive policy
from them - they are display metadata only.

Memory is bounded on purpose: a caller that spoofs a new ``X-Znyx-Sdk`` on every
request must not be able to grow this map without limit. Past the cap new sources
are dropped, and the ones already recorded keep updating.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from threading import Lock
from typing import Dict, List, Optional

# Generous enough for every real identifier ("python-sdk") and version
# ("1.2.1", "2.0.0-rc.1+build.5"), tight enough that a junk header is truncated
# rather than stored whole.
_MAX_SOURCE_LEN = 32
_MAX_VERSION_LEN = 32

# Distinct SDK identifiers retained. Real deployments use a handful; the cap only
# bites when something is generating them, which is precisely when we want a stop.
_MAX_SOURCES = 16

# Deliberately strict: identifiers and semver-ish versions only. Anything else is
# rejected outright rather than sanitised into something that looks legitimate.
_SOURCE_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")


def _clean(value: Optional[str], max_len: int, pattern: re.Pattern) -> Optional[str]:
    """Return a safe value, or None when the header is absent or malformed."""
    if not value:
        return None
    value = value.strip()[:max_len]
    if not value or not pattern.match(value):
        return None
    return value


class SdkRegistry:
    """The SDKs seen calling this runtime, newest report per source.

    Latest-wins by recency rather than by version ordering: two apps on different
    SDK versions both legitimately talk to one runtime, and comparing versions
    would mean shipping a version-parsing dependency to pick a winner that is not
    more correct - "what called most recently" is the honest summary.
    """

    def __init__(self) -> None:
        self._seen: Dict[str, Dict[str, str]] = {}
        self._lock = Lock()

    def record(self, source: Optional[str], version: Optional[str]) -> None:
        """Note one SDK call. Never raises - this runs on the request path."""
        src = _clean(source, _MAX_SOURCE_LEN, _SOURCE_RE)
        if not src:
            return
        ver = _clean(version, _MAX_VERSION_LEN, _VERSION_RE)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            if src not in self._seen and len(self._seen) >= _MAX_SOURCES:
                return
            self._seen[src] = {"source": src, "version": ver or "unknown", "last_seen_at": now}

    def snapshot(self) -> List[Dict[str, str]]:
        """Stable, sorted copy for reporting."""
        with self._lock:
            return sorted(self._seen.values(), key=lambda r: r["source"])

    def clear(self) -> None:
        with self._lock:
            self._seen.clear()


# Process-wide singleton: the middleware writes it, the reporter reads it.
_registry = SdkRegistry()


def get_sdk_registry() -> SdkRegistry:
    return _registry
