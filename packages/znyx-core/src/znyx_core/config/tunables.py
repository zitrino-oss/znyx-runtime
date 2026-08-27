"""Centralised numeric tunables — single source of truth for thresholds,
batch sizes, retry counts, and timeout windows.

Adding a new constant here is preferable to inlining a number in service
code for three reasons:

1. **Discoverability.** A new engineer can grep one file to learn what
   knobs exist and what their defaults are.
2. **Override surface.** Every value reads from an env var with a sensible
   default — ops never has to redeploy to retune a quota.
3. **Type/range safety.** The casts (``int(os.getenv(..., default))``) live
   in one place; adding a min/max guard is one edit, not a campaign.

Convention: every setting is ``UPPER_SNAKE_CASE``, exposed as a module-level
constant. Group by subsystem with a comment header. When a service consumes
one, prefer ``from znyx_core.config.tunables import RETRY_DELAYS_SECONDS`` over
duplicating the literal value.

Scope note: this module ships as part of the OSS ``znyx-core`` package, so it
holds only tunables the runtime engine itself reads. Settings specific to a
particular control-plane deployment (billing cadence, email provider, DB pool
sizing, and similar) belong in that deployment's own config module instead.
"""
from __future__ import annotations

import os
from typing import Tuple


def _int(env: str, default: int) -> int:
    try:
        return int(os.getenv(env, str(default)))
    except (TypeError, ValueError):
        return default


def _float(env: str, default: float) -> float:
    try:
        return float(os.getenv(env, str(default)))
    except (TypeError, ValueError):
        return default


def _str(env: str, default: str = "") -> str:
    return os.getenv(env, default)


def _bool(env: str, default: bool = False) -> bool:
    raw = os.getenv(env)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes")


def _csv_floats(env: str, default: Tuple[float, ...]) -> Tuple[float, ...]:
    raw = os.getenv(env)
    if not raw:
        return default
    try:
        return tuple(float(x.strip()) for x in raw.split(",") if x.strip())
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Runtime telemetry batching
# ---------------------------------------------------------------------------
TELEMETRY_BATCH_SIZE: int = _int("ZNYX_TELEMETRY_BATCH_SIZE", 100)
TELEMETRY_FLUSH_INTERVAL_SECONDS: int = _int("ZNYX_TELEMETRY_FLUSH_INTERVAL", 10)


# ---------------------------------------------------------------------------
# Bundle manager
# ---------------------------------------------------------------------------
BUNDLE_BOOT_RETRY_DELAYS_SECONDS: Tuple[float, ...] = _csv_floats(
    "ZNYX_BUNDLE_BOOT_RETRY_DELAYS", (1.0, 3.0, 5.0)
)


# ---------------------------------------------------------------------------
# Pagination defaults
# ---------------------------------------------------------------------------
MAX_PAGE_SIZE: int = _int("ZNYX_MAX_PAGE_SIZE", 200)


# ---------------------------------------------------------------------------
# HTTP rate limiter (per-IP / per-key sliding window)
# ---------------------------------------------------------------------------
RATE_LIMIT_REQUESTS_PER_MINUTE: int = _int("RATE_LIMIT_REQUESTS_PER_MINUTE", 60)
RATE_LIMIT_BURST: int = _int("RATE_LIMIT_BURST", 10)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
# Max distinct label-sets per metric; new sets beyond this collapse into a
# single "other" series so metric memory stays bounded.
METRICS_MAX_LABEL_SETS: int = _int("METRICS_MAX_LABEL_SETS", 500)


# ---------------------------------------------------------------------------
# Detector execution pool
# ---------------------------------------------------------------------------
# Worker threads for the synchronous detector pipeline. Values above 1 pay off
# when detectors block on I/O (remote detectors, NLI scoring, judge calls);
# 1 restores strict serialization.
DETECTOR_WORKERS: int = _int("ZNYX_DETECTOR_WORKERS", 4)
# Pending admissions allowed beyond the in-flight workers. Past this depth the
# evaluator rejects immediately (callers surface HTTP 503) instead of queueing
# without bound.
DETECTOR_QUEUE_MAX: int = _int("ZNYX_DETECTOR_QUEUE_MAX", 64)
# Per-request deadline for the detector pipeline; a request past it fails
# closed (BLOCK). 0 disables the deadline.
DETECTOR_DEADLINE_SECONDS: float = _float("ZNYX_DETECTOR_DEADLINE_SECONDS", 60.0)
# Max cached detector instances, keyed (name, config digest); least-recently
# used entries are evicted beyond this.
DETECTOR_INSTANCE_CACHE_SIZE: int = _int("ZNYX_DETECTOR_CACHE_SIZE", 256)
