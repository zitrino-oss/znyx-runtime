"""``corpus_ingest`` lifecycle hook (OWASP LLM05).

Corpus writes are not request traffic, so like ``tool_registration`` this is invoked
explicitly by whatever owns the ingest path. Checking at WRITE time is the point: once
poisoned content is indexed, every later reader inherits it, and a retrieval-time check
can only stop its use, never the corruption.

Unlike ``tool_registration``, this hook's detector is STATEFUL: the ingest-burst rule
counts writes per source over a window. Building a detector per call therefore threw the
counter away on every write, and the one rule that needs history could never fire. The
detector is cached per configuration instead, so the counter survives across calls the
way it does for the in-pipeline stateful detectors (``abuse``, ``unbounded_consumption``).

The cache is per PROCESS, which is the same limitation those two carry: under multiple
replicas each holds its own counters, so ``max_writes_per_window`` triggers at roughly
the configured limit times the replica count. Set a shared store via
``set_corpus_state_store()`` to make the counter global.
"""
import json
import threading
from typing import Any, Dict, Optional

from znyx_core.core.models import DetectorResult
from znyx_core.detectors.corpus_poisoning_monitor import CorpusPoisoningMonitorDetector

# Only the settings that shape the counter belong in the cache key. Keying on the whole
# config would split the counter whenever an unrelated threshold was tuned, silently
# resetting the burst history at exactly the moment an operator was adjusting it.
_STATEFUL_KEYS = ("max_writes_per_window", "window_seconds")

_lock = threading.Lock()
_detectors: Dict[str, CorpusPoisoningMonitorDetector] = {}
# Bounded so a caller passing per-request config cannot grow this without limit.
_MAX_CACHED = 64


def _cache_key(cfg: Dict[str, Any]) -> str:
    return json.dumps({k: cfg.get(k) for k in _STATEFUL_KEYS}, sort_keys=True, default=str)


def _detector_for(cfg: Dict[str, Any]) -> CorpusPoisoningMonitorDetector:
    key = _cache_key(cfg)
    with _lock:
        det = _detectors.get(key)
        if det is None:
            if len(_detectors) >= _MAX_CACHED:
                # Drop the oldest rather than refuse: refusing would silently fall back
                # to a stateless detector, which is the bug this cache exists to fix.
                _detectors.pop(next(iter(_detectors)))
            det = CorpusPoisoningMonitorDetector(cfg)
            _detectors[key] = det
        else:
            # Non-stateful thresholds can change between calls; apply them to the cached
            # instance so config edits take effect without discarding the counter.
            det.enabled = cfg.get("enabled", True)
            det.action = (cfg.get("action") or "WARN").upper()
            det.authority_threshold = max(1, int(cfg.get("authority_threshold", 1)))
            det.max_question_repeats = max(2, int(cfg.get("max_question_repeats", 3)))
            det.flag_untrusted_into_trusted = bool(
                cfg.get("flag_untrusted_into_trusted", True))
        return det


def scan_corpus_write(document: str, tenant_id: str = "",
                      metadata: Optional[Dict[str, Any]] = None,
                      config: Optional[Dict[str, Any]] = None) -> DetectorResult:
    """Screen ``document`` before it enters a corpus. Defaults to enabled: the hook is an
    explicit management action, so the caller has already decided to look."""
    cfg = dict(config or {})
    cfg.setdefault("enabled", True)
    return _detector_for(cfg).detect(document, tenant_id=tenant_id, metadata=metadata)


def reset_corpus_ingest_state() -> None:
    """Drop all cached detectors and their counters. For tests and for an operator
    deliberately clearing burst history after a false positive."""
    with _lock:
        _detectors.clear()
