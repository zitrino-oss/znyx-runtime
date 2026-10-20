"""This process's place among the worker processes of one ``znyx serve``.

The supervisor (``znyx.supervisor``) starts N copies of this service on one port
and tells each who it is through ``ZNYX_WORKER_INDEX``, ``ZNYX_WORKER_COUNT`` and
``ZNYX_RUN_DIR``. Run any other way (``znyx-runtime serve``, uvicorn, the old
images) none is set and this is worker 0 of 1, which behaves exactly as before.

* Worker 0 alone runs the jobs that must happen once per instance: delivering the
  spools, the install heartbeat, the benchmark queue, pushing model pins to the
  model server and reporting to the control plane.
* Every worker writes a small liveness record and a metrics snapshot into the run
  directory every few seconds. The supervisor restarts a worker whose record goes
  stale; ``/metrics`` and the dashboard merge the snapshots so the instance reports
  as one.
"""
from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

REPORT_INTERVAL_SECONDS = 2.0
RECENT_DECISIONS = 100


def worker_index() -> int:
    try:
        return max(0, int(os.getenv("ZNYX_WORKER_INDEX", "0")))
    except ValueError:
        return 0


def worker_count() -> int:
    try:
        return max(1, int(os.getenv("ZNYX_WORKER_COUNT", "1")))
    except ValueError:
        return 1


def is_leader() -> bool:
    """Whether this process runs the once-per-instance jobs."""
    return worker_index() == 0


def run_dir() -> Optional[Path]:
    raw = os.getenv("ZNYX_RUN_DIR", "").strip()
    return Path(raw) if raw else None


def _write_json(path: Path, data: Any) -> None:
    """Atomic replace, so a reader never sees half a file. The temporary file is
    created exclusively with an unpredictable name (never a planted symlink), and is
    owner-only."""
    import tempfile

    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(data, separators=(",", ":"), default=str))
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class RecentDecisions:
    """The last few decisions this worker made, metadata only (never text): for the
    local dashboard."""

    def __init__(self, maxlen: int = RECENT_DECISIONS):
        self._items: Deque[Dict[str, Any]] = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def record(self, event: Dict[str, Any]) -> None:
        item = {
            "time": event.get("timestamp"),
            "context": event.get("context"),
            "decision": event.get("decision"),
            "risk_score": event.get("risk_score"),
            "rules": list(event.get("rule_hit_ids") or [])[:10],
            "latency_ms": event.get("latency_ms"),
            "app_id": event.get("app_id"),
            "policy_version": event.get("policy_version"),
            "worker": worker_index(),
        }
        with self._lock:
            self._items.append(item)

    def items(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._items)


class WorkerReporter:
    """Writes this worker's liveness record, metrics snapshot and recent decisions."""

    def __init__(self, collector, recent: RecentDecisions, directory: Path,
                 index: Optional[int] = None, version: str = ""):
        self.collector = collector
        self.recent = recent
        self.directory = directory
        self.index = worker_index() if index is None else index
        self.version = version
        self.started_at = time.time()
        self._task: Optional[asyncio.Task] = None
        # The supervisor judges liveness by this counter changing, on its own monotonic
        # clock, so a wall-clock step can neither kill healthy workers nor hide a hung one.
        self._seq = 0
        # write_now runs from the report loop and from request threads (/metrics, the
        # dashboard); one at a time, so records are never interleaved.
        self._lock = threading.Lock()

    @property
    def liveness_path(self) -> Path:
        return self.directory / "workers" / f"w{self.index}.json"

    @property
    def metrics_path(self) -> Path:
        return self.directory / "metrics" / f"w{self.index}.json"

    def restore(self) -> None:
        """Carry on from a predecessor's totals: the supervisor restarted this worker
        index after a crash, and its counters must not vanish from the instance's."""
        previous = _read_json(self.metrics_path)
        if isinstance(previous, dict) and isinstance(previous.get("metrics"), dict):
            try:
                self.collector.restore(previous["metrics"])
            except Exception:  # noqa: BLE001 - a corrupt snapshot only loses history
                logger.debug("could not restore metrics snapshot", exc_info=True)

    def write_now(self) -> None:
        from znyx_runtime.sdk_registry import get_sdk_registry

        with self._lock:
            for sub in ("workers", "metrics"):
                (self.directory / sub).mkdir(parents=True, exist_ok=True, mode=0o700)
            snapshot = self.collector.snapshot()
            evaluations = sum(v for _, v in snapshot.get("counters", {}).get(
                "guardrails_evaluations_total", []))
            now = time.time()
            self._seq += 1
            from znyx_runtime.gateway.usage import recent_calls
            _write_json(self.metrics_path, {"metrics": snapshot, "recent": self.recent.items(),
                                            "sdk_clients": get_sdk_registry().snapshot(),
                                            "gateway_recent": recent_calls()})
            _write_json(self.liveness_path, {
                "index": self.index, "pid": os.getpid(), "version": self.version, "seq": self._seq,
                "started_at": self.started_at, "beat_at": now,
                "uptime_seconds": round(now - self.started_at), "requests": int(evaluations),
            })

    async def _loop(self) -> None:
        while True:
            try:
                await asyncio.to_thread(self.write_now)
            except Exception:  # noqa: BLE001 - reporting must never take the worker down
                logger.debug("worker report failed", exc_info=True)
            await asyncio.sleep(REPORT_INTERVAL_SECONDS)

    async def start(self) -> None:
        self.restore()
        self._task = asyncio.create_task(self._loop(), name="znyx-worker-report")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        try:
            self.write_now()
        except Exception:  # noqa: BLE001
            pass


def instance_snapshot(collector, recent: Optional[RecentDecisions] = None,
                      reporter: Optional[WorkerReporter] = None) -> Dict[str, Any]:
    """Metrics and recent decisions for the whole instance: this worker's, merged
    with the other workers' latest snapshots when there are several."""
    from znyx_core.middleware.metrics import merge_snapshots

    from znyx_runtime.gateway.usage import recent_calls

    directory = run_dir()
    if reporter is None or directory is None or worker_count() == 1:
        return {"metrics": collector.snapshot(), "recent": recent.items() if recent else [],
                "gateway_recent": recent_calls(), "workers": 1}
    try:
        reporter.write_now()          # this worker's numbers are current, not 2 s old
    except Exception:  # noqa: BLE001
        pass
    snapshots, recent_items, gateway_items = [], [], []
    for path in sorted((directory / "metrics").glob("w*.json")):
        data = _read_json(path)
        if isinstance(data, dict) and isinstance(data.get("metrics"), dict):
            snapshots.append(data["metrics"])
            recent_items.extend(data.get("recent") or [])
            gateway_items.extend(data.get("gateway_recent") or [])
    recent_items.sort(key=lambda d: str(d.get("time") or ""))
    gateway_items.sort(key=lambda d: str(d.get("timestamp") or ""))
    return {"metrics": merge_snapshots(snapshots), "recent": recent_items[-RECENT_DECISIONS:],
            "gateway_recent": gateway_items[-RECENT_DECISIONS:], "workers": len(snapshots)}


def worker_records() -> List[Dict[str, Any]]:
    """Every worker's liveness record (for the dashboard)."""
    directory = run_dir()
    if directory is None:
        return []
    out = []
    now = time.time()
    for path in sorted((directory / "workers").glob("w*.json")):
        data = _read_json(path)
        if isinstance(data, dict):
            data["age_seconds"] = round(now - float(data.get("beat_at") or now), 1)
            out.append(data)
    return out


def _other_snapshots() -> List[Dict[str, Any]]:
    directory = run_dir()
    if directory is None:
        return []
    out = []
    for path in sorted((directory / "metrics").glob("w*.json")):
        data = _read_json(path)
        if isinstance(data, dict):
            out.append(data)
    return out


def instance_sdk_clients(local: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """The SDKs that called any worker (worker 0 reports for the instance): newest
    report per source, as the per-process registry keeps it."""
    if worker_count() == 1:
        return local
    newest: Dict[str, Dict[str, str]] = {}
    for item in list(local) + [c for snap in _other_snapshots() for c in snap.get("sdk_clients") or []]:
        if not isinstance(item, dict) or not item.get("source"):
            continue
        seen = newest.get(item["source"])
        if seen is None or str(item.get("last_seen_at", "")) > str(seen.get("last_seen_at", "")):
            newest[item["source"]] = item
    return sorted(newest.values(), key=lambda r: r["source"])


def instance_evaluations() -> Optional[int]:
    """Evaluations across every worker (from their snapshots), or None with one worker."""
    if worker_count() == 1:
        return None
    total = 0.0
    for snap in _other_snapshots():
        metrics = snap.get("metrics") or {}
        total += sum(v for _, v in metrics.get("counters", {}).get("guardrails_evaluations_total", []))
    return int(total)

