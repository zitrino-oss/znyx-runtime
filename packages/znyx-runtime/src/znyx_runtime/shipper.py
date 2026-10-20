"""Durable shipping of runtime records to the control plane.

Three record streams leave a runtime: evaluation telemetry, egress audit events and
LLM-judge audit events. Telemetry used to sit in a memory buffer that was dropped
whenever a send failed; the two audit spools were only ever read by a control plane
sharing the runtime's disk, so a runtime in a customer's network never delivered them.

``SpoolShipper`` tails an append-only JSON-lines spool file and POSTs new records to a
control-plane endpoint in batches. It remembers how far it has delivered in an offset
file next to the spool, and only advances that offset after the control plane
acknowledges a batch, so a crash or an outage delays records rather than losing
them. The control plane's ingestion is idempotent on each record's event id, so a
batch re-sent after a crash between delivery and the offset write is harmless.

Spools are rotated once fully delivered and larger than ``rotate_bytes``: the file is
renamed, writers (which open the file per append) start a new one, and the renamed
file is deleted. Disk use is bounded by ``max_bytes``: past it, the oldest
undelivered records are dropped and counted, which is the only way records are ever
lost, and only when the control plane has been unreachable for a long time.
"""
import asyncio
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from znyx_core.utils.spool import append_lines, spool_lock

logger = logging.getLogger(__name__)

DEFAULT_BATCH = 500
DEFAULT_INTERVAL = 5.0
DEFAULT_ROTATE_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_BYTES = 256 * 1024 * 1024
MAX_BACKOFF = 300.0

# Returned by a sender when the control plane rejected a batch as malformed (a
# permanent failure): the shipper skips it instead of retrying forever.
DROP = "drop"

# HTTP statuses worth retrying: auth or plan problems an operator can fix, timeouts,
# rate limiting and every server error. Any other 4xx is a malformed batch.
_RETRYABLE_4XX = {401, 403, 408, 409, 425, 429}


class SpoolShipper:
    """Delivers one spool file to one control-plane endpoint.

    ``post`` is an async callable ``(url, records) -> bool`` returning True when the
    control plane accepted the batch. Inject it in tests; the runtime passes an
    httpx-based sender (``http_post``).
    """

    def __init__(self, spool_path: str, url: str, post: Callable, *,
                 name: str = "spool", batch_size: int = DEFAULT_BATCH,
                 interval: float = DEFAULT_INTERVAL, rotate_bytes: int = DEFAULT_ROTATE_BYTES,
                 max_bytes: int = DEFAULT_MAX_BYTES):
        self.path = Path(spool_path)
        self.offset_path = self.path.with_name(self.path.name + ".offset")
        self.url = url
        self.post = post
        self.name = name
        self.batch_size = batch_size
        self.interval = interval
        self.rotate_bytes = rotate_bytes
        self.max_bytes = max_bytes
        self.delivered = 0
        self.dropped = 0
        self.failures = 0
        self._task: Optional[asyncio.Task] = None
        self._backoff = 0.0
        # One shipper per spool, across processes and instances (two instances under one
        # user share the default spool paths): the owner holds this lock while it runs.
        self._owner_fd: Optional[int] = None

    # -- offset bookkeeping -------------------------------------------------

    def _read_offset(self) -> int:
        try:
            return int(self.offset_path.read_text().strip() or 0)
        except (FileNotFoundError, ValueError):
            return 0

    def _write_offset(self, offset: int) -> None:
        import tempfile

        self.offset_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, tmp = tempfile.mkstemp(dir=str(self.offset_path.parent),
                                   prefix=f".{self.offset_path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(str(offset))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.offset_path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _acquire_ownership(self) -> bool:
        """Whether this shipper owns the spool (non-blocking; retried every pass)."""
        if self._owner_fd is not None:
            return True
        path = self.path.with_name(self.path.name + ".shipper")
        try:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        except OSError:
            return False
        try:
            try:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except ImportError:  # pragma: no cover - Windows
                import msvcrt
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            os.close(fd)
            return False
        self._owner_fd = fd
        return True

    def _release_ownership(self) -> None:
        if self._owner_fd is not None:
            try:
                os.close(self._owner_fd)
            except OSError:
                pass
            self._owner_fd = None

    # -- reading ----------------------------------------------------------------

    def _read_batch(self, offset: int) -> Tuple[List[Dict[str, Any]], int]:
        """Up to batch_size complete records from ``offset``. Returns them and the
        byte offset just after the last one. A partial last line (a write still in
        progress) is left for next time; an unparseable line is skipped."""
        records: List[Dict[str, Any]] = []
        try:
            with open(self.path, "rb") as fh:
                fh.seek(offset)
                pos = offset
                while len(records) < self.batch_size:
                    line = fh.readline()
                    if not line or not line.endswith(b"\n"):
                        break
                    pos += len(line)
                    try:
                        records.append(json.loads(line))
                    except ValueError:
                        logger.warning("%s: skipping unparseable spool line", self.name)
                return records, pos
        except FileNotFoundError:
            return [], offset

    # -- one delivery pass ----------------------------------------------------

    async def ship_once(self) -> int:
        """Deliver everything currently pending, batch by batch. Returns the number
        of records delivered; stops at the first failed batch."""
        self._enforce_disk_limit()
        offset = self._read_offset()
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return 0
        if offset > size:      # the spool was replaced underneath us
            offset = 0
        sent = 0
        while True:
            records, new_offset = self._read_batch(offset)
            if not records:
                if new_offset != offset:   # only unparseable lines: move past them
                    self._write_offset(new_offset)
                    offset = new_offset
                    continue
                break
            try:
                ok = await self.post(self.url, records)
            except Exception as exc:
                logger.debug("%s: delivery error: %s", self.name, exc)
                ok = False
            if ok == DROP:
                # The control plane rejected the batch itself (malformed, not a
                # transient failure). Retrying would block every later record
                # behind it, so skip it and count it.
                logger.warning("%s: control plane rejected a batch of %d records; skipping it",
                               self.name, len(records))
                self._write_offset(new_offset)
                offset = new_offset
                self.dropped += len(records)
                continue
            if not ok:
                self.failures += 1
                return sent
            self._write_offset(new_offset)
            offset = new_offset
            sent += len(records)
            self.delivered += len(records)
        self._rotate_if_done(offset)
        return sent

    def _rotate_if_done(self, offset: int) -> None:
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return
        if offset < size or size < self.rotate_bytes:
            return
        # Under the spool's lock no writer, in this process or another, can append
        # between the rename and the copy-back, so nothing lands in `shipped` unseen.
        with spool_lock(self.path):
            shipped = self.path.with_name(self.path.name + ".shipped")
            os.replace(self.path, shipped)
            tail = self._read_tail(shipped, offset)
            if tail:
                fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
                with os.fdopen(fd, "a", encoding="utf-8") as fh:
                    for rec in tail:
                        fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            shipped.unlink(missing_ok=True)
            self._write_offset(0)

    @staticmethod
    def _read_tail(path: Path, offset: int) -> List[Dict[str, Any]]:
        """Complete records in ``path`` after byte ``offset``."""
        out: List[Dict[str, Any]] = []
        with open(path, "rb") as fh:
            fh.seek(offset)
            for line in fh:
                if line.endswith(b"\n"):
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
        return out

    def _enforce_disk_limit(self) -> None:
        """Past max_bytes of undelivered records, drop the oldest and count them."""
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return
        offset = self._read_offset()
        if size - offset <= self.max_bytes:
            return
        target = size - self.max_bytes
        with open(self.path, "rb") as fh:
            fh.seek(target)
            fh.readline()                       # move to a line boundary
            new_offset = fh.tell()
            fh.seek(offset)
            dropped = 0
            pos = offset
            while pos < new_offset:
                line = fh.readline()
                if not line:
                    break
                pos += len(line)
                dropped += 1
        self.dropped += dropped
        logger.warning("%s: control plane unreachable for too long; dropped %d oldest "
                       "undelivered records to stay under %d bytes", self.name, dropped, self.max_bytes)
        self._write_offset(new_offset)

    # -- background loop --------------------------------------------------------

    async def run(self) -> None:
        while True:
            if not self._acquire_ownership():
                # Another process ships this spool; this one stands by and takes over if
                # that one stops. Records appended here are still delivered, by the owner.
                await asyncio.sleep(self.interval)
                continue
            sent_before = self.failures
            try:
                await self.ship_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a delivery loop must survive any error
                # Disk full, a permission change, a vanished directory: count it, back off
                # and try again. A dead loop would stop delivery and the disk cap for good.
                self.failures += 1
                logger.exception("%s: delivery pass failed; retrying", self.name)
            if self.failures > sent_before:
                self._backoff = min(MAX_BACKOFF, max(self.interval, self._backoff * 2 or self.interval))
                await asyncio.sleep(self._backoff * (0.8 + 0.4 * random.random()))
            else:
                self._backoff = 0.0
                await asyncio.sleep(self.interval)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self.run(), name=f"shipper-{self.name}")

    async def stop(self, final_attempt: bool = True) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if final_attempt and self._acquire_ownership():
            try:
                await asyncio.wait_for(self.ship_once(), timeout=5.0)
            except Exception:
                pass
        self._release_ownership()

    def status(self) -> Dict[str, Any]:
        try:
            pending = max(0, self.path.stat().st_size - self._read_offset())
        except FileNotFoundError:
            pending = 0
        return {"name": self.name, "delivered": self.delivered, "dropped": self.dropped,
                "failures": self.failures, "pending_bytes": pending}


def append_record(path: str, record: Dict[str, Any], fsync: bool = False) -> None:
    """Append one JSON record to a spool file, under the spool's lock (several worker
    processes may append while one rotates it)."""
    append_lines(path, [json.dumps(record, separators=(",", ":"), default=str)], fsync=fsync)


def http_post(token: str, timeout: float = 10.0, transport=None) -> Callable:
    """An async sender for SpoolShipper: POST {"events": records} with the runtime
    token (``X-API-Key``, as telemetry has always sent it). True on 2xx, False to
    retry later, ``DROP`` for a batch the control plane rejected as malformed."""
    import httpx

    async def _post(url: str, records: List[Dict[str, Any]]):
        async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
            resp = await client.post(url, json={"events": records},
                                     headers={"X-API-Key": token})
        if resp.status_code < 400:
            return True
        logger.warning("Delivery to %s failed: HTTP %s", url, resp.status_code)
        if resp.status_code < 500 and resp.status_code not in _RETRYABLE_4XX:
            return DROP
        return False

    return _post
