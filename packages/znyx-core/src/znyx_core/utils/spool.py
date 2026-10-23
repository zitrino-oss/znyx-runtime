"""Appending to a JSON-lines spool from several processes at once.

The worker processes of one ``znyx serve`` write the same audit and telemetry spools,
and one of them ships and rotates each spool. Rotation renames the file; a writer in
another process that opened the old name a moment earlier would append to the
renamed file after its tail was copied back, and the record would be lost with it.
Every append and every rotation therefore takes an exclusive advisory lock on a
companion ``<spool>.lock`` file (POSIX ``flock``; ``msvcrt`` on Windows). The lock
is per open file description, so it also orders threads within one process.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

try:  # POSIX
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows
    _fcntl = None
    import msvcrt as _msvcrt

PathLike = Union[str, "os.PathLike[str]"]


def lock_path(path: PathLike) -> Path:
    p = Path(path)
    return p.with_name(p.name + ".lock")


@contextlib.contextmanager
def spool_lock(path: PathLike) -> Iterator[None]:
    """Hold the spool's exclusive lock for the duration of the block."""
    lp = lock_path(path)
    if not lp.parent.exists():
        lp.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(lp, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if _fcntl is not None:
            _fcntl.flock(fd, _fcntl.LOCK_EX)
        else:  # pragma: no cover - Windows
            _msvcrt.locking(fd, _msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            if _fcntl is not None:
                _fcntl.flock(fd, _fcntl.LOCK_UN)
            else:  # pragma: no cover - Windows
                os.lseek(fd, 0, os.SEEK_SET)
                _msvcrt.locking(fd, _msvcrt.LK_UNLCK, 1)
    finally:
        os.close(fd)


def append_lines(path: PathLike, lines: Iterable[str], fsync: bool = True) -> None:
    """Append complete lines (without their newline) under the spool's lock.

    Spools carry rule-hit messages that can quote request fragments, so the file and a
    newly created directory are owner-only. If the file ends in a partial line (a writer
    killed mid-append), a newline is written first: otherwise this record would be glued
    onto that fragment and skipped as unparseable although its caller was told it was
    recorded."""
    p = Path(path)
    if not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    data = "".join(line + "\n" for line in lines).encode("utf-8")
    with spool_lock(p):
        fd = os.open(p, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            size = os.fstat(fd).st_size
            if size:
                with open(p, "rb") as tail:
                    tail.seek(size - 1)
                    if tail.read(1) != b"\n":
                        data = b"\n" + data
            os.write(fd, data)
            if fsync:
                os.fsync(fd)
        finally:
            os.close(fd)


# -- hash-chained spools (AI gateway Phase 5, part 4) ------------------------------------------

GENESIS = "0" * 64


def chain_hash(prev: str, record: Dict[str, Any]) -> str:
    """The hash a chained record carries: of the previous hash, a newline and the record's
    canonical JSON without its ``chain`` field. The control plane recomputes it the same way."""
    payload = {k: v for k, v in record.items() if k != "chain"}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256((prev + "\n" + text).encode("utf-8", "surrogatepass")).hexdigest()


def _chain_state_path(path: Path) -> Path:
    return path.with_name(path.name + ".chain")


def _last_chained(path: Path) -> Optional[Dict[str, Any]]:
    """The chain field of the spool's last complete record, if any."""
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if not size:
        return None
    with open(path, "rb") as handle:
        handle.seek(max(0, size - 65536))
        lines = handle.read().splitlines()
    for raw in reversed(lines):
        try:
            record = json.loads(raw)
        except ValueError:
            continue
        chain = record.get("chain") if isinstance(record, dict) else None
        return chain if isinstance(chain, dict) else None
    return None


def _valid_state(state: Any) -> bool:
    return (isinstance(state, dict) and isinstance(state.get("id"), str) and isinstance(state.get("seq"), int)
            and isinstance(state.get("hash"), str))


def _save_chain_state(state_path: Path, state: Dict[str, Any]) -> None:
    # Not fsynced: the state is a cache of the spool's last record, which is durable; a state
    # lost or left behind by a crash is rebuilt from the spool (its size no longer matches).
    tmp = state_path.with_name(state_path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, json.dumps(state).encode())
    finally:
        os.close(fd)
    os.replace(tmp, state_path)


def carry_chain_over(path: PathLike, shipped: PathLike) -> None:
    """Before a spool's delivered records are removed (the shipper's rotation), make sure the
    chain state is at least at the last record they hold, so the next record continues the
    chain instead of reusing a number. Call it holding the spool's lock."""
    p = Path(path)
    state_path = _chain_state_path(p)
    last = _last_chained(Path(shipped))
    if not _valid_state(last):
        return
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = None
    if not _valid_state(state) or state["id"] != last["id"] or state["seq"] < last["seq"]:
        state = {"id": last["id"], "seq": last["seq"], "hash": last["hash"]}
    try:
        state["size"] = p.stat().st_size
    except OSError:
        state["size"] = 0
    _save_chain_state(state_path, state)


def append_chained(path: PathLike, records: List[Dict[str, Any]], fsync: bool = True) -> List[Dict[str, Any]]:
    """Append records to a hash-chained spool: each gets ``chain: {id, seq, prev, hash}``
    extending the spool's chain. Under the spool's lock (several processes extend one chain);
    the state is kept in ``<spool>.chain`` with the spool's size after the last write. When the
    size differs (a crash between writing a record and saving the state, a rotation) the state is
    checked against the spool's last record, so the chain never forks. Lines are written with
    non-ASCII characters escaped, so any string (a lone surrogate too) can be recorded; the hash
    covers the value, not the escaping. Returns the records as written."""
    import uuid
    p = Path(path)
    if not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    state_path = _chain_state_path(p)
    with spool_lock(p):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = None
        try:
            size_now = p.stat().st_size
        except OSError:
            size_now = 0
        if not (_valid_state(state) and state.get("size") == size_now):
            tail = _last_chained(p)
            tail_ok = _valid_state(tail)
            if not _valid_state(state):
                # No saved state: continue the spool's own chain when it has one, else start one.
                state = ({"id": tail["id"], "seq": tail["seq"], "hash": tail["hash"]} if tail_ok
                         else {"id": uuid.uuid4().hex, "seq": 0, "hash": GENESIS})
            elif tail_ok and tail["id"] == state["id"] and tail["seq"] > state["seq"]:
                state = {"id": state["id"], "seq": tail["seq"], "hash": tail["hash"]}
        written = []
        for record in records:
            record = {k: v for k, v in record.items() if k != "chain"}
            seq = state["seq"] + 1
            digest = chain_hash(state["hash"], record)
            record["chain"] = {"id": state["id"], "seq": seq, "prev": state["hash"], "hash": digest}
            state = {"id": state["id"], "seq": seq, "hash": digest}
            written.append(record)
        data = "".join(json.dumps(r, separators=(",", ":"), default=str) + "\n" for r in written).encode("ascii")
        fd = os.open(p, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            size = os.fstat(fd).st_size
            if size:
                with open(p, "rb") as tail_handle:
                    tail_handle.seek(size - 1)
                    if tail_handle.read(1) != b"\n":
                        data = b"\n" + data
            os.write(fd, data)
            if fsync:
                os.fsync(fd)
            state["size"] = os.fstat(fd).st_size
        finally:
            os.close(fd)
        _save_chain_state(state_path, state)
    return written


def verify_chain(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Check chained records in order: one finding per problem, ``{seq, problem}`` with
    ``problem`` "mismatch" (the record does not hash to what it says) or "gap" (records are
    missing before it). Empty when the chain holds."""
    findings = []
    last: Dict[str, Tuple[int, str]] = {}
    for record in records:
        chain = record.get("chain") if isinstance(record, dict) else None
        if not isinstance(chain, dict):
            findings.append({"seq": None, "problem": "unchained"})
            continue
        cid, seq, prev, digest = chain.get("id"), chain.get("seq"), chain.get("prev"), chain.get("hash")
        if chain_hash(str(prev), record) != digest:
            findings.append({"chain": cid, "seq": seq, "problem": "mismatch"})
        known = last.get(cid)
        if known is not None and (seq != known[0] + 1 or prev != known[1]):
            findings.append({"chain": cid, "seq": seq, "problem": "gap"})
        last[cid] = (seq if isinstance(seq, int) else -1, str(digest))
    return findings
