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
import os
from pathlib import Path
from typing import Iterable, Iterator, Union

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
