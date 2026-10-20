"""Appending to and rotating a spool from several processes at once
(znyx_core.utils.spool, znyx_runtime.shipper).

The worker processes of one ``znyx serve`` append to the same spool while worker 0
ships and rotates it. Rotation renames the file, so without the lock a record
appended at the wrong moment would land in the renamed file and be lost.
"""
import asyncio
import collections
import json
import stat
import subprocess
import sys
import threading
import time

from znyx_core.utils.spool import append_lines, lock_path, spool_lock
from znyx_runtime.shipper import SpoolShipper, append_record

WRITERS = 4
RECORDS_EACH = 300

_WRITER = """
import sys, time
from znyx_runtime.shipper import append_record
path, writer, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
for i in range(count):
    append_record(path, {"id": f"{writer}-{i}", "pad": "x" * 40})
    time.sleep(0.002)    # paced, so the shipper catches up and rotates between appends
"""


class Collector:
    def __init__(self):
        self.ids = []

    async def post(self, url, records):
        self.ids.extend(r["id"] for r in records)
        return True


def _spawn_writers(path):
    return [subprocess.Popen([sys.executable, "-c", _WRITER, str(path), str(w), str(RECORDS_EACH)])
            for w in range(WRITERS)]


def test_every_record_is_delivered_exactly_once_while_rotating(tmp_path):
    spool = tmp_path / "events.spool"
    sink = Collector()
    # A tiny rotate_bytes makes the shipper rotate whenever it catches up, many times
    # over the run, while the writers are still appending.
    shipper = SpoolShipper(str(spool), "http://cp/x", sink.post, name="t", rotate_bytes=2048,
                           batch_size=50)
    rotations = []
    original = shipper._rotate_if_done

    def counting(offset):
        before = spool.stat().st_size if spool.exists() else 0
        original(offset)
        if not spool.exists() or spool.stat().st_size < before:
            rotations.append(before)

    shipper._rotate_if_done = counting
    writers = _spawn_writers(spool)

    async def ship_until_writers_finish():
        deadline = time.monotonic() + 120
        while any(w.poll() is None for w in writers):
            assert time.monotonic() < deadline, "writers did not finish"
            await shipper.ship_once()
            await asyncio.sleep(0.002)
        for w in writers:
            assert w.returncode == 0
        for _ in range(3):                       # drain, and let rotation settle
            await shipper.ship_once()

    asyncio.run(ship_until_writers_finish())

    expected = {f"{w}-{i}" for w in range(WRITERS) for i in range(RECORDS_EACH)}
    counts = collections.Counter(sink.ids)
    duplicates = {k: v for k, v in counts.items() if v > 1}
    assert not duplicates, f"delivered more than once: {list(duplicates)[:5]}"
    assert set(counts) == expected, f"lost {len(expected - set(counts))} of {len(expected)} records"
    assert shipper.dropped == 0
    assert shipper.delivered == len(expected)
    assert len(rotations) >= 3, "the run did not exercise rotation under concurrent appends"


def test_the_shipper_rotated_during_the_run(tmp_path):
    """The scenario above only proves something if rotation actually happened."""
    spool = tmp_path / "events.spool"
    sink = Collector()
    shipper = SpoolShipper(str(spool), "http://cp/x", sink.post, name="t", rotate_bytes=512)
    rotations = []
    original = shipper._rotate_if_done

    def counting(offset):
        before = spool.stat().st_size if spool.exists() else 0
        original(offset)
        after = spool.stat().st_size if spool.exists() else 0
        if after < before:
            rotations.append(before)

    shipper._rotate_if_done = counting

    async def scenario():
        for i in range(40):
            append_record(spool, {"id": str(i), "pad": "y" * 60})
            await shipper.ship_once()

    asyncio.run(scenario())
    assert rotations, "no rotation happened"
    assert sink.ids == [str(i) for i in range(40)]


def test_rotation_keeps_a_record_appended_just_before_it_takes_the_lock(tmp_path, monkeypatch):
    """A writer can append between the shipper's "fully delivered" check and its
    acquiring the lock; that record must be carried over, not lost with the rename."""
    import contextlib

    import znyx_runtime.shipper as shipper_module

    spool = tmp_path / "events.spool"
    real_lock = shipper_module.spool_lock
    sneaked = []

    @contextlib.contextmanager
    def lock_after_a_late_writer(path):
        if not sneaked:
            sneaked.append(True)
            append_record(path, {"id": "late"})      # lands before the shipper holds the lock
        with real_lock(path):
            yield

    monkeypatch.setattr(shipper_module, "spool_lock", lock_after_a_late_writer)
    sink = Collector()
    shipper = SpoolShipper(str(spool), "http://cp/x", sink.post, name="t", rotate_bytes=1)
    for i in range(3):
        append_record(spool, {"id": f"a{i}"})

    async def scenario():
        assert await shipper.ship_once() == 3        # delivers a0..a2, then rotates
        assert await shipper.ship_once() == 1        # "late" survived the rotation

    asyncio.run(scenario())
    assert sink.ids == ["a0", "a1", "a2", "late"]


# ---------------------------------------------------------------------------
# The lock itself
# ---------------------------------------------------------------------------

def test_spool_lock_serialises_threads(tmp_path):
    spool = tmp_path / "events.spool"
    inside = 0
    overlaps = []
    mu = threading.Lock()

    def worker():
        nonlocal inside
        for _ in range(40):
            with spool_lock(spool):
                with mu:
                    inside += 1
                    overlaps.append(inside)
                time.sleep(0.0005)
                with mu:
                    inside -= 1

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert not any(t.is_alive() for t in threads), "deadlock"
    assert len(overlaps) == 160
    assert max(overlaps) == 1, "two threads were inside the lock at once"


def test_spool_lock_is_released_after_an_exception(tmp_path):
    spool = tmp_path / "events.spool"
    try:
        with spool_lock(spool):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    done = threading.Event()

    def again():
        with spool_lock(spool):
            done.set()

    t = threading.Thread(target=again)
    t.start()
    assert done.wait(5), "the lock was left held"
    t.join()


def test_append_lines_writes_complete_lines_and_a_private_lock_file(tmp_path):
    spool = tmp_path / "nested" / "dir" / "events.spool"
    append_lines(spool, ['{"a":1}', '{"a":2}'])
    append_lines(spool, ['{"a":3}'], fsync=False)
    text = spool.read_text()
    assert text == '{"a":1}\n{"a":2}\n{"a":3}\n'
    assert [json.loads(line)["a"] for line in text.splitlines()] == [1, 2, 3]
    lock = lock_path(spool)
    assert lock.name == "events.spool.lock" and lock.is_file()
    assert stat.S_IMODE(lock.stat().st_mode) == 0o600


def test_append_record_serialises_one_json_line_with_unusual_values(tmp_path):
    spool = tmp_path / "events.spool"
    append_record(spool, {"when": object, "n": 1})          # default=str keeps it writable
    [line] = spool.read_text().splitlines()
    assert json.loads(line)["n"] == 1


def test_concurrent_appenders_never_interleave_a_line(tmp_path):
    spool = tmp_path / "events.spool"

    def writer(tag):
        for i in range(200):
            append_record(spool, {"id": f"{tag}-{i}", "pad": tag * 500})

    threads = [threading.Thread(target=writer, args=(c,)) for c in "abcd"]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    lines = spool.read_text().splitlines()
    assert len(lines) == 800
    for line in lines:
        rec = json.loads(line)                              # every line is whole JSON
        assert rec["pad"] == rec["id"][0] * 500
