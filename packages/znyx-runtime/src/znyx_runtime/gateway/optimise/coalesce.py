"""Request coalescing (technique 2): identical requests in flight at the same moment
share one upstream call.

Only for requests the planner found deterministic (temperature 0, one choice, not
streamed, not stored). The key covers the caller's gateway key, the provider URL, the
headers sent upstream (so the provider credential and API version) and the whole
request body after the input guard, so only truly identical calls of one caller meet.
Each caller still runs its own output guard on its own copy of the answer.

The shared call keeps running while anyone waits for it: a caller that leaves stops
waiting, and the call is cancelled only when no caller is left. The provider bills the
call once, so exactly one caller pays for it: the first to receive the answer (the one
that started it, unless it left). A call cancelled because everyone left is paid, as an
estimate, by the last caller to leave.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple


class Flight:
    """One upstream call and the callers waiting for it."""

    def __init__(self, task: "asyncio.Future", shared: bool, on_close: Callable[["Flight"], None]):
        self.task = task
        self.shared = shared
        self.waiters = 0
        self.closing = False
        self.paid = False
        self.started = time.perf_counter()
        self.finished: Optional[float] = None
        self._on_close = on_close

    def leave(self) -> bool:
        """A caller stopped waiting (its client went away, or its handler was cancelled).
        Returns True when it was the last one: the call is then cancelled and nobody new
        can join it."""
        self.waiters -= 1
        if self.waiters > 0 or self.task.done():
            return False
        self.closing = True
        self._on_close(self)
        self.task.cancel()
        return True

    def pay(self) -> bool:
        """True for the one caller that pays for the call (the first to ask)."""
        if self.paid:
            return False
        self.paid = True
        return True

    @property
    def upstream_ms(self) -> Optional[int]:
        if self.finished is None:
            return None
        return int((self.finished - self.started) * 1000)


class Coalescer:
    def __init__(self) -> None:
        self._flights: Dict[str, Flight] = {}

    @staticmethod
    def key(caller: str, url: str, headers: Dict[str, str], body: Dict[str, Any]) -> str:
        material = json.dumps([caller, url, sorted(headers.items()), body], sort_keys=True,
                              separators=(",", ":"), ensure_ascii=False, default=str)
        return hashlib.sha256(material.encode("utf-8", "surrogatepass")).hexdigest()

    def join(self, key: Optional[str], start: Callable[[], Awaitable[Any]]) -> Tuple[Flight, bool]:
        """The flight for ``key`` (a new one, started now, when none is in flight or the
        one in flight is being cancelled) and whether this caller started it. ``key``
        None: a private flight, never shared."""
        flight = self._flights.get(key) if key is not None else None
        if flight is not None and (flight.closing or flight.task.done()):
            flight = None
        leader = flight is None
        if leader:
            def close(f: Flight, key=key) -> None:
                if key is not None and self._flights.get(key) is f:
                    del self._flights[key]

            flight = Flight(asyncio.ensure_future(start()), shared=key is not None, on_close=close)

            def done(_task, flight=flight) -> None:
                flight.finished = time.perf_counter()
                close(flight)

            flight.task.add_done_callback(done)
            if key is not None:
                self._flights[key] = flight
        flight.waiters += 1
        return flight, leader

    def in_flight(self) -> int:
        return len(self._flights)
