"""Dynamic batching: a bounded-latency queue that coalesces concurrent
/v1/infer calls into batches handed to a runner. A single worker drains the queue so
model inference is serialized (no re-entrancy), batches form up to ``max_batch_size``
within a ``max_wait_ms`` window, and the service stays within a per-task latency budget -
returning 429 (Saturated) instead of blocking past it. The queue is hard-bounded
(``max_queue``, env ZNYX_INFERENCE_MAX_QUEUE): a full queue rejects immediately, and an
item whose waiter has already timed out is dropped at dequeue instead of being scored
into the void. Dependency-free (asyncio only).
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from znyx_inference.runners.base import InferOutput, Runner


class Saturated(Exception):
    """The queue is full or a request exceeded the latency budget → caller returns 429."""


class BatchProcessor:
    def __init__(self, runner: Runner, *, max_batch_size: int = 16,
                 max_wait_ms: int = 10, max_queue: int = 256, budget_ms: int = 2000,
                 cache_scope: str = ""):
        self.runner = runner
        self.max_batch_size = max(1, max_batch_size)
        self.max_wait_ms = max(0, max_wait_ms)
        self.max_queue = max(1, max_queue)
        self.budget_ms = max(1, budget_ms)
        # Opaque identity of the runner configuration this batcher serves (runner kind +
        # spec fingerprint); the service folds it into cache keys. Set by the registry.
        self.cache_scope = cache_scope
        # Hard bound: put_nowait raises QueueFull → reject fast, never queue past the cap.
        self._queue: "asyncio.Queue[tuple]" = asyncio.Queue(maxsize=self.max_queue)
        self._worker: asyncio.Task | None = None
        self._running = False
        # Parameterized requests can't join the coalescing queue, but they must not
        # bypass its bounds either - this counts their in-flight work (see run_direct).
        self._direct_inflight = 0
        # observable metrics (acceptance: batching + saturation are visible)
        self.batches_processed = 0
        self.items_processed = 0
        self.rejected = 0
        self.expired_dropped = 0

    async def start(self) -> None:
        if self._worker is None:
            self._running = True
            self._worker = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

    async def submit(self, text: str) -> InferOutput:
        """Enqueue one item and await its result. Raises Saturated (→429) when the queue
        is full or the latency budget is exceeded - never blocks past the budget."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        try:
            self._queue.put_nowait((text, fut, loop.time() + self.budget_ms / 1000.0))
        except asyncio.QueueFull:
            self.rejected += 1
            raise Saturated("inference queue saturated") from None
        try:
            return await asyncio.wait_for(fut, timeout=self.budget_ms / 1000.0)
        except asyncio.TimeoutError:
            # wait_for cancelled fut, so the worker drops the item at dequeue.
            self.rejected += 1
            raise Saturated("inference exceeded latency budget") from None

    async def run_direct(self, texts: List[str],
                         params: Optional[Dict[str, Any]] = None) -> List[InferOutput]:
        """Score outside the coalescing queue - per-request params can't be batched with
        other callers' texts - under the SAME bounds: at most ``max_queue`` requests in
        flight and the same latency budget, raising Saturated (→429) past either."""
        if self._direct_inflight >= self.max_queue:
            self.rejected += 1
            raise Saturated("inference queue saturated")
        self._direct_inflight += 1
        task = asyncio.ensure_future(asyncio.to_thread(self.runner.infer_batch, texts, params))
        task.add_done_callback(self._direct_done)
        try:
            # shield: the thread can't be interrupted anyway; on timeout the caller gets
            # a fast 429 while the counter keeps the abandoned work bounded to max_queue.
            return await asyncio.wait_for(asyncio.shield(task), timeout=self.budget_ms / 1000.0)
        except asyncio.TimeoutError:
            self.rejected += 1
            raise Saturated("inference exceeded latency budget") from None

    def _direct_done(self, task: "asyncio.Task") -> None:
        self._direct_inflight -= 1
        if not task.cancelled():
            task.exception()  # retrieve, so an abandoned failure doesn't warn at GC

    def _drop_if_stale(self, item: tuple, now: float) -> bool:
        """True when the item must not be scored: its waiter already gave up (future
        done/cancelled) or its deadline passed while queued. Dropping at dequeue keeps
        expired work from consuming model time."""
        _text, fut, deadline = item
        if fut.done():
            self.expired_dropped += 1
            return True
        if now > deadline:
            self.expired_dropped += 1
            fut.set_exception(Saturated("inference request expired in queue"))
            return True
        return False

    async def _fill_batch(self, batch: List[tuple]) -> List[tuple]:
        """Grow ``batch`` up to max_batch_size within the max_wait_ms window. Bounded:
        never awaits past the window, so the caller needs no outer timeout (an outer
        cancel between dequeue and dispatch would strand items)."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.max_wait_ms / 1000.0
        while len(batch) < self.max_batch_size:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            if not self._drop_if_stale(item, loop.time()):
                batch.append(item)
        return batch

    async def _run(self) -> None:
        while self._running:
            try:
                first = await asyncio.wait_for(self._queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            loop = asyncio.get_running_loop()
            if self._drop_if_stale(first, loop.time()):
                continue
            batch = await self._fill_batch([first])
            texts = [t for t, _, _ in batch]
            try:
                # Runner inference is sync/CPU-bound → offload so the loop keeps serving.
                outputs = await asyncio.to_thread(self.runner.infer_batch, texts)
            except Exception as exc:  # noqa: BLE001 - propagate to every waiter in the batch
                for _, fut, _ in batch:
                    if not fut.done():
                        fut.set_exception(exc)
                continue
            self.batches_processed += 1
            self.items_processed += len(batch)
            for (_, fut, _), out in zip(batch, outputs):
                if not fut.done():
                    fut.set_result(out)

    def stats(self) -> dict:
        return {
            "batches_processed": self.batches_processed,
            "items_processed": self.items_processed,
            "rejected": self.rejected,
            "expired_dropped": self.expired_dropped,
            "queue_depth": self._queue.qsize(),
            "direct_inflight": self._direct_inflight,
            "max_batch_size": self.max_batch_size,
            "avg_batch_size": (round(self.items_processed / self.batches_processed, 2)
                               if self.batches_processed else 0.0),
        }
