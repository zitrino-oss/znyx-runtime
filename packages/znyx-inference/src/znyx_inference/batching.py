"""Dynamic batching: a bounded-latency queue that coalesces concurrent
/v1/infer calls into batches handed to a runner. A single worker drains the queue so
model inference is serialized (no re-entrancy), batches form up to ``max_batch_size``
within a ``max_wait_ms`` window, and the service stays within a per-task latency budget —
returning 429 (Saturated) instead of blocking past it. Dependency-free (asyncio only).
"""
from __future__ import annotations

import asyncio
from typing import List

from znyx_inference.runners.base import InferOutput, Runner


class Saturated(Exception):
    """The queue is full or a request exceeded the latency budget → caller returns 429."""


class BatchProcessor:
    def __init__(self, runner: Runner, *, max_batch_size: int = 16,
                 max_wait_ms: int = 10, max_queue: int = 256, budget_ms: int = 2000):
        self.runner = runner
        self.max_batch_size = max(1, max_batch_size)
        self.max_wait_ms = max(0, max_wait_ms)
        self.max_queue = max(1, max_queue)
        self.budget_ms = max(1, budget_ms)
        self._queue: "asyncio.Queue[tuple]" = asyncio.Queue()
        self._worker: asyncio.Task | None = None
        self._running = False
        # observable metrics (acceptance: batching + saturation are visible)
        self.batches_processed = 0
        self.items_processed = 0
        self.rejected = 0

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
        is full or the latency budget is exceeded — never blocks past the budget."""
        if self._queue.qsize() >= self.max_queue:
            self.rejected += 1
            raise Saturated("inference queue saturated")
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await self._queue.put((text, fut))
        try:
            return await asyncio.wait_for(fut, timeout=self.budget_ms / 1000.0)
        except asyncio.TimeoutError:
            self.rejected += 1
            raise Saturated("inference exceeded latency budget")

    async def _collect_batch(self) -> List[tuple]:
        first = await self._queue.get()
        batch = [first]
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.max_wait_ms / 1000.0
        while len(batch) < self.max_batch_size:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
            except asyncio.TimeoutError:
                break
        return batch

    async def _run(self) -> None:
        while self._running:
            try:
                batch = await asyncio.wait_for(self._collect_batch(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            texts = [t for t, _ in batch]
            try:
                # Runner inference is sync/CPU-bound → offload so the loop keeps serving.
                outputs = await asyncio.to_thread(self.runner.infer_batch, texts)
            except Exception as exc:  # noqa: BLE001 — propagate to every waiter in the batch
                for _, fut in batch:
                    if not fut.done():
                        fut.set_exception(exc)
                continue
            self.batches_processed += 1
            self.items_processed += len(batch)
            for (_, fut), out in zip(batch, outputs):
                if not fut.done():
                    fut.set_result(out)

    def stats(self) -> dict:
        return {
            "batches_processed": self.batches_processed,
            "items_processed": self.items_processed,
            "rejected": self.rejected,
            "queue_depth": self._queue.qsize(),
            "max_batch_size": self.max_batch_size,
            "avg_batch_size": (round(self.items_processed / self.batches_processed, 2)
                               if self.batches_processed else 0.0),
        }
