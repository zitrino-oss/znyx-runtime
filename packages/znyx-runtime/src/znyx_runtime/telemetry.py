"""
Lightweight telemetry emitter for the Guardrails Runtime.

Collects metadata-only evaluation events (no raw text) and sends
them to the control plane in batches. Fire-and-forget - never blocks evaluation.
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional
from collections import deque

logger = logging.getLogger(__name__)

# Tunable centralisation — see app/config/tunables.py
from znyx_core.config.tunables import (
    TELEMETRY_BATCH_SIZE as BATCH_SIZE,
    TELEMETRY_FLUSH_INTERVAL_SECONDS as FLUSH_INTERVAL,
)

# Max events buffered before dropping oldest. Kept inline because it's a
# memory-pressure ceiling rather than a behaviour knob.
MAX_BUFFER_SIZE = 10000


class TelemetryEmitter:
    """Async telemetry emitter that batches events to the control plane."""

    def __init__(self, control_plane_url: str, runtime_token: str,
                 enabled: bool = False):
        self.control_plane_url = control_plane_url.rstrip("/")
        self.runtime_token = runtime_token
        self.enabled = enabled
        self._buffer: deque = deque(maxlen=MAX_BUFFER_SIZE)
        self._flush_task: Optional[asyncio.Task] = None

    def emit(self, event: Dict[str, Any]) -> None:
        """Add an event to the buffer (non-blocking)."""
        if not self.enabled:
            return
        self._buffer.append(event)

    async def start(self) -> None:
        """Start the background flush loop."""
        if not self.enabled:
            return
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("Telemetry emitter started")

    async def stop(self) -> None:
        """Stop the flush loop and send remaining events."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        # Final flush
        if self._buffer:
            await self._flush()

    async def _flush_loop(self) -> None:
        """Periodically flush buffered events."""
        while True:
            await asyncio.sleep(FLUSH_INTERVAL)
            if self._buffer:
                await self._flush()

    async def _flush(self) -> None:
        """Send buffered events to the control plane."""
        if not self._buffer:
            return

        # Drain up to BATCH_SIZE events
        batch: List[Dict[str, Any]] = []
        while self._buffer and len(batch) < BATCH_SIZE:
            batch.append(self._buffer.popleft())

        try:
            import httpx
            url = f"{self.control_plane_url}/v1/telemetry/events"
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    url,
                    json={"events": batch},
                    headers={
                        "X-API-Key": self.runtime_token,
                        "Content-Type": "application/json",
                    },
                )
                if resp.status_code >= 400:
                    logger.warning(f"Telemetry send failed: {resp.status_code}")
                else:
                    logger.debug(f"Sent {len(batch)} telemetry events")
        except Exception as e:
            logger.debug(f"Telemetry flush failed: {e}")
            # Events are lost - acceptable for fire-and-forget telemetry
