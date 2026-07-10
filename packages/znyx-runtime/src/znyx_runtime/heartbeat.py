"""
Anonymous install heartbeat for ZNYX Runtime.

Sends a daily anonymous ping with non-sensitive metadata:
- install_id (random UUID, persisted via shared install state)
- version, mode, OS, Python version
- detector count, total evaluation count, run_count

No PII, no request content, no tenant data.
Opt-in: set ZNYX_TELEMETRY=true and ZNYX_HEARTBEAT_URL (off by default; the
runtime never phones home unless you point it at your own receiver).
"""
import asyncio
import logging
import os
import platform
from datetime import datetime, timezone
from typing import Optional

from znyx_runtime.install_state import get_install_id

logger = logging.getLogger(__name__)

HEARTBEAT_ENDPOINT = os.getenv(
    # Opt-in only. Empty by default so the runtime never phones home. Set
    # ZNYX_HEARTBEAT_URL to your own telemetry receiver to enable heartbeats.
    "ZNYX_HEARTBEAT_URL",
    "",
)
HEARTBEAT_INTERVAL = 86400  # 24 hours
VERSION = "1.0.0"


def _build_payload(
    install_id: str,
    mode: str,
    detector_count: int = 0,
    eval_count: int = 0,
    run_count: int = 0,
    event_type: str = "heartbeat",
) -> dict:
    """Build the heartbeat payload - no PII, no content."""
    return {
        "install_id": install_id,
        "version": VERSION,
        "event_type": event_type,
        "mode": mode,
        "source": "runtime",
        "os": platform.system(),
        "os_version": platform.release(),
        "arch": platform.machine(),
        "python_version": platform.python_version(),
        "detector_count": detector_count,
        "eval_count": eval_count,
        "run_count": run_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


class Heartbeat:
    """Anonymous heartbeat that sends a daily ping."""

    def __init__(self, enabled: bool = True, mode: str = "local"):
        self.enabled = enabled
        self.mode = mode
        self.install_id = get_install_id() if enabled else ""
        self.eval_count = 0
        self._task: Optional[asyncio.Task] = None

    def increment_eval_count(self) -> None:
        """Call after each evaluation to track usage count."""
        self.eval_count += 1

    async def start(self) -> None:
        """Start the background heartbeat loop."""
        if not self.enabled:
            logger.info("Anonymous telemetry disabled (ZNYX_TELEMETRY=false)")
            return

        logger.info(f"Anonymous telemetry enabled (install_id={self.install_id[:8]}...)")
        # Send initial ping immediately
        await self._send_ping()
        # Schedule daily pings
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Stop the heartbeat loop."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        """Send a ping every 24 hours."""
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            await self._send_ping()

    async def _send_ping(self, event_type: str = "heartbeat", run_count: int = 0) -> None:
        """Send a single heartbeat ping. Fire-and-forget."""
        try:
            import httpx
            from znyx_runtime.install_state import get_run_count

            payload = _build_payload(
                install_id=self.install_id,
                mode=self.mode,
                eval_count=self.eval_count,
                run_count=run_count or get_run_count(),
                event_type=event_type,
            )

            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    HEARTBEAT_ENDPOINT,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                if resp.status_code < 400:
                    logger.debug("Heartbeat sent")
                else:
                    logger.debug(f"Heartbeat response: {resp.status_code}")
        except Exception as e:
            # Never fail the runtime because of telemetry
            logger.debug(f"Heartbeat failed (non-fatal): {e}")

    async def send_first_run_ping(self, run_count: int = 1) -> None:
        """Send a one-shot first_run ping (called from installer, not the loop)."""
        await self._send_ping(event_type="first_run", run_count=run_count)
