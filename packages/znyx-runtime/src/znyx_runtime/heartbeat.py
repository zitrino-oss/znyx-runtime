"""
Anonymous install heartbeat for ZNYX Runtime.

Sends a daily anonymous ping with non-sensitive metadata:
- install_id (random UUID, persisted via shared install state)
- version, mode, OS, Python version
- detector count, total evaluation count, run_count

No PII, no request content, no tenant data.

Opt-in for the runtime: nothing is sent unless ZNYX_TELEMETRY=true. The destination
defaults to the ZNYX receiver and is overridable with ZNYX_TELEMETRY_URL (set it to a
receiver you operate, or to an empty string to remove the destination altogether).
Note the SDKs use the opposite enable-default (opt-out) - see TELEMETRY.md.
See TELEMETRY.md at the repo root for the exact payload and endpoint.
"""
import asyncio
import logging
import os
import platform
from datetime import datetime, timezone
from typing import Optional

from znyx_runtime.install_state import get_install_id

logger = logging.getLogger(__name__)

DEFAULT_HEARTBEAT_ENDPOINT = "https://cp.znyx.ai/v1/install-telemetry"


def heartbeat_endpoint() -> str:
    """Where install pings go.

    Defaults to the ZNYX receiver. This is a DESTINATION only and does not enable
    anything: sending is gated on ZNYX_TELEMETRY, which is off by default for the
    runtime (see RuntimeConfig.heartbeat_enabled), so an unconfigured install stays
    silent. Point ZNYX_TELEMETRY_URL at your own receiver to self-host it, or set it
    to an empty string to remove the destination entirely.

    Read at call time rather than at import so a test or an embedding app can change
    it after import.
    """
    for name in ("ZNYX_TELEMETRY_URL", "ZNYX_HEARTBEAT_URL"):
        value = os.getenv(name)
        if value is not None:
            # An explicitly empty value is an intentional opt-out of the destination,
            # not "fall through to the default".
            return value.strip()
    return DEFAULT_HEARTBEAT_ENDPOINT


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

    def __init__(self, enabled: bool = False, mode: str = "local"):
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
        endpoint = heartbeat_endpoint()
        if not endpoint:
            # No destination configured. Nothing to do -- see the module docstring:
            # this package deliberately has no built-in receiver, so an unconfigured
            # install is silent rather than phoning somewhere by default.
            logger.debug("Heartbeat skipped: no ZNYX_TELEMETRY_URL configured")
            return
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
                    endpoint,
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
        """Send a one-shot first_run ping (fired once, on an install's first startup)."""
        await self._send_ping(event_type="first_run", run_count=run_count)
