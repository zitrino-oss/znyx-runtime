"""
Lightweight telemetry emitter for the Guardrails Runtime.

Collects metadata-only evaluation events (no raw text) and delivers them to the
control plane in batches through a durable local spool. Never blocks evaluation.
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from znyx_runtime.shipper import SpoolShipper, append_record, http_post

logger = logging.getLogger(__name__)

# Tunable centralisation — see app/config/tunables.py
from znyx_core.config.tunables import (
    TELEMETRY_BATCH_SIZE as BATCH_SIZE,
    TELEMETRY_FLUSH_INTERVAL_SECONDS as FLUSH_INTERVAL,
)

DEFAULT_SPOOL = Path.home() / ".znyx" / "telemetry.spool"


def metadata_only(event: Dict[str, Any]) -> Dict[str, Any]:
    """The event as it may leave the customer's network: no request or response text.

    Some rule-hit messages quote what they matched (a URL, a quoted snippet, an
    equation, a hostname), so they are replaced by the rule id before the event is
    spooled for the control plane. Everything else in the event is already metadata.
    """
    results = event.get("detector_results")
    if not isinstance(results, list):
        return event
    cleaned = []
    for result in results:
        if isinstance(result, dict) and isinstance(result.get("rule_hits"), list):
            result = {**result, "rule_hits": [
                {**hit, "message": hit.get("rule_id", "")} if isinstance(hit, dict) else hit
                for hit in result["rule_hits"]]}
        cleaned.append(result)
    return {**event, "detector_results": cleaned}


class TelemetryEmitter:
    """Telemetry emitter that delivers events to the control plane durably.

    ``emit`` appends the event to a local spool (no fsync: it survives a process
    crash, and telemetry does not need to survive power loss). A ``SpoolShipper``
    delivers the spool in batches and only advances past a batch once the control
    plane accepts it, so an outage delays telemetry instead of dropping it. Events
    used to sit in a memory buffer that was discarded whenever a send failed.
    """

    def __init__(self, control_plane_url: str, runtime_token: str,
                 enabled: bool = False, spool_path: Optional[str] = None, ship: bool = True):
        self.control_plane_url = control_plane_url.rstrip("/")
        self.runtime_token = runtime_token
        self.enabled = enabled
        self.spool_path = str(spool_path or DEFAULT_SPOOL)
        # With several worker processes, every worker appends to the one spool and
        # only worker 0 delivers it (two shippers would send each event twice).
        self.ship = ship
        self._shipper = None

    def emit(self, event: Dict[str, Any]) -> None:
        """Record an event for delivery (non-blocking apart from a small file append)."""
        if not self.enabled:
            return
        try:
            append_record(self.spool_path, metadata_only(event))
        except OSError as exc:
            logger.debug("Telemetry spool write failed: %s", exc)

    async def start(self) -> None:
        """Start delivering the spool."""
        if not self.enabled or not self.ship:
            return
        self._shipper = SpoolShipper(
            self.spool_path, f"{self.control_plane_url}/v1/telemetry/events",
            http_post(self.runtime_token), name="telemetry",
            batch_size=BATCH_SIZE, interval=FLUSH_INTERVAL,
        )
        self._shipper.start()
        logger.info("Telemetry emitter started")

    async def stop(self) -> None:
        """Stop delivering, with one last attempt to send what is pending."""
        if self._shipper is not None:
            await self._shipper.stop(final_attempt=True)
            self._shipper = None

    def status(self) -> Dict[str, Any]:
        return self._shipper.status() if self._shipper else {"name": "telemetry", "enabled": self.enabled}
