import asyncio
import os
import logging
import httpx
from .models import HeartbeatPayload
from .validator import get_license, get_evaluation_count, get_uptime_seconds

logger = logging.getLogger(__name__)


async def start_heartbeat(interval_seconds: int = 300):
    """Send periodic heartbeats to the license server."""
    license_server_base = (
        os.getenv("ZNYX_LICENSE_SERVER_URL", "")
        or os.getenv("GUARDRAILS_LICENSE_SERVER_URL", "")
        or "https://api.znyx.ai/v1/licenses/validate"
    )
    heartbeat_url = license_server_base.replace("/validate", "/heartbeat")

    while True:
        await asyncio.sleep(interval_seconds)
        license = get_license()
        if not license.license_key:
            continue  # Free tier, no heartbeat needed

        payload = HeartbeatPayload(
            license_key=license.license_key,
            evaluation_count=get_evaluation_count(),
            uptime_seconds=get_uptime_seconds(),
        )

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(heartbeat_url, json=payload.model_dump())
                if resp.status_code == 200:
                    logger.debug("Heartbeat sent successfully")
                else:
                    logger.debug(f"Heartbeat response: {resp.status_code}")
        except Exception as e:
            logger.debug(f"Heartbeat failed: {e}")
