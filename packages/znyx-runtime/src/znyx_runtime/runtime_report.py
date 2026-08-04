"""Reports which bundle this runtime is serving, and what its sidecar has loaded.

The control plane cannot dial into a customer network, so it has no way to know whether a
published bundle was ever picked up. Without this, a bundle published five seconds ago and
one published last week that was never fetched look identical in the console. This closes
that gap with a single small POST per cycle.

One report carries BOTH facts because the runtime already holds both: it knows its own
active bundle, and the sidecar's loaded-model list came back in the reconcile response it
just made (see ``inference_sync``). Splitting them into two reporting paths would mean two
endpoints, two tables and a join in the console for no gain.

The runtime already exposes the same bundle metadata on ``GET /v1/bundle/status``, but that
is inbound and unreachable from the control plane. Everything here flows outward, matching
every other channel.

Best-effort throughout: an unreachable control plane, an older control plane without the
endpoint (404), or a malformed response must never affect policy enforcement. Silent when
no control-plane URL is configured, which is the normal OSS/YAML case - there is simply
nowhere to report to, and that is not an error.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 10.0


def _runtime_version() -> Optional[str]:
    try:
        from importlib.metadata import version
        return version("znyx-runtime")
    except Exception:  # noqa: BLE001 - version is informational only
        return None


class RuntimeReporter:
    """Posts bundle + sidecar state to the control plane once per bundle cycle."""

    def __init__(self, control_plane_url: str, runtime_token: str, mode: str = "managed"):
        self.control_plane_url = (control_plane_url or "").rstrip("/")
        self.runtime_token = runtime_token or ""
        self.mode = mode
        self._unsupported = False    # set once a control plane answers 404

    @property
    def enabled(self) -> bool:
        # Needs somewhere to report and something to authenticate with. Both absent is the
        # ordinary self-hosted case, not a misconfiguration.
        return bool(self.control_plane_url and self.runtime_token)

    async def report(self, bundle_info: Dict[str, Any],
                     sidecar_models: Optional[List[Dict[str, Any]]] = None,
                     sidecar_version: Optional[str] = None) -> bool:
        """Send one report. Returns True when the control plane accepted it."""
        if not self.enabled or self._unsupported:
            return False

        payload: Dict[str, Any] = {
            "models": sidecar_models or [],
            "sidecar_version": sidecar_version,
            "runtime_version": _runtime_version(),
            "mode": self.mode,
            "reporter": "runtime",
            "bundle_id": bundle_info.get("bundle_id"),
            "policy_hash": bundle_info.get("policy_hash"),
            # Kept for compatibility with control planes that already store this field from
            # the sidecar-shaped payload; it carries the same value as policy_hash.
            "bundle_etag": bundle_info.get("policy_hash"),
        }
        # A project-scoped token serves several environments, so name which one this is.
        # Prefer what the bundle itself says over the env var: the bundle is authoritative
        # about the scope it was published for.
        env_name = (bundle_info.get("environment")
                    or (os.getenv("ZNYX_ENVIRONMENT_NAME", "") or "").strip())
        if env_name:
            payload["env"] = env_name

        try:
            import httpx

            async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
                resp = await client.post(
                    f"{self.control_plane_url}/v1/inference/heartbeat",
                    json=payload,
                    headers={"X-API-Key": self.runtime_token},
                )
            if resp.status_code == 404:
                # Older control plane. Stop trying rather than logging every cycle forever.
                self._unsupported = True
                logger.debug("runtime-report: control plane has no heartbeat endpoint; "
                             "disabling reports for this process")
                return False
            if resp.status_code >= 400:
                logger.debug("runtime-report: rejected: %s %s",
                             resp.status_code, resp.text[:200])
                return False
            return True
        except Exception as exc:  # noqa: BLE001 - reporting must never affect enforcement
            logger.debug("runtime-report: failed (will retry next cycle): %s", exc)
            return False
