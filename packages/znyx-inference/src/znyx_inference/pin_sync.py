"""Pull-based model pin sync + heartbeat — the sidecar's desired-state loop.

The control plane cannot reach a sidecar running inside a customer's network, so desired
state flows the same direction as everything else: OUT of the customer's deployment. This
service polls the control plane's runtime bundle endpoint (the channel the runtime already
uses), reads ``runtime_policy.inference.pins`` — which model each task should serve for this
token's project/environment — installs anything missing into the local artifact cache, and
hot-loads it as a registry VARIANT (never displacing the operator's active slot). After each
cycle it reports what is actually loaded back to the control plane (heartbeat), which is what
the console renders for deployments it cannot dial.

Posture:
  * Fetches are the same explicit, shortlist-enforced, sha256-pinned path as the operator
    install endpoint (``runners/_fetch.py``) — never an implicit startup download.
  * Off-shortlist pins are skipped and logged unless ZNYX_INFERENCE_ALLOW_UNVETTED=true.
  * Every failure is contained to the cycle: log, heartbeat what we have, retry next tick.

Enabled automatically when ZNYX_CONTROL_PLANE_URL + ZNYX_RUNTIME_TOKEN are set (the same
variables the runtime deployment already carries); force on/off with ZNYX_INFERENCE_PIN_SYNC.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from znyx_inference.registry import RunnerRegistry

logger = logging.getLogger(__name__)


def _sidecar_version() -> Optional[str]:
    try:
        from importlib.metadata import version
        return version("znyx-inference")
    except Exception:  # noqa: BLE001 — version is informational only
        return None


class PinSyncConfig:
    def __init__(self, *, control_plane_url: str = "", token: str = "",
                 interval_s: float = 60.0, enabled: bool = False,
                 allow_unvetted: bool = False):
        self.control_plane_url = control_plane_url.rstrip("/")
        self.token = token
        self.interval_s = interval_s
        self.enabled = enabled
        self.allow_unvetted = allow_unvetted

    @classmethod
    def from_env(cls) -> "PinSyncConfig":
        url = (os.getenv("ZNYX_CONTROL_PLANE_URL", "") or "").strip()
        token = (os.getenv("ZNYX_RUNTIME_TOKEN", "") or "").strip()
        flag = (os.getenv("ZNYX_INFERENCE_PIN_SYNC", "") or "").strip().lower()
        if flag in ("false", "0", "no"):
            enabled = False
        elif flag in ("true", "1", "yes"):
            enabled = bool(url and token)   # forced on still needs somewhere to sync from
        else:
            enabled = bool(url and token)   # auto: on when the CP channel is configured
        try:
            interval = float(os.getenv("ZNYX_INFERENCE_PIN_SYNC_INTERVAL_S", "60"))
        except (TypeError, ValueError):
            interval = 60.0
        allow_unvetted = (os.getenv("ZNYX_INFERENCE_ALLOW_UNVETTED", "") or "").lower() \
            in ("true", "1", "yes")
        return cls(control_plane_url=url, token=token, interval_s=max(5.0, interval),
                   enabled=enabled, allow_unvetted=allow_unvetted)


class PinSyncService:
    def __init__(self, registry: RunnerRegistry, config: PinSyncConfig):
        self.registry = registry
        self.config = config
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._etag: Optional[str] = None
        self._last_pins: Dict[str, Dict[str, Any]] = {}

    # ── Lifecycle ───────────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if not self.config.enabled:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._loop(), name="inference-pin-sync")
        logger.info("pin-sync enabled: polling %s every %.0fs",
                    self.config.control_plane_url, self.config.interval_s)

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self.sync_once()
            except Exception as exc:  # noqa: BLE001 — a failed cycle must never kill the loop
                logger.warning("pin-sync cycle failed: %s", exc)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.config.interval_s)
            except asyncio.TimeoutError:
                pass

    # ── One cycle: fetch pins → apply → heartbeat ───────────────────────────────────

    async def sync_once(self) -> Dict[str, Any]:
        """One desired-state cycle. Returns a small summary (useful for tests/ops)."""
        pins = await self._fetch_pins()
        applied: List[str] = []
        skipped: List[str] = []
        if pins is not None:                 # None → 304 / unchanged
            self._last_pins = pins
            applied, skipped = await self._apply(pins)
        await self._heartbeat()
        return {"pins": self._last_pins, "applied": applied, "skipped": skipped}

    async def _fetch_pins(self) -> Optional[Dict[str, Dict[str, Any]]]:
        headers = {"X-API-Key": self.config.token}
        if self._etag:
            headers["If-None-Match"] = self._etag
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"{self.config.control_plane_url}/v1/bundles/latest",
                                    headers=headers)
        if resp.status_code == 304:
            return None
        resp.raise_for_status()
        self._etag = resp.headers.get("ETag") or self._etag
        body = resp.json()
        policies = body.get("policies") if isinstance(body, dict) else None
        rp = (policies or {}).get("runtime_policy") if isinstance(policies, dict) else None
        inference = (rp or {}).get("inference") if isinstance(rp, dict) else None
        pins = (inference or {}).get("pins") if isinstance(inference, dict) else None
        return pins if isinstance(pins, dict) else {}

    async def _apply(self, pins: Dict[str, Dict[str, Any]]) -> tuple[List[str], List[str]]:
        applied: List[str] = []
        skipped: List[str] = []
        for task, pin in pins.items():
            if not isinstance(pin, dict) or not pin.get("model_id"):
                continue
            model_id = str(pin["model_id"])
            revision = str(pin.get("revision") or "main")
            if self.registry.serves(task, model_id, revision):
                continue
            try:
                await self._install_and_load(task, model_id, revision, pin.get("threshold"))
                applied.append(f"{task}:{model_id}@{revision}")
            except Exception as exc:  # noqa: BLE001 — one bad pin must not block the rest
                logger.warning("pin-sync: could not serve %s=%s@%s: %s",
                               task, model_id, revision, exc)
                skipped.append(f"{task}:{model_id}@{revision}")
        return applied, skipped

    async def _install_and_load(self, task: str, model_id: str, revision: str,
                                threshold: Any) -> None:
        from znyx_inference.runners._fetch import fetch_model, resolve_fetch_target

        # Shortlist enforcement lives in resolve_fetch_target — an off-shortlist pin
        # raises ValueError here and is skipped (unless the operator opted out).
        target = resolve_fetch_target(task, model_id=model_id, revision=revision,
                                      allow_unvetted=self.config.allow_unvetted)
        dest = Path(target["dest_dir"])
        sha: Optional[str] = None
        if not (dest.exists() and any(dest.iterdir())):
            logger.info("pin-sync: fetching %s@%s for task %s", model_id, revision, task)
            sha = await asyncio.to_thread(
                fetch_model, target["model_id"], target["revision"], target["dest_dir"],
                runner=target["runner"])

        spec: Dict[str, Any] = {
            "runner": target["runner"],
            "model_id": target["model_id"],
            "revision": target["revision"],
        }
        if sha:
            spec["sha256"] = sha
        if threshold is not None:
            spec["threshold"] = threshold
        info = await self.registry.load_variant(task, spec)
        if not info.available:
            raise RuntimeError(info.detail or "variant failed to load")

    async def _heartbeat(self) -> None:
        """Report loaded models to the control plane. Best-effort: an older control plane
        without the endpoint (404) or a network blip must never affect serving."""
        payload = {
            "models": [m.model_dump() for m in self.registry.list_models()],
            "pins": self._last_pins,
            "sidecar_version": _sidecar_version(),
            "bundle_etag": self._etag,
        }
        env_name = (os.getenv("ZNYX_ENVIRONMENT_NAME", "") or "").strip()
        if env_name:
            payload["env"] = env_name    # project-scoped tokens: which env this sidecar is
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{self.config.control_plane_url}/v1/inference/heartbeat",
                    json=payload, headers={"X-API-Key": self.config.token})
            if resp.status_code == 404:
                logger.debug("pin-sync: control plane has no heartbeat endpoint (older CP)")
            elif resp.status_code >= 400:
                logger.warning("pin-sync: heartbeat rejected: %s %s",
                               resp.status_code, resp.text[:200])
        except Exception as exc:  # noqa: BLE001
            logger.debug("pin-sync: heartbeat failed: %s", exc)
