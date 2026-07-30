"""Pull-based model provisioning + heartbeat — the sidecar's desired-state loop.

The control plane cannot reach a sidecar running inside a customer's network, so desired
state flows the same direction as everything else: OUT of the customer's deployment. On
startup and every interval thereafter this service pings ``GET /v1/inference/models`` — which
model each task should serve for this key's project/environment — installs anything missing
into the local artifact cache, and hot-loads it as a registry VARIANT (never displacing the
operator's active slot). After each cycle it reports what is actually loaded back to the
control plane (heartbeat), which is what the console renders for deployments it cannot dial.

The whole loop needs exactly two settings: ``ZNYX_CONTROL_PLANE_URL`` and
``ZNYX_INFERENCE_API_KEY``. Nothing addresses the sidecar — the runtime reaches it at a
fixed convention address (same-pod loopback, or ``http://inference:9000`` in the shipped
compose), so there is no URL for an operator to configure or for the control plane to store.

Desired state deliberately does NOT come from the policy bundle. It used to be read out of
``runtime_policy.inference.pins`` in ``GET /v1/bundles/latest``, which meant a project that
had never published a bundle could never provision a model. The dedicated endpoint has no
such coupling — and it takes a narrowly scoped ``inference`` key that cannot read policy or
write telemetry, rather than the runtime's own token.

Posture:
  * Fetches are the same explicit, shortlist-enforced, sha256-pinned path as the operator
    install endpoint (``runners/_fetch.py``) — never an implicit startup download.
  * Off-shortlist pins are skipped and logged unless ZNYX_INFERENCE_ALLOW_UNVETTED=true.
  * Every failure is contained to the cycle: log, heartbeat what we have, retry next tick.

Enabled automatically when ZNYX_CONTROL_PLANE_URL + a key are set; force on/off with
ZNYX_INFERENCE_PIN_SYNC.
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
        # The dedicated inference key wins. ZNYX_RUNTIME_TOKEN stays as a fallback so a
        # deployment that pre-dates the inference key keeps syncing after an image bump —
        # the heartbeat accepts either credential, and the models endpoint accepts a runtime
        # token only if it happens to carry inference:sync.
        token = (os.getenv("ZNYX_INFERENCE_API_KEY", "") or "").strip() \
            or (os.getenv("ZNYX_RUNTIME_TOKEN", "") or "").strip()
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
        evicted: List[str] = []
        if pins is not None:                 # None → 304 / unchanged
            previous = self._last_pins
            self._last_pins = pins
            applied, skipped = await self._apply(pins)
            evicted = await self._evict_removed(previous, pins)
            # Only when something changed — unchanged desired state (304), or desired state
            # whose models are all already served, produces no line, so this doesn't spam the
            # log every poll interval; the per-model lines below already cover the download.
            if applied or skipped or evicted:
                logger.info("pin-sync: cycle complete — applied=%s skipped=%s evicted=%s",
                            applied, skipped, evicted)
        await self._heartbeat()
        return {"pins": self._last_pins, "applied": applied, "skipped": skipped, "evicted": evicted}

    async def _fetch_pins(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """Ping the control plane for this deployment's desired models.

        Returns ``{task: pin}`` (the shape ``_apply``/``_evict_removed`` consume), or None
        when the control plane reports 304 Not Modified — nothing has changed since the last
        cycle, so there is nothing to apply or evict.

        The endpoint scopes itself from the key: an environment-scoped key names its own
        environment, while a project-scoped key (the normal case — one key per project) needs
        ``?env=`` to pick up environment-specific pins, falling back to the project-wide
        cascade when ZNYX_ENVIRONMENT_NAME is unset.
        """
        headers = {"X-API-Key": self.config.token}
        if self._etag:
            headers["If-None-Match"] = self._etag
        params = {}
        env_name = (os.getenv("ZNYX_ENVIRONMENT_NAME", "") or "").strip()
        if env_name:
            params["env"] = env_name
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"{self.config.control_plane_url}/v1/inference/models",
                                    headers=headers, params=params or None)
        if resp.status_code == 304:
            return None
        resp.raise_for_status()
        self._etag = resp.headers.get("ETag") or self._etag
        body = resp.json()
        models = body.get("models") if isinstance(body, dict) else None
        if not isinstance(models, list):
            return {}

        # Flatten back to task-keyed form. The endpoint also supplies `runner` and `sha256`,
        # which _install_and_load uses to build the spec instead of re-deriving them.
        pins: Dict[str, Dict[str, Any]] = {}
        for entry in models:
            if not isinstance(entry, dict):
                continue
            task, model_id = entry.get("task"), entry.get("model_id")
            if not task or not model_id:
                continue
            pins[str(task)] = {
                "model_id": str(model_id),
                "revision": str(entry.get("revision") or "main"),
                "threshold": entry.get("threshold"),
                "runner": entry.get("runner"),
                "sha256": entry.get("sha256"),
            }
        return pins

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
                await self._install_and_load(task, model_id, revision, pin.get("threshold"),
                                             expected_sha256=pin.get("sha256"))
                applied.append(f"{task}:{model_id}@{revision}")
            except Exception as exc:  # noqa: BLE001 — one bad pin must not block the rest
                logger.warning("pin-sync: could not serve %s=%s@%s: %s",
                               task, model_id, revision, exc)
                skipped.append(f"{task}:{model_id}@{revision}")
        return applied, skipped

    async def _evict_removed(self, previous: Dict[str, Dict[str, Any]],
                             current: Dict[str, Dict[str, Any]]) -> List[str]:
        """Unload any variant this service loaded for a task whose pin has since been
        removed or changed to a different model. Safe without provenance tracking —
        ``registry.load_variant`` has exactly one caller in the whole package (this
        service), so every entry in the registry's variant tables was put there by a
        pin, never by the manual Install/Reload path (that only ever replaces the
        ACTIVE slot via ``reload_task``)."""
        evicted: List[str] = []
        for task, prev_pin in previous.items():
            if not isinstance(prev_pin, dict) or not prev_pin.get("model_id"):
                continue
            model_id = str(prev_pin["model_id"])
            revision = str(prev_pin.get("revision") or "main")
            cur_pin = current.get(task)
            still_pinned = (
                isinstance(cur_pin, dict)
                and str(cur_pin.get("model_id") or "") == model_id
                and str(cur_pin.get("revision") or "main") == revision
            )
            if still_pinned:
                continue
            if await self.registry.unload_variant(task, model_id, revision):
                evicted.append(f"{task}:{model_id}@{revision}")
                logger.info("pin-sync: unloaded %s@%s for task %s (no longer pinned)",
                            model_id, revision, task)
        return evicted

    async def _install_and_load(self, task: str, model_id: str, revision: str,
                                threshold: Any,
                                expected_sha256: Optional[str] = None) -> None:
        from znyx_inference.runners._fetch import fetch_model, resolve_fetch_target

        # Shortlist enforcement lives in resolve_fetch_target — an off-shortlist pin
        # raises ValueError here and is skipped (unless the operator opted out).
        target = resolve_fetch_target(task, model_id=model_id, revision=revision,
                                      allow_unvetted=self.config.allow_unvetted)
        dest = Path(target["dest_dir"])
        sha: Optional[str] = None
        if not (dest.exists() and any(dest.iterdir())):
            logger.info("pin-sync: fetching %s@%s for task %s (this can take a while for "
                        "large models — huggingface_hub's own progress bar is disabled on a "
                        "non-tty stream, so nothing more will print here until it finishes)",
                        model_id, revision, task)
            sha = await asyncio.to_thread(
                fetch_model, target["model_id"], target["revision"], target["dest_dir"],
                runner=target["runner"])
            logger.info("pin-sync: fetch complete for %s@%s (sha256=%s)",
                        model_id, revision, sha)
            # The control plane told us which artifact this pin means. Refuse a different
            # one rather than loading it and reporting the wrong digest as fact.
            if expected_sha256 and sha and sha != expected_sha256:
                raise RuntimeError(
                    f"sha256 mismatch for {model_id}@{revision}: control plane expects "
                    f"{expected_sha256}, fetched artifact is {sha}")
        else:
            logger.info("pin-sync: %s@%s for task %s already cached, loading",
                        model_id, revision, task)

        spec: Dict[str, Any] = {
            "runner": target["runner"],
            "model_id": target["model_id"],
            "revision": target["revision"],
        }
        # Prefer the digest we just computed; otherwise carry the control plane's expectation
        # so the runner still verifies an artifact that was already on disk (previously a
        # cached model loaded with no sha256 in its spec, i.e. unverified).
        if sha or expected_sha256:
            spec["sha256"] = sha or expected_sha256
        if threshold is not None:
            spec["threshold"] = threshold
        info = await self.registry.load_variant(task, spec)
        if not info.available:
            raise RuntimeError(info.detail or "variant failed to load")
        logger.info("pin-sync: now serving %s@%s for task %s", model_id, revision, task)

    async def _heartbeat(self) -> None:
        """Report loaded models to the control plane. Best-effort: an older control plane
        without the endpoint (404) or a network blip must never affect serving.

        ``bundle_etag`` keeps its wire name for compatibility with control planes already
        storing it, but now carries the ETag of ``GET /v1/inference/models`` — the version of
        the desired state this report corresponds to, which is what makes it diagnostic.
        """
        payload = {
            "models": [m.model_dump() for m in self.registry.list_models()],
            "pins": self._last_pins,
            "sidecar_version": _sidecar_version(),
            "bundle_etag": self._etag,
        }
        env_name = (os.getenv("ZNYX_ENVIRONMENT_NAME", "") or "").strip()
        if env_name:
            payload["env"] = env_name    # project-scoped keys: which env this sidecar is
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
