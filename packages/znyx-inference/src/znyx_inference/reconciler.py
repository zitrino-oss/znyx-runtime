"""Model reconciler: makes the loaded model set match a desired pin set.

This service is DRIVEN, not scheduled. The runtime owns the control-plane channel and
hands us the desired pins over the deployment's own network on every bundle cycle; we
converge to them and report back what is actually loaded. There is deliberately no
timer, no control-plane client and no credential here.

Why the runtime drives it (this replaced a poll loop that lived in this package):

  * A pin lives INSIDE the policy bundle, so anything allowed to fetch pins is allowed
    to read the whole policy. There is no narrower credential to hand a sidecar, so the
    only way to keep policy out of this process is for it never to call the control
    plane at all. It now needs no token and no outbound internet access.
  * One reader means desired state cannot drift. Two independent pollers can each be
    individually healthy while disagreeing about which model is wanted, and nothing
    surfaces the disagreement.
  * Convergence is free. The runtime re-pushes every cycle whether or not anything
    changed, so a fetch that failed once is retried on the next push. The old poll loop
    skipped reconciliation entirely on an HTTP 304, which meant a single transient
    download failure left a model permanently unserved until someone republished or
    restarted the sidecar.

Posture (unchanged from the poll-based version):
  * Fetches go through the same explicit, shortlist-enforced, sha256-pinned path as the
    operator install endpoint (``runners/_fetch.py``). Never an implicit download.
  * Off-shortlist pins are skipped and logged unless ZNYX_INFERENCE_ALLOW_UNVETTED=true.
  * A pinned model loads as a registry VARIANT and never displaces the operator's active
    slot, so several projects sharing one sidecar are each served the model they pinned.
  * Every failure is contained to one pin: it is logged and skipped, and the rest of the
    desired set still applies.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from znyx_inference.registry import RunnerRegistry

logger = logging.getLogger(__name__)


def sidecar_version() -> Optional[str]:
    try:
        from importlib.metadata import version
        return version("znyx-inference")
    except Exception:  # noqa: BLE001 - version is informational only
        return None


def _allow_unvetted() -> bool:
    return (os.getenv("ZNYX_INFERENCE_ALLOW_UNVETTED", "") or "").lower() in ("true", "1", "yes")


class ModelReconciler:
    """Converges the registry's loaded models onto a desired pin set."""

    def __init__(self, registry: RunnerRegistry):
        self.registry = registry
        # The pin set from the previous reconcile, needed to spot a pin that has been
        # REMOVED (or repointed) so its variant can be evicted. A push carries the full
        # desired set, so anything here but not in the new set is genuinely gone.
        self._last_pins: Dict[str, Dict[str, Any]] = {}
        # Serialises reconcile cycles. REQUIRED, not defensive: this is now driven by an HTTP
        # endpoint, so two pushes can arrive concurrently (a slow cycle still downloading while
        # the runtime's next 30s tick fires, or two runtime replicas sharing one sidecar).
        # Without it, two cycles can fetch the same model into the same directory at once and
        # corrupt the artifact, and the read-modify-write of _last_pins races so eviction sees
        # a half-updated previous set. The predecessor design could not hit this because it was
        # a single timer loop; moving to push-driven reconciliation introduced the exposure.
        self._lock = asyncio.Lock()

    @property
    def last_pins(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._last_pins)

    async def reconcile(self, pins: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """Make the loaded set match ``pins``. Returns a summary plus what is loaded.

        Idempotent by design: the common case is a push identical to the last one, where
        every pin is already served, nothing is evicted, and this is a dictionary lookup
        per task. That is what makes per-cycle re-pushing cheap enough to rely on for
        retry.

        Serialised: concurrent callers queue rather than interleave. A caller arriving during a
        multi-minute model download waits for it, which is correct - the alternative is two
        threads writing the same artifact directory.
        """
        async with self._lock:
            return await self._reconcile_locked(pins)

    async def _reconcile_locked(self, pins: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
        pins = pins if isinstance(pins, dict) else {}
        previous = self._last_pins
        self._last_pins = pins

        applied, skipped = await self._apply(pins)
        evicted = await self._evict_removed(previous, pins)

        # Only log when something actually changed, so a steady state does not emit a
        # line on every push. The per-model lines below already cover a download.
        if applied or skipped or evicted:
            logger.info("reconcile: applied=%s skipped=%s evicted=%s", applied, skipped, evicted)

        return {
            "applied": applied,
            "skipped": skipped,
            "evicted": evicted,
            "pins": pins,
            "models": [m.model_dump() for m in self.registry.list_models()],
            "sidecar_version": sidecar_version(),
        }

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
            except Exception as exc:  # noqa: BLE001 - one bad pin must not block the rest
                logger.warning("reconcile: could not serve %s=%s@%s: %s",
                               task, model_id, revision, exc)
                skipped.append(f"{task}:{model_id}@{revision}")
        return applied, skipped

    async def _evict_removed(self, previous: Dict[str, Dict[str, Any]],
                             current: Dict[str, Dict[str, Any]]) -> List[str]:
        """Unload any variant we loaded for a task whose pin has since been removed or
        repointed at a different model.

        Safe without provenance tracking: ``registry.load_variant`` has exactly one
        caller in the whole package (``_install_and_load`` below), so every entry in the
        registry's variant tables was put there by a pin, never by the manual
        install/reload path (which only ever replaces the ACTIVE slot via
        ``reload_task``).
        """
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
                logger.info("reconcile: unloaded %s@%s for task %s (no longer pinned)",
                            model_id, revision, task)
        return evicted

    async def _install_and_load(self, task: str, model_id: str, revision: str,
                                threshold: Any,
                                expected_sha256: Optional[str] = None) -> None:
        from znyx_inference.runners._fetch import fetch_model, resolve_fetch_target

        # Shortlist enforcement lives in resolve_fetch_target - an off-shortlist pin
        # raises ValueError here and is skipped (unless the operator opted out).
        target = resolve_fetch_target(task, model_id=model_id, revision=revision,
                                      allow_unvetted=_allow_unvetted())
        dest = Path(target["dest_dir"])
        sha: Optional[str] = None
        if not (dest.exists() and any(dest.iterdir())):
            logger.info("reconcile: fetching %s@%s for task %s (this can take a while for "
                        "large models - huggingface_hub's own progress bar is disabled on a "
                        "non-tty stream, so nothing more will print here until it finishes)",
                        model_id, revision, task)
            sha = await asyncio.to_thread(
                fetch_model, target["model_id"], target["revision"], target["dest_dir"],
                runner=target["runner"])
            logger.info("reconcile: fetch complete for %s@%s (sha256=%s)", model_id, revision, sha)
            # The control plane told us which artifact this pin means. Refuse a different
            # one rather than loading it and reporting the wrong digest as fact.
            if expected_sha256 and sha and sha != expected_sha256:
                raise RuntimeError(
                    f"sha256 mismatch for {model_id}@{revision}: control plane expects "
                    f"{expected_sha256}, fetched artifact is {sha}")
        else:
            logger.info("reconcile: %s@%s for task %s already cached, loading",
                        model_id, revision, task)

        spec: Dict[str, Any] = {
            "runner": target["runner"],
            "model_id": target["model_id"],
            "revision": target["revision"],
        }
        # Prefer the digest we just computed; otherwise carry the control plane's
        # expectation so the runner still verifies an artifact that was already on disk
        # (previously a cached model loaded with no sha256 in its spec, i.e. unverified).
        if sha or expected_sha256:
            spec["sha256"] = sha or expected_sha256
        if threshold is not None:
            spec["threshold"] = threshold
        info = await self.registry.load_variant(task, spec)
        if not info.available:
            raise RuntimeError(info.detail or "variant failed to load")
        logger.info("reconcile: now serving %s@%s for task %s", model_id, revision, task)
