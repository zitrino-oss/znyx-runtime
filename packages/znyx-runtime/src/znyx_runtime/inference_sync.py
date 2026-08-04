"""Hands the inference sidecar its desired models, and collects what it has loaded.

The runtime is the ONLY component that talks to the control plane. It already fetches the
policy bundle, and that bundle carries ``runtime_policy.inference`` describing which model
each inference task should serve. This module forwards that to the sidecar over the
deployment's own network and keeps the sidecar's reply for the runtime's heartbeat.

Consequences of putting it here rather than in the sidecar (where it used to live as a
poll loop):

  * The sidecar needs no control-plane credential and no outbound internet access. A pin
    lives inside the policy, so any credential able to fetch pins could read the entire
    policy; there was no narrower key to issue.
  * Desired state cannot drift, because one process reads it.
  * Retry is automatic. ``push`` runs on every bundle cycle whether or not the bundle
    changed, so a model whose download failed is retried a cycle later. The sidecar's old
    loop skipped reconciliation on an HTTP 304 and therefore never retried.

Reads the same declarative block the control plane injects at publish time, exactly like
``judge_audit_sink`` reads ``runtime_policy.judge_budgets``. In YAML mode an operator can
write the block by hand and get the same behaviour with no control plane at all.

Address precedence, matching what the inference call path already does in
``znyx_core.engine.backends.build_strategy``:
  1. ``runtime_policy.inference.base_url`` from the bundle, when set.
  2. ``ZNYX_INFERENCE_URL``.
  3. ``http://localhost:9000`` (the co-located convention).
So a co-located sidecar needs no configuration at all, and no console setting is required
for model provisioning to work.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# The sidecar reconcile is a local call, but a cold start may block behind a large model
# download in a previous cycle, so allow more than a typical loopback timeout.
_PUSH_TIMEOUT_SECONDS = 30.0

# Must stay <= the control plane's own bound on the heartbeat models field, so we never build a
# report the control plane will reject.
_MAX_REPORTED_MODELS = 128


def _sidecar_base_url(policy: Optional[Dict[str, Any]]) -> str:
    """Where the sidecar is. See the address precedence in the module docstring."""
    from znyx_core.engine.ml_catalog import inference_url

    inference = _inference_block(policy)
    base = str((inference or {}).get("base_url") or "").strip()
    return (base or inference_url()).rstrip("/")


def _inference_block(policy: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(policy, dict):
        return None
    runtime_policy = policy.get("runtime_policy")
    if not isinstance(runtime_policy, dict):
        return None
    inference = runtime_policy.get("inference")
    return inference if isinstance(inference, dict) else None


def pins_from_policy(policy: Optional[Dict[str, Any]]) -> Optional[Dict[str, Dict[str, Any]]]:
    """Extract the desired pins, distinguishing "none" from "not stated".

    Three-state, matching what the control plane publishes:

    * ``{...}`` - install these.
    * ``{}``    - nothing is pinned. Meaningful and still pushed: it is what tells the
                  sidecar to evict models it previously loaded.
    * ``None``  - the bundle says nothing about pins, because the control plane could not
                  read them (or this policy predates the manifest). Say nothing to the
                  sidecar and leave the running state alone.

    Collapsing the last two would turn a transient control-plane read failure into
    "unload every pinned model", silently stripping ML enforcement from a healthy
    deployment. That is why an absent key is not an empty map.
    """
    inference = _inference_block(policy)
    if inference is None or "pins" not in inference:
        return None
    pins = inference.get("pins")
    if not isinstance(pins, dict):
        return None
    clean: Dict[str, Dict[str, Any]] = {}
    for task, pin in pins.items():
        if isinstance(pin, dict) and pin.get("model_id"):
            clean[str(task)] = pin
    return clean


class InferenceSync:
    """Pushes desired pins to the sidecar and remembers its last report."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._last_models: List[Dict[str, Any]] = []
        self._last_error: Optional[str] = None
        self._sidecar_version: Optional[str] = None

    @property
    def reported_models(self) -> List[Dict[str, Any]]:
        """What the sidecar last said it has loaded. Empty until a push succeeds."""
        return list(self._last_models)

    @property
    def sidecar_version(self) -> Optional[str]:
        return self._sidecar_version

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    async def push(self, policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Send the desired pins to the sidecar and keep its reply.

        Best-effort on purpose. A sidecar that is down, starting, or absent entirely must
        never affect policy enforcement: the detector's own strategy fallback already
        covers an unavailable ML backend, and the next cycle retries. Returns the sidecar's
        summary, or an empty dict when the push did not land.
        """
        if not self.enabled:
            return {}

        pins = pins_from_policy(policy)
        # The bundle stated nothing about pins (unreadable upstream, or an older policy).
        # Pushing an empty set here would evict models the deployment is legitimately
        # serving, so do nothing at all.
        if pins is None:
            return {}
        # No pins and nothing ever loaded is the common OSS case: skip the call rather than
        # logging a connection failure every cycle against a sidecar that may not exist.
        # Once something HAS been loaded we always push, including an empty set, so
        # eviction still reaches the sidecar.
        if not pins and not self._last_models:
            return {}

        base = _sidecar_base_url(policy)
        try:
            import httpx

            async with httpx.AsyncClient(timeout=_PUSH_TIMEOUT_SECONDS) as client:
                resp = await client.post(f"{base}/v1/models/desired", json={"pins": pins})
            if resp.status_code == 404:
                # An older sidecar without the reconcile endpoint. Nothing we can do from
                # here; it is a debug line rather than a warning because the operator's fix
                # is an image upgrade, not a runtime change.
                logger.debug("inference-sync: sidecar at %s has no /v1/models/desired "
                             "(older image); skipping", base)
                self._last_error = "sidecar too old"
                return {}
            resp.raise_for_status()
            summary = resp.json() if resp.content else {}
        except Exception as exc:  # noqa: BLE001 - never let this affect enforcement
            self._last_error = str(exc)
            logger.debug("inference-sync: push to %s failed (will retry next cycle): %s",
                         base, exc)
            return {}

        self._last_error = None
        if isinstance(summary, dict):
            models = summary.get("models")
            if isinstance(models, list):
                # Truncate rather than relay verbatim. The control plane bounds this field, so
                # an unexpectedly long list would make every heartbeat 422 and we would silently
                # stop reporting at all. A deployment serves a handful of tasks plus variants, so
                # hitting this means something is wrong upstream, not a real configuration.
                if len(models) > _MAX_REPORTED_MODELS:
                    logger.warning("inference-sync: sidecar reported %d models; truncating to %d",
                                   len(models), _MAX_REPORTED_MODELS)
                    models = models[:_MAX_REPORTED_MODELS]
                self._last_models = models
            version = summary.get("sidecar_version")
            if isinstance(version, str):
                self._sidecar_version = version
            if summary.get("applied") or summary.get("skipped") or summary.get("evicted"):
                logger.info("inference-sync: sidecar applied=%s skipped=%s evicted=%s",
                            summary.get("applied"), summary.get("skipped"),
                            summary.get("evicted"))
            return summary
        return {}
