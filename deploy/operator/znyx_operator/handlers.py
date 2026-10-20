"""kopf handlers: apply what render.py produces, keep it applied, report readiness.

Every child is written with server-side apply under the field manager ``znyx-operator``
(conflicts forced), so the operator owns exactly the fields it renders: a hand edit to one
of them is reverted on the next reconcile, while fields other controllers own (a HPA's
replica count excepted, see README) are left alone.

Run with ``kopf run --standalone`` semantics (see main.py): no peering objects are needed,
so run exactly one replica of the operator.
"""
from __future__ import annotations

import base64
import datetime
import logging
import threading
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional

import kopf
from kubernetes import client, config
from kubernetes.client.rest import ApiException

from . import render

PLURAL = "znyxinstances"
RECONCILE_INTERVAL = 60.0   # seconds between drift-correcting re-applies
STATUS_INTERVAL = 15.0      # seconds between readiness refreshes

_KINDS = {
    "Secret": ("v1", "secrets"),
    "ConfigMap": ("v1", "configmaps"),
    "PersistentVolumeClaim": ("v1", "persistentvolumeclaims"),
    "Service": ("v1", "services"),
    "Deployment": ("apps/v1", "deployments"),
    "PodDisruptionBudget": ("policy/v1", "poddisruptionbudgets"),
    "NetworkPolicy": ("networking.k8s.io/v1", "networkpolicies"),
}

_api_client: Optional[client.ApiClient] = None
_api_lock = threading.Lock()
# One reconcile per instance at a time: the create/update handler, the drift timer and the
# status timer all run in threads, and two reconciles racing on a first-time Secret could
# otherwise each generate a key.
_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _api() -> client.ApiClient:
    global _api_client
    with _api_lock:
        if _api_client is None:
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()
            _api_client = client.ApiClient()
        return _api_client


def _path(kind: str, namespace: str, name: str) -> str:
    api_version, plural = _KINDS[kind]
    base = "/api/v1" if api_version == "v1" else f"/apis/{api_version}"
    return f"{base}/namespaces/{namespace}/{plural}/{name}"


def _call(method: str, path: str, *, body: Any = None, query: Any = None,
          content_type: str = "application/json") -> Any:
    return _api().call_api(
        path, method, query_params=query or [],
        header_params={"Accept": "application/json", "Content-Type": content_type},
        body=body, response_type="object", auth_settings=["BearerToken"],
        _return_http_data_only=True, _preload_content=True)


def get_object(kind: str, namespace: str, name: str) -> Optional[Dict[str, Any]]:
    try:
        return _call("GET", _path(kind, namespace, name))
    except ApiException as exc:
        if exc.status == 404:
            return None
        raise


def apply_manifest(manifest: Mapping[str, Any]) -> None:
    meta = manifest["metadata"]
    _call("PATCH", _path(manifest["kind"], meta["namespace"], meta["name"]), body=manifest,
          query=[("fieldManager", render.FIELD_MANAGER), ("force", "true")],
          content_type="application/apply-patch+yaml")


def delete_object(kind: str, namespace: str, name: str) -> None:
    try:
        _call("DELETE", _path(kind, namespace, name))
    except ApiException as exc:
        if exc.status != 404:
            raise


def _existing_api_key(namespace: str, name: str) -> Optional[str]:
    secret = get_object("Secret", namespace, f"{name}-secrets")
    value = ((secret or {}).get("data") or {}).get(render.SECRET_API_KEY)
    return base64.b64decode(value).decode("utf-8") if value else None


def _prune(manifests: List[Mapping[str, Any]], namespace: str, name: str,
           log: logging.Logger) -> None:
    """Delete a PodDisruptionBudget / NetworkPolicy the spec no longer asks for."""
    rendered = {m["kind"] for m in manifests}
    for kind in render.PRUNABLE_KINDS:
        if kind in rendered:
            continue
        current = get_object(kind, namespace, name)
        labels = ((current or {}).get("metadata") or {}).get("labels") or {}
        if current and labels.get("app.kubernetes.io/managed-by") == "znyx-operator":
            log.info("removing %s %s/%s (no longer in the spec)", kind, namespace, name)
            delete_object(kind, namespace, name)


def _set_status(patch: kopf.Patch, current: Mapping[str, Any], new: Mapping[str, Any]) -> None:
    for key, value in new.items():
        if current.get(key) != value:
            patch.status[key] = value


def _reconcile(*, spec: Mapping[str, Any], name: str, namespace: str, uid: str,
               generation: int, status: Mapping[str, Any], patch: kopf.Patch,
               log: logging.Logger, body: Optional[Mapping[str, Any]] = None,
               apply: bool = True) -> None:
    """Validate, apply (when ``apply``), then publish status from the live Deployment."""
    lock = _locks[f"{namespace}/{name}"]
    with lock:
        errors = render.validate(spec, name)
        normalized = render.normalize(spec)
        version = str(normalized["version"])
        replicas = normalized["replicas"] if isinstance(normalized["replicas"], int) else 0
        previous = list(status.get("conditions") or [])
        try:
            if errors:
                log.warning("invalid spec, nothing applied: %s", "; ".join(errors))
                if body is not None and apply:
                    kopf.warn(body, reason="InvalidSpec", message="; ".join(errors))
                new = render.build_status(generation=generation, version=version,
                                          replicas=replicas, deployment=None, errors=errors,
                                          now=_now(), previous_conditions=previous)
                _set_status(patch, status, new)
                return
            if apply:
                manifests = render.render(spec, name, namespace, uid,
                                          existing_api_key=_existing_api_key(namespace, name))
                for manifest in manifests:
                    apply_manifest(manifest)
                _prune(manifests, namespace, name, log)
            deployment = get_object("Deployment", namespace, name)
        except ApiException as exc:
            raise kopf.TemporaryError(
                f"Kubernetes API returned {exc.status} {exc.reason}", delay=15) from exc
        new = render.build_status(generation=generation, version=version, replicas=replicas,
                                  deployment=deployment, errors=[], now=_now(),
                                  previous_conditions=previous)
        _set_status(patch, status, new)


@kopf.on.startup()
def configure(settings: kopf.OperatorSettings, **_: Any) -> None:
    # Keep kopf's bookkeeping in annotations: the CRD's structural schema prunes unknown
    # status fields, and the status stays ours alone.
    settings.persistence.progress_storage = kopf.AnnotationsProgressStorage(prefix="znyx.ai")
    settings.persistence.diffbase_storage = kopf.AnnotationsDiffBaseStorage(
        prefix="znyx.ai", key="last-handled-configuration")
    settings.posting.level = logging.INFO


@kopf.on.create(render.GROUP, render.VERSION, PLURAL)
@kopf.on.update(render.GROUP, render.VERSION, PLURAL)
@kopf.on.resume(render.GROUP, render.VERSION, PLURAL)
def reconcile(spec, name, namespace, uid, meta, status, patch, body, logger, **_):
    _reconcile(spec=spec, name=name, namespace=namespace, uid=uid,
               generation=meta.get("generation", 0), status=status, patch=patch,
               log=logger, body=body)


@kopf.timer(render.GROUP, render.VERSION, PLURAL, interval=RECONCILE_INTERVAL,
            initial_delay=RECONCILE_INTERVAL)
def correct_drift(spec, name, namespace, uid, meta, status, patch, logger, **_):
    """Re-apply everything: reverts hand edits and recreates deleted children."""
    _reconcile(spec=spec, name=name, namespace=namespace, uid=uid,
               generation=meta.get("generation", 0), status=status, patch=patch, log=logger)


@kopf.timer(render.GROUP, render.VERSION, PLURAL, interval=STATUS_INTERVAL,
            initial_delay=STATUS_INTERVAL)
def refresh_status(spec, name, namespace, uid, meta, status, patch, logger, **_):
    """Read the Deployment and publish readiness, without writing anything else."""
    _reconcile(spec=spec, name=name, namespace=namespace, uid=uid,
               generation=meta.get("generation", 0), status=status, patch=patch, log=logger,
               apply=False)
