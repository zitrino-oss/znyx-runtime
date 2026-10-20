"""Pure rendering for the Znyx operator: a ZnyxInstance spec in, Kubernetes manifests out.

No Kubernetes client and no kopf is imported here, so the whole of "what does an instance
look like" is unit-testable without a cluster (tests/test_operator_render.py). The handlers
apply what this module returns and nothing else.

This module is the single description of the resources an instance consists of. The Helm
chart (deploy/helm) and the Terraform module (deploy/terraform) render the same resources,
so a change to a name, label, probe or default here must be made there as well.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import re
import secrets
from typing import Any, Dict, List, Mapping, Optional

import yaml

GROUP = "znyx.ai"
VERSION = "v1alpha1"
KIND = "ZnyxInstance"
API_VERSION = f"{GROUP}/{VERSION}"
FIELD_MANAGER = "znyx-operator"

DEFAULT_VERSION = "1.0.0"
DEFAULT_REPOSITORY = "znyx/znyx"
DEFAULT_REPLICAS = 2
HTTP_PORT = 8080
SERVICE_PORT = 80
SECRET_API_KEY = "RUNTIME_API_KEY"
SECRET_TOKEN = "ZNYX_RUNTIME_TOKEN"
SECRET_REDIS = "ZNYX_REDIS_URL"

# Kinds the operator may stop rendering (replicas back to 1, policy switched off) and
# therefore must delete, because a leftover PodDisruptionBudget blocks node drains and a
# leftover NetworkPolicy keeps denying traffic.
PRUNABLE_KINDS = ("PodDisruptionBudget", "NetworkPolicy")

_DEFAULT_CONFIG: Dict[str, Any] = {
    "version": 1,
    "server": {"host": "0.0.0.0", "port": HTTP_PORT, "workers": "auto"},
    "mode": "local",
    "dashboard": {"enabled": True},
}

_DEFAULT_RESOURCES = {
    "requests": {"cpu": "1", "memory": "512Mi"},
    "limits": {"cpu": "2", "memory": "1Gi"},
}
_DEFAULT_RESOURCES_ML = {
    "requests": {"cpu": "2", "memory": "2Gi"},
    "limits": {"cpu": "4", "memory": "4Gi"},
}

# The settings znyx.yaml accepts (mirrors znyx.config.SETTINGS; kept as data so the
# operator does not depend on the znyx package). An unknown key fails validation here
# instead of crash-looping the pods, which is what the runtime does with it.
_CONFIG_KEYS = frozenset({
    "server.host", "server.port", "server.workers", "server.graceful_timeout",
    "server.worker_timeout", "server.run_dir", "server.allowed_origins",
    "mode", "policy", "fail_mode",
    "control_plane.url", "control_plane.token", "control_plane.bundle_public_key",
    "control_plane.require_signed_bundles", "control_plane.poll_interval",
    "auth.require", "auth.api_key", "auth.api_key_file",
    "ml.enabled", "ml.port", "ml.artifacts_dir", "ml.url",
    "state.backend", "state.redis_url", "state.fail_mode",
    "telemetry.enabled", "telemetry.heartbeat",
    "dashboard.enabled", "gateway.config", "airgapped", "airgapped_allow", "log_level",
})
_CONFIG_SECTIONS = frozenset(k.split(".")[0] for k in _CONFIG_KEYS if "." in k)
_CONFIG_CHOICES = {
    "mode": ("local", "managed"),
    "fail_mode": ("open", "closed"),
    "state.backend": ("auto", "memory", "sqlite", "redis"),
    "state.fail_mode": ("local", "closed"),
    "log_level": ("critical", "error", "warning", "info", "debug"),
}

_DNS_LABEL = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
# The longest child name is "<name>-secrets"; labels hold at most 63 characters and the
# Deployment name is the pod-name prefix, so stay well inside both.
_MAX_NAME = 52
_PULL_POLICIES = ("Always", "IfNotPresent", "Never")


# ---------------------------------------------------------------------------
# Spec normalisation and validation
# ---------------------------------------------------------------------------

def _section(spec: Mapping[str, Any], key: str) -> Dict[str, Any]:
    value = spec.get(key)
    return dict(value) if isinstance(value, Mapping) else {}


def normalize(spec: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The spec with every default filled in (the CRD schema defaults the same values)."""
    spec = spec or {}
    image = _section(spec, "image")
    ml = _section(spec, "ml")
    net = _section(spec, "networkPolicy")
    persistence = _section(spec, "persistence")
    state = _section(spec, "state")
    replicas = spec.get("replicas")
    return {
        "version": spec.get("version") if spec.get("version") is not None else DEFAULT_VERSION,
        "image": {
            "repository": image.get("repository") or DEFAULT_REPOSITORY,
            "pullPolicy": image.get("pullPolicy") or "IfNotPresent",
            "pullSecrets": list(image.get("pullSecrets") or []),
        },
        "replicas": DEFAULT_REPLICAS if replicas is None else replicas,
        "ml": {
            "enabled": bool(ml.get("enabled", False)),
            "modelsClaim": ml.get("modelsClaim") or "",
            "modelsSizeLimit": ml.get("modelsSizeLimit") or "",
        },
        "airgapped": bool(spec.get("airgapped", False)),
        "config": copy.deepcopy(spec.get("config")) if spec.get("config") is not None else {},
        "policy": spec.get("policy") or "",
        "secretRef": spec.get("secretRef") or "",
        "state": {"redis": bool(state.get("redis", False))},
        "resources": copy.deepcopy(spec["resources"]) if spec.get("resources") else None,
        "networkPolicy": {
            "enabled": bool(net.get("enabled", False)),
            "ingressFrom": list(net.get("ingressFrom") or []),
            "egressTo": list(net.get("egressTo") or []),
        },
        "persistence": {
            "enabled": bool(persistence.get("enabled", False)),
            "size": persistence.get("size") or "1Gi",
            "storageClassName": persistence.get("storageClassName") or "",
        },
    }


def _check_config(config: Any) -> List[str]:
    if not isinstance(config, Mapping):
        return [f"spec.config must be a mapping of znyx.yaml settings, got {type(config).__name__}"]
    errors: List[str] = []
    flat: Dict[str, Any] = {}
    for key, value in config.items():
        key = str(key)
        if key == "version":
            if value != 1:
                errors.append(f"spec.config.version {value!r} is not supported (use 1)")
        elif key in _CONFIG_SECTIONS:
            if value is None:
                continue
            if not isinstance(value, Mapping):
                errors.append(f"spec.config.{key} must be a section of settings, got {value!r}")
                continue
            for sub, sub_value in value.items():
                flat[f"{key}.{sub}"] = sub_value
        else:
            flat[key] = value
    for key, value in flat.items():
        if key not in _CONFIG_KEYS:
            errors.append(f"spec.config: unknown setting {key}")
            continue
        choices = _CONFIG_CHOICES.get(key)
        if choices and str(value).strip().lower() not in choices:
            errors.append(f"spec.config: {key} must be one of {', '.join(choices)}, got {value!r}")
    port = flat.get("server.port")
    if port is not None and str(port) != str(HTTP_PORT):
        errors.append(f"spec.config: server.port must be {HTTP_PORT} under the operator "
                      f"(the Service and probes target it), got {port!r}")
    return errors


def validate(spec: Optional[Mapping[str, Any]], name: Optional[str] = None) -> List[str]:
    """Everything wrong with a spec, as readable messages. Empty means renderable."""
    errors: List[str] = []
    if name is not None:
        if not _DNS_LABEL.match(name) or len(name) > _MAX_NAME:
            errors.append(f"metadata.name {name!r} must be a DNS label of at most "
                          f"{_MAX_NAME} characters (it prefixes child resource names)")
    s = normalize(spec)

    replicas = s["replicas"]
    if isinstance(replicas, bool) or not isinstance(replicas, int):
        errors.append(f"spec.replicas must be an integer, got {replicas!r}")
        replicas = None
    elif replicas < 1:
        errors.append(f"spec.replicas must be at least 1, got {replicas}")

    if not isinstance(s["version"], str) or not s["version"].strip():
        errors.append("spec.version must be a non-empty image tag")
    elif not re.match(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$", s["version"]):
        errors.append(f"spec.version {s['version']!r} is not a valid image tag")
    elif len(s["version"]) > 63:
        errors.append("spec.version is longer than 63 characters (it is also a label value)")
    if s["image"]["pullPolicy"] not in _PULL_POLICIES:
        errors.append(f"spec.image.pullPolicy must be one of {', '.join(_PULL_POLICIES)}")

    if s["persistence"]["enabled"] and (replicas is None or replicas > 1):
        errors.append("spec.persistence.enabled needs spec.replicas 1: a ReadWriteOnce volume "
                      "cannot be shared between replicas (use state.redis for shared limits)")
    if s["state"]["redis"] and not s["secretRef"]:
        errors.append("spec.state.redis needs spec.secretRef with a ZNYX_REDIS_URL key")
    if s["policy"] and not isinstance(s["policy"], str):
        errors.append("spec.policy must be the text of a policies.yaml file")
    elif s["policy"]:
        try:
            parsed = yaml.safe_load(s["policy"])
        except yaml.YAMLError as exc:
            errors.append(f"spec.policy is not valid YAML: {exc}")
        else:
            if not isinstance(parsed, dict):
                errors.append("spec.policy must be a YAML mapping")

    errors.extend(_check_config(s["config"]))
    if isinstance(s["config"], Mapping):
        heartbeat = (s["config"].get("telemetry") or {}).get("heartbeat") \
            if isinstance(s["config"].get("telemetry"), Mapping) else None
        if s["airgapped"] and heartbeat is True:
            errors.append("spec.airgapped and config telemetry.heartbeat contradict each other")
        if s["ml"]["enabled"] and isinstance(s["config"].get("ml"), Mapping) \
                and s["config"]["ml"].get("url"):
            errors.append("spec.ml.enabled and config ml.url are mutually exclusive")
    return errors


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def generate_api_key() -> str:
    """A new runtime API key, in the format `znyx serve` generates."""
    return "zx_" + secrets.token_urlsafe(32)


def _deep_merge(base: Dict[str, Any], over: Mapping[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _config_yaml(config: Mapping[str, Any]) -> str:
    return yaml.safe_dump(_deep_merge(_DEFAULT_CONFIG, config), sort_keys=True,
                          default_flow_style=False)


def selector_labels(name: str) -> Dict[str, str]:
    return {"app.kubernetes.io/name": "znyx", "app.kubernetes.io/instance": name}


def labels_for(name: str, version: str) -> Dict[str, str]:
    return {**selector_labels(name),
            "app.kubernetes.io/version": version,
            "app.kubernetes.io/component": "gateway",
            "app.kubernetes.io/managed-by": "znyx-operator"}


def config_checksum(data: Mapping[str, str]) -> str:
    """sha256 of the rendered ConfigMap data, stamped on the pods so a config change rolls them."""
    return hashlib.sha256(json.dumps(dict(data), sort_keys=True).encode("utf-8")).hexdigest()


def image_ref(s: Mapping[str, Any]) -> str:
    suffix = "-ml" if s["ml"]["enabled"] else ""
    return f"{s['image']['repository']}:{s['version']}{suffix}"


def _env(s: Mapping[str, Any], secret_name: str) -> List[Dict[str, Any]]:
    env: List[Dict[str, Any]] = []
    if s["policy"]:
        env.append({"name": "ZNYX_POLICY_PATH", "value": "/etc/znyx/policies.yaml"})
    if s["ml"]["enabled"]:
        env.append({"name": "ZNYX_ML", "value": "true"})
    if s["airgapped"]:
        env.append({"name": "ZNYX_AIRGAPPED", "value": "true"})
    if s["state"]["redis"]:
        env.append({"name": "ZNYX_STATE_BACKEND", "value": "redis"})

    def from_secret(key: str, optional: bool) -> Dict[str, Any]:
        ref: Dict[str, Any] = {"name": secret_name, "key": key}
        if optional:
            ref["optional"] = True
        return {"name": key, "valueFrom": {"secretKeyRef": ref}}

    # The API key is required (every replica must share one); with a user-supplied
    # secret that is the user's contract, so a missing key shows up as a pod error
    # instead of each replica quietly generating its own.
    env.append(from_secret(SECRET_API_KEY, False))
    env.append(from_secret(SECRET_TOKEN, True))
    env.append(from_secret(SECRET_REDIS, True))
    return env


def render(spec: Optional[Mapping[str, Any]], name: str, namespace: str, uid: str = "",
           *, existing_api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """The manifests for one ZnyxInstance, each carrying an ownerReference to it.

    ``existing_api_key`` is the RUNTIME_API_KEY already stored in ``<name>-secrets``, if
    any: it is reused so reconciling never rotates the key. A new one is generated only
    when there is none and the spec has no ``secretRef``.

    Raises ValueError listing every validation error.
    """
    errors = validate(spec, name)
    if errors:
        raise ValueError("; ".join(errors))
    s = normalize(spec)
    version = s["version"]
    labels = labels_for(name, version)
    selector = selector_labels(name)
    owner = [{
        "apiVersion": API_VERSION, "kind": KIND, "name": name, "uid": uid,
        "controller": True, "blockOwnerDeletion": True,
    }]

    def meta(obj_name: str) -> Dict[str, Any]:
        return {"name": obj_name, "namespace": namespace, "labels": dict(labels),
                "ownerReferences": copy.deepcopy(owner)}

    out: List[Dict[str, Any]] = []
    ml = s["ml"]["enabled"]
    replicas = s["replicas"]

    # -- Secret (only when the user did not bring one) ---------------------------------
    user_secret = bool(s["secretRef"])
    secret_name = s["secretRef"] if user_secret else f"{name}-secrets"
    if not user_secret:
        key = existing_api_key or generate_api_key()
        out.append({
            "apiVersion": "v1", "kind": "Secret", "metadata": meta(secret_name),
            "type": "Opaque",
            "data": {SECRET_API_KEY: base64.b64encode(key.encode("utf-8")).decode("ascii")},
        })

    # -- ConfigMap ---------------------------------------------------------------------
    config_data = {"znyx.yaml": _config_yaml(s["config"])}
    if s["policy"]:
        config_data["policies.yaml"] = s["policy"]
    out.append({"apiVersion": "v1", "kind": "ConfigMap", "metadata": meta(f"{name}-config"),
                "data": config_data})

    # -- PersistentVolumeClaim ---------------------------------------------------------
    persistence = s["persistence"]
    if persistence["enabled"]:
        pvc_spec: Dict[str, Any] = {
            "accessModes": ["ReadWriteOnce"],
            "resources": {"requests": {"storage": persistence["size"]}},
        }
        if persistence["storageClassName"]:
            pvc_spec["storageClassName"] = persistence["storageClassName"]
        out.append({"apiVersion": "v1", "kind": "PersistentVolumeClaim",
                    "metadata": meta(f"{name}-data"), "spec": pvc_spec})

    # -- Deployment --------------------------------------------------------------------
    volumes: List[Dict[str, Any]] = [
        {"name": "config", "configMap": {"name": f"{name}-config"}},
        {"name": "data", "persistentVolumeClaim": {"claimName": f"{name}-data"}}
        if persistence["enabled"] else {"name": "data", "emptyDir": {}},
        {"name": "tmp", "emptyDir": {}},
    ]
    mounts: List[Dict[str, Any]] = [
        {"name": "config", "mountPath": "/etc/znyx", "readOnly": True},
        {"name": "data", "mountPath": "/data"},
        {"name": "tmp", "mountPath": "/tmp"},
    ]
    if ml:
        if s["ml"]["modelsClaim"]:
            volumes.append({"name": "models",
                            "persistentVolumeClaim": {"claimName": s["ml"]["modelsClaim"]}})
        else:
            empty: Dict[str, Any] = {}
            if s["ml"]["modelsSizeLimit"]:
                empty["sizeLimit"] = s["ml"]["modelsSizeLimit"]
            volumes.append({"name": "models", "emptyDir": empty})
        mounts.append({"name": "models", "mountPath": "/models"})

    resources = s["resources"] or copy.deepcopy(_DEFAULT_RESOURCES_ML if ml else _DEFAULT_RESOURCES)
    container_security = {
        "runAsNonRoot": True, "runAsUser": 1000, "runAsGroup": 1000,
        "readOnlyRootFilesystem": True, "allowPrivilegeEscalation": False,
        "capabilities": {"drop": ["ALL"]},
        "seccompProfile": {"type": "RuntimeDefault"},
    }
    container = {
        "name": "znyx",
        "image": image_ref(s),
        "imagePullPolicy": s["image"]["pullPolicy"],
        "args": ["znyx", "serve", "--config", "/etc/znyx/znyx.yaml"],
        "env": _env(s, secret_name),
        "ports": [{"name": "http", "containerPort": HTTP_PORT, "protocol": "TCP"}],
        "readinessProbe": {"httpGet": {"path": "/readyz", "port": "http"}},
        "livenessProbe": {"httpGet": {"path": "/healthz", "port": "http"}},
        "startupProbe": {"httpGet": {"path": "/healthz", "port": "http"},
                         "failureThreshold": 30, "periodSeconds": 2},
        "securityContext": container_security,
        "resources": resources,
        "volumeMounts": mounts,
    }
    pod_spec: Dict[str, Any] = {
        "containers": [container],
        "volumes": volumes,
        "terminationGracePeriodSeconds": 40,
        # The gateway never talks to the Kubernetes API: do not hand its pods a token.
        "automountServiceAccountToken": False,
        "securityContext": {
            "runAsNonRoot": True, "runAsUser": 1000, "runAsGroup": 1000, "fsGroup": 1000,
            "seccompProfile": {"type": "RuntimeDefault"},
        },
    }
    if s["image"]["pullSecrets"]:
        pod_spec["imagePullSecrets"] = [
            {"name": p["name"] if isinstance(p, Mapping) else str(p)}
            for p in s["image"]["pullSecrets"]]
    deployment_spec: Dict[str, Any] = {
        "replicas": replicas,
        "selector": {"matchLabels": dict(selector)},
        "template": {
            "metadata": {
                "labels": dict(labels),
                "annotations": {"checksum/config": config_checksum(config_data)},
            },
            "spec": pod_spec,
        },
    }
    if persistence["enabled"]:
        # One ReadWriteOnce volume: the old pod must release it before the new one starts.
        deployment_spec["strategy"] = {"type": "Recreate"}
    out.append({"apiVersion": "apps/v1", "kind": "Deployment", "metadata": meta(name),
                "spec": deployment_spec})

    # -- Service -----------------------------------------------------------------------
    out.append({
        "apiVersion": "v1", "kind": "Service", "metadata": meta(name),
        "spec": {"type": "ClusterIP", "selector": dict(selector),
                 "ports": [{"name": "http", "port": SERVICE_PORT, "targetPort": "http",
                            "protocol": "TCP"}]},
    })

    # -- PodDisruptionBudget -----------------------------------------------------------
    if replicas > 1:
        out.append({
            "apiVersion": "policy/v1", "kind": "PodDisruptionBudget", "metadata": meta(name),
            "spec": {"minAvailable": 1, "selector": {"matchLabels": dict(selector)}},
        })

    # -- NetworkPolicy -----------------------------------------------------------------
    net = s["networkPolicy"]
    if net["enabled"] or s["airgapped"]:
        egress: List[Dict[str, Any]] = [
            {"ports": [{"protocol": "UDP", "port": 53}, {"protocol": "TCP", "port": 53}]},
        ]
        if not s["airgapped"]:
            egress.append({"ports": [{"protocol": "TCP", "port": 443}]})
        if net["egressTo"]:
            egress.append({"to": copy.deepcopy(net["egressTo"])})
        out.append({
            "apiVersion": "networking.k8s.io/v1", "kind": "NetworkPolicy",
            "metadata": meta(name),
            "spec": {
                "podSelector": {"matchLabels": dict(selector)},
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [{
                    "from": copy.deepcopy(net["ingressFrom"]) or [{"podSelector": {}}],
                    "ports": [{"protocol": "TCP", "port": HTTP_PORT}],
                }],
                "egress": egress,
            },
        })
    return out


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def _condition(kind: str, ok: bool, reason: str, message: str, now: str,
               previous: Optional[List[Mapping[str, Any]]]) -> Dict[str, Any]:
    status = "True" if ok else "False"
    since = now
    for old in previous or []:
        if old.get("type") == kind and old.get("status") == status:
            since = old.get("lastTransitionTime") or now
    return {"type": kind, "status": status, "reason": reason, "message": message,
            "lastTransitionTime": since}


def build_status(*, generation: int, version: str, replicas: int,
                 deployment: Optional[Mapping[str, Any]], errors: List[str], now: str,
                 previous_conditions: Optional[List[Mapping[str, Any]]] = None) -> Dict[str, Any]:
    """The ZnyxInstance status from the Deployment's state (or the validation errors).

    ``deployment`` is the Deployment as a dict (metadata.generation, status.*), or None
    when it does not exist yet. ``lastTransitionTime`` only moves when a condition flips.
    """
    if errors:
        valid = _condition("SpecValid", False, "InvalidSpec", "; ".join(errors), now,
                           previous_conditions)
        ready = _condition("Ready", False, "InvalidSpec", "the spec is invalid; nothing was "
                           "applied", now, previous_conditions)
        return {"ready": False, "readyReplicas": 0, "observedGeneration": generation,
                "version": version, "replicas": replicas, "conditions": [ready, valid]}
    valid = _condition("SpecValid", True, "Valid", "spec accepted", now, previous_conditions)
    if deployment is None:
        ready = _condition("Ready", False, "DeploymentMissing", "the Deployment does not exist "
                           "yet", now, previous_conditions)
        return {"ready": False, "readyReplicas": 0, "observedGeneration": generation,
                "version": version, "replicas": replicas, "conditions": [ready, valid]}
    dstatus = deployment.get("status") or {}
    ready_replicas = int(dstatus.get("readyReplicas") or 0)
    updated = int(dstatus.get("updatedReplicas") or 0)
    observed = int(dstatus.get("observedGeneration") or 0)
    current = int((deployment.get("metadata") or {}).get("generation") or 0)
    is_ready = observed >= current and updated >= replicas and ready_replicas >= replicas
    if is_ready:
        ready = _condition("Ready", True, "Available",
                           f"{ready_replicas}/{replicas} replicas ready", now, previous_conditions)
    else:
        ready = _condition("Ready", False, "Progressing",
                           f"{ready_replicas}/{replicas} replicas ready, {updated} updated", now,
                           previous_conditions)
    return {"ready": is_ready, "readyReplicas": ready_replicas, "observedGeneration": generation,
            "version": version, "replicas": replicas, "conditions": [ready, valid]}
