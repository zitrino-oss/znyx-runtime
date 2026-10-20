"""Unit tests for the operator's pure renderer (deploy/operator/znyx_operator/render.py).

No cluster and no kopf: render.py imports only the standard library and PyYAML.
"""
import base64
import sys
from pathlib import Path

import pytest
import yaml

_OPERATOR_DIR = Path(__file__).resolve().parents[1] / "deploy" / "operator"
sys.path.insert(0, str(_OPERATOR_DIR))

from znyx_operator import render  # noqa: E402


def objects(spec=None, name="gw", namespace="prod", uid="uid-1", **kw):
    return render.render(spec or {}, name, namespace, uid, **kw)


def by_kind(manifests):
    return {m["kind"]: m for m in manifests}


def pod_spec(manifests):
    return by_kind(manifests)["Deployment"]["spec"]["template"]["spec"]


def container(manifests):
    return pod_spec(manifests)["containers"][0]


def env_of(manifests):
    return {e["name"]: e for e in container(manifests)["env"]}


def test_defaults():
    m = objects()
    kinds = by_kind(m)
    assert [x["kind"] for x in m] == ["Secret", "ConfigMap", "Deployment", "Service",
                                      "PodDisruptionBudget"]
    dep = kinds["Deployment"]
    assert dep["metadata"]["name"] == "gw"
    assert dep["spec"]["replicas"] == 2
    assert dep["metadata"]["labels"] == {
        "app.kubernetes.io/name": "znyx", "app.kubernetes.io/instance": "gw",
        "app.kubernetes.io/version": "1.0.0", "app.kubernetes.io/component": "gateway",
        "app.kubernetes.io/managed-by": "znyx-operator"}
    assert dep["spec"]["selector"]["matchLabels"] == {
        "app.kubernetes.io/name": "znyx", "app.kubernetes.io/instance": "gw"}
    c = container(m)
    assert c["name"] == "znyx"
    assert c["image"] == "znyx/znyx:1.0.0"
    assert c["args"] == ["znyx", "serve", "--config", "/etc/znyx/znyx.yaml"]
    assert c["ports"] == [{"name": "http", "containerPort": 8080, "protocol": "TCP"}]
    assert c["readinessProbe"]["httpGet"] == {"path": "/readyz", "port": "http"}
    assert c["livenessProbe"]["httpGet"] == {"path": "/healthz", "port": "http"}
    assert c["startupProbe"]["httpGet"] == {"path": "/healthz", "port": "http"}
    assert c["startupProbe"]["failureThreshold"] == 30
    assert c["startupProbe"]["periodSeconds"] == 2
    assert pod_spec(m)["terminationGracePeriodSeconds"] == 40
    assert c["resources"] == {"requests": {"cpu": "1", "memory": "512Mi"},
                              "limits": {"cpu": "2", "memory": "1Gi"}}
    svc = kinds["Service"]
    assert svc["metadata"]["name"] == "gw"
    assert svc["spec"]["type"] == "ClusterIP"
    assert svc["spec"]["ports"][0]["port"] == 80
    assert svc["spec"]["ports"][0]["targetPort"] == "http"
    assert svc["spec"]["selector"] == dep["spec"]["selector"]["matchLabels"]
    assert "NetworkPolicy" not in kinds and "PersistentVolumeClaim" not in kinds
    assert set(env_of(m)) == {"RUNTIME_API_KEY", "ZNYX_RUNTIME_TOKEN", "ZNYX_REDIS_URL"}


def test_configmap_merges_config_over_defaults():
    cm = by_kind(objects({"config": {"server": {"workers": 3}, "log_level": "debug"}}))["ConfigMap"]
    assert cm["metadata"]["name"] == "gw-config"
    cfg = yaml.safe_load(cm["data"]["znyx.yaml"])
    assert cfg == {"version": 1,
                   "server": {"host": "0.0.0.0", "port": 8080, "workers": 3},
                   "mode": "local", "dashboard": {"enabled": True}, "log_level": "debug"}
    assert "policies.yaml" not in cm["data"]


def test_config_volume_and_mounts():
    m = objects()
    vols = {v["name"]: v for v in pod_spec(m)["volumes"]}
    assert vols["config"]["configMap"]["name"] == "gw-config"
    assert vols["data"] == {"name": "data", "emptyDir": {}}
    assert vols["tmp"] == {"name": "tmp", "emptyDir": {}}
    mounts = {x["mountPath"]: x for x in container(m)["volumeMounts"]}
    assert mounts["/etc/znyx"]["readOnly"] is True
    assert set(mounts) == {"/etc/znyx", "/data", "/tmp"}


def test_ml_suffix_resources_and_models_volume():
    m = objects({"ml": {"enabled": True}, "version": "1.2.3",
                 "image": {"repository": "reg.example/znyx"}})
    c = container(m)
    assert c["image"] == "reg.example/znyx:1.2.3-ml"
    assert env_of(m)["ZNYX_ML"]["value"] == "true"
    assert c["resources"] == {"requests": {"cpu": "2", "memory": "2Gi"},
                              "limits": {"cpu": "4", "memory": "4Gi"}}
    vols = {v["name"]: v for v in pod_spec(m)["volumes"]}
    assert vols["models"] == {"name": "models", "emptyDir": {}}
    assert {"name": "models", "mountPath": "/models"} in c["volumeMounts"]
    # the version label keeps the plain tag
    assert by_kind(m)["Deployment"]["metadata"]["labels"]["app.kubernetes.io/version"] == "1.2.3"


def test_ml_models_claim_and_size_limit():
    m = objects({"ml": {"enabled": True, "modelsClaim": "weights"}})
    vols = {v["name"]: v for v in pod_spec(m)["volumes"]}
    assert vols["models"]["persistentVolumeClaim"] == {"claimName": "weights"}
    m = objects({"ml": {"enabled": True, "modelsSizeLimit": "5Gi"}})
    vols = {v["name"]: v for v in pod_spec(m)["volumes"]}
    assert vols["models"]["emptyDir"] == {"sizeLimit": "5Gi"}


def test_no_ml_means_no_models_volume_or_env():
    m = objects()
    assert "ZNYX_ML" not in env_of(m)
    assert "models" not in {v["name"] for v in pod_spec(m)["volumes"]}


def test_custom_resources_override_defaults():
    res = {"requests": {"cpu": "250m"}, "limits": {"cpu": "1"}}
    assert container(objects({"resources": res}))["resources"] == res


def test_airgapped_forces_network_policy_without_443():
    m = objects({"airgapped": True})
    assert env_of(m)["ZNYX_AIRGAPPED"]["value"] == "true"
    np = by_kind(m)["NetworkPolicy"]
    assert np["spec"]["policyTypes"] == ["Ingress", "Egress"]
    ports = [(p["protocol"], p["port"]) for rule in np["spec"]["egress"]
             for p in rule.get("ports", [])]
    assert ("UDP", 53) in ports and ("TCP", 53) in ports
    assert all(port != 443 for _, port in ports)


def test_network_policy_when_enabled_allows_443_and_extras():
    peer = {"podSelector": {"matchLabels": {"app": "redis"}}}
    m = objects({"networkPolicy": {"enabled": True, "egressTo": [peer],
                                   "ingressFrom": [{"namespaceSelector": {}}]}})
    np = by_kind(m)["NetworkPolicy"]
    egress = np["spec"]["egress"]
    ports = [p["port"] for rule in egress for p in rule.get("ports", [])]
    assert 53 in ports and 443 in ports
    assert {"to": [peer]} in egress
    assert np["spec"]["ingress"] == [{"from": [{"namespaceSelector": {}}],
                                      "ports": [{"protocol": "TCP", "port": 8080}]}]


def test_network_policy_default_ingress_is_same_namespace():
    np = by_kind(objects({"networkPolicy": {"enabled": True}}))["NetworkPolicy"]
    assert np["spec"]["ingress"][0]["from"] == [{"podSelector": {}}]
    assert np["spec"]["podSelector"]["matchLabels"]["app.kubernetes.io/instance"] == "gw"


def test_no_network_policy_by_default():
    assert "NetworkPolicy" not in by_kind(objects())


def test_policy_adds_configmap_key_and_env():
    policy = "version: 1\nrules: []\n"
    m = objects({"policy": policy})
    assert by_kind(m)["ConfigMap"]["data"]["policies.yaml"] == policy
    assert env_of(m)["ZNYX_POLICY_PATH"]["value"] == "/etc/znyx/policies.yaml"
    assert "ZNYX_POLICY_PATH" not in env_of(objects())


def test_checksum_changes_with_config_and_is_stable_otherwise():
    def checksum(spec):
        dep = by_kind(objects(spec, existing_api_key="k"))["Deployment"]
        return dep["spec"]["template"]["metadata"]["annotations"]["checksum/config"]

    base = checksum({"config": {"log_level": "info"}})
    assert base == checksum({"config": {"log_level": "info"}})
    assert len(base) == 64
    assert base != checksum({"config": {"log_level": "debug"}})
    assert base != checksum({"config": {"log_level": "info"}, "policy": "version: 1\n"})
    # unrelated spec changes do not roll the pods
    assert base == checksum({"config": {"log_level": "info"}, "replicas": 5})


def test_checksum_matches_configmap_data():
    m = objects({"policy": "version: 1\n"})
    cm = by_kind(m)["ConfigMap"]
    ann = by_kind(m)["Deployment"]["spec"]["template"]["metadata"]["annotations"]
    assert ann["checksum/config"] == render.config_checksum(cm["data"])


def test_secret_created_only_without_secret_ref():
    m = objects()
    secret = by_kind(m)["Secret"]
    assert secret["metadata"]["name"] == "gw-secrets"
    key = base64.b64decode(secret["data"]["RUNTIME_API_KEY"]).decode()
    assert key.startswith("zx_") and len(key) > 20
    ref = env_of(m)["RUNTIME_API_KEY"]["valueFrom"]["secretKeyRef"]
    assert ref == {"name": "gw-secrets", "key": "RUNTIME_API_KEY"}

    m = objects({"secretRef": "mine"})
    assert "Secret" not in by_kind(m)
    for name in ("RUNTIME_API_KEY", "ZNYX_RUNTIME_TOKEN", "ZNYX_REDIS_URL"):
        assert env_of(m)[name]["valueFrom"]["secretKeyRef"]["name"] == "mine"
    assert "optional" not in env_of(m)["RUNTIME_API_KEY"]["valueFrom"]["secretKeyRef"]
    assert env_of(m)["ZNYX_RUNTIME_TOKEN"]["valueFrom"]["secretKeyRef"]["optional"] is True
    assert env_of(m)["ZNYX_REDIS_URL"]["valueFrom"]["secretKeyRef"]["optional"] is True


def test_random_key_is_reused_when_existing_value_passed():
    first = base64.b64decode(by_kind(objects())["Secret"]["data"]["RUNTIME_API_KEY"]).decode()
    second = base64.b64decode(by_kind(objects())["Secret"]["data"]["RUNTIME_API_KEY"]).decode()
    assert first != second  # genuinely random without an existing value
    reused = by_kind(objects(existing_api_key=first))["Secret"]["data"]["RUNTIME_API_KEY"]
    assert base64.b64decode(reused).decode() == first
    # a user-supplied secret ignores any stale existing value
    assert "Secret" not in by_kind(objects({"secretRef": "mine"}, existing_api_key=first))


def test_redis_state_backend_env():
    m = objects({"secretRef": "mine", "state": {"redis": True}})
    assert env_of(m)["ZNYX_STATE_BACKEND"]["value"] == "redis"
    assert "ZNYX_STATE_BACKEND" not in env_of(objects())


def test_pdb_only_with_more_than_one_replica():
    assert "PodDisruptionBudget" not in by_kind(objects({"replicas": 1}))
    pdb = by_kind(objects({"replicas": 2}))["PodDisruptionBudget"]
    assert pdb["spec"]["minAvailable"] == 1
    assert pdb["spec"]["selector"]["matchLabels"]["app.kubernetes.io/instance"] == "gw"


def test_persistence_renders_pvc_and_uses_it():
    m = objects({"replicas": 1, "persistence": {"enabled": True, "size": "5Gi",
                                                "storageClassName": "fast"}})
    pvc = by_kind(m)["PersistentVolumeClaim"]
    assert pvc["metadata"]["name"] == "gw-data"
    assert pvc["spec"]["accessModes"] == ["ReadWriteOnce"]
    assert pvc["spec"]["resources"]["requests"]["storage"] == "5Gi"
    assert pvc["spec"]["storageClassName"] == "fast"
    vols = {v["name"]: v for v in pod_spec(m)["volumes"]}
    assert vols["data"]["persistentVolumeClaim"]["claimName"] == "gw-data"
    assert by_kind(m)["Deployment"]["spec"]["strategy"] == {"type": "Recreate"}


def test_image_pull_settings():
    m = objects({"image": {"pullPolicy": "Always", "pullSecrets": ["regcred"]}})
    assert container(m)["imagePullPolicy"] == "Always"
    assert pod_spec(m)["imagePullSecrets"] == [{"name": "regcred"}]


def test_validation_rejects_persistence_with_multiple_replicas():
    errors = render.validate({"replicas": 2, "persistence": {"enabled": True}})
    assert any("persistence" in e for e in errors)
    # the default replica count (2) is also too many
    assert render.validate({"persistence": {"enabled": True}})
    assert render.validate({"replicas": 1, "persistence": {"enabled": True}}) == []
    with pytest.raises(ValueError, match="persistence"):
        objects({"replicas": 3, "persistence": {"enabled": True}})


@pytest.mark.parametrize("replicas", [0, -1])
def test_validation_rejects_replicas_below_one(replicas):
    errors = render.validate({"replicas": replicas})
    assert any("replicas" in e for e in errors)
    with pytest.raises(ValueError):
        objects({"replicas": replicas})


def test_validation_rejects_non_integer_replicas():
    assert render.validate({"replicas": "2"})
    assert render.validate({"replicas": True})


def test_validation_rejects_unknown_mode_and_keys():
    assert any("mode" in e for e in render.validate({"config": {"mode": "banana"}}))
    assert render.validate({"config": {"mode": "managed"}}) == []
    assert any("fail_mod" in e for e in render.validate({"config": {"fail_mod": "open"}}))
    assert any("server.prot" in e
               for e in render.validate({"config": {"server": {"prot": 1}}}))
    assert render.validate({"config": {"server": {"port": 9090}}})
    assert render.validate({"config": {"version": 2}})


def test_validation_other_rules():
    assert render.validate({"state": {"redis": True}})  # needs secretRef
    assert render.validate({"state": {"redis": True}, "secretRef": "s"}) == []
    assert render.validate({"policy": "a: [unclosed"})
    assert render.validate({"policy": "- just\n- a list\n"})
    assert render.validate({"airgapped": True, "config": {"telemetry": {"heartbeat": True}}})
    assert render.validate({"image": {"pullPolicy": "Sometimes"}})
    assert render.validate({}, name="Bad_Name")
    assert render.validate({}, name="x" * 60)
    assert render.validate({}, name="ok-name") == []


def test_config_keys_match_the_znyx_package():
    """The operator keeps its own copy of the settings list; fail when it drifts."""
    config = pytest.importorskip("znyx.config")
    assert render._CONFIG_KEYS == {s.key for s in config.SETTINGS}


def test_owner_references_on_every_child():
    full = {"replicas": 1, "airgapped": True, "policy": "version: 1\n",
            "persistence": {"enabled": True}}
    for spec in ({}, full, {"replicas": 3, "networkPolicy": {"enabled": True}}):
        m = objects(spec, uid="abc-123")
        assert m
        for obj in m:
            refs = obj["metadata"]["ownerReferences"]
            assert refs == [{"apiVersion": "znyx.ai/v1alpha1", "kind": "ZnyxInstance",
                             "name": "gw", "uid": "abc-123", "controller": True,
                             "blockOwnerDeletion": True}], obj["kind"]
            assert obj["metadata"]["namespace"] == "prod"
            assert obj["metadata"]["labels"]["app.kubernetes.io/managed-by"] == "znyx-operator"


def test_security_context():
    m = objects()
    pod = pod_spec(m)["securityContext"]
    assert pod["runAsNonRoot"] is True
    assert pod["runAsUser"] == 1000 and pod["runAsGroup"] == 1000 and pod["fsGroup"] == 1000
    assert pod["seccompProfile"] == {"type": "RuntimeDefault"}
    sc = container(m)["securityContext"]
    assert sc["readOnlyRootFilesystem"] is True
    assert sc["allowPrivilegeEscalation"] is False
    assert sc["capabilities"] == {"drop": ["ALL"]}
    assert sc["runAsNonRoot"] is True and sc["runAsUser"] == 1000
    assert sc["seccompProfile"] == {"type": "RuntimeDefault"}
    assert pod_spec(m)["automountServiceAccountToken"] is False


def test_render_is_deterministic_for_the_same_input():
    assert objects(existing_api_key="k") == objects(existing_api_key="k")


def test_manifests_are_yaml_serialisable():
    m = objects({"ml": {"enabled": True}, "airgapped": True, "policy": "version: 1\n"})
    assert list(yaml.safe_load_all(yaml.safe_dump_all(m))) == m
    for obj in m:
        assert obj["apiVersion"] and obj["kind"] and obj["metadata"]["name"]


def _dep(ready, updated=2, generation=3, observed=3):
    return {"metadata": {"generation": generation},
            "status": {"readyReplicas": ready, "updatedReplicas": updated,
                       "observedGeneration": observed}}


def test_status_ready_progressing_and_invalid():
    t0, t1 = "2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z"
    st = render.build_status(generation=4, version="1.0.0", replicas=2, deployment=_dep(2),
                             errors=[], now=t0)
    assert st["ready"] is True and st["readyReplicas"] == 2 and st["observedGeneration"] == 4
    cond = {c["type"]: c for c in st["conditions"]}
    assert cond["Ready"]["status"] == "True" and cond["SpecValid"]["status"] == "True"

    # still ready: the transition time does not move
    again = render.build_status(generation=4, version="1.0.0", replicas=2, deployment=_dep(2),
                                errors=[], now=t1, previous_conditions=st["conditions"])
    assert {c["type"]: c for c in again["conditions"]}["Ready"]["lastTransitionTime"] == t0

    rolling = render.build_status(generation=5, version="1.0.1", replicas=2,
                                  deployment=_dep(2, updated=1), errors=[], now=t1,
                                  previous_conditions=st["conditions"])
    assert rolling["ready"] is False
    ready = {c["type"]: c for c in rolling["conditions"]}["Ready"]
    assert ready["reason"] == "Progressing" and ready["lastTransitionTime"] == t1

    stale = render.build_status(generation=5, version="1.0.0", replicas=2,
                                deployment=_dep(2, generation=4, observed=3), errors=[], now=t1)
    assert stale["ready"] is False

    bad = render.build_status(generation=5, version="1.0.0", replicas=0, deployment=None,
                              errors=["spec.replicas must be at least 1, got 0"], now=t1)
    assert bad["ready"] is False
    cond = {c["type"]: c for c in bad["conditions"]}
    assert cond["SpecValid"]["status"] == "False"
    assert "replicas" in cond["SpecValid"]["message"]

    missing = render.build_status(generation=1, version="1.0.0", replicas=2, deployment=None,
                                  errors=[], now=t1)
    assert missing["ready"] is False
