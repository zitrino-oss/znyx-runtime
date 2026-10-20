# Znyx operator

A small Kubernetes operator for the single-container Znyx install. You declare a
`ZnyxInstance`; the operator renders and keeps applied the resources that make up a
gateway:

- a `Deployment` of the one `znyx` container (`znyx serve`, port 8080),
- a `ConfigMap` with `znyx.yaml` (and `policies.yaml` when you give a policy),
- a `Service` (ClusterIP, 80 to the `http` port),
- a `Secret` with a generated API key (unless you bring your own),
- a `PodDisruptionBudget` (when `replicas` is above 1),
- a `NetworkPolicy` (when enabled, or always when air-gapped),
- a `PersistentVolumeClaim` (when `persistence.enabled`).

Every child carries an ownerReference to the `ZnyxInstance`, so deleting the instance
removes all of them (including the data volume, if you enabled persistence).

## Operator or Helm chart

Both render the same resources (`znyx_operator/render.py` is the reference; `deploy/helm`
and `deploy/terraform` follow it). The operator adds what a chart cannot:

- **Continuous reconciliation.** Every 60 seconds the operator re-applies each instance
  (server-side apply, field manager `znyx-operator`, conflicts forced). A hand edit to a
  field it owns is reverted, and a deleted child is recreated. Helm only acts on
  `helm upgrade`.
- **Config changes roll the pods.** The pod template carries `checksum/config`, the sha256
  of the rendered ConfigMap data, so editing `spec.config` or `spec.policy` restarts the
  gateway.
- **Status.** `kubectl get znyx` shows readiness; `.status.conditions` says why not.
- **Validation with a readable error.** An invalid spec is not applied; the reason is in
  `status.conditions` (`SpecValid`) and an event.

Use the chart if you manage releases from CI and want no controller running in the
cluster. Use the operator if a platform team wants to offer `ZnyxInstance` as an API to
other teams.

## Install

```sh
# 1. build and push the operator image (Python 3.12, non-root, uid 1000)
docker build -t znyx/znyx-operator:0.1.0 deploy/operator

# 2. install the CRD, RBAC and the operator
kubectl apply -f deploy/operator/crd.yaml
kubectl apply -f deploy/operator/rbac.yaml
kubectl apply -f deploy/operator/operator.yaml

# 3. create an instance
kubectl apply -f deploy/operator/examples/basic.yaml
kubectl get znyx
```

The generated API key of an instance named `znyx-basic`:

```sh
kubectl get secret znyx-basic-secrets -o jsonpath='{.data.RUNTIME_API_KEY}' | base64 -d
```

Examples: `examples/basic.yaml`, `examples/ml.yaml` (ML image, weights on a claim, your own
Secret), `examples/airgapped.yaml`.

To watch only some namespaces set `ZNYX_OPERATOR_NAMESPACES=team-a,team-b` on the operator
Deployment (empty, the default, watches all). The shipped RBAC is a ClusterRole; for a
namespaced install replace it with a Role and RoleBinding per watched namespace (keep the
cluster-level `customresourcedefinitions` and `namespaces` list/watch rules that kopf needs).

## ZnyxInstance reference (`znyx.ai/v1alpha1`)

| Field | Default | Meaning |
|---|---|---|
| `spec.version` | `1.0.0` | Image tag. Also the `app.kubernetes.io/version` label. |
| `spec.image.repository` | `znyx/znyx` | Image repository. |
| `spec.image.pullPolicy` | `IfNotPresent` | `Always`, `IfNotPresent` or `Never`. |
| `spec.image.pullSecrets` | none | Names of image pull Secrets. |
| `spec.replicas` | `2` | At least 1. |
| `spec.ml.enabled` | `false` | Use the ML image (`<repository>:<version>-ml`); sets `ZNYX_ML=true`; larger default resources. |
| `spec.ml.modelsClaim` | none | Existing PVC mounted at `/models`. Without it `/models` is an emptyDir. |
| `spec.ml.modelsSizeLimit` | none | Size limit of the `/models` emptyDir. |
| `spec.airgapped` | `false` | Sets `ZNYX_AIRGAPPED=true` and forces a NetworkPolicy with DNS egress only (no 443). |
| `spec.config` | `{}` | The `znyx.yaml` content, deep-merged over `{version: 1, server: {host: 0.0.0.0, port: 8080, workers: auto}, mode: local, dashboard: {enabled: true}}`. |
| `spec.policy` | none | The `policies.yaml` text. Adds the ConfigMap key and `ZNYX_POLICY_PATH=/etc/znyx/policies.yaml`. Without it the image's built-in policy is used. |
| `spec.secretRef` | none | Secret with `RUNTIME_API_KEY` (required) and optionally `ZNYX_RUNTIME_TOKEN`, `ZNYX_REDIS_URL`. Without it the operator creates `<name>-secrets`. |
| `spec.state.redis` | `false` | Sets `ZNYX_STATE_BACKEND=redis`. Needs `secretRef` with `ZNYX_REDIS_URL`. |
| `spec.resources` | see below | Container resources, replacing the defaults entirely. |
| `spec.networkPolicy.enabled` | `false` | Render a NetworkPolicy (always rendered when air-gapped). |
| `spec.networkPolicy.ingressFrom` | own namespace | NetworkPolicyPeers allowed to reach port 8080. |
| `spec.networkPolicy.egressTo` | none | Extra NetworkPolicyPeers reachable on any port (for example an in-cluster Redis). DNS (53) is always open; 443 to anywhere is open unless air-gapped. |
| `spec.persistence.enabled` | `false` | Keep `/data` on the PVC `<name>-data`. Needs `replicas: 1`. |
| `spec.persistence.size` | `1Gi` | PVC size. |
| `spec.persistence.storageClassName` | cluster default | PVC storage class. |

Default resources: requests 1 CPU / 512Mi, limits 2 CPU / 1Gi. With `ml.enabled`:
requests 2 / 2Gi, limits 4 / 4Gi.

The pods run as uid/gid 1000, non-root, read-only root filesystem, no privilege escalation,
all capabilities dropped, seccomp `RuntimeDefault`, with writable `/data` and `/tmp`. The
service account token is not mounted (the gateway never calls the Kubernetes API).

Status: `ready`, `replicas`, `readyReplicas`, `observedGeneration`, `version` and the
conditions `Ready` and `SpecValid`. Readiness is refreshed every 15 seconds from the
Deployment.

## Validation

A spec that fails validation is not applied (existing children are left as they are) and
is reported in the status. Checked: `replicas` is an integer of at least 1;
`persistence.enabled` with more than one replica (a ReadWriteOnce volume cannot be
shared; use `state.redis` for shared limits instead); `state.redis` without `secretRef`;
unknown `znyx.yaml` keys and invalid choices such as an unknown `mode`; `server.port` other
than 8080; a `policy` that is not a YAML mapping; air-gapped together with
`telemetry.heartbeat`; an invalid image tag or pull policy; an instance name that is not a
DNS label of at most 52 characters.

## How it runs

Handlers are in `znyx_operator/handlers.py` ([kopf](https://kopf.readthedocs.io), pinned in
`requirements.txt`). `main.py` starts kopf in `--standalone` mode: no peering objects, no
leader election, so **run exactly one operator replica** (the Deployment uses
`strategy: Recreate`). Rendering is pure and lives in `znyx_operator/render.py`; its tests
are `tests/test_operator_render.py` and need no cluster.

## Limitations

- Single operator replica, no leader election (standalone mode).
- The `znyx.yaml` key list is a copy of `znyx.config.SETTINGS`, checked by a test; a new
  setting needs adding to `render.py` first or validation rejects it.
- A secret the operator created is only generated once. If you delete `<name>-secrets`, a
  new key is generated on the next reconcile and the pods pick it up on their next restart.
- The operator does not check that the Secret named by `secretRef` exists; pods fail with
  `CreateContainerConfigError` until it does (visible in `Ready: False`, `Progressing`).
- Rendering sets `spec.replicas`, so an HPA or `kubectl scale` on the Deployment is
  reverted. Scale through `spec.replicas`.
- Deleting a `ZnyxInstance` deletes its PVC (ownerReference garbage collection).
- With `persistence.enabled` the Deployment uses `strategy: Recreate`, so a rollout has
  brief downtime.
- Only a PodDisruptionBudget or NetworkPolicy that the spec stops asking for is deleted;
  a Secret or PVC left behind by toggling `secretRef` or `persistence` is not.
- v1alpha1: the schema may change.
