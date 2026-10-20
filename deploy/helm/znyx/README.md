# Znyx Helm chart

Deploys the single-container Znyx install on Kubernetes: one image, one port (8080), no
sidecar. The container serves every API and the local dashboard, runs worker processes
sized to its CPU limit, and with `ml.enabled` also starts the ML model server inside the
same container on loopback (never exposed, holds no credential).

The chart renders a ConfigMap, a Secret, a Deployment, a Service and a PodDisruptionBudget by
default, plus an HPA, ServiceMonitor, Ingress, NetworkPolicy and a PVC when you ask for them.
The Kubernetes operator in `deploy/operator` renders the same resources; the plain manifests
in `deploy/k8s` are this chart's defaults for a release named `znyx`.

Requirements: Kubernetes 1.25 or newer, Helm 3.

## Install

```bash
helm install znyx ./deploy/helm/znyx --namespace znyx --create-namespace
```

With your own policy and a fixed API key:

```bash
helm install znyx ./deploy/helm/znyx -n znyx --create-namespace \
  --set-file policy=./policies.yaml \
  --set secrets.apiKey="$(openssl rand -base64 32 | tr -d '=+/')"
```

If you set neither `secrets.apiKey` nor `existingSecret`, the chart generates a random key on
the first install and keeps it across upgrades (it reads the existing Secret). Read it with:

```bash
kubectl -n znyx get secret znyx-secrets -o jsonpath='{.data.RUNTIME_API_KEY}' | base64 -d; echo
```

Then try it:

```bash
kubectl -n znyx port-forward svc/znyx 8080:80
curl -s localhost:8080/healthz          # always 200
open http://localhost:8080/dashboard
helm test znyx -n znyx                  # curls /healthz from inside the cluster
```

In-cluster address for your apps and SDKs: `http://znyx.znyx.svc.cluster.local`.

Pods become ready (`/readyz`) once a policy is loaded. With no `policy` value the image's
built-in starter policy is used, which is fine for a first look and not for production.

> `helm template`, Argo CD and other offline renderers cannot look up the existing Secret, so a
> generated key would change on every render. In GitOps, set `secrets.apiKey` or `existingSecret`.

## How configuration reaches the container

| Resource | What it holds |
|---|---|
| ConfigMap `<name>-config` | `znyx.yaml` from `.Values.config`, and `policies.yaml` from `.Values.policy` when set. Mounted read-only at `/etc/znyx`. |
| Secret `<name>-secrets` (or `existingSecret`) | `RUNTIME_API_KEY`, `ZNYX_RUNTIME_TOKEN`, `ZNYX_REDIS_URL` (only non-empty keys). Injected as environment variables. |

The container runs `znyx serve --config /etc/znyx/znyx.yaml`. Precedence is flags, then
environment variables, then the file, then the image's own defaults (which only make it
listen on 0.0.0.0 and point at the built-in policy). The chart sets a few variables that
therefore beat the file, so use the chart values for these, not `config`:

| Variable | Set from |
|---|---|
| `ZNYX_POLICY_PATH` | `policy` (only when set) |
| `ZNYX_ML` | `ml.enabled` |
| `ZNYX_AIRGAPPED` | `airgapped` |
| `ZNYX_STATE_BACKEND=redis` | `secrets.redisUrl` (or `state.redis.enabled`) |
| `ZNYX_METRICS_REQUIRE_KEY` | `metrics.requireKey` |
| `ZNYX_RUN_DIR`, `ZNYX_API_KEY_FILE`, spool paths | the image, all under `/data` |

A change to the ConfigMap changes the `checksum/config` pod annotation and rolls the pods.
Secret changes do not: run `kubectl rollout restart deployment/<name>` (or use a reloader).

## Values

Every value is documented in [`values.yaml`](values.yaml) and validated by
[`values.schema.json`](values.schema.json). The ones you will touch most:

| Value | Default | Description |
|---|---|---|
| `image.registry` | `""` | Registry prefix, for a private mirror. |
| `image.repository` | `znyx/znyx` | Repository path. |
| `image.tag` | `""` (chart `appVersion`) | Image tag. `-ml` is appended when `ml.enabled`. |
| `imagePullSecrets` | `[]` | Pull secrets for a private registry. |
| `replicaCount` | `2` | Pods (ignored with `autoscaling.enabled`). |
| `config` | see file | `znyx.yaml` as a map. Defaults: `version: 1`, `server: {host: 0.0.0.0, port: 8080, workers: auto}`, `mode: local`, `dashboard: {enabled: true}`. `mode` is `local` or `managed`. |
| `policy` | `""` | Policy file contents (local mode). Empty: the image's built-in policy. |
| `existingSecret` | `""` | Use your own Secret instead of the chart-created one. |
| `secrets.apiKey` | `""` | Client API key. Empty: generated once and kept. |
| `secrets.runtimeToken` | `""` | Console runtime token (managed mode). |
| `secrets.redisUrl` | `""` | Redis URL; shares limits and budgets across replicas. |
| `state.redis.enabled` | `false` | Declare that `existingSecret` holds `ZNYX_REDIS_URL`. |
| `airgapped` | `false` | No outbound calls; forces the NetworkPolicy on. |
| `ml.enabled` | `false` | Use the `-ml` image and run the model server in the pod. |
| `ml.models.existingClaim` | `""` | PVC mounted at `/models` (pre-fetched weights). |
| `ml.models.readOnly` | `false` | Mount that claim read-only. |
| `ml.models.sizeLimit` | `25Gi` | Size limit of the per-pod emptyDir used without a claim. |
| `persistence.enabled` | `false` | PVC for `/data` instead of an emptyDir. |
| `metrics.requireKey` | `false` | Require the API key on `/metrics`. |
| `resources` | `{}` (mode defaults) | Rules only: 1 CPU / 512Mi requests, 2 CPU / 1Gi limits. ML: 2 / 2Gi and 4 / 4Gi. |
| `autoscaling.*` | disabled | HPA on CPU. |
| `podDisruptionBudget.*` | enabled | `minAvailable: 1`, rendered when replicas > 1. |
| `topologySpread.*` | enabled, soft | Spread across nodes (`ScheduleAnyway`). |
| `service.*` | ClusterIP, 80 | Service type and port (targets the `http` container port). |
| `ingress.*` | disabled | Standard Ingress. |
| `serviceMonitor.*` | disabled | Prometheus Operator scrape of `/metrics`. |
| `networkPolicy.*` | disabled | See below. |
| `extraEnv`, `extraVolumes`, `extraVolumeMounts` | empty | Passthrough. |

Resources: worker processes are sized from the CPU limit (`workers: auto` is one per CPU, up
to four, about 140 MB each). If you raise the CPU limit, raise the memory limit with it.

Security: the pod runs as uid 1000 with a read-only root filesystem, no capabilities, no
privilege escalation, the `RuntimeDefault` seccomp profile and no service account token. It
writes only to `/data` and `/tmp` (emptyDirs).

## Network policy

Set `networkPolicy.enabled=true` (always on when `airgapped=true`).

- Ingress: only to the `http` port, only from `networkPolicy.ingressFrom` (default: pods in the
  same namespace). Add your ingress controller's and Prometheus's namespaces there.
- Egress: DNS to kube-dns (UDP and TCP 53) is always allowed. Not air-gapped: TCP 443 to
  anywhere (console, model downloads, external judges), or only to `networkPolicy.egressTo`
  when set. Air-gapped: DNS, `egressTo` on `egressPorts`, and `extraEgress`, nothing else.
- The ML model server runs inside the pod on loopback, so it needs no egress rule and no
  ingress rule.

## Multiple replicas: Redis

Without Redis each replica enforces its own rate limits and budgets, so the effective limit
is the replica count times what you configured. The install notes warn about this. Point all
replicas at one Redis:

```bash
helm upgrade --install znyx ./deploy/helm/znyx -n znyx --reuse-values \
  --set secrets.redisUrl=redis://redis.data.svc.cluster.local:6379/0
```

This sets `ZNYX_STATE_BACKEND=redis` and `ZNYX_REDIS_URL`. With your own Secret, add a
`ZNYX_REDIS_URL` key to it and set `state.redis.enabled=true`. If Redis is unreachable, limits
fall back to per replica (`config.state.fail_mode: local`) or requests fail closed (`closed`).

## Managed mode (console-managed policy)

```bash
helm install znyx ./deploy/helm/znyx -n znyx --create-namespace \
  --set config.mode=managed \
  --set config.control_plane.url=https://console.example.com \
  --set config.control_plane.require_signed_bundles=true \
  --set secrets.runtimeToken="$RUNTIME_TOKEN"
```

The chart refuses to render managed mode without the URL and a token. `policy` is ignored in
managed mode. Add `config.control_plane.bundle_public_key` to verify signed bundles. In an
air-gapped cluster, allow the in-cluster console with `networkPolicy.egressTo`.

## Air-gapped install

No outbound calls at all: no model downloads, no heartbeat, no external judge calls.

1. Mirror the images into your registry. The tag for `ml.enabled` has the `-ml` suffix:

   ```bash
   docker pull znyx/znyx:1.0.0-ml
   docker tag  znyx/znyx:1.0.0-ml registry.internal.example.com:5000/znyx/znyx:1.0.0-ml
   docker push registry.internal.example.com:5000/znyx/znyx:1.0.0-ml
   # for `helm test`:
   #   mirror curlimages/curl:8.11.1 too, or set test.image.repository
   ```

2. Fetch the sha256-pinned model weights on a connected machine (see
   `packages/znyx-inference/MODELS.md`) and copy them onto a PersistentVolume. Create a
   PVC, for example `znyx-models`, holding that tree (ReadOnlyMany or ReadWriteMany if you run
   several replicas).

3. Install:

   ```bash
   kubectl create namespace znyx
   kubectl -n znyx create secret docker-registry regcred \
     --docker-server=registry.internal.example.com:5000 --docker-username=... --docker-password=...

   helm install znyx ./deploy/helm/znyx -n znyx \
     --set airgapped=true \
     --set image.registry=registry.internal.example.com:5000 \
     --set 'imagePullSecrets[0].name=regcred' \
     --set ml.enabled=true \
     --set ml.models.existingClaim=znyx-models \
     --set ml.models.readOnly=true \
     --set-file policy=./policies.yaml
   ```

`image.registry` is prepended to `image.repository`; if your mirror uses a different path,
set `image.repository` to the full path and leave `registry` empty. With `airgapped=true`
the NetworkPolicy allows DNS only; add your in-cluster Redis or console with
`networkPolicy.egressTo` / `networkPolicy.extraEgress`. See `ci/airgapped-values.yaml`.

## Upgrading from the old `znyx/runtime` + inference sidecar layout

The old manifests ran two containers in the pod (`runtime` on 8080, `inference` on 9000) from
the `znyx/runtime` and `znyx/inference` images, with the sidecar reached on
`http://localhost:9000`. The new layout is one container from `znyx/znyx` (add `ml.enabled`
for the model server), still on port 8080, with the model server on loopback inside it.

| Old | New |
|---|---|
| `znyx/runtime` image | `znyx/znyx` |
| `znyx/runtime` + `znyx/inference` sidecar | `znyx/znyx` with `ml.enabled=true` (`-ml` tag) |
| `ZNYX_MODE`, `ZNYX_POLICY_PATH`, `ZNYX_CONTROL_PLANE_URL`, ... env vars | the same variables still work (`extraEnv`), but prefer `config` |
| Secret `znyx-runtime-secrets` with key `RUNTIME_API_KEY` | `existingSecret=znyx-runtime-secrets`, or `secrets.apiKey` |
| `ZNYX_INFERENCE_URL` for an external model server | `config.ml.url` (and leave `ml.enabled=false`) |
| Deployment and Service `znyx-runtime` | `znyx` (override with `fullnameOverride=znyx-runtime` to keep the name) |
| readiness on `/healthz` | readiness on `/readyz`, liveness and startup on `/healthz` |

Steps:

1. Keep your API key: `--set existingSecret=znyx-runtime-secrets` (the Secret needs the
   `RUNTIME_API_KEY` key, as before), or pass the key as `secrets.apiKey`.
2. Keep the service address your apps use: `--set fullnameOverride=znyx-runtime`, or update the
   clients to `http://znyx.<namespace>.svc.cluster.local`.
3. Delete the old Deployment first if it has the same name (`kubectl delete deploy znyx-runtime`),
   since Helm will not adopt resources it did not create, or install under a new name and move
   traffic across.
4. If you ran the sidecar, add `--set ml.enabled=true`. Weights are fetched into `/models` as
   before; the model server no longer has a pod-level address to manage, and the egress gate
   treats it as in-boundary because it is on loopback.

The old images keep working for one deprecation window; this chart only deploys the new one.

## Development

```bash
helm lint deploy/helm/znyx
for f in deploy/helm/znyx/ci/*.yaml; do
  helm lint deploy/helm/znyx -f "$f"
  helm template znyx deploy/helm/znyx -f "$f" | kubeconform -strict -summary -ignore-missing-schemas
done
```

The files in `ci/` cover the defaults, ML, air-gapped, and managed mode with Redis, HPA,
ServiceMonitor, Ingress and persistence.
