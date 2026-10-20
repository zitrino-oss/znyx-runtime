# Znyx Terraform module

Installs the single-container Znyx gateway on an existing Kubernetes cluster by wrapping the
Helm chart in `deploy/helm/znyx`. It creates, optionally, the namespace and the Secret that
holds the API key, then a `helm_release`.

| Resource | When |
|---|---|
| `kubernetes_namespace_v1` | `create_namespace` is true (default) |
| `random_password` | no `api_key` and no `existing_secret` given |
| `kubernetes_secret_v1` `<release>-secrets` | no `existing_secret` given: `RUNTIME_API_KEY`, plus `ZNYX_RUNTIME_TOKEN` and `ZNYX_REDIS_URL` when set |
| `helm_release` | always |

Credentials live in the Secret rather than in the Helm release values (Helm stores release
values in a cluster Secret), and the chart is given `existingSecret`. A short hash of the
module-created Secret's contents is added as a pod annotation, so changing the key, token or
Redis URL rolls the pods.

Requirements: Terraform 1.5 or newer, providers `hashicorp/helm` 3.x, `hashicorp/kubernetes`
2.30 or newer and `hashicorp/random` 3.5 or newer (validated with helm 3.3, kubernetes 3.4 and
random 3.9). You configure the `helm` and `kubernetes` providers in the root module.

## Example

```hcl
provider "kubernetes" {
  config_path = "~/.kube/config"
}

provider "helm" {
  kubernetes = {
    config_path = "~/.kube/config"
  }
}

module "znyx" {
  source = "github.com/zitrino-oss/znyx-runtime//deploy/terraform/znyx"
  # or, from a checkout: source = "./deploy/terraform/znyx"

  namespace = "znyx"
  policy    = file("${path.module}/policies.yaml")
  replicas  = 2
  redis_url = "redis://redis.data.svc.cluster.local:6379/0"   # share limits across replicas
}

output "znyx_url" {
  value = module.znyx.internal_url      # http://znyx.znyx.svc.cluster.local
}
```

A runnable version is in [`examples/basic`](examples/basic). Read the generated key with
`terraform output -raw api_key`.

By default the module installs the chart from this repository (`<module>/../../helm/znyx`,
which exists with a repository checkout or a git module source). To install a published
chart instead, set `chart_repository` (and `chart_version`), or point `chart_path` at a copy:

```hcl
module "znyx" {
  source = "./deploy/terraform/znyx"

  chart_repository = "oci://registry.internal.example.com/charts"
  chart_version    = "0.1.0"
}
```

## Other setups

Managed mode (policies from the console):

```hcl
module "znyx" {
  source            = "./deploy/terraform/znyx"
  mode              = "managed"
  control_plane_url = "https://console.example.com"
  runtime_token     = var.znyx_runtime_token
}
```

ML model server in the pod, models pre-fetched on a claim:

```hcl
module "znyx" {
  source                   = "./deploy/terraform/znyx"
  ml_enabled               = true
  ml_models_existing_claim = "znyx-models"
  ml_models_read_only      = true
}
```

Air-gapped, with a private registry mirror, and the in-cluster console allowed through the
network policy (pass anything else the chart supports with `extra_values`):

```hcl
module "znyx" {
  source                   = "./deploy/terraform/znyx"
  airgapped                = true
  image_registry           = "registry.internal.example.com:5000"
  image_pull_secrets       = ["regcred"]
  ml_enabled               = true
  ml_models_existing_claim = "znyx-models"
  extra_values = [yamlencode({
    networkPolicy = {
      egressTo = [{ namespaceSelector = { matchLabels = { "kubernetes.io/metadata.name" = "znyx-console" } } }]
    }
  })]
}
```

Your own Secret (External Secrets, Vault, ...): set `existing_secret = "my-secret"` with a
`RUNTIME_API_KEY` key. The module then creates no Secret; add `redis_in_existing_secret = true`
if it also holds `ZNYX_REDIS_URL`.

## Inputs

| Name | Type | Default | Description |
|---|---|---|---|
| `release_name` | string | `"znyx"` | Release, Deployment and Service name. |
| `namespace` | string | `"znyx"` | Target namespace. |
| `create_namespace` | bool | `true` | Create the namespace. |
| `chart_path` | string | local chart | Chart directory (used when `chart_repository` is null). |
| `chart_repository` | string | `null` | Chart repository URL or `oci://` registry. |
| `chart_name` | string | `"znyx"` | Chart name in the repository. |
| `chart_version` | string | `null` | Chart version. |
| `extra_values` | list(string) | `[]` | Extra Helm values (YAML), applied last. |
| `wait` | bool | `true` | Wait for readiness. |
| `atomic` | bool | `false` | Roll back on failure. |
| `timeout_seconds` | number | `600` | Readiness timeout. |
| `image_registry` | string | `""` | Registry prefix for a private mirror. |
| `image_repository` | string | `"znyx/znyx"` | Image repository. |
| `image_tag` | string | `""` | Image tag (empty: chart appVersion; `-ml` added for ML). |
| `image_pull_secrets` | list(string) | `[]` | Existing pull Secrets. |
| `mode` | string | `"local"` | `local` or `managed` (validated). |
| `policy` | string | `""` | Policy file contents (local mode). |
| `control_plane_url` | string | `null` | Console URL (managed mode). |
| `replicas` | number | `2` | Pod count (whole number, at least 1). |
| `ml_enabled` | bool | `false` | `-ml` image and model server in the pod. |
| `ml_models_existing_claim` | string | `""` | PVC with pre-fetched weights at `/models`. |
| `ml_models_read_only` | bool | `false` | Mount that claim read-only. |
| `airgapped` | bool | `false` | No outbound calls; forces the NetworkPolicy on. |
| `network_policy_enabled` | bool | `false` | Create the NetworkPolicy. |
| `existing_secret` | string | `null` | Your own Secret; the module creates none. |
| `api_key` | string (sensitive) | `null` | API key; null generates one. |
| `runtime_token` | string (sensitive) | `null` | Console token (managed mode). |
| `redis_url` | string (sensitive) | `null` | Redis URL (`redis://` or `rediss://`). |
| `redis_in_existing_secret` | bool | `false` | `existing_secret` holds `ZNYX_REDIS_URL`. |

Managed mode without `control_plane_url`, or without a token (when the module creates the
Secret), fails at plan time with a message saying what to set.

## Outputs

| Name | Description |
|---|---|
| `release_name` | Helm release name. |
| `namespace` | Namespace of the release. |
| `service_name` | The ClusterIP Service name (same as the release name). |
| `internal_url` | `http://<service>.<namespace>.svc.cluster.local` for apps and SDKs. |
| `secret_name` | Secret holding the credentials. |
| `api_key` | Client API key (sensitive; null with `existing_secret`). |
| `chart_version` | Installed chart version. |
| `status` | Helm release status. |

## State and secrets

The generated API key and any `runtime_token` or `redis_url` are stored in the Terraform
state, as with any Terraform-managed Secret. Use an encrypted remote backend, or supply
`existing_secret` from your secret manager so the module never sees them.
