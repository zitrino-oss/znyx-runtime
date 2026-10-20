locals {
  namespace  = var.create_namespace ? kubernetes_namespace_v1.this[0].metadata[0].name : var.namespace
  chart      = var.chart_repository == null ? coalesce(var.chart_path, "${path.module}/../../helm/znyx") : var.chart_name
  repository = var.chart_repository

  # The module creates the Secret unless one is supplied, so credentials stay out of the Helm
  # release (which stores its values in a cluster Secret) and the chart is given existingSecret.
  create_secret = var.existing_secret == null
  secret_name   = local.create_secret ? "${var.release_name}-secrets" : var.existing_secret
  api_key       = var.api_key != null ? var.api_key : try(random_password.api_key[0].result, null)
  redis         = local.create_secret ? var.redis_url != null : var.redis_in_existing_secret

  secret_data = merge(
    { RUNTIME_API_KEY = local.api_key },
    var.runtime_token != null ? { ZNYX_RUNTIME_TOKEN = var.runtime_token } : {},
    var.redis_url != null ? { ZNYX_REDIS_URL = var.redis_url } : {},
  )

  config = merge(
    { mode = var.mode },
    var.control_plane_url != null ? { control_plane = { url = var.control_plane_url } } : {},
  )

  values = yamlencode({
    fullnameOverride = var.release_name
    replicaCount     = var.replicas
    image = {
      registry   = var.image_registry
      repository = var.image_repository
      tag        = var.image_tag
    }
    imagePullSecrets = [for name in var.image_pull_secrets : { name = name }]
    config           = local.config
    policy           = var.policy
    existingSecret   = local.secret_name
    state            = { redis = { enabled = local.redis } }
    airgapped        = var.airgapped
    ml = {
      enabled = var.ml_enabled
      models = {
        existingClaim = var.ml_models_existing_claim
        readOnly      = var.ml_models_read_only
      }
    }
    networkPolicy = { enabled = var.network_policy_enabled }
    # Pods do not restart on a Secret change by themselves; a short hash of the module's own
    # Secret contents as a pod annotation rolls them when the key, token or URL changes.
    podAnnotations = local.create_secret ? {
      "checksum/secret" = nonsensitive(substr(sha256(jsonencode(local.secret_data)), 0, 16))
    } : {}
  })
}

resource "kubernetes_namespace_v1" "this" {
  count = var.create_namespace ? 1 : 0

  metadata {
    name = var.namespace
    labels = {
      "app.kubernetes.io/name"       = "znyx"
      "app.kubernetes.io/managed-by" = "terraform"
    }
  }
}

resource "random_password" "api_key" {
  count = local.create_secret && var.api_key == null ? 1 : 0

  length  = 40
  special = false
}

resource "kubernetes_secret_v1" "this" {
  count = local.create_secret ? 1 : 0

  metadata {
    name      = local.secret_name
    namespace = local.namespace
    labels = {
      "app.kubernetes.io/name"       = "znyx"
      "app.kubernetes.io/instance"   = var.release_name
      "app.kubernetes.io/component"  = "gateway"
      "app.kubernetes.io/managed-by" = "terraform"
    }
  }

  type = "Opaque"
  data = local.secret_data
}

resource "helm_release" "this" {
  name       = var.release_name
  namespace  = local.namespace
  chart      = local.chart
  repository = local.repository
  version    = var.chart_version

  create_namespace = false
  wait             = var.wait
  atomic           = var.atomic
  timeout          = var.timeout_seconds

  values = concat([local.values], var.extra_values)

  depends_on = [kubernetes_secret_v1.this]

  lifecycle {
    precondition {
      condition     = var.mode != "managed" || var.control_plane_url != null
      error_message = "mode is managed: set control_plane_url to the console URL."
    }
    precondition {
      condition     = var.mode != "managed" || !local.create_secret || var.runtime_token != null
      error_message = "mode is managed: set runtime_token (or existing_secret with a ZNYX_RUNTIME_TOKEN key)."
    }
  }
}
