# -- Placement ----------------------------------------------------------------------------

variable "release_name" {
  description = "Helm release name. It is also the name of the Deployment and the Service (the module sets fullnameOverride to it), so the internal URL is http://<release_name>.<namespace>.svc.cluster.local."
  type        = string
  default     = "znyx"

  validation {
    condition     = can(regex("^[a-z0-9]([-a-z0-9]{0,50}[a-z0-9])?$", var.release_name))
    error_message = "release_name must be a lowercase DNS label of at most 52 characters (letters, digits and dashes, starting and ending with a letter or digit)."
  }
}

variable "namespace" {
  description = "Namespace to install into."
  type        = string
  default     = "znyx"

  validation {
    condition     = can(regex("^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$", var.namespace))
    error_message = "namespace must be a valid Kubernetes namespace name (a lowercase DNS label)."
  }
}

variable "create_namespace" {
  description = "Create the namespace (kubernetes_namespace_v1). Set false when it already exists."
  type        = bool
  default     = true
}

# -- Chart --------------------------------------------------------------------------------

variable "chart_path" {
  description = "Path to the chart on disk. Used when chart_repository is null. The default is the chart in this repository (deploy/helm/znyx)."
  type        = string
  default     = null
}

variable "chart_repository" {
  description = "Chart repository URL (or oci:// registry). When set, the chart is pulled from there instead of chart_path."
  type        = string
  default     = null
}

variable "chart_name" {
  description = "Chart name inside chart_repository. Ignored when installing from chart_path."
  type        = string
  default     = "znyx"
}

variable "chart_version" {
  description = "Chart version to install. Null: the local chart's own version, or the latest in the repository."
  type        = string
  default     = null
}

variable "extra_values" {
  description = "Additional Helm values as YAML documents, applied after the values this module sets (later entries win). Use it for anything the module has no variable for, for example ingress, serviceMonitor, resources or networkPolicy.egressTo."
  type        = list(string)
  default     = []
}

variable "wait" {
  description = "Wait until the pods are ready before the apply finishes."
  type        = bool
  default     = true
}

variable "atomic" {
  description = "Roll back the release if the install or upgrade fails (implies wait)."
  type        = bool
  default     = false
}

variable "timeout_seconds" {
  description = "How long to wait for the release to become ready."
  type        = number
  default     = 600

  validation {
    condition     = var.timeout_seconds >= 30
    error_message = "timeout_seconds must be at least 30."
  }
}

# -- Image --------------------------------------------------------------------------------

variable "image_registry" {
  description = "Registry prefix for the image, for a private mirror (for example registry.internal.example.com:5000). Empty: Docker Hub."
  type        = string
  default     = ""
}

variable "image_repository" {
  description = "Image repository path."
  type        = string
  default     = "znyx/znyx"
}

variable "image_tag" {
  description = "Image tag. Empty: the chart's appVersion. The chart appends -ml when ml_enabled."
  type        = string
  default     = ""
}

variable "image_pull_secrets" {
  description = "Names of existing image pull Secrets in the namespace."
  type        = list(string)
  default     = []
}

# -- Behaviour ------------------------------------------------------------------------------

variable "mode" {
  description = "local: policies from the policy variable (or the image's starter policy). managed: signed policy bundles from the console (needs control_plane_url and runtime_token)."
  type        = string
  default     = "local"

  validation {
    condition     = contains(["local", "managed"], var.mode)
    error_message = "mode must be \"local\" or \"managed\"."
  }
}

variable "policy" {
  description = "Contents of the policy file (local mode). Empty: the image's built-in starter policy. Use file(\"policies.yaml\") to load one."
  type        = string
  default     = ""
}

variable "control_plane_url" {
  description = "Console URL (managed mode)."
  type        = string
  default     = null
}

variable "replicas" {
  description = "Number of pods. More than one needs redis_url (or redis_in_existing_secret) so the replicas share limits and budgets."
  type        = number
  default     = 2

  validation {
    condition     = var.replicas >= 1 && var.replicas == floor(var.replicas)
    error_message = "replicas must be a whole number of at least 1."
  }
}

variable "ml_enabled" {
  description = "Use the -ml image and run the ML model server inside the pod on loopback."
  type        = bool
  default     = false
}

variable "ml_models_existing_claim" {
  description = "Existing PersistentVolumeClaim with pre-fetched model weights, mounted at /models. Empty: a per-pod emptyDir."
  type        = string
  default     = ""
}

variable "ml_models_read_only" {
  description = "Mount ml_models_existing_claim read-only (recommended for air-gapped, pre-fetched weights)."
  type        = bool
  default     = false
}

variable "airgapped" {
  description = "No outbound calls at all. Forces the chart's NetworkPolicy on (DNS only plus networkPolicy.egressTo set through extra_values)."
  type        = bool
  default     = false
}

variable "network_policy_enabled" {
  description = "Create the NetworkPolicy (always created when airgapped)."
  type        = bool
  default     = false
}

# -- Credentials ----------------------------------------------------------------------------------

variable "existing_secret" {
  description = "Name of a Secret you manage yourself (RUNTIME_API_KEY, optionally ZNYX_RUNTIME_TOKEN and ZNYX_REDIS_URL). When set, the module creates no Secret and ignores api_key, runtime_token and redis_url."
  type        = string
  default     = null
}

variable "api_key" {
  description = "Client API key stored in the module-created Secret. Null: a random key is generated (random_password) and kept in the Terraform state."
  type        = string
  default     = null
  sensitive   = true
}

variable "runtime_token" {
  description = "Console runtime token (managed mode), stored in the module-created Secret."
  type        = string
  default     = null
  sensitive   = true
}

variable "redis_url" {
  description = "Redis URL (redis:// or rediss://) so replicas share limits and budgets. Stored in the module-created Secret."
  type        = string
  default     = null
  sensitive   = true

  validation {
    condition     = var.redis_url == null || can(regex("^rediss?://", var.redis_url))
    error_message = "redis_url must start with redis:// or rediss://."
  }
}

variable "redis_in_existing_secret" {
  description = "Declare that existing_secret holds a ZNYX_REDIS_URL key, so the pods use the Redis state backend."
  type        = bool
  default     = false
}
