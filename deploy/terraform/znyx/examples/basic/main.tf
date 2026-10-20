# Znyx on an existing cluster: a namespace, a generated API key and the chart's defaults
# (two replicas, rules only). Run from this directory:
#
#   terraform init
#   terraform apply
#   terraform output -raw api_key
#
# Needs a reachable cluster in the kubeconfig below. `terraform validate` needs none.

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    helm = {
      source  = "hashicorp/helm"
      version = ">= 3.0.0, < 4.0.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = ">= 2.30.0, < 4.0.0"
    }
    random = {
      source  = "hashicorp/random"
      version = ">= 3.5.0, < 4.0.0"
    }
  }
}

variable "kubeconfig_path" {
  description = "Path to the kubeconfig of the target cluster."
  type        = string
  default     = "~/.kube/config"
}

provider "kubernetes" {
  config_path = var.kubeconfig_path
}

provider "helm" {
  kubernetes = {
    config_path = var.kubeconfig_path
  }
}

module "znyx" {
  source = "../.."

  namespace = "znyx"

  # Your own policy (local mode); remove to use the image's built-in starter policy.
  # policy = file("${path.module}/policies.yaml")

  # Several replicas should share limits and budgets:
  # redis_url = "redis://redis.data.svc.cluster.local:6379/0"
}

output "internal_url" {
  value = module.znyx.internal_url
}

output "api_key" {
  value     = module.znyx.api_key
  sensitive = true
}
