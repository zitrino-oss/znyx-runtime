output "release_name" {
  description = "Helm release name."
  value       = helm_release.this.name
}

output "namespace" {
  description = "Namespace the release is installed in."
  value       = local.namespace
}

output "service_name" {
  description = "Name of the ClusterIP Service (the same as the release name)."
  value       = var.release_name
}

output "internal_url" {
  description = "In-cluster URL for apps and SDKs (Service port 80)."
  value       = "http://${var.release_name}.${local.namespace}.svc.cluster.local"
}

output "secret_name" {
  description = "Secret holding RUNTIME_API_KEY (and the token and Redis URL when set)."
  value       = local.secret_name
}

output "api_key" {
  description = "The client API key, or null when existing_secret is used (read it from that Secret). Sensitive: `terraform output -raw api_key`."
  value       = local.create_secret ? local.api_key : null
  sensitive   = true
}

output "chart_version" {
  description = "Chart version of the installed release."
  value       = helm_release.this.metadata.version
}

output "status" {
  description = "Helm release status."
  value       = helm_release.this.status
}
