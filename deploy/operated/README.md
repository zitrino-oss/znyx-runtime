# Znyx operated inside the customer's cloud

The data plane, Redis, media models, provider credentials, audit spools and request content
stay in the customer's account. The customer owns the cluster, network policy and secret
store. Znyx receives metadata and only the explicitly enabled hash-only shared-defence records.

**Azure first:** use the [AKS/Entra/PIM runbook](azure/README.md), native access/audit Bicep
templates and Azure deployment overlay. AWS is retained as a secondary option below.

## Install and access

1. Use an existing private Kubernetes cluster. On AKS, configure Entra/PIM and audit delivery as
   described in the Azure runbook. On EKS, enable API access entries and control-plane
   `audit`, `authenticator` and `api` logging. Enable CloudTrail management events in the customer
   account and retain them in a customer-controlled, encrypted, versioned audit bucket. Capture
   Kubernetes audit logs in the same retention regime. These are customer prerequisites; this kit
   does not silently alter the customer's cluster or create a public endpoint.
2. Install the existing `deploy/terraform/znyx` module or Helm chart with this directory's
   `values.yaml` and customer values. Put `ZNYX_RUNTIME_TOKEN`, `RUNTIME_API_KEY` and
   `ZNYX_REDIS_URL` into the existing `znyx-runtime-secrets` Secret through the customer's secret
   manager. Configure the control-plane URL, pinned release, media service, TLS ingress and
   network-policy egress explicitly. Replicas must share Redis; budgets fail closed if it is down.
3. The customer reviews `rbac.yaml` and applies it in the dedicated `znyx` namespace (change both
   namespace fields together if using another namespace). The role cannot read Secrets, logs,
   exec into pods, create workloads or modify RBAC. No cluster-wide privileges are granted.
4. For Azure, prepare the actual Entra object IDs and reviewed namespace grant using
   `azure/prepare_access.py` and `azure/README.md`. The generic group name in `rbac.yaml` is not
   an Entra object ID. For AWS, deploy `aws-access.yaml`, naming the exact approved operator role,
   a new random external ID and an expiry. Do not output the external ID or store it in Git.
   The access entry maps only to the diagnostics group. On other Kubernetes clouds, map the
   customer's federated operator identity to the same group through the native identity service.
5. Azure operators activate customer-approved PIM access. AWS operators assume the role with a
   unique `znyx-<ticket>` source identity. STS credentials last
   at most an hour; the trust expires at the customer's chosen time. Record access in CloudTrail
   and Kubernetes audit logs. Monitoring uses metadata endpoints and customer-owned alerts.

## Upgrade and incident operation

Znyx prepares a version-pinned Helm/Terraform change and validation evidence for the customer's
GitOps pipeline. The customer approves and executes it; diagnostics access cannot modify workloads.
Validate readiness, signed-bundle polling, Redis, media extraction, a blocked test attack, clean
requests, usage delivery and multi-replica limits before promotion. Retain the previous image and
configuration so the customer's pipeline can revert the deployment. This is an operational runbook,
not Phase 6's traffic canary or automatic rollback engine.

For an incident, use the existing signed-bundle kill switch, or customer ingress/network isolation.
Access beyond diagnostics requires a separately approved, expiring customer grant and an incident
ticket; do not give the diagnostics role Secrets, pod-exec or administrator access.

## Revoke and offboard

For Azure, remove namespace access, deactivate PIM and remove cluster-user access; verify actual
denials after propagation and isolate the private route during urgent incidents.
For AWS, delete the RoleBinding and EKS access entry, revoke existing role sessions, then
deletes the IAM trust/role. Removing the trust alone does not terminate already-issued STS sessions.
Rotate runtime/provider credentials if incident access could have exposed them. Disable shared
defence consent and revoke the runtime token. Archive the customer's audit spools and cluster/CloudTrail
logs before removing workloads, volumes or Redis. The customer decides retention and deletion.

## Marketplace delivery

`deploy/marketplace/build_delivery.py` defaults to Azure and assembles a BYOL review archive with
native CPA manifest, Marketplace image values, ARM/portal inputs and this operated option.
It accepts release/image digests and chosen publisher metadata; it does not invent
seller or product identifiers. The seller uploads the signed images/chart into the marketplace's
assigned repositories, submits the product and completes cloud-provider validation. Customer
deployment and listing publication are external acceptance steps, not accomplished by this repository.

AWS supports container images and standard Helm delivery; Helm Quick Launch was discontinued on
2026-03-01. See [AWS container products](https://docs.aws.amazon.com/marketplace/latest/userguide/container-based-products.html)
and [EKS access entries](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-eks-accessentry.html).
