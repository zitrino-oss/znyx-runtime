# Azure operated deployment

Azure is the first operated deployment and marketplace target. The customer owns an existing
private AKS cluster, its Entra tenant, Secrets, volumes, Redis, audit destination and access
grants. Workload changes use the customer's reviewed GitOps pipeline. This kit neither creates
an operator account nor grants standing administrator access.

## Customer prerequisites

- Private AKS, managed Entra authentication, local accounts disabled, and an approved private
  operator network path. Keep the existing authorization mode: Azure RBAC or Kubernetes RBAC.
- A dedicated customer security group governed by Entra PIM for Groups, with eligible operator
  memberships only, customer approval, MFA and ticket requirements. Invite any external
  operators through the customer's normal B2B process. PIM licensing/policy is customer-owned.
- Namespace `znyx`; an existing Secret with `RUNTIME_API_KEY`, `ZNYX_RUNTIME_TOKEN` and a private
  TLS `ZNYX_REDIS_URL`; a shared ReadWriteMany runtime-data PVC; and, for speech, an approved
  read-only model PVC with `/speech` and `ZNYX_SPEECH_MODEL_SHA256` in that Secret. Images, provider
  credentials and model assets stay in customer-controlled storage. Do not put credentials
  into Helm values, extension configuration settings or portal text fields.
  Azure's overlay projects all keys from this dedicated runtime Secret into the container,
  including approved provider key variables and bundle-verification configuration. Include
  only runtime-specific keys; do not reuse a general application or control-plane Secret.
- A private Redis endpoint compatible with the runtime's Redis client, its actual IPv4 and
  TLS port, and customer DNS/network policy. The supplied Redis egress rule is a single `/32`.
  Runtime media defaults to local extraction. No model download is needed during deployment.
- Existing customer Log Analytics workspace and retention/access policy. The diagnostics role
  receives no Log Analytics/Key Vault access.

## Prepare access locally

The customer reads the real AKS configuration, then generates files without granting access:

```sh
az aks show --resource-group "$AKS_RESOURCE_GROUP" --name "$AKS_NAME" --output json > /tmp/znyx-aks.json
python deploy/operated/azure/prepare_access.py \
  --cluster-id "$AKS_RESOURCE_ID" --cluster-description /tmp/znyx-aks.json \
  --group-id "$CUSTOMER_PIM_GROUP_OBJECT_ID" --principal-id "$OPERATOR_GUEST_OBJECT_ID" \
  --authorization-mode azure --start-at "$ACTIVATION_START_UTC" --minutes 60 \
  --ticket "$INCIDENT_TICKET" --output /tmp/znyx-access-review
```

Use `--authorization-mode kubernetes` only when the cluster already uses Entra authentication
with Kubernetes RBAC. Preflight checks the cluster ID, private endpoint, managed Entra,
disabled local accounts and matching authorization mode. It refuses a mismatch before writing.
It does not query Azure itself; review a freshly exported description from the actual cluster.
Use a fresh output directory. IDs are customer-tenant object IDs, not display names or app IDs.

The files contain a self-activation request with a maximum 60-minute duration and ticket, and
ARM parameters for the customer-owned group. PIM still enforces the customer's eligibility and
approval policy. The generator never posts to Graph or Azure.

## Customer applies the reviewed grants

First run an ARM what-if, then deploy `access.bicep` with the generated parameters after customer
review. This grants only the AKS Cluster User role on this cluster, for Entra kubeconfig
retrieval. The template requires an explicit preflight acknowledgement from the generator;
that acknowledgement is not an independent Azure-side configuration audit.

```sh
az deployment group what-if --resource-group "$AKS_RESOURCE_GROUP" \
  --template-file deploy/operated/azure/access.bicep \
  --parameters @/tmp/znyx-access-review/access.parameters.json

az deployment group create --resource-group "$AKS_RESOURCE_GROUP" \
  --template-file deploy/operated/azure/access.bicep \
  --parameters @/tmp/znyx-access-review/access.parameters.json
```

For **Azure RBAC**, customer administrators create `namespace-role.json` with
`az role definition create --role-definition @/tmp/znyx-access-review/namespace-role.json`,
then assign its role-definition ID to the group at the exact namespace scope recorded in
`namespace-assignment.json`. Use that file's `name` for idempotency:

```sh
az role assignment create --name "$NAMESPACE_ASSIGNMENT_NAME" \
  --assignee-object-id "$CUSTOMER_PIM_GROUP_OBJECT_ID" --assignee-principal-type Group \
  --role "$NAMESPACE_ROLE_DEFINITION_ID" --scope "$AKS_RESOURCE_ID/namespaces/znyx"
```

This custom role grants deployment, service, event and related workload diagnostics. It has
no wildcard, mutation, pod, log, exec, Secret or RBAC permissions. Pod APIs are intentionally
absent from the Azure role. Do not substitute the broader built-in Reader/Writer/Admin roles.

For **Kubernetes RBAC**, the customer applies the generated `rbac.yaml` instead. It binds the
real Entra group UUID only in `znyx`; read-only pod metadata is allowed, while `pods/log`,
`pods/exec`, Secrets, mutations and RBAC are absent. Do not apply this binding in Azure-RBAC mode.

The eligible operator submits the reviewed activation JSON to
`https://graph.microsoft.com/v1.0/identityGovernance/privilegedAccess/group/assignmentScheduleRequests`
using their customer-tenant identity and the approved PIM process. No tokens belong in this
repository. After approval, obtain user credentials with `az aks get-credentials` and kubelogin;
never use `--admin`. Verify an allowed read and denied Secret/log/exec/mutation requests, plus
denial outside the namespace, before accepting the grant.

## Install and audit

Install the Helm chart using `operated/values.yaml`, `operated/azure/values.yaml` and customer
overrides, or use the generated native marketplace package. Set the actual pinned Linux AMD64
media image, control-plane URL, Secret/PVC names and Redis egress `/32`/port. The `.invalid` URL
and `10.0.0.1/32` are packaging defaults; replace them with reviewed customer settings. Direct
Helm users must do this themselves; the marketplace UI requires these inputs.

The service is ClusterIP. Label approved client namespaces `znyx.ai/gateway-client=true` to
allow their traffic; calls still need a gateway key. Customer ingress/TLS and any additional
egress destinations remain explicit configuration. Run the clean/blocked channel, signed-feed,
multi-replica Redis and media checks in the parent runbook before serving production traffic.

`diagnostics.bicep` routes `kube-audit` and `guard` to an existing customer workspace using
resource-specific tables. Deploy it with the actual cluster/workspace names after a what-if.
Also retain Azure Activity Log role/deployment changes, Entra sign-in/audit/PIM events and
runtime audit spools under the customer's retention policy. Correlate principal/object IDs,
activation times, assignment IDs and incident ticket. Full audit records may contain sensitive
data; only customer-authorized audit readers should have access. The operator role cannot read them.

## Expiry, revoke and offboard

PIM activation expires; existing tokens and authorization caches can outlive the membership
change. Expiry is not an instantaneous session kill. For revocation, the customer removes the
namespace Azure role assignment (or Kubernetes RoleBinding), deactivates PIM membership and
removes the cluster-user assignment. Validate actual denials after propagation. For an urgent
incident, isolate the operator's private network route while authorization changes propagate.

Retire eligibility and B2B access through the customer process. Rotate any credential that was
exposed through a separately approved incident grant. Disable sharing and revoke the runtime
token when offboarding; archive customer audit records before deleting workloads/volumes.
Delete the custom role definition only after its assignments have been removed. No grant in
this kit gives the operator workload-write, role-assignment-write or cluster-admin authority.

See [AKS authorization](https://learn.microsoft.com/en-us/azure/aks/entra-id-authorization),
[Entra/Kubernetes RBAC](https://learn.microsoft.com/en-us/azure/aks/kubernetes-rbac-entra-id),
[PIM activation API](https://learn.microsoft.com/en-us/graph/api/privilegedaccessgroup-post-assignmentschedulerequests?view=graph-rest-1.0)
and [Azure packaging](../../marketplace/README.md). Local compilation/rendering is complete;
actual installation, PIM expiry/revocation and audit delivery need the customer's real account.
