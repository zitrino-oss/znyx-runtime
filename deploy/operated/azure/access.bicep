targetScope = 'resourceGroup'

@description('Existing customer-owned private AKS cluster in this resource group.')
param aksClusterName string

@description('Object ID of the customer-owned Entra security group governed by PIM. No permanent operator members.')
@minLength(36)
@maxLength(36)
param operatorGroupObjectId string

@description('Set by prepare_access.py only after validating the customer AKS description. Review that preflight before deployment.')
@allowed([true])
param confirmedPrivateEntraCluster bool

resource aks 'Microsoft.ContainerService/managedClusters@2024-10-01' existing = {
  name: aksClusterName
}

// This grant retrieves the Entra user kubeconfig only. Namespace authorization is
// applied separately; no cluster-admin credential or Kubernetes data permission.
var clusterUserRoleId = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '4abbcc35-e782-43d8-92c5-2d3f1bd2253f')
resource clusterUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (confirmedPrivateEntraCluster) {
  name: guid(aks.id, operatorGroupObjectId, clusterUserRoleId)
  scope: aks
  properties: {
    principalId: operatorGroupObjectId
    principalType: 'Group'
    roleDefinitionId: clusterUserRoleId
    description: 'Customer-controlled Znyx diagnostics group. Activation requires customer Entra PIM policy.'
  }
}

output preflightAcknowledged bool = confirmedPrivateEntraCluster
output clusterResourceId string = aks.id
output authorizationMode string = (aks.properties.?aadProfile.?enableAzureRBAC ?? false) ? 'azure' : 'kubernetes'
output assignmentId string = confirmedPrivateEntraCluster ? clusterUser!.id : ''
