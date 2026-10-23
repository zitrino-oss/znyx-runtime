targetScope = 'resourceGroup'

@description('Existing customer-owned AKS cluster in this resource group.')
param aksClusterName string

@description('Existing customer-controlled Log Analytics workspace name.')
param workspaceName string

@description('Resource group of that workspace, in the same subscription.')
param workspaceResourceGroup string = resourceGroup().name

resource aks 'Microsoft.ContainerService/managedClusters@2024-10-01' existing = {
  name: aksClusterName
}

resource workspace 'Microsoft.OperationalInsights/workspaces@2023-09-01' existing = {
  name: workspaceName
  scope: resourceGroup(workspaceResourceGroup)
}

resource audit 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  name: 'znyx-operated-audit'
  scope: aks
  properties: {
    workspaceId: workspace.id
    logAnalyticsDestinationType: 'Dedicated'
    logs: [
      { category: 'kube-audit', enabled: true }
      { category: 'guard', enabled: true }
    ]
  }
}

output diagnosticSettingId string = audit.id
