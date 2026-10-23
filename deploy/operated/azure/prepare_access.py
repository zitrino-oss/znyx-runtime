"""Render customer-reviewed AKS/PIM access files locally; never call Azure or Graph."""
from __future__ import annotations

import argparse
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

OPERATIONS = (
    'services', 'events', 'events.k8s.io/events', 'endpoints', 'persistentvolumeclaims',
    'apps/deployments', 'apps/replicasets', 'apps/statefulsets',
    'autoscaling/horizontalpodautoscalers', 'policy/poddisruptionbudgets',
)


def render(output, *, cluster_id, cluster_description, group_id, principal_id, start_at, ticket,
           namespace='znyx', minutes=60, authorization_mode='azure'):
    parts = re.fullmatch(r'/subscriptions/([0-9a-fA-F-]{36})/resourceGroups/([\w.()-]+)/providers/'
                        r'Microsoft.ContainerService/managedClusters/([A-Za-z0-9_-]+)', cluster_id)
    if not parts:
        raise ValueError('Supply the complete existing AKS resource ID')
    uuid.UUID(parts[1])
    group_id, principal_id = str(uuid.UUID(group_id)), str(uuid.UUID(principal_id))
    if not re.fullmatch(r'[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?', namespace):
        raise ValueError('Supply a valid Kubernetes namespace')
    if type(minutes) is not int or not 1 <= minutes <= 60:
        raise ValueError('PIM activation must be 1..60 minutes')
    if authorization_mode not in ('azure', 'kubernetes'):
        raise ValueError('Choose azure or kubernetes authorization')
    if cluster_description.get('id', '').lower() != cluster_id.lower():
        raise ValueError('AKS description must identify the selected cluster')
    aad = cluster_description.get('aadProfile') or {}
    if (aad.get('managed') is not True or cluster_description.get('disableLocalAccounts') is not True
            or (cluster_description.get('apiServerAccessProfile') or {}).get('enablePrivateCluster') is not True):
        raise ValueError('Require private AKS, managed Entra authentication and disabled local accounts')
    azure_mode = aad.get('enableAzureRbac', aad.get('enableAzureRBAC', False))
    if (azure_mode is True) != (authorization_mode == 'azure'):
        raise ValueError('Selected authorization mode differs from AKS; do not change the cluster implicitly')
    start = datetime.fromisoformat(start_at.replace('Z', '+00:00'))
    if start.tzinfo is None or start.utcoffset().total_seconds() != 0:
        raise ValueError('Activation start must include the UTC timezone')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:/-]{0,99}', ticket):
        raise ValueError('Supply a ticket reference without free-form content')
    body = {'accessId': 'member', 'principalId': principal_id, 'groupId': group_id,
        'action': 'selfActivate', 'scheduleInfo': {
            'startDateTime': start.astimezone(timezone.utc).isoformat(),
            'expiration': {'type': 'afterDuration', 'duration': f'PT{minutes}M'}},
        'justification': 'Znyx diagnostics: ' + ticket,
        'ticketInfo': {'ticketNumber': ticket, 'ticketSystem': 'Customer incident system'}}
    files = {'pim-activation.json': body,
        'access.parameters.json': {
            '$schema': 'https://schema.management.azure.com/schemas/2019-04-01/deploymentParameters.json#',
            'contentVersion': '1.0.0.0', 'parameters': {
                'aksClusterName': {'value': parts[3]}, 'operatorGroupObjectId': {'value': group_id}}}}
    files['access.parameters.json']['parameters']['confirmedPrivateEntraCluster'] = {'value': True}
    scope = cluster_id + '/namespaces/' + namespace
    if authorization_mode == 'azure':
        # Pod subresources are intentionally absent: this role grants workload/service
        # diagnostics, without any pod, log, exec, secret or RBAC data permissions.
        role_id = str(uuid.uuid5(uuid.NAMESPACE_URL, scope + '/znyx-operated-diagnostics'))
        files['namespace-role.json'] = {'Name': 'Znyx diagnostics ' + role_id[:8], 'Id': role_id,
            'IsCustom': True, 'Description': 'Customer-owned namespace diagnostics without request content',
            'Actions': [], 'NotActions': [], 'DataActions': [
                'Microsoft.ContainerService/managedClusters/' + operation + '/read' for operation in OPERATIONS],
            'NotDataActions': [], 'AssignableScopes': [cluster_id.split('/providers/')[0]]}
        files['namespace-assignment.json'] = {'scope': scope, 'principalId': group_id, 'principalType': 'Group',
            'roleDefinitionId': role_id,
            'name': str(uuid.uuid5(uuid.NAMESPACE_URL, scope + group_id + role_id))}
    output = Path(output)
    # Refuse mixing two authorization modes in the same output directory.
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    for filename, value in files.items():
        (output / filename).write_text(json.dumps(value, indent=2) + '\n')
    if authorization_mode == 'kubernetes':
        objects = list(yaml.safe_load_all((Path(__file__).resolve().parents[1] / 'rbac.yaml').read_text()))
        for obj in objects:
            obj['metadata']['namespace'] = namespace
        objects[1]['subjects'][0]['name'] = group_id
        (output / 'rbac.yaml').write_text(yaml.safe_dump_all(objects, sort_keys=False))
    return files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cluster-id', 'group-id', 'principal-id', 'start-at', 'ticket'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--namespace', default='znyx')
    parser.add_argument('--minutes', type=int, default=60)
    parser.add_argument('--authorization-mode', choices=('azure', 'kubernetes'), default='azure')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cluster-description', type=Path, required=True)
    args = vars(parser.parse_args())
    args['cluster_description'] = json.loads(args['cluster_description'].read_text())
    render(**args)
