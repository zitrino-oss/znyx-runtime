"""Azure delivery assets: permission boundaries, native inputs and real Helm rendering."""
import importlib.util
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CLUSTER = '/subscriptions/12345678-1234-4234-8234-123456789012/resourceGroups/customer/providers/Microsoft.ContainerService/managedClusters/private-aks'
GROUP = '12345678-1234-4234-8234-123456789013'
PERSON = '12345678-1234-4234-8234-123456789014'
IMAGE = 'exampleacr.azurecr.io/znyx/runtime@sha256:' + 'a' * 64


def module(relative):
    spec = importlib.util.spec_from_file_location('azure_review', ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def access_args(mode='azure'):
    return dict(cluster_id=CLUSTER, cluster_description={'id': CLUSTER, 'disableLocalAccounts': True,
        'aadProfile': {'managed': True, 'enableAzureRbac': mode == 'azure'},
        'apiServerAccessProfile': {'enablePrivateCluster': True}},
        group_id=GROUP, principal_id=PERSON, start_at='2026-10-23T18:00:00Z', ticket='INC-1234',
        authorization_mode=mode)


def test_azure_access_is_customer_scoped_expiring_and_has_no_secret_or_pod_permissions(tmp_path):
    files = module('deploy/operated/azure/prepare_access.py').render(tmp_path, **access_args())
    request = files['pim-activation.json']
    assert request['action'] == 'selfActivate' and request['accessId'] == 'member'
    assert request['groupId'] == GROUP and request['principalId'] == PERSON
    assert request['scheduleInfo']['expiration'] == {'type': 'afterDuration', 'duration': 'PT60M'}
    role = files['namespace-role.json']
    assert role['Actions'] == []
    assert role['AssignableScopes'] == [CLUSTER.split('/providers/')[0]]
    assert all(operation.endswith('/read') and '*' not in operation for operation in role['DataActions'])
    assert not any(word in operation.lower() for operation in role['DataActions']
                   for word in ('secrets', 'pods/', 'logs/', 'exec/', 'rolebindings', 'roleassignments'))
    assignment = files['namespace-assignment.json']
    assert assignment['scope'] == CLUSTER + '/namespaces/znyx' and assignment['principalId'] == GROUP
    assert not (tmp_path / 'rbac.yaml').exists()


def test_entra_kubernetes_mode_uses_the_real_group_object_id_and_namespace(tmp_path):
    files = module('deploy/operated/azure/prepare_access.py').render(tmp_path, namespace='customer-znyx', **access_args('kubernetes'))
    role, binding = list(yaml.safe_load_all((tmp_path / 'rbac.yaml').read_text()))
    assert binding['subjects'][0]['name'] == GROUP
    assert role['metadata']['namespace'] == binding['metadata']['namespace'] == 'customer-znyx'
    assert 'namespace-role.json' not in files


@pytest.mark.parametrize('change', [dict(minutes=0), dict(minutes=61), dict(minutes=True),
    dict(group_id='operator-display-name'), dict(namespace='default/other'), dict(start_at='2026-10-23T18:00:00'),
    dict(authorization_mode='kubernetes'), dict(cluster_description={'id': CLUSTER}), dict(ticket='raw prompt text')])
def test_access_generator_refuses_unsafe_or_unverified_grants_before_writing(tmp_path, change):
    args = {**access_args(), **change}
    with pytest.raises(ValueError):
        module('deploy/operated/azure/prepare_access.py').render(tmp_path / 'output', **args)
    assert not (tmp_path / 'output').exists()


def build(output):
    return module('deploy/marketplace/build_delivery.py').build(output, 'azure', IMAGE, '1.2.3',
        azure_application_name='com.example.znyx', azure_publisher='Example Publisher')


def test_azure_delivery_has_native_relative_assets_and_is_reproducible(tmp_path):
    manifest = build(tmp_path / 'one')
    build(tmp_path / 'two')
    assert (tmp_path / 'one/znyx-delivery.tar.gz').read_bytes() == (tmp_path / 'two/znyx-delivery.tar.gz').read_bytes()
    assert manifest['azure']['architecture'] == 'linux/amd64'
    assert manifest['azure']['cnab_status'] == 'requires-authenticated-cpa-build-and-seller-validation'
    native = yaml.safe_load((tmp_path / 'one/azure/manifest.yaml').read_text())
    for key in ('helmChart', 'clusterArmTemplate', 'uiDefinition'):
        assert (tmp_path / 'one/azure' / native[key]).exists()
    template = json.loads((tmp_path / 'one/azure/mainTemplate.json').read_text())
    ui = json.loads((tmp_path / 'one/azure/createUIDefinition.json').read_text())
    assert set(template['parameters']) - {'extensionVersion'} == set(ui['parameters']['outputs'])
    extension, = template['resources']
    assert extension['type'] == 'Microsoft.KubernetesConfiguration/extensions'
    assert extension['properties']['autoUpgradeMinorVersion'] is False
    assert extension['properties']['configurationProtectedSettings'] == {}
    assert 'DONOTMODIFY' in template['variables'].values()
    with tarfile.open(tmp_path / 'one/znyx-delivery.tar.gz') as archive:
        assert 'azure/manifest.yaml' in archive.getnames()
        assert 'operated/azure/access.bicep' in archive.getnames()
    for name, digest in manifest['files'].items():
        import hashlib
        with tarfile.open(tmp_path / 'one/znyx-delivery.tar.gz') as archive:
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest


@pytest.mark.parametrize('fields', [{}, {'azure_application_name': 'com.example.znyx'},
    {'azure_application_name': '../bad', 'azure_publisher': 'Publisher'}])
def test_azure_packaging_refuses_missing_or_invalid_publisher_metadata(tmp_path, fields):
    with pytest.raises(ValueError):
        module('deploy/marketplace/build_delivery.py').build(tmp_path, 'azure', IMAGE, '1.2.3', **fields)


@pytest.mark.parametrize('image,version', [('docker.io/znyx/runtime@sha256:' + 'a' * 64, '1.2.3'),
    ('exampleacr.azurecr.io/znyx/runtime:latest@sha256:' + 'a' * 64, '1.2.3'), (IMAGE, '1.2.3-preview')])
def test_azure_packaging_refuses_wrong_registry_mutable_repository_or_version(tmp_path, image, version):
    with pytest.raises(ValueError):
        module('deploy/marketplace/build_delivery.py').build(tmp_path, 'azure', image, version,
            azure_application_name='com.example.znyx', azure_publisher='Example Publisher')


@pytest.mark.parametrize('digest', [True, False])
def test_real_helm_render_follows_marketplace_rehosted_image_for_every_container(tmp_path, digest):
    helm = os.getenv('HELM_BINARY') or shutil.which('helm')
    if not helm:
        pytest.skip('Set HELM_BINARY to validate the native marketplace chart')
    build(tmp_path)
    chart = tmp_path / 'azure/chart'
    values = yaml.safe_load((chart / 'values.yaml').read_text())
    image = values['global']['azure']['images']['runtime']
    image['registry'], image['image'] = 'marketplace.azurecr.io', 'assigned/rehosted-runtime'
    if digest:
        image['digest'] = 'sha256:' + 'b' * 64
        expected = 'marketplace.azurecr.io/assigned/rehosted-runtime@sha256:' + 'b' * 64
    else:
        image.pop('digest')
        image['tag'] = 'rehosted-1.2.3'
        expected = 'marketplace.azurecr.io/assigned/rehosted-runtime:rehosted-1.2.3'
    (chart / 'values.yaml').write_text(yaml.safe_dump(values))
    result = subprocess.run([helm, 'template', 'znyx', str(chart), '--namespace', 'znyx',
        '--set', 'networkPolicy.extraEgress[0].to[0].ipBlock.cidr=10.1.2.3/32'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    objects = [obj for obj in yaml.safe_load_all(result.stdout) if obj]
    deployment = next(obj for obj in objects if obj['kind'] == 'Deployment')
    spec = deployment['spec']['template']['spec']
    assert spec['nodeSelector']['kubernetes.io/arch'] == 'amd64'
    assert all(container['image'] == expected for container in spec['containers'])
    assert spec['containers'][0]['envFrom'] == [{'secretRef': {'name': 'znyx-runtime-secrets'}}]
    health = next(obj for obj in objects if obj['kind'] == 'Pod')
    assert health['spec']['containers'][0]['image'] == expected
    models = next(volume for volume in spec['volumes'] if volume['name'] == 'models')
    assert models['persistentVolumeClaim']['readOnly'] is True
    assert not any(obj['kind'] in ('Secret', 'ClusterRole', 'ClusterRoleBinding') for obj in objects)
