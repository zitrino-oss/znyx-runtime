"""Build a reproducible BYOL marketplace review package, without publishing anything."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import tarfile
from pathlib import Path


def prepare_azure(output, root, image, version, application_name, publisher):
    """Produce native CPA inputs; authenticated CNAB build/push remains a seller action."""
    import yaml
    if not re.fullmatch(r'[a-z][a-z0-9]*(?:\.[a-z][a-z0-9-]*){2,}', application_name or ''):
        raise ValueError('Azure needs the chosen reverse-domain application name')
    if not publisher or publisher != publisher.strip() or len(publisher) > 100 or any(ord(c) < 32 for c in publisher):
        raise ValueError('Azure needs the publisher display name')
    repository, digest = image.split('@', 1)
    registry, separator, image_path = repository.partition('/')
    if not separator or not re.fullmatch(r'[a-z0-9]{5,50}\.azurecr\.io', registry):
        raise ValueError('Azure packaging requires a public-Azure ACR image reference')
    if not re.fullmatch(r'[a-z0-9]+(?:[._/-][a-z0-9]+)*', image_path):
        raise ValueError('Use a lowercase ACR repository path without a tag')
    if not re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+', version):
        raise ValueError('Azure marketplace versions must have three numeric components')
    azure = output / 'azure'
    if azure.exists():
        raise ValueError('Use a fresh Azure delivery directory')
    azure.mkdir(parents=True)
    shutil.copytree(root / 'helm/znyx', azure / 'chart')
    def merge(target, source):
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = value
    values = yaml.safe_load((azure / 'chart/values.yaml').read_text())
    for relative in ('operated/values.yaml', 'operated/azure/values.yaml'):
        merge(values, yaml.safe_load((root / relative).read_text()))
    values['global'] = {'azure': {'images': {'runtime': {
        'registry': registry, 'image': image_path, 'digest': digest}}}}
    values['image'] = {'registry': '', 'repository': repository, 'digest': digest,
                       'tag': '', 'pullPolicy': 'IfNotPresent'}
    values['test']['useRuntimeImage'] = True
    (azure / 'chart/values.yaml').write_text(yaml.safe_dump(values, sort_keys=False))
    chart = yaml.safe_load((azure / 'chart/Chart.yaml').read_text())
    chart['version'] = chart['appVersion'] = version
    (azure / 'chart/Chart.yaml').write_text(yaml.safe_dump(chart, sort_keys=False))
    template = json.loads((root / 'marketplace/azure/mainTemplate.json').read_text())
    template['parameters']['extensionVersion']['defaultValue'] = version
    (azure / 'mainTemplate.json').write_text(json.dumps(template, indent=2) + '\n')
    shutil.copyfile(root / 'marketplace/azure/createUIDefinition.json', azure / 'createUIDefinition.json')
    native = {'applicationName': application_name, 'publisher': publisher,
        'description': 'Znyx guarded AI gateway inside customer-owned AKS', 'version': version,
        'helmChart': './chart', 'clusterArmTemplate': './mainTemplate.json',
        'uiDefinition': './createUIDefinition.json', 'registryServer': registry,
        'extensionRegistrationParameters': {'defaultScope': 'cluster', 'namespace': 'znyx'}}
    (azure / 'manifest.yaml').write_text(yaml.safe_dump(native, sort_keys=False))
    return azure


def build(output: Path, cloud: str, image: str, version: str, *, azure_application_name=None, azure_publisher=None):
    if cloud not in ("aws", "azure", "gcp"):
        raise ValueError("Choose aws, azure or gcp")
    if not re.fullmatch(r"[a-zA-Z0-9./:_-]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("Supply the marketplace-assigned image repository pinned by SHA-256 digest")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][a-zA-Z0-9.-]+)?", version):
        raise ValueError("Supply a semantic release version")
    root = Path(__file__).resolve().parents[1]
    output = Path(output).resolve()
    if any(output.is_relative_to(root / source) for source in ('helm', 'operated', 'marketplace')):
        raise ValueError('Delivery output must be outside the source asset directories')
    files = sorted(path for directory in ("helm/znyx", "operated") for path in (root / directory).rglob("*") if path.is_file())
    files = [path for path in files if '__pycache__' not in path.parts]
    generated = []
    if cloud == 'azure':
        azure = prepare_azure(output, root, image, version, azure_application_name, azure_publisher)
        generated = sorted(path for path in azure.rglob('*') if path.is_file())
    manifest = {"schema_version": 1, "purpose": "seller-review-package", "cloud": cloud,
        "license_model": "BYOL", "release": version, "image": image,
        "publication_status": "requires-seller-submission", "files": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}
    output.mkdir(parents=True, exist_ok=True)
    manifest['files'].update({str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest() for path in generated})
    if cloud == 'azure':
        manifest['azure'] = {'architecture': 'linux/amd64', 'native_inputs': 'azure/manifest.yaml',
                             'cnab_status': 'requires-authenticated-cpa-build-and-seller-validation'}
    manifest_path = output / "delivery.json"
    repository, digest = image.split("@", 1)
    pinned = output / "marketplace-values.json"
    pinned.write_text(json.dumps({"image": {"registry": "", "repository": repository, "digest": digest},
                                 "test": {"useRuntimeImage": True}}, indent=2) + "\n")
    manifest["files"]["marketplace-values.json"] = hashlib.sha256(pinned.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    def normalized(info):
        info.uid = info.gid = info.mtime = 0
        info.uname = info.gname = ""
        info.mode = 0o644
        return info
    with (output / "znyx-delivery.tar.gz").open("wb") as handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path in files:
                    archive.add(path, arcname=str(path.relative_to(root)), filter=normalized)
                for path in generated:
                    archive.add(path, arcname=str(path.relative_to(output)), filter=normalized)
                archive.add(manifest_path, arcname="delivery.json", filter=normalized)
                archive.add(pinned, arcname="marketplace-values.json", filter=normalized)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloud", default='azure', choices=("azure", "aws", "gcp"))
    parser.add_argument("--image", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument('--azure-application-name')
    parser.add_argument('--azure-publisher')
    args = parser.parse_args()
    build(args.output, args.cloud, args.image, args.version,
          azure_application_name=args.azure_application_name, azure_publisher=args.azure_publisher)
