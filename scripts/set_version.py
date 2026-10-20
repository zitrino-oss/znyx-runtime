"""Stamp one version into every package: ``python scripts/set_version.py 1.2.0``.

Znyx is released as one version. ``znyx`` pins ``znyx-core``, ``znyx-runtime`` and
``znyx-inference`` at exactly its own version, and the services require
``znyx-core>=`` that version, so all four manifests must change together. The
publish workflow runs this with the version derived from the release tag; run it by
hand to bump a release locally. ``--check`` only reports whether the manifests agree.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ("znyx-core", "znyx-runtime", "znyx-inference", "znyx")
_SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+([-.+][0-9A-Za-z.+-]+)?$")


def manifest(pkg: str) -> Path:
    return ROOT / "packages" / pkg / "pyproject.toml"


def current_versions() -> dict:
    out = {}
    for pkg in PACKAGES:
        m = re.search(r'^version = "([^"]+)"', manifest(pkg).read_text(encoding="utf-8"), re.M)
        out[pkg] = m.group(1) if m else None
    return out


def stamp(text: str, version: str) -> str:
    text = re.sub(r'^version = "[^"]*"', f'version = "{version}"', text, count=1, flags=re.M)
    # Floors on the engine (the services) and exact pins (the znyx package).
    text = re.sub(r'znyx-core>=[^",\]\s]+', f"znyx-core>={version}", text)
    text = re.sub(r'(znyx-(?:core|runtime|inference)(?:\[[a-z,]+\])?)==[^",\]\s]+',
                  rf"\g<1>=={version}", text)
    return text


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("version", nargs="?")
    parser.add_argument("--check", action="store_true", help="Only check the manifests agree.")
    args = parser.parse_args(argv)
    if args.check:
        versions = current_versions()
        print("\n".join(f"{pkg}: {v}" for pkg, v in versions.items()))
        return 0 if len(set(versions.values())) == 1 else 1
    if not args.version or not _SEMVER.match(args.version.lstrip("v")):
        parser.error("give a version like 1.2.0")
    version = args.version.lstrip("v")
    for pkg in PACKAGES:
        path = manifest(pkg)
        path.write_text(stamp(path.read_text(encoding="utf-8"), version), encoding="utf-8")
        print(f"{pkg}: {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
