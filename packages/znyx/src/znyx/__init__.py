"""Znyx: one install for AI guardrails.

The ``znyx`` package is the product shell: the ``znyx`` command, the ``znyx.yaml``
config file and the supervisor that runs the runtime's worker processes and the
optional ML model server on one port. The services themselves live in
``znyx_runtime``, ``znyx_inference`` and ``znyx_core``, released at the same version.
"""
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("znyx")
except PackageNotFoundError:  # not installed (source-tree import)
    __version__ = "0.0.0+unknown"

COMPONENTS = ("znyx-core", "znyx-runtime", "znyx-inference")


def component_versions() -> dict:
    """Installed version of each component; ``None`` when one is missing."""
    out = {}
    for dist in COMPONENTS:
        try:
            out[dist] = _pkg_version(dist)
        except PackageNotFoundError:
            out[dist] = None
    return out


__all__ = ["__version__", "component_versions", "COMPONENTS"]
