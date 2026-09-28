"""ZNYX Core - detection engine, policy schema and evaluation primitives.

`__version__` is read from installed package metadata rather than hardcoded, so
the version a running service reports is always the version that was actually
installed. A literal here would have to be kept in step with pyproject.toml by
hand, and the two drift silently the moment a release bumps one and not the
other.

The fallback only fires when the package is imported from a source tree that was
never installed (no .dist-info) - a bare PYTHONPATH import in a test or a
checkout. It is deliberately "0.0.0+unknown" rather than a plausible-looking
number: an about screen showing an obviously-unreal version is a better failure
than one confidently showing the wrong release.
"""
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("znyx-core")
except PackageNotFoundError:  # not installed (source-tree import)
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
