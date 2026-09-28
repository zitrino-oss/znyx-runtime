"""ZNYX Runtime - the data-plane service that serves policy bundles and evaluates traffic.

See znyx_core.__init__ for why `__version__` comes from installed metadata rather
than a literal.
"""
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("znyx-runtime")
except PackageNotFoundError:  # not installed (source-tree import)
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
