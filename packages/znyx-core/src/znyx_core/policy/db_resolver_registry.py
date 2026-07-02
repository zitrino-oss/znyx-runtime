"""Registry seam for the database-backed policy resolver.

The OSS engine never constructs a DB-backed resolver itself — that path belongs
to the control plane, which owns the database schema. The control plane registers
a factory here at import time (see ``app/control_plane/__init__.py``); the engine
calls :func:`get_db_policy_resolver` only when a caller supplies a DB session.

This keeps ``znyx_core`` free of any ``app.control_plane`` import, so the shared
engine ships in the OSS ``znyx-core`` package with a clean import closure.
"""
from typing import Any, Callable, Optional

_factory: Optional[Callable[..., Any]] = None


def register_db_policy_resolver(factory: Callable[..., Any]) -> None:
    """Register the control-plane factory used to build a DB-backed resolver."""
    global _factory
    _factory = factory


def get_db_policy_resolver(db: Any, cache_ttl: int = 60) -> Any:
    """Build a DB-backed policy resolver via the registered factory.

    Raises RuntimeError if no factory is registered — i.e. the DB-backed policy
    path was taken in a process that never loaded the control plane. The OSS
    runtime never passes a ``db`` session, so it never reaches this call.
    """
    if _factory is None:
        raise RuntimeError(
            "No DB policy resolver registered. The database-backed policy path "
            "requires the control plane; import app.control_plane to register it."
        )
    return _factory(db, cache_ttl=cache_ttl)
