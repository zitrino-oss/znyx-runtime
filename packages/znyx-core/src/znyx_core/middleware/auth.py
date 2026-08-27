"""Shared API-key authentication with pluggable validation backend.

Historically this module was coupled to one specific database schema, which
made the runtime impossible to deploy on its own. The abstraction here solves
that: the shared layer declares a ``AuthValidator`` protocol and a
module-level registry, and each process (control plane, runtime, tests)
registers the validator it wants to use at startup.

- A control plane registers a DB-backed validator that looks up APIKey rows.
- Runtime (if it ever needs bearer-auth) can register an in-memory validator
  or an HTTP-proxy validator that calls a control plane over the wire.
- Tests register a stub validator that returns a hand-crafted APIKeyRecord.

If nothing is registered, resolving a validator raises: authentication fails
closed rather than falling back to anything implicit.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, List, Optional, Protocol

from fastapi import Depends, Header, HTTPException, status

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data model for validated keys.
#
# This mirrors the fields callers currently read off the CP ``APIKey`` ORM
# object. We keep it as a simple dataclass so runtime code doesn't need to
# import SQLAlchemy or any CP-internal model to assert against it.
# ---------------------------------------------------------------------------

@dataclass
class APIKeyRecord:
    """Validated key payload returned by an ``AuthValidator``.

    Fields mirror the previously-returned ORM object so call sites don't need
    to change. New fields can be added here as additional validator backends
    surface them (e.g. an HMAC-signed JWT-style API key with embedded scopes).
    """

    id: Optional[Any] = None
    key_prefix: str = ""
    app_id: str = ""
    tenant_id: Optional[str] = None
    org_id: Optional[Any] = None
    project_id: Optional[Any] = None
    environment_id: Optional[Any] = None
    allowed_app_ids: List[str] = field(default_factory=list)
    scopes: List[str] = field(default_factory=list)
    is_active: bool = True
    expires_at: Optional[datetime] = None


class AuthValidator(Protocol):
    """Validate a raw API key string and return a key record.

    Raise :class:`HTTPException` (401/403) on failure.
    """

    async def validate(self, raw_key: str) -> APIKeyRecord: ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_registered_validator: Optional[AuthValidator] = None
_default_validator_factory: Optional[Callable[[], AuthValidator]] = None


def register_auth_validator(validator: AuthValidator) -> None:
    """Register the process-wide validator.

    Call once at startup. Subsequent calls replace the prior registration —
    useful for tests that need to swap in a stub.
    """

    global _registered_validator
    _registered_validator = validator
    logger.info("Auth validator registered: %s", type(validator).__name__)


def register_default_auth_validator_factory(factory: Callable[[], AuthValidator]) -> None:
    """Register the fallback validator factory used when none is explicitly set.

    A control plane registers its DB-backed validator here on import. Keeping
    it in a registry rather than a direct import means ``znyx_core`` carries no
    control-plane import, so the shared engine ships cleanly in OSS
    ``znyx-core``.
    """

    global _default_validator_factory
    _default_validator_factory = factory


def _default_validator() -> AuthValidator:
    """Resolve the fallback validator via the registered factory.

    Raises if no factory is registered — i.e. a process that never loaded the
    control plane and never called :func:`register_auth_validator` tried to
    authenticate. The OSS runtime registers its own validator (or leaves auth
    optional), so it does not depend on this fallback.
    """
    if _default_validator_factory is None:
        raise RuntimeError(
            "No auth validator registered and no default factory available. "
            "Register one via register_auth_validator(), or run behind a "
            "control plane that registers its own validator."
        )
    return _default_validator_factory()


def _get_validator() -> AuthValidator:
    return _registered_validator or _default_validator()


# ---------------------------------------------------------------------------
# Public FastAPI dependencies — unchanged signatures.
# ---------------------------------------------------------------------------

async def verify_api_key(x_api_key: str = Header(...)) -> APIKeyRecord:
    """Validate the ``X-API-Key`` header and return the key record.

    Delegates to whichever :class:`AuthValidator` is registered. Raises 401
    if the header is missing or the validator rejects the key.
    """

    if not x_api_key or not x_api_key.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
        )
    return await _get_validator().validate(x_api_key.strip())


async def get_current_app_id(x_api_key: str = Header(...)) -> List[str]:
    """Return the list of app IDs the current key is allowed to modify.

    Legacy keys without scopes are scoped down to their own ``app_id``.
    """

    api_key = await verify_api_key(x_api_key)

    if not api_key.scopes:
        logger.warning(
            "Legacy key %s has no scopes - restricting to own app_id",
            api_key.key_prefix,
        )
        return [api_key.app_id] if api_key.app_id else []

    return list(api_key.allowed_app_ids or [])


async def require_app_permission(
    app_id: str,
    allowed_app_ids: List[str] = Depends(get_current_app_id),
):
    """Raise 403 if the caller's key is not authorised for ``app_id``."""

    if app_id not in allowed_app_ids:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Not authorized to access app '{app_id}'",
        )


# ---------------------------------------------------------------------------
# Utility used by multiple validators.
# ---------------------------------------------------------------------------

def hash_api_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()
