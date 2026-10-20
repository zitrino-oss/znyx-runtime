"""Secret references, resolved inside the customer's network.

Provider keys and other credentials are configured, and shipped in policy bundles,
as references, never as the secret itself:

    env:OPENAI_API_KEY          an environment variable of the runtime
    file:/run/secrets/openai    a file's contents (Kubernetes / Docker secrets);
                                surrounding whitespace is trimmed
    literal:sk-...              the value itself; accepted for local experiments only,
                                and reported as a warning

A value without a scheme is rejected rather than guessed at, so a key pasted into a
config file is caught instead of silently shipped around. Cloud secret stores (Key
Vault, KMS, Secrets Manager) plug in through ``register_resolver``.

Who may name what. Configuration the operator writes (local mode) may reference any
variable or file. A signed bundle from the control plane (managed mode) is limited
to what the operator allows on this side: variables named in ``ZNYX_SECRET_ENV_ALLOW``
(default: ``ZNYX_SECRET_*`` and the standard provider key names) and files under
``ZNYX_SECRETS_DIR`` (default ``/run/secrets`` and ``/var/run/secrets``). A bundle may
not carry ``literal:`` values (the control plane never issues them), and plugged-in
schemes are usable from a bundle only when listed in ``ZNYX_SECRET_SCHEMES_ALLOW``.
Without this, a compromised control plane could name any of the host's variables or
files as a "provider key" and point that provider at itself.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)

_RESOLVERS: Dict[str, Callable[[str], Optional[str]]] = {}
_restricted = False

DEFAULT_ENV_ALLOW = ("ZNYX_SECRET_*", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AZURE_OPENAI_API_KEY",
                     "GOOGLE_API_KEY")
DEFAULT_SECRETS_DIRS = ("/run/secrets", "/var/run/secrets")


def restrict_references(enabled: bool) -> None:
    """Managed mode: references (which then come from bundles) may only name what the
    operator allows on this side. See the module docstring."""
    global _restricted
    _restricted = bool(enabled)


def _env_allowed(name: str) -> bool:
    raw = os.environ.get("ZNYX_SECRET_ENV_ALLOW")
    patterns = [p.strip() for p in raw.split(",") if p.strip()] if raw else list(DEFAULT_ENV_ALLOW)
    return any(name.startswith(p[:-1]) if p.endswith("*") else name == p for p in patterns)


def _scheme_allowed(scheme: str) -> bool:
    raw = os.environ.get("ZNYX_SECRET_SCHEMES_ALLOW") or ""
    return scheme in {s.strip().lower() for s in raw.split(",") if s.strip()}


def _file_allowed(path: Path) -> bool:
    raw = os.environ.get("ZNYX_SECRETS_DIR")
    dirs = [d for d in raw.split(os.pathsep) if d] if raw else list(DEFAULT_SECRETS_DIRS)
    try:
        real = path.resolve(strict=True)
    except (OSError, RuntimeError):
        return False
    for directory in dirs:
        try:
            base = Path(directory).expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if real == base or base in real.parents:
            return True
    return False


class SecretRefError(ValueError):
    """A reference is malformed or cannot be resolved. The message never contains
    the secret."""


def register_resolver(scheme: str, resolver: Callable[[str], Optional[str]]) -> None:
    """Add a scheme, e.g. ``register_resolver("vault", lookup)`` for ``vault:path``."""
    _RESOLVERS[scheme.lower()] = resolver


def is_reference(value: object) -> bool:
    if not isinstance(value, str) or ":" not in value:
        return False
    scheme = value.split(":", 1)[0].lower()
    return scheme in ("env", "file", "literal", "none") or scheme in _RESOLVERS


def resolve(ref: Optional[str], *, what: str = "secret") -> Optional[str]:
    """The secret a reference names. ``None``, ``""`` and ``none:`` mean "no secret"
    (an endpoint that needs no key). Raises ``SecretRefError`` when the reference is
    malformed or names something that does not exist."""
    if ref is None or ref == "":
        return None
    if not isinstance(ref, str) or ":" not in ref:
        raise SecretRefError(
            f"{what} must be a reference such as env:NAME or file:/path, not a plain value "
            "(a key written straight into configuration ends up in files and bundles)")
    scheme, _, rest = ref.partition(":")
    scheme = scheme.lower()
    if scheme == "none":
        return None
    if scheme == "env":
        name = rest.strip()
        if not name:
            raise SecretRefError(f"{what}: env: needs a variable name")
        if _restricted and not _env_allowed(name):
            raise SecretRefError(f"{what}: the bundle names variable {name}, which this runtime "
                                 "does not allow (ZNYX_SECRET_ENV_ALLOW)")
        value = os.environ.get(name)
        if value is None or value == "":
            raise SecretRefError(f"{what}: environment variable {name} is not set")
        return value
    if scheme == "file":
        path = Path(rest.strip()).expanduser()
        if _restricted and not _file_allowed(path):
            raise SecretRefError(f"{what}: the bundle names a file outside this runtime's secrets "
                                 "directories (ZNYX_SECRETS_DIR)")
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise SecretRefError(f"{what}: cannot read {path} ({exc.strerror or exc})") from None
        if not value:
            raise SecretRefError(f"{what}: {path} is empty")
        return value
    if scheme == "literal":
        if _restricted:
            raise SecretRefError(f"{what}: a bundle may not carry a literal value; use env: or file:")
        logger.warning("%s is configured as a literal value; use env: or file: outside "
                       "local experiments", what)
        return rest
    resolver = _RESOLVERS.get(scheme)
    if resolver is None:
        raise SecretRefError(f"{what}: unknown reference scheme {scheme!r} "
                             "(env:, file:, literal:, none:)")
    if _restricted and not _scheme_allowed(scheme):
        raise SecretRefError(f"{what}: the bundle uses the {scheme}: scheme, which this runtime "
                             "does not allow (ZNYX_SECRET_SCHEMES_ALLOW)")
    try:
        value = resolver(rest)
    except SecretRefError:
        raise
    except Exception as exc:  # noqa: BLE001 - never echo the secret or the store's reply
        raise SecretRefError(f"{what}: {scheme}: lookup failed ({type(exc).__name__})") from None
    if not value:
        raise SecretRefError(f"{what}: {scheme}: reference resolved to nothing")
    return value


def describe(ref: Optional[str]) -> str:
    """A loggable description of a reference (never the secret)."""
    if not ref:
        return "none"
    if not isinstance(ref, str) or ":" not in ref:
        return "(plain value)"
    scheme, _, rest = ref.partition(":")
    return f"{scheme}:{'***' if scheme.lower() == 'literal' else rest}"


_reported: set = set()


def maybe_resolve(value, *, what: str = "secret"):
    """For credentials a policy or bundle carries: a reference is resolved; anything
    else (a plaintext credential from a control plane that still inlines them) is
    returned unchanged, so existing deployments keep working. A reference that cannot
    be resolved gives "" (no credential: the call fails or falls back as the policy
    says) and is logged once."""
    if not is_reference(value):
        return value
    try:
        return resolve(value, what=what) or ""
    except SecretRefError as exc:
        if value not in _reported:
            _reported.add(value)
            logger.error("%s", exc)
        return ""

