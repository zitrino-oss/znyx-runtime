"""Double-submit-cookie CSRF protection for state-changing requests.

Design rationale — why not just rely on ``SameSite=Lax``?

``SameSite=Lax`` protects against cross-site ``POST`` from a simple ``<form>``
submission, but it does **not** block cross-site ``fetch()`` with
``credentials: 'include'`` if the attacker's origin is somehow whitelisted
by CORS — and our CORS config uses ``allow_credentials=True``. Belt and
braces: we *also* require a header-based CSRF token that matches a
non-``HttpOnly`` cookie. The attacker can't read the cookie cross-origin
(that's enforced by the browser's same-origin policy on ``document.cookie``),
so they can't craft the matching header.

Flow:

1. ``/v1/auth/login`` sets ``znyx_session`` (``HttpOnly``) **and**
   ``znyx_csrf`` (not ``HttpOnly``) on the response.
2. The console's bootstrap reads ``znyx_csrf`` from ``document.cookie`` and
   attaches it as ``X-CSRF-Token`` on every mutation.
3. This middleware enforces the match on every non-safe method.

Exemptions:

- ``Authorization: Bearer`` requests (SDK + CI) bypass the check — they don't
  ride on a cookie session, so CSRF doesn't apply.
- ``X-API-Key`` requests likewise bypass — API-key auth is intentional.
- GET / HEAD / OPTIONS are never checked.
- Webhook receiver paths (``/v1/billing/webhook``) are exempt — they carry
  their own HMAC signature and are not cookie-driven.
"""
from __future__ import annotations

import hmac
import logging
import os
import secrets
from typing import Iterable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}

SESSION_COOKIE_NAME = "znyx_session"
CSRF_COOKIE_NAME = "znyx_csrf"
CSRF_HEADER_NAME = "x-csrf-token"

# Paths exempt from CSRF even when cookie-authenticated.
DEFAULT_EXEMPT_PREFIXES: tuple[str, ...] = (
    "/v1/billing/webhook",           # Stripe webhook, HMAC-signed
    "/v1/licenses/heartbeat",        # license heartbeat, unauthenticated
    "/scim/",                        # SCIM, bearer token only
    "/v1/auth/login",                # login itself sets the cookies
    "/v1/auth/register",             # register sets them too
    "/v1/auth/mfa/",                 # MFA flows bootstrap the session
    "/v1/admin/auth/login",          # super-admin login, sets cookies
    "/v1/admin/auth/logout",         # clearing cookies; CSRF-forcing a logout is harmless
    "/v1/auth/forgot-password",      # public — no session required
    "/v1/auth/reset-password",       # public — token-based auth
    "/v1/auth/resend-verification",  # public — no session required
    "/v1/auth/verify-email",         # public — token-based auth
)


def generate_csrf_token() -> str:
    """128-bit URL-safe token. Cryptographically random, not derivable."""
    return secrets.token_urlsafe(32)


def _request_has_bearer_or_apikey(request: Request) -> bool:
    """Bearer-token / API-key requests aren't cookie-driven; CSRF doesn't apply."""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return True
    if request.headers.get("x-api-key"):
        return True
    return False


def _path_exempt(path: str, exempt: Iterable[str]) -> bool:
    for prefix in exempt:
        if prefix.endswith("/"):
            if path.startswith(prefix):
                return True
        elif path == prefix or path.startswith(prefix + "/"):
            return True
    return False


class CSRFMiddleware(BaseHTTPMiddleware):
    """Enforce double-submit-cookie CSRF on cookie-authenticated mutations."""

    def __init__(self, app, extra_exempt_prefixes: Optional[tuple[str, ...]] = None):
        super().__init__(app)
        self._exempt = DEFAULT_EXEMPT_PREFIXES + (extra_exempt_prefixes or ())
        # Test mode: used by integration tests that don't want to round-trip
        # a login just to POST. Deliberately env-gated so you can't disable
        # it in prod by flag alone.
        self._disabled = os.getenv("CSRF_DISABLED", "false").lower() in ("1", "true", "yes")
        from znyx_core.utils.env import is_production
        if self._disabled and is_production():
            logger.error("CSRF_DISABLED=true rejected in production; re-enabling")
            self._disabled = False

    async def dispatch(self, request: Request, call_next):
        if self._disabled or request.method.upper() in SAFE_METHODS:
            return await call_next(request)

        if _path_exempt(request.url.path, self._exempt):
            return await call_next(request)

        # Bearer / API-key flows skip the check — they aren't browser cookies.
        if _request_has_bearer_or_apikey(request):
            return await call_next(request)

        session_cookie = request.cookies.get(SESSION_COOKIE_NAME)
        if not session_cookie:
            # Not a cookie session — auth middleware downstream will reject
            # anonymous mutations with 401. CSRF check is moot.
            return await call_next(request)

        cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
        header_token = request.headers.get(CSRF_HEADER_NAME)

        if not cookie_token or not header_token or not hmac.compare_digest(
            cookie_token, header_token
        ):
            logger.warning(
                "CSRF rejection on %s %s", request.method, request.url.path
            )
            return JSONResponse(
                status_code=403,
                content={
                    "detail": {
                        "error": "csrf_token_missing_or_invalid",
                        "message": (
                            "Missing or mismatched CSRF token. Send the "
                            f"'{CSRF_HEADER_NAME}' header matching the "
                            f"'{CSRF_COOKIE_NAME}' cookie."
                        ),
                    }
                },
            )

        return await call_next(request)
