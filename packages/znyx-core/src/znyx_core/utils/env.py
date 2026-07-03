"""Canonical environment detection — single source of truth.

Historically "are we in production?" was decided by two env-var families with
OPPOSITE defaults:

  * ``ZNYX_ENV`` / ``ENVIRONMENT``           — default: NOT prod (runtime, CSRF,
                                                rate-limit, main, crypto)
  * ``ZNYX_ENVIRONMENT`` / ``GUARDRAILS_ENVIRONMENT`` — default: production
                                                (auth, license, billing)

Setting only one family could leave CSRF-disable, HSTS, cookie ``Secure``, or
the rate-limit fail-fast in the wrong mode. ``is_production()`` reads all four
so the decision can never desync.

Migration note: existing modules still use their local checks; new code should
prefer this helper, and call sites should migrate to it over time (each
migration must be validated against the local-run env-var conventions).
"""
from __future__ import annotations

import os

# Order is informational only; the rule below does not depend on it.
ENV_VARS = ("ZNYX_ENV", "ZNYX_ENVIRONMENT", "ENVIRONMENT", "GUARDRAILS_ENVIRONMENT")
_NONPROD = {"dev", "development", "local", "test", "testing", "ci", "staging", "stage"}


def is_production() -> bool:
    """Return True unless ANY recognized env var explicitly names a non-prod env.

    Secure-by-default: with nothing set we assume production. A single explicit
    non-prod value (e.g. ``ZNYX_ENVIRONMENT=development``) flips the whole
    process to non-prod, so the families can never disagree.
    """
    for var in ENV_VARS:
        value = (os.getenv(var) or "").strip().lower()
        if value in _NONPROD:
            return False
    return True
