"""Centralised numeric tunables — single source of truth for thresholds,
batch sizes, retry counts, and timeout windows.

Adding a new constant here is preferable to inlining a number in service
code for three reasons:

1. **Discoverability.** A new engineer can grep one file to learn what
   knobs exist and what their defaults are.
2. **Override surface.** Every value reads from an env var with a sensible
   default — ops never has to redeploy to retune a quota.
3. **Type/range safety.** The casts (``int(os.getenv(..., default))``) live
   in one place; adding a min/max guard is one edit, not a campaign.

Convention: every setting is ``UPPER_SNAKE_CASE``, exposed as a module-level
constant. Group by subsystem with a comment header. When a service consumes
one, prefer ``from app.config.tunables import RETRY_DELAYS_SECONDS`` over
duplicating the literal value.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple


def _int(env: str, default: int) -> int:
    try:
        return int(os.getenv(env, str(default)))
    except (TypeError, ValueError):
        return default


def _float(env: str, default: float) -> float:
    try:
        return float(os.getenv(env, str(default)))
    except (TypeError, ValueError):
        return default


def _str(env: str, default: str = "") -> str:
    return os.getenv(env, default)


def _bool(env: str, default: bool = False) -> bool:
    raw = os.getenv(env)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes")


def _csv_floats(env: str, default: Tuple[float, ...]) -> Tuple[float, ...]:
    raw = os.getenv(env)
    if not raw:
        return default
    try:
        return tuple(float(x.strip()) for x in raw.split(",") if x.strip())
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Webhook delivery
# ---------------------------------------------------------------------------
WEBHOOK_MAX_RETRIES: int = _int("ZNYX_WEBHOOK_MAX_RETRIES", 3)
WEBHOOK_RETRY_DELAYS_SECONDS: Tuple[float, ...] = _csv_floats(
    "ZNYX_WEBHOOK_RETRY_DELAYS", (1.0, 5.0, 30.0)
)
WEBHOOK_REQUEST_TIMEOUT_SECONDS: float = _float("ZNYX_WEBHOOK_TIMEOUT", 10.0)


# ---------------------------------------------------------------------------
# Runtime telemetry batching
# ---------------------------------------------------------------------------
TELEMETRY_BATCH_SIZE: int = _int("ZNYX_TELEMETRY_BATCH_SIZE", 100)
TELEMETRY_FLUSH_INTERVAL_SECONDS: int = _int("ZNYX_TELEMETRY_FLUSH_INTERVAL", 10)


# ---------------------------------------------------------------------------
# Bundle manager
# ---------------------------------------------------------------------------
BUNDLE_BOOT_RETRY_DELAYS_SECONDS: Tuple[float, ...] = _csv_floats(
    "ZNYX_BUNDLE_BOOT_RETRY_DELAYS", (1.0, 3.0, 5.0)
)
BUNDLE_POLL_INTERVAL_SECONDS: float = _float("ZNYX_BUNDLE_POLL_INTERVAL", 30.0)


# ---------------------------------------------------------------------------
# Billing — subscription state machine
# ---------------------------------------------------------------------------
BILLING_GRACE_DAYS: int = _int("ZNYX_BILLING_GRACE_DAYS", 7)
# DEPRECATED: trials are removed — no plan grants a trial and nothing reads
# this value anymore. Kept only so existing env/config files don't error.
# The ``trialing`` status is left inert for any pre-existing trialing orgs.
BILLING_TRIAL_DAYS: int = _int("ZNYX_BILLING_TRIAL_DAYS", 14)


# ---------------------------------------------------------------------------
# Retention defaults (per plan, in days). Keep alongside other tunables so
# ops can override per-tier without editing the service body.
# ---------------------------------------------------------------------------
RETENTION_FREE_DAYS: int = _int("ZNYX_RETENTION_FREE_DAYS", 30)
RETENTION_GROWTH_DAYS: int = _int("ZNYX_RETENTION_GROWTH_DAYS", 90)
RETENTION_ENTERPRISE_DAYS: int = _int("ZNYX_RETENTION_ENTERPRISE_DAYS", 365)


# ---------------------------------------------------------------------------
# Auth / brute-force protection
# ---------------------------------------------------------------------------
AUTH_MAX_FAILED_ATTEMPTS: int = _int("MAX_FAILED_LOGIN_ATTEMPTS", 10)
AUTH_LOCKOUT_DURATION_MINUTES: int = _int("LOGIN_LOCKOUT_MINUTES", 15)
AUTH_LOGIN_RATE_LIMIT_PER_MINUTE: int = _int("LOGIN_RATE_LIMIT_PER_MINUTE", 10)
AUTH_REGISTER_RATE_LIMIT_PER_MINUTE: int = _int("REGISTER_RATE_LIMIT_PER_MINUTE", 5)


# ---------------------------------------------------------------------------
# Remote detector HTTP client
# ---------------------------------------------------------------------------
REMOTE_DETECTOR_TIMEOUT_SECONDS: float = _float("ZNYX_REMOTE_DETECTOR_TIMEOUT", 5.0)
REMOTE_DETECTOR_MAX_RETRIES: int = _int("ZNYX_REMOTE_DETECTOR_MAX_RETRIES", 2)
REMOTE_DETECTOR_CIRCUIT_THRESHOLD: int = _int(
    "ZNYX_REMOTE_DETECTOR_CIRCUIT_THRESHOLD", 5
)


# ---------------------------------------------------------------------------
# Scheduler intervals (seconds)
# ---------------------------------------------------------------------------
SCHEDULER_RETENTION_INTERVAL_SECONDS: int = _int("SCHEDULER_RETENTION_INTERVAL", 3600)
SCHEDULER_ALERT_INTERVAL_SECONDS: int = _int("SCHEDULER_ALERT_INTERVAL", 300)
SCHEDULER_SUBSCRIPTION_INTERVAL_SECONDS: int = _int(
    "SCHEDULER_SUBSCRIPTION_INTERVAL", 3600
)
# Auto-invoice generation: how often to check for ended billing periods.
SCHEDULER_BILLING_INTERVAL_SECONDS: int = _int("SCHEDULER_BILLING_INTERVAL", 3600)


# ---------------------------------------------------------------------------
# Pagination defaults
# ---------------------------------------------------------------------------
DEFAULT_PAGE_SIZE: int = _int("ZNYX_DEFAULT_PAGE_SIZE", 50)
MAX_PAGE_SIZE: int = _int("ZNYX_MAX_PAGE_SIZE", 200)


# ---------------------------------------------------------------------------
# Database connection pool
# ---------------------------------------------------------------------------
DB_POOL_SIZE: int = _int("DB_POOL_SIZE", 20)
DB_MAX_OVERFLOW: int = _int("DB_MAX_OVERFLOW", 40)


# ---------------------------------------------------------------------------
# HTTP rate limiter (per-IP / per-key sliding window)
# ---------------------------------------------------------------------------
RATE_LIMIT_REQUESTS_PER_MINUTE: int = _int("RATE_LIMIT_REQUESTS_PER_MINUTE", 60)
RATE_LIMIT_BURST: int = _int("RATE_LIMIT_BURST", 10)


# ---------------------------------------------------------------------------
# Server ports
# ---------------------------------------------------------------------------
CONTROL_PLANE_PORT: int = _int("CONTROL_PLANE_PORT", _int("PORT", 8000))
RUNTIME_PORT: int = _int("RUNTIME_PORT", _int("PORT", 8080))


# ---------------------------------------------------------------------------
# Email — Microsoft Graph API
# ---------------------------------------------------------------------------
MS_GRAPH_TENANT_ID: str = _str("MICROSOFT_TENANT_ID", "")
MS_GRAPH_CLIENT_ID: str = _str("MICROSOFT_CLIENT_ID", "")
MS_GRAPH_CLIENT_SECRET: str = _str("MICROSOFT_CLIENT_SECRET", "")
MS_GRAPH_SENDER_EMAIL: str = _str("MICROSOFT_SENDER_EMAIL", "")
EMAIL_FROM_NAME: str = _str("EMAIL_FROM_NAME", "ZNYX Guardrails")
# Inbox that website contact-form enquiries (POST /v1/contact) are delivered to.
# Mirrors the website's NEXT_PUBLIC_ENQUIRY_EMAIL default.
CONTACT_ENQUIRY_EMAIL: str = _str("CONTACT_ENQUIRY_EMAIL", "").strip() or "enquiry@zitrino.com"
# NB: a *present but empty* env var (e.g. an unset GitHub Actions variable
# expanded into a ConfigMap as CONSOLE_BASE_URL='') makes os.getenv return ""
# rather than the default, which would yield relative links like
# "/console/verify-email?..." in emails — broken in Outlook and other clients.
# Strip + `or default` so an empty value still resolves to an absolute URL.
CONSOLE_BASE_URL: str = _str("CONSOLE_BASE_URL", "").strip().rstrip("/") or "http://localhost:3701"
PASSWORD_RESET_EXPIRES_MINUTES: int = _int("PASSWORD_RESET_EXPIRES_MINUTES", 60)


# ---------------------------------------------------------------------------
# Azure Storage
# ---------------------------------------------------------------------------
AZURE_STORAGE_ACCOUNT_NAME: str = _str("AZURE_STORAGE_ACCOUNT_NAME", "")
AZURE_STORAGE_ACCOUNT_KEY: str = _str("AZURE_STORAGE_ACCOUNT_KEY", "")
AZURE_STORAGE_CONTAINER_NAME: str = _str("AZURE_STORAGE_CONTAINER_NAME", "")
AZURE_PUBLIC_STORAGE_CONTAINER: str = _str("AZURE_PUBLIC_STORAGE_CONTAINER", "publicmailtemplates")
KEY_VAULT_NAME: str = _str("KEY_VAULT_NAME", "")
