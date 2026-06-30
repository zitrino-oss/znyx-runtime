import os
import json
import time
import logging
import httpx
from pathlib import Path
from typing import Optional
from .models import LicenseInfo, LicenseTier, LicenseValidationResponse

logger = logging.getLogger(__name__)

LICENSE_CACHE_FILE = Path(".guardrails_license_cache.json")
GRACE_PERIOD_DAYS = 7

_current_license: Optional[LicenseInfo] = None
_evaluation_count: int = 0
_start_time: float = time.time()


def get_license() -> LicenseInfo:
    global _current_license
    if _current_license is None:
        _current_license = LicenseInfo(
            license_key="",
            tier=LicenseTier.free,
            org_id="local",
            org_name="Local",
            max_evaluations_per_month=1000,
            features=[],
        )
    return _current_license


def is_feature_enabled(feature: str) -> bool:
    license = get_license()
    if license.tier == LicenseTier.enterprise:
        return True
    return feature in license.features


def increment_evaluation_count():
    global _evaluation_count
    _evaluation_count += 1


def get_evaluation_count() -> int:
    return _evaluation_count


def get_uptime_seconds() -> int:
    return int(time.time() - _start_time)


async def validate_license() -> LicenseInfo:
    global _current_license

    license_key = (
        os.getenv("ZNYX_LICENSE_KEY", "").strip()
        or os.getenv("GUARDRAILS_LICENSE_KEY", "").strip()
    )
    license_server = (
        os.getenv("ZNYX_LICENSE_SERVER_URL", "")
        or os.getenv("GUARDRAILS_LICENSE_SERVER_URL", "")
        or "https://api.znyx.ai/v1/licenses/validate"
    )

    if not license_key:
        logger.info("No license key set - running in Free tier (local mode)")
        _current_license = LicenseInfo(
            license_key="",
            tier=LicenseTier.free,
            org_id="local",
            org_name="Local",
            max_evaluations_per_month=1000,
            features=[],
        )
        return _current_license

    # Try to validate with license server
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(license_server, json={"license_key": license_key})
            if resp.status_code == 200:
                data = resp.json()
                validation = LicenseValidationResponse(**data)
                if validation.valid:
                    _current_license = LicenseInfo(
                        license_key=license_key,
                        tier=LicenseTier(validation.tier or "free"),
                        org_id=validation.org_id or "",
                        org_name=validation.org_name or "",
                        max_evaluations_per_month=validation.max_evaluations_per_month,
                        features=validation.features,
                        expires_at=validation.expires_at,
                        is_valid=True,
                    )
                    # Cache the valid license
                    _save_cache(_current_license)
                    logger.info(
                        f"License validated: {_current_license.tier.value} tier, "
                        f"org: {_current_license.org_name}"
                    )
                    return _current_license
                else:
                    logger.warning(f"License invalid: {validation.message}")
            else:
                logger.warning(f"License server returned HTTP {resp.status_code}")
    except Exception as e:
        logger.warning(f"Cannot reach license server: {e}")

    # Fallback to cache
    cached = _load_cache(license_key)
    if cached:
        logger.info(f"Using cached license: {cached.tier.value} tier (grace period)")
        _current_license = cached
        return _current_license

    # No cache, no server - default to free
    logger.warning("License validation failed and no cache available - falling back to Free tier")
    _current_license = LicenseInfo(
        license_key=license_key,
        tier=LicenseTier.free,
        org_id="unknown",
        org_name="Unknown",
        max_evaluations_per_month=1000,
        features=[],
        is_valid=False,
    )
    return _current_license


def _save_cache(license: LicenseInfo):
    try:
        data = license.model_dump()
        data["cached_at"] = time.time()
        LICENSE_CACHE_FILE.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logger.debug(f"Could not save license cache: {e}")


def _load_cache(license_key: str) -> Optional[LicenseInfo]:
    try:
        if not LICENSE_CACHE_FILE.exists():
            return None
        data = json.loads(LICENSE_CACHE_FILE.read_text())
        if data.get("license_key") != license_key:
            return None
        cached_at = data.get("cached_at", 0)
        if time.time() - cached_at > GRACE_PERIOD_DAYS * 86400:
            logger.warning("License cache expired (past grace period)")
            return None
        data.pop("cached_at", None)
        return LicenseInfo(**data)
    except Exception:
        return None
