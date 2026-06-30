from pydantic import BaseModel
from typing import Optional
from enum import Enum


class LicenseTier(str, Enum):
    free = "free"
    growth = "growth"
    enterprise = "enterprise"


class LicenseInfo(BaseModel):
    license_key: str
    tier: LicenseTier
    org_id: str
    org_name: str
    max_evaluations_per_month: int  # -1 = unlimited
    features: list[str] = []  # enabled feature flags
    expires_at: Optional[str] = None
    is_valid: bool = True


class HeartbeatPayload(BaseModel):
    license_key: str
    runtime_version: str = "1.0.0"
    evaluation_count: int = 0
    detector_count: int = 0
    uptime_seconds: int = 0


class LicenseValidationResponse(BaseModel):
    valid: bool
    tier: Optional[str] = None
    org_id: Optional[str] = None
    org_name: Optional[str] = None
    max_evaluations_per_month: int = 0
    features: list[str] = []
    expires_at: Optional[str] = None
    message: str = ""
