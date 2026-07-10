"""The inference scoring contract — the request + response shapes the
extended RemoteDetector speaks. Kept dependency-free (pydantic only)."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Canonical decision set — the inference contract speaks the same vocabulary as the
# rest of ZNYX (single source of truth), so a runner can't emit an unknown decision.
from znyx_core.core.models import Decision


class InferRequest(BaseModel):
    """One scoring request. Provide EXACTLY ONE non-empty input form: ``text`` for a
    single item OR ``texts`` for an explicit batch (a 422 otherwise)."""
    text: Optional[str] = None
    texts: Optional[List[str]] = None
    model_id: Optional[str] = None       # optional pin assertion (must match the loaded model)
    revision: Optional[str] = None

    @model_validator(mode="after")
    def _validate(self):
        has_text, has_texts = self.text is not None, self.texts is not None
        if has_text == has_texts:
            raise ValueError("provide exactly one of 'text' or 'texts'")
        if has_text and not (self.text or "").strip():
            raise ValueError("'text' must be non-empty")
        if has_texts and (not self.texts or any(not (t or "").strip() for t in self.texts)):
            raise ValueError("'texts' must be a non-empty list of non-empty strings")
        # A revision pin is meaningless without the model it pins — reject it (else a
        # revision-only pin would be silently ignored, scoring against the wrong model).
        if self.revision is not None and not self.model_id:
            raise ValueError("'revision' pin requires 'model_id'")
        return self

    def items(self) -> List[str]:
        if self.texts is not None:
            return list(self.texts)
        return [self.text or ""]


class InferResult(BaseModel):
    """The confidence contract for a single scored item."""
    decision: Decision                              # canonical: ALLOW|WARN|BLOCK|REDACT|TRANSFORM
    risk_score: int = Field(ge=0, le=100)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    label_scores: Optional[Dict[str, float]] = None
    calibrated_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    threshold: Optional[float] = None
    model_version: str                              # model_id@revision (stable)

    @field_validator("label_scores")
    @classmethod
    def _label_scores_in_unit_range(cls, v):
        if v is not None:
            for key, val in v.items():
                if not (0.0 <= float(val) <= 1.0):
                    raise ValueError(f"label score for '{key}' must be in [0, 1]")
        return v


class InferResponse(InferResult):
    """Single-item response = the contract + the service-measured latency."""
    latency_ms: int = 0
    cached: bool = False


class BatchInferResponse(BaseModel):
    results: List[InferResult]
    latency_ms: int = 0
    model_version: str


class ModelInfo(BaseModel):
    task: str
    model_version: str
    runner: str                # "stub" | "classifier" | "embedding" | "nli" | "guard_llm"
    available: bool            # False when heavy deps / artifacts are missing
    model_id: Optional[str] = None
    revision: Optional[str] = None
    sha256: Optional[str] = None
    detail: Optional[str] = None
