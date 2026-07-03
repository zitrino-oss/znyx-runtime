"""Shared skeleton for heavy (ML) runners (F3). Keeps the no-egress / fail-closed
posture in one place: lazy-import the stack (missing → RunnerUnavailable), resolve the
LOCAL pinned artifact dir, verify its sha256, then build the model with
``local_files_only=True``. No heavy import at module load — only inside ``load()``.
"""
from __future__ import annotations

from typing import Any, Dict, List

from znyx_inference.runners._artifacts import resolve_artifact_dir, verify_pinned
from znyx_inference.runners.base import InferOutput, Runner, RunnerUnavailable


class HeavyRunner(Runner):
    runner_kind = "heavy"

    def __init__(self, task: str, spec: Dict[str, Any]):
        self.task = task
        self.spec = spec or {}
        self.model_id = self.spec.get("model_id") or task
        self.revision = self.spec.get("revision") or "main"
        self.threshold = float(self.spec.get("threshold", 0.5))
        self.model_version = f"{self.model_id}@{self.revision}"
        self._ready = False
        self._artifact_dir = ""

    # subclasses implement these
    def _import_stack(self) -> None:
        """Lazy-import the ML deps; raise RunnerUnavailable if absent."""
        raise NotImplementedError

    def _build(self, path: str) -> None:
        """Load model objects from the verified local ``path``."""
        raise NotImplementedError

    def load(self) -> None:
        self._import_stack()                                   # deps present?
        self._artifact_dir = resolve_artifact_dir(self.spec)   # local only
        verify_pinned(self._artifact_dir, self.spec.get("sha256"))  # exists + sha256 ok
        self._build(self._artifact_dir)
        self._ready = True

    def _decision(self, unsafe_prob: float) -> str:
        if unsafe_prob >= self.threshold:
            return "BLOCK"
        if unsafe_prob >= self.threshold / 2:
            return "WARN"
        return "ALLOW"

    def _output(self, unsafe_prob: float, label_scores: Dict[str, float] | None = None) -> InferOutput:
        unsafe_prob = max(0.0, min(1.0, float(unsafe_prob)))
        # Defensively clamp every label score into the contract's [0,1] range so a
        # runner's raw value (e.g. a negative cosine) can't turn a successful inference
        # into a response-validation 500. Runners should still emit probabilities here.
        if label_scores:
            label_scores = {k: round(max(0.0, min(1.0, float(v))), 4)
                            for k, v in label_scores.items()}
        return InferOutput(
            decision=self._decision(unsafe_prob),
            risk_score=int(round(unsafe_prob * 100)),
            confidence=round(unsafe_prob, 4),
            calibrated_score=round(unsafe_prob, 4),
            label_scores=label_scores,
            threshold=self.threshold,
        )

    def infer_batch(self, texts: List[str]) -> List[InferOutput]:
        raise NotImplementedError


def require(import_fn, what: str):
    """Run a lazy import, converting ImportError into RunnerUnavailable so a missing ML
    stack degrades the task to 'unavailable' instead of crashing the service."""
    try:
        return import_fn()
    except ImportError as exc:  # noqa: PERF203
        raise RunnerUnavailable(f"{what} not installed (inference image only): {exc}") from exc
