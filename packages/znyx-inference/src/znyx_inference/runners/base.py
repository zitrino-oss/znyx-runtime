"""Runner protocol + the dependency-free StubRunner.

A Runner turns a batch of texts into the confidence contract for one task. Heavy
runners lazy-import their ML stack in ``load()``; the StubRunner needs nothing and lets
the whole service (contract / cache / batching / registry / 429) run and be tested with
zero ML dependencies installed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class RunnerUnavailable(Exception):
    """Raised by ``load()`` when a runner's heavy deps or pinned model artifacts are
    absent. The registry catches this and marks the task unavailable (503), rather than
    crashing the service."""


@dataclass
class InferOutput:
    """A runner's per-item output — the model-native part of the confidence contract."""
    decision: str
    risk_score: int
    confidence: Optional[float] = None
    label_scores: Optional[Dict[str, float]] = None
    calibrated_score: Optional[float] = None
    threshold: Optional[float] = None


class Runner:
    """Base runner. Subclasses set ``task``/``model_version`` and implement
    ``load`` + ``infer_batch``. ``infer_batch`` MUST return one output per input, in
    order (the batcher relies on positional alignment)."""
    task: str = "generic"
    runner_kind: str = "base"
    model_version: str = "base@v0"

    def load(self) -> None:
        """Prepare the runner (load weights, verify sha256). May raise RunnerUnavailable."""

    def infer_batch(self, texts: List[str], params: Optional[Dict[str, Any]] = None) -> List[InferOutput]:
        raise NotImplementedError


@dataclass
class StubRunner(Runner):
    """Deterministic, dependency-free runner. Keyword/heuristic scoring so the service is
    fully functional (and testable) without torch/transformers. NOT for production
    detection quality — it exists so the infra runs anywhere and as a safe default."""
    task: str = "prompt_injection"
    threshold: float = 0.5
    model_version: str = "stub@v1"
    runner_kind: str = "stub"
    # task → list of (regex, weight) signals contributing to the risk probability.
    _signals: Dict[str, List] = field(default_factory=dict)

    # Heuristic signal sets per task. Weights sum-capped at 1.0.
    _DEFAULT_SIGNALS = {
        "prompt_injection": [
            (r"ignore (?:all |the )?(?:previous|prior|above) instructions", 0.7),
            (r"\bdisregard\b.*\b(?:rules?|instructions?|policy)\b", 0.6),
            (r"\bdeveloper mode\b|\bDAN\b", 0.6),
            (r"reveal|show|print.*(?:system prompt|instructions)", 0.5),
            (r"</?(?:system|assistant)>", 0.4),
        ],
        "toxicity": [
            (r"\b(?:idiot|stupid|hate you|kill yourself|moron)\b", 0.7),
            (r"\b(?:slur|racist|sexist)\b", 0.6),
        ],
        "jailbreak": [
            (r"pretend you are|act as if|no restrictions|without any filter", 0.6),
            (r"ignore (?:all |the )?(?:previous|prior) instructions", 0.7),
        ],
    }

    def __post_init__(self):
        if not self._signals:
            self._signals = {
                t: [(re.compile(p, re.IGNORECASE), w) for p, w in sigs]
                for t, sigs in self._DEFAULT_SIGNALS.items()
            }

    def _score_one(self, text: str) -> InferOutput:
        sigs = self._signals.get(self.task, [])
        prob = 0.0
        hits: Dict[str, float] = {}
        for pattern, weight in sigs:
            if pattern.search(text or ""):
                prob = min(1.0, prob + weight)
                hits[pattern.pattern[:24]] = weight
        risk = int(round(prob * 100))
        decision = "BLOCK" if prob >= self.threshold else ("WARN" if prob > 0 else "ALLOW")
        return InferOutput(
            decision=decision,
            risk_score=risk,
            confidence=round(prob, 4),
            calibrated_score=round(prob, 4),
            label_scores={"unsafe": round(prob, 4), "safe": round(1.0 - prob, 4)},
            threshold=self.threshold,
        )

    def infer_batch(self, texts: List[str], params: Optional[Dict[str, Any]] = None) -> List[InferOutput]:
        return [self._score_one(t) for t in texts]
