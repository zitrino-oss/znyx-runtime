"""Quality scorer - orchestrates all quality metric modules.

Runs enabled quality metrics on output text and produces a QualityReport.
Quality scores are informational only (never block). They feed into traces,
analytics, and the UI.

Supports two evaluation modes:
- **Deterministic** (default): rule-based scorers using NLP heuristics.
- **Judge-style** (optional): delegates scoring to an LLM "judge" endpoint
  for richer evaluation of groundedness, relevance, and safety.
"""
import logging
import os
from datetime import datetime, timezone
from typing import Callable, Dict, Any, List, Optional

from app.shared.core.models import QualityReport, QualityScore
from app.shared.engine.quality.groundedness import score_groundedness
from app.shared.engine.quality.relevance import score_relevance
from app.shared.engine.quality.coherence import score_coherence
from app.shared.engine.quality.fluency import score_fluency
from app.shared.engine.quality.intent_resolution import score_intent_resolution
from app.shared.engine.quality.task_adherence import score_task_adherence
from app.shared.engine.quality.tool_accuracy import score_tool_accuracy

logger = logging.getLogger(__name__)

# Registry for judge-style evaluator functions
# Each function signature: (input_text, output_text, metadata) -> QualityScore
_judge_evaluators: Dict[str, Callable] = {}

_DEFAULT_METRICS = [
    "groundedness", "relevance", "coherence", "fluency",
    "intent_resolution", "task_adherence",
]

_DEFAULT_WEIGHTS = {
    "groundedness": 0.25,
    "relevance": 0.25,
    "coherence": 0.15,
    "fluency": 0.10,
    "intent_resolution": 0.15,
    "task_adherence": 0.10,
    "tool_call_accuracy": 0.0,  # only in tool context
}


def register_judge_evaluator(metric: str, evaluator: Callable) -> None:
    """Register a judge-style evaluator function for a quality metric.

    Judge evaluators are called when ``judge_mode`` is enabled in the
    QualityScorer config.  They take precedence over deterministic scorers
    for the metrics they cover.

    Args:
        metric: Metric name (e.g., ``"groundedness"``, ``"safety"``).
        evaluator: ``(input_text: str, output_text: str, metadata: dict) -> QualityScore``.
    """
    _judge_evaluators[metric] = evaluator
    logger.info(f"Registered judge evaluator for metric: {metric}")


def unregister_judge_evaluator(metric: str) -> None:
    """Remove a judge-style evaluator."""
    _judge_evaluators.pop(metric, None)


class QualityScorer:
    """Orchestrates quality metric scoring on AI output.

    When ``judge_mode`` is enabled in config, registered judge evaluators
    are used in preference to deterministic scorers.  If no judge evaluator
    is registered for a metric, the deterministic scorer is used as fallback.
    """

    def __init__(self, config: Dict[str, Any]):
        self.enabled = config.get("enabled", False)
        self.metrics: List[str] = config.get("metrics", _DEFAULT_METRICS)
        self.weights: Dict[str, float] = {**_DEFAULT_WEIGHTS, **config.get("weights", {})}
        self.judge_mode: bool = config.get("judge_mode", False)
        self.judge_endpoint: Optional[str] = config.get(
            "judge_endpoint", os.getenv("QUALITY_JUDGE_ENDPOINT"),
        )
        self.judge_model: Optional[str] = config.get(
            "judge_model", os.getenv("QUALITY_JUDGE_MODEL"),
        )

    def score(
        self,
        input_text: str,
        output_text: str,
        metadata: Optional[Dict[str, Any]] = None,
        tool_context: Optional[Dict[str, Any]] = None,
    ) -> QualityReport:
        """Run all enabled quality metrics and return a QualityReport."""
        if not self.enabled:
            return QualityReport()

        metadata = metadata or {}
        scores: List[QualityScore] = []

        for metric in self.metrics:
            try:
                qs = self._score_metric(metric, input_text, output_text, metadata, tool_context)
                if qs:
                    scores.append(qs)
            except Exception:
                logger.debug(f"Quality metric '{metric}' failed", exc_info=True)

        # Compute weighted overall score
        overall = self._weighted_average(scores)

        return QualityReport(
            scores=scores,
            overall_score=round(overall, 3),
            evaluated_at=datetime.now(timezone.utc).isoformat(),
        )

    def _score_metric(
        self,
        metric: str,
        input_text: str,
        output_text: str,
        metadata: Dict[str, Any],
        tool_context: Optional[Dict[str, Any]],
    ) -> Optional[QualityScore]:
        """Dispatch to the correct scorer module.

        If ``judge_mode`` is enabled and a judge evaluator is registered for
        the metric, it is used instead of the deterministic scorer.
        """
        # Try judge evaluator first if judge_mode is enabled
        if self.judge_mode and metric in _judge_evaluators:
            try:
                result = _judge_evaluators[metric](input_text, output_text, metadata)
                if result:
                    logger.debug(f"Judge evaluator scored '{metric}': {result.score}")
                    return result
            except Exception:
                logger.warning(f"Judge evaluator for '{metric}' failed, falling back to deterministic", exc_info=True)

        # Deterministic fallback
        if metric == "groundedness":
            return score_groundedness(output_text, metadata)
        if metric == "relevance":
            return score_relevance(input_text, output_text, metadata)
        if metric == "coherence":
            return score_coherence(output_text, metadata)
        if metric == "fluency":
            return score_fluency(output_text, metadata)
        if metric == "intent_resolution":
            return score_intent_resolution(input_text, output_text, metadata)
        if metric == "task_adherence":
            return score_task_adherence(input_text, output_text, metadata)
        if metric == "tool_call_accuracy":
            return score_tool_accuracy(tool_context, metadata)
        logger.warning(f"Unknown quality metric: {metric}")
        return None

    def _weighted_average(self, scores: List[QualityScore]) -> float:
        """Compute weighted average of quality scores."""
        if not scores:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0
        for qs in scores:
            w = self.weights.get(qs.metric, 0.1)
            weighted_sum += qs.score * w
            total_weight += w

        return weighted_sum / total_weight if total_weight > 0 else 0.0
