"""Streaming evaluator - evaluate text chunks with a sliding window.

Buffers incoming chunks and runs detectors when the window fills or on
explicit flush. Designed for Server-Sent Events (SSE) streaming responses
from LLMs where text arrives incrementally.

Usage:
    evaluator = StreamingEvaluator(policy, window_size=100, overlap=20)
    for chunk in llm_stream:
        events = evaluator.push(chunk)
        for event in events:
            yield event   # SSE event dict
    final = evaluator.flush()
    yield final
"""
import logging
import time
from typing import Any, Dict, List, Optional

from znyx_core.core.models import (
    Decision, EvaluationRequest, RuleHit,
)
from znyx_core.engine.detector_registry import default_registry
from znyx_core.engine.orchestrator import DetectorOrchestrator

logger = logging.getLogger(__name__)


class StreamingEvaluator:
    """Sliding-window streaming evaluator.

    Buffers text chunks and runs the detector pipeline when enough text
    accumulates (``window_size`` characters). A configurable ``overlap``
    ensures boundary tokens between windows are re-evaluated.

    Each ``push()`` call returns a list of SSE event dicts:
    - ``{"event": "chunk", "data": {...}}`` for each chunk forwarded
    - ``{"event": "guardrail", "data": {...}}`` when a window is evaluated
    - ``{"event": "block", "data": {...}}`` if a BLOCK decision is reached

    ``flush()`` evaluates any remaining buffered text and returns a final
    summary event.
    """

    def __init__(
        self,
        policy: Dict[str, Any],
        context: str = "output",
        window_size: int = 200,
        overlap: int = 40,
        request: Optional[EvaluationRequest] = None,
    ):
        self.policy = policy
        self.context = context
        self.window_size = max(50, window_size)
        self.overlap = min(overlap, self.window_size // 2)

        self._buffer = ""
        self._full_text = ""
        self._chunk_count = 0
        self._window_count = 0
        self._blocked = False
        self._all_hits: List[RuleHit] = []
        self._max_risk = 0
        self._total_latency_ms = 0

        self._request = request or EvaluationRequest(
            request_id="stream-0", tenant_id="stream", app_id="stream", text="",
        )
        self._orchestrator = DetectorOrchestrator(default_registry)

    @property
    def is_blocked(self) -> bool:
        return self._blocked

    def push(self, chunk: str) -> List[Dict[str, Any]]:
        """Push a text chunk. Returns list of SSE events."""
        if self._blocked:
            return [{"event": "block", "data": {"reason": "Stream blocked by previous window evaluation"}}]

        self._buffer += chunk
        self._full_text += chunk
        self._chunk_count += 1

        events: List[Dict[str, Any]] = []

        # Forward the chunk
        events.append({
            "event": "chunk",
            "data": {"text": chunk, "chunk_index": self._chunk_count},
        })

        # Evaluate when buffer exceeds window size
        while len(self._buffer) >= self.window_size and not self._blocked:
            window_text = self._buffer[:self.window_size]
            self._buffer = self._buffer[self.window_size - self.overlap:]
            eval_events = self._evaluate_window(window_text)
            events.extend(eval_events)

        return events

    def flush(self) -> Dict[str, Any]:
        """Flush remaining buffer and return final summary event."""
        events = []
        if self._buffer and not self._blocked:
            eval_events = self._evaluate_window(self._buffer)
            events.extend(eval_events)
            self._buffer = ""

        summary = {
            "event": "done",
            "data": {
                "total_chunks": self._chunk_count,
                "windows_evaluated": self._window_count,
                "final_decision": "BLOCK" if self._blocked else "ALLOW",
                "max_risk_score": self._max_risk,
                "total_rule_hits": len(self._all_hits),
                "total_latency_ms": self._total_latency_ms,
                "full_text_length": len(self._full_text),
            },
        }
        return summary

    def _evaluate_window(self, text: str) -> List[Dict[str, Any]]:
        """Run detectors on a single window of text."""
        self._window_count += 1
        events = []

        t0 = time.perf_counter()
        if self.context == "input":
            orch = self._orchestrator.run_input_detectors(text, self.policy, self._request)
        else:
            orch = self._orchestrator.run_output_detectors(text, self.policy, self._request)

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        self._total_latency_ms += elapsed_ms

        # Aggregate results
        from znyx_core.core.decision import DecisionAggregator
        result = DecisionAggregator.aggregate(orch.results)

        self._max_risk = max(self._max_risk, result.risk_score)
        self._all_hits.extend(result.rule_hits)

        decision = result.decision or Decision.ALLOW

        event_data = {
            "window_index": self._window_count,
            "decision": decision.value,
            "risk_score": result.risk_score,
            "rule_hits": [{"rule_id": h.rule_id, "message": h.message} for h in result.rule_hits],
            "latency_ms": elapsed_ms,
            "text_preview": text[:80] + "..." if len(text) > 80 else text,
        }

        if decision == Decision.BLOCK:
            self._blocked = True
            events.append({"event": "block", "data": event_data})
        else:
            events.append({"event": "guardrail", "data": event_data})

        return events
