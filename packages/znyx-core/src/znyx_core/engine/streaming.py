"""Streaming evaluator - evaluate text chunks with a sliding window.

Buffers incoming chunks and runs detectors when the window fills or on
explicit flush. Designed for Server-Sent Events (SSE) streaming responses
from LLMs where text arrives incrementally.

Text is never forwarded before it has been evaluated: ``push`` releases only
the part of a window that no later window will re-examine, and ``flush``
evaluates whatever is still buffered before releasing it.

Usage:
    evaluator = StreamingEvaluator(policy, window_size=100, overlap=20)
    for chunk in llm_stream:
        for event in evaluator.push(chunk):
            yield event   # SSE event dict
    for event in evaluator.flush():
        yield event       # trailing chunk/guardrail/block events, then `done`
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
    - ``{"event": "guardrail", "data": {...}}`` when a window is evaluated
    - ``{"event": "chunk", "data": {...}}`` carrying text that window cleared
    - ``{"event": "block", "data": {...}}`` if a BLOCK decision is reached

    A ``chunk`` event is only ever emitted *after* the window covering its text
    has been evaluated and allowed, and always after that window's
    ``guardrail`` event, so a consumer that stops on a non-ALLOW verdict never
    renders the text the verdict rejected. The overlap tail of each window is
    withheld until the following window clears it, so a pattern straddling a
    window boundary cannot leak its first half before the window that catches
    it runs.

    ``flush()`` evaluates any remaining buffered text, releases it if it is
    clean, and returns those trailing events followed by the ``done`` summary.
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
        self._released = ""
        self._release_count = 0
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

    @property
    def released_text(self) -> str:
        """The text released to the caller so far, i.e. everything a consumer
        following the ``chunk`` events has been cleared to show. Always an
        evaluated-and-allowed prefix of ``_full_text``."""
        return self._released

    def push(self, chunk: str) -> List[Dict[str, Any]]:
        """Push a text chunk. Returns list of SSE events.

        The chunk is buffered, not forwarded. Whenever the buffer holds a full
        window that window is evaluated, and only on an ALLOW is the part of it
        the next window will not re-examine released as a ``chunk`` event. The
        overlap tail stays buffered so a boundary-straddling pattern cannot
        escape ahead of the window that detects it. On a BLOCK nothing further
        is released; the buffered remainder is dropped with the stream.
        """
        if self._blocked:
            return [{"event": "block", "data": {"reason": "Stream blocked by previous window evaluation"}}]

        self._buffer += chunk
        self._full_text += chunk
        self._chunk_count += 1

        events: List[Dict[str, Any]] = []

        # Evaluate whole windows as they become available. `advance` is the
        # prefix this window is solely responsible for: the overlap tail is left
        # in the buffer for the next window and released only once that clears.
        # self.overlap is clamped to window_size // 2 in __init__, so advance is
        # always >= half a window and the loop always makes progress.
        advance = self.window_size - self.overlap
        while len(self._buffer) >= self.window_size and not self._blocked:
            events.extend(self._evaluate_window(self._buffer[:self.window_size]))
            if self._blocked:
                break
            events.append(self._release(self._buffer[:advance]))
            self._buffer = self._buffer[advance:]

        return events

    def flush(self) -> List[Dict[str, Any]]:
        """Evaluate whatever is still buffered, then return the trailing events.

        Returns the events for the final window (its ``guardrail``/``block``
        verdict, plus a ``chunk`` carrying the tail if it was allowed) followed
        by the ``done`` summary. The block verdict used to be computed here and
        then discarded, which let a stream shorter than one window be forwarded
        in full while its BLOCK never reached the caller.
        """
        events: List[Dict[str, Any]] = []

        if self._buffer and not self._blocked:
            tail = self._buffer
            self._buffer = ""
            events.extend(self._evaluate_window(tail))
            if not self._blocked:
                events.append(self._release(tail))

        events.append({
            "event": "done",
            "data": {
                "total_chunks": self._chunk_count,
                "windows_evaluated": self._window_count,
                "final_decision": "BLOCK" if self._blocked else "ALLOW",
                "max_risk_score": self._max_risk,
                "total_rule_hits": len(self._all_hits),
                "total_latency_ms": self._total_latency_ms,
                "full_text_length": len(self._full_text),
                # What actually reached the caller. On ALLOW this equals
                # full_text_length; on BLOCK it is the clean prefix released
                # before the offending window, and the gap is the withheld text.
                "released_text_length": len(self._released),
            },
        })
        return events

    def _release(self, text: str) -> Dict[str, Any]:
        """Build the ``chunk`` event for evaluated, allowed text.

        The only place a ``chunk`` event is created, so every byte the caller
        receives has passed a detector. ``chunk_index`` counts releases (not
        ``push`` calls): windows do not line up with chunk boundaries, so it is
        the release order that tells a consumer how to reassemble the text.
        """
        self._released += text
        self._release_count += 1
        return {
            "event": "chunk",
            "data": {"text": text, "chunk_index": self._release_count},
        }

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
        }

        if decision == Decision.BLOCK:
            self._blocked = True
            # No text_preview on a BLOCK. The verdict event must not carry the
            # very text the stream is withholding: a proxy that forwards or logs
            # every SSE event would otherwise re-open the channel `chunk`
            # suppression just closed. Callers correlate on window_index and
            # rule_hits instead.
            events.append({"event": "block", "data": event_data})
        else:
            # Safe on an ALLOW/WARN: this text is released in the `chunk` event
            # that follows, so the preview reveals nothing extra.
            event_data["text_preview"] = (
                text[:80] + "..." if len(text) > 80 else text
            )
            events.append({"event": "guardrail", "data": event_data})

        return events
