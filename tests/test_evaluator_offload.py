"""Blocking work stays off the event loop.

Quality scoring can call the NLI sidecar and LLM judges, and ask_human
remediation makes an fsync'd spool write. Both used to run inside the async
handler, stalling every other request in the process while they waited.
"""
import asyncio
import threading
import time

import pytest

import znyx_core.engine.evaluator as evaluator_mod
from znyx_core.core.models import Decision, EvaluationRequest
from znyx_core.engine.evaluator import GuardrailsEvaluator
from znyx_core.engine.quality_scorer import QualityScorer
from znyx_core.engine.remediation import RemediationHandler

QUALITY_POLICY = {"quality_scoring": {"enabled": True}}


def _request(text="The answer is forty two."):
    return EvaluationRequest(request_id="r1", tenant_id="t", app_id="a", text=text,
                             metadata={"original_input": "What is the answer?"})


def test_quality_scoring_runs_on_a_detector_worker(monkeypatch):
    seen = {}
    real = QualityScorer.score

    def spy(self, *a, **k):
        seen["thread"] = threading.current_thread().name
        return real(self, *a, **k)

    monkeypatch.setattr(QualityScorer, "score", spy)
    response = asyncio.run(GuardrailsEvaluator().evaluate(
        _request(), context="output", policy=QUALITY_POLICY))
    assert response.quality is not None
    assert seen["thread"].startswith("detector")


def test_slow_quality_scoring_does_not_stall_the_event_loop(monkeypatch):
    def slow(self, *a, **k):
        time.sleep(0.4)
        return None

    monkeypatch.setattr(QualityScorer, "score", slow)

    async def main():
        gaps = []

        async def ticker():
            last = time.perf_counter()
            for _ in range(30):
                await asyncio.sleep(0.01)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now

        await asyncio.gather(
            GuardrailsEvaluator().evaluate(_request(), context="output", policy=QUALITY_POLICY),
            ticker(),
        )
        return max(gaps)

    assert asyncio.run(main()) < 0.15


def test_quality_scoring_past_the_deadline_is_skipped_not_failed(monkeypatch):
    monkeypatch.setattr(evaluator_mod, "_DEADLINE_S", 0.05)
    real = QualityScorer.score

    def slow(self, *a, **k):
        time.sleep(0.3)
        return real(self, *a, **k)   # a real score, so only the deadline can drop it

    monkeypatch.setattr(QualityScorer, "score", slow)
    response = asyncio.run(GuardrailsEvaluator().evaluate(
        _request(), context="output", policy=QUALITY_POLICY))
    assert response.decision == Decision.ALLOW
    assert response.quality is None


def test_remediation_runs_off_the_event_loop(monkeypatch, tmp_path):
    seen = {}
    real = RemediationHandler.apply

    def spy(self, *a, **k):
        try:
            asyncio.get_running_loop()
            seen["on_loop"] = True
        except RuntimeError:
            seen["on_loop"] = False
        return real(self, *a, **k)

    monkeypatch.setattr(RemediationHandler, "apply", spy)
    monkeypatch.setenv("ZNYX_ASK_HUMAN_SPOOL", str(tmp_path / "ask-human.spool"))
    policy = {"pii": {"enabled": True, "on_fail": "ask_human"}}
    response = asyncio.run(GuardrailsEvaluator().evaluate(
        _request("Please email jane.doe" + "@example.com today."), context="output", policy=policy))
    assert response.decision == Decision.BLOCK
    assert seen["on_loop"] is False


def test_allow_decisions_skip_the_remediation_thread_hop(monkeypatch):
    calls = []
    monkeypatch.setattr(asyncio, "to_thread", lambda *a, **k: calls.append(a) or pytest.fail("hop"))
    response = asyncio.run(GuardrailsEvaluator().evaluate(
        _request("A perfectly ordinary answer."), context="output", policy={"pii": {"enabled": True}}))
    assert response.decision == Decision.ALLOW
    assert calls == []
