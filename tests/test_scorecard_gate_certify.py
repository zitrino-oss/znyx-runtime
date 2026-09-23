"""The ``certify_model`` escape hatch on the scorecard enforcement gate.

Context: a model-backed detector's BLOCK/REDACT is downgraded to advisory WARN until its
policy carries a passing ``_scorecard_gate`` stamp. That gate is circular for the benchmark
that PRODUCES the stamp: the run measures the model in order to decide whether it may
enforce, so gating it records a WARN for a model that actually said BLOCK. ``certify_model``
skips the gate for exactly those runs.

Scores are unaffected either way — ``eval_metrics`` scores "flagged vs ALLOW" and WARN is
flagged — which the last test pins, because certification depends on that invariant.
"""
from znyx_core.core.models import Decision, DetectorResult
from znyx_core.engine.detector_registry import default_registry
from znyx_core.engine.orchestrator import DetectorOrchestrator

# A model-backed detector config (is_model_backed reads strategy.order), with a stamp that
# does NOT permit enforcement — the state every detector is in before it is certified.
FAILING_STAMP = {
    "enabled": True,
    "action": "BLOCK",
    "strategy": {"order": ["local_deterministic", "local_ml"]},
    "backends": {"local_ml": {"task": "toxicity", "model_id": "unitary/toxic-bert",
                              "revision": "main"}},
    "_scorecard_gate": {"enforcement_passed": False},
}
UNSTAMPED = {k: v for k, v in FAILING_STAMP.items() if k != "_scorecard_gate"}
PASSING_STAMP = {**FAILING_STAMP, "_scorecard_gate": {"enforcement_passed": True}}


def _orch():
    return DetectorOrchestrator(default_registry)


def _block():
    return DetectorResult(decision=Decision.BLOCK, risk_score=100)


def test_gate_downgrades_block_to_warn_by_default():
    """Production behaviour, unchanged: a failing stamp pins BLOCK to advisory WARN."""
    out = _orch()._verify_and_apply_scorecard_gate("toxicity", FAILING_STAMP, _block())
    assert out.decision == Decision.WARN


def test_certify_model_preserves_the_models_block():
    """A certification run records what the model decided, not the gated downgrade."""
    out = _orch()._verify_and_apply_scorecard_gate(
        "toxicity", FAILING_STAMP, _block(), certify_model=True)
    assert out.decision == Decision.BLOCK


def test_certify_model_preserves_block_when_unstamped():
    """The fail-closed no-stamp path is skipped too — an uncertified detector is precisely
    the one being benchmarked, so it has no stamp yet."""
    orch = _orch()
    assert orch._verify_and_apply_scorecard_gate(
        "toxicity", UNSTAMPED, _block()).decision == Decision.WARN
    assert orch._verify_and_apply_scorecard_gate(
        "toxicity", UNSTAMPED, _block(), certify_model=True).decision == Decision.BLOCK


def test_redact_is_preserved_under_certify():
    """REDACT is the other enforcing action the gate downgrades (e.g. the pii NER layer)."""
    result = DetectorResult(decision=Decision.REDACT, risk_score=80)
    out = _orch()._verify_and_apply_scorecard_gate(
        "pii", FAILING_STAMP, result, certify_model=True)
    assert out.decision == Decision.REDACT


def test_passing_stamp_enforces_with_or_without_certify():
    """certify_model only ever skips a downgrade; it never changes an already-enforcing
    detector, so a certified model behaves identically in both paths."""
    orch = _orch()
    for certify in (False, True):
        out = orch._verify_and_apply_scorecard_gate(
            "toxicity", PASSING_STAMP, _block(), certify_model=certify)
        assert out.decision == Decision.BLOCK


def test_certify_does_not_leak_across_calls_on_a_shared_orchestrator():
    """The bypass is per call, never instance state. GuardrailsEvaluator builds ONE
    orchestrator and shares it across every request in the process — and in the runtime the
    benchmark worker and tenant traffic live in that same process. Were the flag held on the
    instance, a tenant request evaluated during a benchmark would lose its gate."""
    orch = _orch()
    assert orch._verify_and_apply_scorecard_gate(
        "toxicity", FAILING_STAMP, _block(), certify_model=True).decision == Decision.BLOCK
    # interleaved tenant-shaped call: still downgraded
    assert orch._verify_and_apply_scorecard_gate(
        "toxicity", FAILING_STAMP, _block()).decision == Decision.WARN
    # and again after, to catch a flag set on first use
    assert orch._verify_and_apply_scorecard_gate(
        "toxicity", FAILING_STAMP, _block(), certify_model=True).decision == Decision.BLOCK


def test_certify_model_is_not_reachable_from_request_or_policy():
    """The bypass must stay a Python keyword argument. If it ever becomes an
    EvaluationRequest field or a policy key, any caller could switch off its own
    enforcement gate — so pin both surfaces."""
    from znyx_core.core.models import EvaluationRequest
    assert "certify_model" not in EvaluationRequest.model_fields
    # a policy-supplied key must not be honoured: same config, plus the flag as data
    hostile = {**FAILING_STAMP, "certify_model": True}
    out = _orch()._verify_and_apply_scorecard_gate("toxicity", hostile, _block())
    assert out.decision == Decision.WARN


def test_warn_scores_as_flagged_so_metrics_are_gate_independent():
    """Why certify_model changes no score: the positive class is "not ALLOW", so a gated
    WARN and an ungated BLOCK land in the same bucket. Certification relies on this, and
    nothing else pins it — if this breaks, a detector could never earn enforcement."""
    from znyx_core.engine import eval_metrics

    gated = eval_metrics.classification_metrics({"BLOCK": {"WARN": 56, "ALLOW": 7},
                                                 "ALLOW": {"WARN": 8, "ALLOW": 76}})
    ungated = eval_metrics.classification_metrics({"BLOCK": {"BLOCK": 56, "ALLOW": 7},
                                                   "ALLOW": {"BLOCK": 8, "ALLOW": 76}})
    for key in ("precision", "recall", "f1", "fp_rate"):
        assert gated[key] == ungated[key], key
    assert gated["f1"] is not None and gated["f1"] > 0.8
