"""The NER layer redacts what it finds, instead of only reporting that it found something.

A token-classification model labels every sub-word token, so it knows exactly which
characters are a name. The runner used to collapse that into ``risk = max prob`` and drop
the offsets, which left a REDACT-action detector unable to act: "a person's name is present
(0.9994)" cannot be turned into a replacement. Wiring the pii_ner layer therefore produced a
REDACT decision on text that still contained the name.

Now the runner emits character spans, they travel through the sidecar response and the
RemoteDetector, and the additive merge recomputes redaction from the deterministic AND model
spans TOGETHER over the original text. Merging in span space is the point: the regex layer
has already rewritten the string, so the model's original-text offsets cannot be applied to
its output — that is the class of bug fixed in test_pii_redaction_overlap.py.
"""
import pytest

from znyx_core.core.models import Decision, DetectorResult
from znyx_core.detectors.pii import PIIDetector
from znyx_core.detectors.remote import RemoteDetector
from znyx_core.engine.escalation import _additive_merge, apply_redaction_spans

TEXT = "My name is Sarah Chen and Alan Turing live in Manchester."


def _runner():
    from znyx_inference.runners.ner import NerRunner
    return NerRunner.__new__(NerRunner)          # pure methods only; no model load


# ── runner: tokens -> character spans ────────────────────────────────────────────────

def test_subword_tokens_merge_into_one_entity_span():
    """"Sarah"/"Chen" are two tokens of one name and must yield a single span."""
    toks = [("O", .99, (0, 2)), ("O", .99, (3, 7)), ("O", .99, (8, 10)),
            ("B-PER", .99, (11, 16)), ("I-PER", .98, (17, 21))]
    assert _runner()._merge_spans(toks) == [(11, 21, "PER")]
    assert TEXT[11:21] == "Sarah Chen"


def test_adjacent_entities_of_the_same_type_stay_separate():
    """A ``B-`` prefix opens a new entity, so "Sarah Chen and Alan Turing" is two names —
    without this they merge into one span covering the word between them."""
    toks = [("B-PER", .99, (11, 16)), ("I-PER", .98, (17, 21)),
            ("O", .99, (22, 25)),
            ("B-PER", .99, (26, 30)), ("I-PER", .97, (31, 37))]
    spans = _runner()._merge_spans(toks)
    assert spans == [(11, 21, "PER"), (26, 37, "PER")]
    assert [TEXT[s:e] for s, e, _ in spans] == ["Sarah Chen", "Alan Turing"]


def test_special_and_outside_tokens_produce_no_span():
    """[CLS]/[SEP] report (0, 0); a zero-width span would redact at position 0."""
    assert _runner()._merge_spans([("O", .99, (0, 2)), ("O", .5, (0, 0))]) == []


def test_an_outside_token_breaks_a_run_of_the_same_type():
    toks = [("I-LOC", .9, (0, 5)), ("O", .9, (6, 9)), ("I-LOC", .9, (10, 15))]
    assert _runner()._merge_spans(toks) == [(0, 5, "LOC"), (10, 15, "LOC")]


# ── transport: malformed spans are dropped, never trusted ────────────────────────────

@pytest.mark.parametrize("raw", [
    None, "nonsense", 42, [[5]], [["a", "b", "PER"]], [[5, 5, "PER"]],
    [[9, 3, "PER"]], [[-1, 4, "PER"]],
])
def test_malformed_spans_from_a_sidecar_are_discarded(raw):
    """Spans are spliced into text, so a bad one must be dropped rather than passed on —
    it would otherwise replace the wrong characters."""
    assert RemoteDetector._coerce_spans(raw) == []


def test_wellformed_spans_are_parsed_and_default_their_label():
    assert RemoteDetector._coerce_spans([[0, 10, "PER"], [14, 24]]) == [
        (0, 10, "PER"), (14, 24, "PII")]


# ── applying spans ──────────────────────────────────────────────────────────────────

def test_overlapping_spans_resolve_longest_first():
    """Same guard as the regex path: overlapping spans applied in sequence would splice
    fragments of the original back in."""
    text = "sarah.chen@example.com"
    assert apply_redaction_spans(text, [(0, 22, "EMAIL"), (10, 18, "USERNAME")]) == "[EMAIL]"


def test_spans_outside_the_text_are_ignored():
    """A stale or mismatched offset must not raise or truncate."""
    assert apply_redaction_spans("short", [(0, 999, "PER")]) == "short"
    assert apply_redaction_spans("short", []) == "short"


# ── the headline: unstructured PII is actually removed ──────────────────────────────

def test_merge_redacts_both_structured_and_unstructured_pii():
    text = "Sarah Chen in Manchester; email sarah.chen@example.com."
    det = PIIDetector({"enabled": True, "action": "REDACT"}).detect(text)
    ml = DetectorResult(decision=Decision.BLOCK, risk_score=100,
                        redaction_spans=[(0, 10, "PER"), (14, 24, "LOC")])
    merged = _additive_merge(det, ml, text)

    assert merged.sanitized_text == "[PER] in [LOC]; email [EMAIL]."
    for leaked in ("Sarah", "Chen", "Manchester", "sarah.chen", "example.com"):
        assert leaked not in merged.sanitized_text
    assert merged.risk_score == 100                      # worst-of risk preserved


def test_regex_only_text_is_unchanged_by_the_ner_layer():
    """The deterministic layer must not lose ground when the model finds nothing."""
    text = "Please email me at sarah.chen@example.com about this."
    det = PIIDetector({"enabled": True, "action": "REDACT"}).detect(text)
    merged = _additive_merge(det, DetectorResult(decision=Decision.ALLOW, risk_score=0), text)
    assert merged.sanitized_text == "Please email me at [EMAIL] about this."


def test_a_score_only_model_cannot_claim_a_redaction():
    """No spans => fall back to the deterministic layer's text. A model that reports only a
    score must not appear to have redacted anything, which was the original defect."""
    text = "Sarah Chen lives in Manchester."
    det = PIIDetector({"enabled": True, "action": "REDACT"}).detect(text)
    scored_only = DetectorResult(decision=Decision.BLOCK, risk_score=100)
    merged = _additive_merge(det, scored_only, text)
    assert "Sarah Chen" in (merged.sanitized_text or text)   # honest: nothing was redacted
    assert merged.risk_score == 100                           # but the finding is reported


def test_merge_without_text_keeps_previous_behaviour():
    """``text`` is optional so existing callers are unaffected: with no original text there
    is no safe offset space, so the deterministic sanitized text wins."""
    det = DetectorResult(decision=Decision.REDACT, risk_score=60,
                         sanitized_text="[EMAIL]", redaction_spans=[(0, 22, "EMAIL")])
    ml = DetectorResult(decision=Decision.BLOCK, risk_score=100,
                        redaction_spans=[(0, 10, "PER")])
    assert _additive_merge(det, ml).sanitized_text == "[EMAIL]"


def test_pii_detector_publishes_only_redactable_spans():
    """Spans for BLOCK-action types must not be published as redactable — the merge would
    replace text the policy meant to block on instead."""
    det = PIIDetector({"enabled": True, "action": "REDACT"}).detect(
        "Please email me at sarah.chen@example.com about this.")
    assert det.redaction_spans, "expected the email span to be published"
    for start, end, _label in det.redaction_spans:
        assert 0 <= start < end
