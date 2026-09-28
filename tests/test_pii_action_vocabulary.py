"""The PII detector must honour the action vocabulary the console actually writes.

The Config page labels the redaction option "Redact" but stores it as ``TRANSFORM``
(PiiDetectorConfig.tsx), and log-only as ``ALLOW``. This detector was written against
``REDACT``/``BLOCK`` only, so ``TRANSFORM`` matched neither branch:

    blocked_types stayed empty          -> not BLOCK
    _redact_pii's `== 'REDACT'` gate    -> every span skipped, text returned untouched

and the result still reported ``Decision.REDACT``. PII went out in the clear labelled as
having been removed. It stayed hidden because an untouched policy has no ``action`` key at
all and falls back to the literal ``'REDACT'`` — the bug only appeared once someone used
the dropdown, and ``BLOCK`` (the one value both sides spell alike) kept working throughout.
"""
import pytest

from znyx_core.core.models import Decision
from znyx_core.detectors.pii import PIIDetector

TEXT = "My email is test@example.com - can you check my account?"


def _detect(text=TEXT, **cfg):
    return PIIDetector({"enabled": True, **cfg}).detect(text)


@pytest.mark.parametrize("action", ["REDACT", "TRANSFORM", "transform", "Transform", "MASK"])
def test_redacting_actions_rewrite_the_text(action):
    """Every spelling of "redact" must actually remove the address."""
    res = _detect(action=action)
    assert res.decision is Decision.REDACT
    assert res.sanitized_text is not None
    assert "test@example.com" not in res.sanitized_text
    assert "[EMAIL]" in res.sanitized_text


@pytest.mark.parametrize("action", ["BLOCK", "block", "DENY"])
def test_blocking_actions_block(action):
    assert _detect(action=action).decision is Decision.BLOCK


@pytest.mark.parametrize("action", ["ALLOW", "allow", "WARN", "LOG", "NONE"])
def test_allow_reports_without_rewriting(action):
    """ALLOW is log-only: findings are reported, the text is left alone.

    It must not claim REDACT — a REDACT carrying untouched text misleads the caller and
    outranks genuine ALLOWs in the "worst wins" aggregation downstream.
    """
    res = _detect(action=action)
    assert res.decision is Decision.ALLOW
    assert res.rule_hits, "findings must still be reported under ALLOW"
    assert not res.sanitized_text


@pytest.mark.parametrize("action", ["REDCAT", "", "  ", None, 42, {"a": 1}])
def test_unrecognised_actions_fail_safe_to_redaction(action):
    """A typo must strip the PII, never emit it."""
    res = _detect(action=action)
    assert res.decision is Decision.REDACT
    assert "test@example.com" not in (res.sanitized_text or "")


def test_per_type_action_is_normalized_too():
    """A per-type TRANSFORM must redact, and a per-type BLOCK still wins over it."""
    text = "Email test@example.com and SSN 123-45-6789 here"

    res = _detect(text, action="ALLOW", types={"ssn": {"enabled": True, "action": "TRANSFORM"}})
    assert res.decision is Decision.REDACT
    assert "123-45-6789" not in res.sanitized_text
    assert "test@example.com" in res.sanitized_text, "email was set to ALLOW"

    res = _detect(text, action="TRANSFORM", types={"ssn": {"enabled": True, "action": "BLOCK"}})
    assert res.decision is Decision.BLOCK


def test_absent_action_key_still_defaults_to_redaction():
    """The pre-dropdown path that always worked must keep working."""
    res = _detect()
    assert res.decision is Decision.REDACT
    assert "[EMAIL]" in res.sanitized_text
