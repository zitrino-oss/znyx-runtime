"""PII redaction must not splice the original PII back into its own output.

``_redact_pii`` replaced spans from the end of the string forwards, which keeps earlier
offsets valid ONLY while the spans are disjoint. They are not: several patterns match inside
one another. For "sarah.chen@example.com" the collected spans were

    (  0, 22) EMAIL     'sarah.chen@example.com'
    ( 10, 18) USERNAME  '@example'

so the USERNAME replacement rewrote the string first and the EMAIL replacement then indexed
the rewritten string using offsets from the original, emitting '[EMAIL]om' — a fragment of
the live address surviving its own redaction. 'hello@acme.com' produced '[EMAIL]].com',
leaking a stray bracket as well. `pii` ships enabled with action REDACT, so this was live.
"""
import re

import pytest

from znyx_core.detectors.pii import PIIDetector


def _redact(text, **cfg):
    det = PIIDetector({"enabled": True, "action": "REDACT", **cfg})
    return det.detect(text).sanitized_text or text


# The exact strings that leaked, with what they used to emit.
@pytest.mark.parametrize("text,was", [
    ("sarah.chen@example.com", "[EMAIL]om"),
    ("bob@test.co.uk", "[EMAIL]co.uk"),
    ("user@mail.example.org", "[EMAIL]e.org"),
    ("admin@acme.co.uk", "[EMAIL]co.uk"),
    ("j.doe@sub.domain.example.com", "[EMAIL]le.com"),
    ("contact us at hello@acme.com please", "contact us at [EMAIL]].com please"),
])
def test_email_redaction_leaves_no_fragment(text, was):
    out = _redact(text)
    assert out != was, "regression: the pre-fix leaky output is back"
    # nothing of the original address may survive
    local, _, domain = text.partition("@")
    for piece in (local.split()[-1], domain.rstrip(" .")):
        assert piece not in out, f"leaked {piece!r} in {out!r}"
    # and no fragment may be glued to a redaction tag
    assert not re.search(r"\][A-Za-z0-9.\-]", out), f"fragment after tag in {out!r}"
    assert "]]" not in out and "][" not in out, f"stray bracket in {out!r}"


def test_overlapping_spans_collapse_to_the_widest_match():
    """Longest-match-wins: the EMAIL span must swallow the USERNAME sub-match rather than
    both being applied."""
    assert _redact("sarah.chen@example.com") == "[EMAIL]"
    assert _redact("contact us at hello@acme.com please") == "contact us at [EMAIL] please"


def test_multiple_disjoint_pii_still_all_redacted():
    """Overlap resolution must not drop genuinely separate findings."""
    out = _redact("two: a@x.com and b@y.co.uk done")
    assert out.count("[EMAIL]") == 2
    assert "x.com" not in out and "y.co.uk" not in out


@pytest.mark.parametrize("text,expected", [
    ("card 4111-1111-1111-1111 exp 12/26", "card [CREDIT_CARD] exp 12/26"),
    ("SSN 123-45-6789 phone 555-123-4567", "SSN [SSN] phone [PHONE]"),
    ("IBAN GB29NWBK60161331926819 ref 60161331", "IBAN [IBAN] ref [BANK_ACCOUNT]"),
    ("visit https://acme.com/u/sarah.chen?id=99", "visit [URL]"),
    ("a@b.io", "[EMAIL]"),
])
def test_cases_that_already_worked_are_unchanged(text, expected):
    """These redacted correctly before the fix; pin them so the overlap guard doesn't
    regress the non-overlapping path."""
    assert _redact(text) == expected


def test_nothing_to_redact_returns_text_unchanged():
    clean = "The download link in your email has expired."
    assert _redact(clean) == clean


def test_a_non_redacting_type_cannot_suppress_an_overlapping_redaction():
    """Only REDACT-action types rewrite text, so they are filtered BEFORE overlap
    resolution. Were a BLOCK/WARN-action span allowed to win an overlap, the PII it covers
    would be left in the clear — a worse outcome than the leak this fix addresses."""
    det = PIIDetector({"enabled": True, "action": "REDACT"})
    text = "sarah.chen@example.com"
    spans = [
        (0, 22, "EMAIL", text, "email"),          # REDACT
        (0, 22, "CREDIT_CARD", text, "credit_card"),  # typically BLOCK — must not win
    ]
    out = det._redact_pii(text, spans)
    assert text not in out, f"PII survived redaction: {out!r}"
