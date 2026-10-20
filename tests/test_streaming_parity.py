"""Streaming parity: whatever the full-text check blocks or redacts must also be
blocked or redacted when the same text is streamed, wherever the sensitive value
lands relative to the window boundaries and whatever the chunk size.

Regression cover for three defects in the original sliding window:
  * REDACT was never applied: windows released raw text after a REDACT verdict.
  * A pattern longer than the overlap that straddled a release point escaped:
    its first part was released before any window saw it whole.
  * Every window consumed an abuse rate-limit token, so a long, harmless
    answer was blocked as rate-limited part-way through.

Sample secrets are assembled at runtime so no pattern-matching literal appears in
the source tree (the repo's secret scanner would flag it).
"""
import base64
import json

import pytest

from znyx_core.core.decision import DecisionAggregator
from znyx_core.core.models import Decision, EvaluationRequest
from znyx_core.engine.detector_registry import default_registry
from znyx_core.engine.orchestrator import DetectorOrchestrator
from znyx_core.engine.streaming import MIN_HOLDBACK, StreamingEvaluator

POLICY = {"secrets": {"enabled": True}, "pii": {"enabled": True}}

_SEG = "abcdefghijKLMNOPQRSTuvwxyz0123456789"
_JWT_HEAD = "eyJ" + "hbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
_JWT_SIG = "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"


def _b64url_json(obj):
    return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()


BLOCK_SAMPLES = {
    "jwt": _JWT_HEAD + "." + _b64url_json({"sub": "1234567890", "name": "John"}) + "." + _JWT_SIG,
    # A well-formed JWT longer than a whole default window: it must still be
    # evaluated in one piece rather than split across windows.
    "long_jwt": _JWT_HEAD + "." + _b64url_json({"sub": "1", "name": "n" * 300}) + "." + _JWT_SIG,
    "aws_key": "AKIA" + "IOSFODNN7EXAMPLQ",
    "private_key": "-----BEGIN RSA " + "PRIVATE KEY-----",
    "keyword_api_key": "api_key: " + (_SEG * 2)[:40],
}

REDACT_SAMPLES = {
    "email": "jane.doe" + "@example.com",
    "phone": "555-123-4567",
    "card": "4111 1111 1111 1111",
    # Token-shaped but not a valid JWT, so the PII detector redacts rather than the
    # secrets detector blocking. Longer than a default window.
    "long_token": _JWT_HEAD + "." + "eyJ" + ("zdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4i" * 8)
                  + "fQ" + "." + _JWT_SIG,
}

# Ordinary prose, no digits, so any digit run in released text came from a sample.
FILLER = ("the quarterly review covers planning goals and team updates for the next "
          "release cycle while keeping the roadmap focused on customer outcomes ")

CONFIGS = [(200, 48), (50, 10), (320, 100)]
CHUNK_SIZES = [1, 17, 200]
OFFSETS = range(0, 400, 13)


def _full_text_verdict(text):
    orch = DetectorOrchestrator(default_registry).run_output_detectors(
        text, POLICY, EvaluationRequest(request_id="full", tenant_id="parity", app_id="a", text=text))
    return DecisionAggregator.aggregate(orch.results).decision, orch.current_text


def _stream(text, window, overlap, chunk):
    ev = StreamingEvaluator(POLICY, context="output", window_size=window, overlap=overlap,
                            request=EvaluationRequest(request_id="s", tenant_id="parity",
                                                      app_id="a", text=""))
    events = []
    for i in range(0, len(text), chunk):
        events.extend(ev.push(text[i:i + chunk]))
        if ev.is_blocked:
            break
    events.extend(ev.flush())
    return ev, events


def _fragment_leaked(secret, released, n):
    """True when any n-character slice of ``secret`` appears in ``released``."""
    return any(secret[i:i + n] in released for i in range(0, len(secret) - n + 1))


def _texts(sample):
    for offset in OFFSETS:
        # The prefix can end mid-word; the space keeps the sample a separate word,
        # as the detectors' word-boundary anchors require in full text too.
        prefix = (FILLER * 4)[:offset] + " "
        yield offset, prefix + sample + " " + FILLER * 2


@pytest.mark.parametrize("name", sorted(BLOCK_SAMPLES))
def test_streaming_blocks_whatever_full_text_blocks(name):
    sample = BLOCK_SAMPLES[name]
    for offset, text in _texts(sample):
        decision, _ = _full_text_verdict(text)
        assert decision == Decision.BLOCK, f"precondition: full text blocks {name}"
        for window, overlap in CONFIGS:
            for chunk in CHUNK_SIZES:
                ev, events = _stream(text, window, overlap, chunk)
                where = f"{name} offset={offset} window={window}/{overlap} chunk={chunk}"
                assert ev.is_blocked, f"stream not blocked: {where}"
                assert events[-1]["data"]["final_decision"] == "BLOCK", where
                assert not _fragment_leaked(sample, ev.released_text, 8), f"fragment leaked: {where}"


@pytest.mark.parametrize("name", sorted(REDACT_SAMPLES))
def test_streaming_redacts_whatever_full_text_redacts(name):
    sample = REDACT_SAMPLES[name]
    n = min(8, len(sample) // 2)
    for offset, text in _texts(sample):
        decision, sanitized = _full_text_verdict(text)
        assert decision == Decision.REDACT, f"precondition: full text redacts {name}"
        for window, overlap in CONFIGS:
            for chunk in CHUNK_SIZES:
                ev, events = _stream(text, window, overlap, chunk)
                where = f"{name} offset={offset} window={window}/{overlap} chunk={chunk}"
                assert not ev.is_blocked, where
                assert not _fragment_leaked(sample, ev.released_text, n), f"fragment leaked: {where}"
                assert ev.released_text == sanitized, f"differs from full-text redaction: {where}"
                assert events[-1]["data"]["final_decision"] == "REDACT", where
                # The window preview must never show what the chunk event withheld.
                for e in events:
                    if e["event"] == "guardrail":
                        assert not _fragment_leaked(sample, e["data"].get("text_preview", ""), n), where


def test_clean_text_streams_unchanged():
    text = FILLER * 6
    for window, overlap in CONFIGS:
        for chunk in CHUNK_SIZES:
            ev, events = _stream(text, window, overlap, chunk)
            assert ev.released_text == text
            assert events[-1]["data"]["final_decision"] == "ALLOW"


def test_long_answer_consumes_one_rate_limit_token():
    """A long, harmless answer spans far more windows than the abuse limit allows
    requests per minute, and must not be blocked as rate-limited."""
    policy = {"abuse": {"enabled": True, "rate_limit_per_minute": 5}}
    text = FILLER * 80
    ev = StreamingEvaluator(policy, context="output",
                            request=EvaluationRequest(request_id="s", tenant_id="abuse-parity",
                                                      app_id="a", text="", metadata={"user_id": "u1"}))
    for i in range(0, len(text), 40):
        ev.push(text[i:i + 40])
    done = ev.flush()[-1]["data"]
    assert done["windows_evaluated"] > 5
    assert not ev.is_blocked
    assert ev.released_text == text


def test_small_overlap_is_raised_to_the_safe_minimum():
    ev = StreamingEvaluator(POLICY, window_size=50, overlap=0)
    assert ev.overlap == MIN_HOLDBACK
    assert ev.window_size >= 2 * ev.overlap
    done = ev.flush()[-1]["data"]
    assert done["effective_overlap"] == MIN_HOLDBACK
    assert done["effective_window_size"] == ev.window_size


def test_long_unbroken_run_is_evaluated_whole_without_hanging():
    """A run far longer than a window (but under MAX_RUN) is evaluated in one
    piece and then released; nothing is split and nothing hangs."""
    text = "x" * 9700 + " tail words here"
    ev, events = _stream(text, 200, 48, 700)
    assert ev.released_text == text
    assert events[-1]["data"]["final_decision"] == "ALLOW"


def test_unbroken_run_longer_than_max_fails_closed(monkeypatch):
    """Splitting an unbroken token is how a JWT escaped, so a token longer than
    MAX_RUN is not split: the stream fails closed instead."""
    import znyx_core.engine.streaming as streaming
    monkeypatch.setattr(streaming, "MAX_RUN", 5000)
    text = "x" * 6000 + " tail words here"
    ev, events = _stream(text, 200, 48, 700)
    assert ev.is_blocked
    assert ev.released_text == ""
    assert events[-1]["data"]["final_decision"] == "BLOCK"


def test_unbroken_run_is_held_until_it_ends():
    """No part of an unbroken run is released while the run may still be growing."""
    ev = StreamingEvaluator(POLICY, window_size=200, overlap=48)
    ev.push("intro words " + "z" * 500)
    assert "z" not in ev.released_text
    ev.push(" and then more words " * 20)
    assert "z" * 500 in ev.released_text


# ---------------------------------------------------------------------------
# Regression cover from the adversarial review of this change (2026-10-19).
# ---------------------------------------------------------------------------
import random  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

_FILL = "the quarterly review covers planning goals and team updates for the next release cycle "


def _stream_with(policy, text, window=200, overlap=48, chunk=1, **kw):
    ev = StreamingEvaluator(policy, context="output", window_size=window, overlap=overlap,
                            request=EvaluationRequest(request_id="s", tenant_id="review",
                                                      app_id="a", text=""), **kw)
    events = []
    for i in range(0, len(text), chunk):
        events.extend(ev.push(text[i:i + chunk]))
        if ev.is_blocked:
            break
    events.extend(ev.flush())
    return ev, events


_FUZZ_WORDS = ("the review covers planning goals and team updates for next release cycle while "
               "keeping roadmap focused on customer outcomes with clear owners").split()
_FUZZ_PII = {
    "email": lambda r: r.choice(["jane.doe", "m.ross", "ops-team"]) + "@" + r.choice(["example.com", "acme.io"]),
    "phone": lambda r: r.choice(["555-123-4567", "(415) 555-0199", "+1 212 555 0147"]),
    "ssn": lambda r: r.choice(["123-45-6789", "078 05 1120"]),
    "card": lambda r: r.choice(["4539 1488 0343 6467", "5555-5555-5555-4444"]),
    "ip": lambda r: r.choice(["203.0.113.77", "198.51.100.23"]),
    "name": lambda r: r.choice(["Dear Sarah Connor", "Mr. Edward Norton", "patient Marcus Delgado",
                                "my name is Olivia Brennan"]),
    "dob": lambda r: r.choice(["March 5, 1985", "04/12/1990"]),
    "iban": lambda r: r.choice(["DE89 3704 0044 0532 0130 00", "GB29 NWBK 6016 1331 9268 19"]),
}
_PLACEHOLDER = re.compile(r"\[[A-Z_0-9]+\]")


def _words(t):
    return _PLACEHOLDER.sub(" ", t).split()


def _is_subsequence(small, big):
    it = iter(big)
    return all(any(x == y for y in it) for x in small)


def test_fuzz_dense_pii_never_leaks_and_never_garbles():
    """Dense random PII (names, addresses-like values, spaced IBANs, partial
    shapes) at random window settings. The stream must never release a raw
    value the full-text path removed, and wherever its output differs from the
    full-text output it may only have redacted *more* (no repeated or invented
    text)."""
    rng = random.Random(20261019)
    for _ in range(200):
        parts, values = [], []
        for _ in range(rng.randint(30, 90)):
            if rng.random() < 0.12:
                v = _FUZZ_PII[rng.choice(sorted(_FUZZ_PII))](rng)
                parts.append(v)
                values.append(v)
            else:
                parts.append(rng.choice(_FUZZ_WORDS))
        text = " ".join(parts) + " ."
        _, sanitized = _full_text_verdict(text)
        window = rng.choice([50, 96, 120, 200, 333])
        overlap = rng.choice([0, 48, 60, 100])
        chunk = rng.choice([1, 3, 11, 40, 500])
        ev, _ = _stream(text, window, overlap, chunk)
        if ev.is_blocked:
            continue
        released = ev.released_text
        where = f"window={window}/{overlap} chunk={chunk} text={text[:80]!r}"
        for v in values:
            for tok in v.replace("(", " ").replace(")", " ").split():
                if len(tok) >= 4 and tok not in sanitized and tok not in _FUZZ_WORDS:
                    assert tok not in released, f"raw {tok!r} released: {where}"
        if released != sanitized:
            sw, fw = _words(released), _words(sanitized)
            assert _is_subsequence(sw, fw) and len(sw) <= len(fw), f"garbled: {where}"


def test_keyword_and_value_split_by_a_long_gap_is_blocked():
    value = "Zq7bY2kLm9Xp4RtW" + "8sVn3HgJ5dFc6AeB1uKo0iPy"
    for gap in (30, 70, 120):
        text = _FILL * 2 + "aws_secret_access_key =" + " " * gap + value + " " + _FILL
        assert _full_text_verdict(text)[0] == Decision.BLOCK
        ev, _ = _stream_with(POLICY, text)
        assert ev.is_blocked, f"gap={gap}"
        assert value not in ev.released_text


def test_private_key_header_padded_with_whitespace_is_blocked():
    # The private_key pattern allows up to 64 characters (one line) between BEGIN and
    # PRIVATE; within that, padding must not let the stream release the key.
    for pad in (20, 60):
        header = "-----BEGIN" + " " * pad + "RSA " + "PRIVATE KEY-----"
        text = _FILL + header + "\nMIIEowIBAAKCAQEAw1x2abc\n-----END RSA PRIVATE KEY-----\n" + _FILL
        assert _full_text_verdict(text)[0] == Decision.BLOCK
        ev, _ = _stream_with(POLICY, text)
        assert ev.is_blocked, f"pad={pad}"
        assert "MIIEowIBAAKCAQEA" not in ev.released_text


def test_full_holdback_after_a_long_run():
    """The window evaluated when a long run ends must still hold back the full
    overlap, or a spaced value right after the run is released half-redacted."""
    policy = {"pii": {"enabled": True}}
    for runlen in (190, 240):
        text = "-" * runlen + "DE89" + " 3704 0044 0532 0130 00 is the account. " + _FILL
        ev, _ = _stream_with(policy, text)
        assert "DE89" not in ev.released_text, f"runlen={runlen}"
        assert "3704" not in ev.released_text, f"runlen={runlen}"


def test_on_fail_remediation_applies_to_streams(tmp_path):
    from znyx_core.engine.remediation import RemediationHandler
    text = _FILL + "contact jane.doe" + "@example.com for details. " + _FILL * 2
    handler = RemediationHandler(ask_human_spool_path=str(tmp_path / "ask-human.spool"))
    for policy in ({"pii": {"enabled": True, "on_fail": "refrain"}},
                   {"pii": {"enabled": True}, "on_fail": "ask_human"}):
        ev, events = _stream_with(policy, text, remediation=handler)
        assert ev.is_blocked, policy
        assert ev.released_text == "" or "jane.doe" not in ev.released_text
        assert events[-1]["data"]["final_decision"] == "BLOCK"


def test_text_without_spaces_streams_incrementally():
    """Scripts that do not use spaces (here Chinese) are not one giant token."""
    ev = StreamingEvaluator({"pii": {"enabled": True}}, window_size=200, overlap=48)
    text = "这是一个关于季度规划和团队更新的回顾。" * 300
    released_before_flush = 0
    for i in range(0, len(text), 20):
        released_before_flush += sum(1 for e in ev.push(text[i:i + 20]) if e["event"] == "chunk")
    assert released_before_flush > 10
    ev.flush()
    assert ev.released_text == text


def test_rule_hits_are_counted_once_per_value():
    text = _FILL + "write to jane.doe" + "@example.com today. " + _FILL * 3
    ev, events = _stream_with({"pii": {"enabled": True}}, text, chunk=7)
    ids = sorted(h.rule_id for h in ev.rule_hits)
    assert ids == sorted(set(ids))
    assert events[-1]["data"]["total_rule_hits"] == len(ids)


def test_mapping_is_linear_on_a_long_run_of_addresses():
    """A long unbroken run full of redactions used to cost minutes of diff work."""
    text = "x@ab.co," * 1024 + " end."
    t0 = time.perf_counter()
    ev, _ = _stream_with({"pii": {"enabled": True}}, text, chunk=64)
    assert time.perf_counter() - t0 < 5.0
    assert "x@ab.co" not in ev.released_text


def test_jwt_is_not_split_at_a_large_run_boundary():
    """A JWT at the end of a long run of padding is evaluated whole."""
    jwt = BLOCK_SAMPLES["jwt"]
    text = "=" * 9000 + jwt + " " + _FILL
    assert _full_text_verdict(text)[0] == Decision.BLOCK
    ev, _ = _stream_with(POLICY, text, chunk=512)
    assert ev.is_blocked
    assert jwt.split(".")[1] not in ev.released_text


# ---------------------------------------------------------------------------
# Regression cover from adversarial review round 2 (2026-10-19).
# ---------------------------------------------------------------------------
import json as _json  # noqa: E402


def _b64j(obj):
    return base64.urlsafe_b64encode(_json.dumps(obj).encode()).decode().rstrip("=")


def test_jwt_after_a_password_keyword_is_blocked_at_every_offset():
    """pii.password matched `token: eyJ...` up to the JWT's first dot; the cut moved
    to that point, inside the token, and the rest of the JWT streamed raw."""
    jwt = (_b64j({"alg": "HS256", "typ": "JWT"}) + "." +
           _b64j({"sub": "user-8812", "role": "admin", "scope": "billing:write users:read",
                  "iat": 1716239022, "exp": 1716842822}) + "." + _JWT_SIG)
    payload = jwt.split(".")[1]
    for offset in range(0, 400, 7):
        text = ("Sure. " * 80)[:offset] + "Here is your token: " + jwt + " - keep it private."
        assert _full_text_verdict(text)[0] == Decision.BLOCK
        for chunk in (1, 16, 500):
            ev, _ = _stream_with(POLICY, text, chunk=chunk)
            assert ev.is_blocked, f"offset={offset} chunk={chunk}"
            assert payload[:24] not in ev.released_text


def test_keyword_hidden_by_an_earlier_redaction_still_blocks_the_value():
    rng = random.Random(7)
    token = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/")
                    for _ in range(356))
    pre = ("Here are the temporary credentials you asked for, export them before running "
           "the deploy script. " * 2)[:129]
    for text in ("Set AWS_SESSION_TOKEN: " + token + " then deploy.",
                 pre + " AWS_SESSION_TOKEN: " + token + " then deploy."):
        assert _full_text_verdict(text)[0] == Decision.BLOCK
        for window in (96, 200, 400):
            ev, _ = _stream_with(POLICY, text, window=window, chunk=64)
            assert ev.is_blocked
            assert token[-40:] not in ev.released_text


def test_partial_masking_streams_and_matches_full_text():
    """partial strategy used to force a coarse region that stalled streaming."""
    policy = {"pii": {"enabled": True, "redaction_strategy": "partial"}}
    rng = random.Random(1)
    text = "Contacts: " + " ".join(
        f"jane{i} {rng.randint(100, 999)}-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"
        for i in range(400)) + " end."
    full = DetectorOrchestrator(default_registry).run_output_detectors(
        text, policy, EvaluationRequest(request_id="f", tenant_id="t", app_id="a", text=text)).current_text
    t0 = time.perf_counter()
    ev = StreamingEvaluator(policy, request=EvaluationRequest(request_id="s", tenant_id="t", app_id="a", text=""))
    released_early = 0
    for i in range(0, len(text), 20):
        released_early += sum(len(e["data"]["text"]) for e in ev.push(text[i:i + 20]) if e["event"] == "chunk")
    ev.flush()
    assert time.perf_counter() - t0 < 5.0
    assert released_early > len(text) // 2
    assert ev.released_text == full


def test_custom_pii_pattern_streams_and_matches_full_text():
    policy = {"pii": {"enabled": True, "patterns": {"emp_id": r"EMP-\d{6}"}}}
    text = "Roster: " + " ".join(f"EMP-{100000 + i} user{i}" + "@corp.com" for i in range(300)) + " end."
    full = DetectorOrchestrator(default_registry).run_output_detectors(
        text, policy, EvaluationRequest(request_id="f", tenant_id="t", app_id="a", text=text)).current_text
    ev, _ = _stream_with(policy, text, chunk=50)
    assert ev.released_text == full


def test_growing_windows_stay_linear():
    """A URL match that runs on through text the token rule treats as breaks
    (here Chinese) keeps the release point waiting. Each retry doubles the window,
    so the work stays linear instead of re-evaluating ever larger windows."""
    body = ("这是一个很长的中文段落用于说明问题" * 2000)[:32000]
    text = "See https://example.com/" + body + " end."
    t0 = time.perf_counter()
    ev, events = _stream_with({"pii": {"enabled": True}}, text, chunk=1000)
    assert time.perf_counter() - t0 < 5.0
    assert events[-1]["data"]["windows_evaluated"] < 40


def test_many_edits_compose_quickly():
    policy = {"pii": {"enabled": True},
              "sensitive_business_data": {"enabled": True, "action": "REDACT", "block_threshold": 0}}
    text = "list: " + ",".join(f"user{i}" + "@ex.com" for i in range(6000)) + " see the term sheet."
    t0 = time.perf_counter()
    ev, _ = _stream_with(policy, text, chunk=500)
    assert time.perf_counter() - t0 < 5.0
    assert "@ex.com" not in ev.released_text


def test_user_agent_is_not_released_raw():
    ua = ("Mozilla/5.0 (Linux; Android 14; Pixel 8 Pro; wv; en-US; carrier Verizon; build channel "
          "beta; managed profile for the field sales team; enterprise enrolment; Mobile Safari)")
    filler = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron "
    text = filler + ua + " " + filler * 3
    for window, overlap in ((50, 0), (400, 160)):
        ev, _ = _stream_with({"pii": {"enabled": True}}, text, window=window, overlap=overlap, chunk=5)
        assert "Pixel 8 Pro" not in ev.released_text, f"window={window}"


def _fake_rewriters(monkeypatch, with_spans):
    """Fake transforming detectors exercising insertion, deletion, multi-word
    replacement and a rewrite of an earlier placeholder, with or without spans
    (spans that do not reproduce the output force the aligned fallback)."""
    from znyx_core.core.models import DetectorResult
    from znyx_core.detectors.competitor import CompetitorDetector
    from znyx_core.detectors.malicious_url import MaliciousURLDetector
    from znyx_core.detectors.pii import PIIDetector
    from znyx_core.detectors.sensitive_business_data import SensitiveBusinessDataDetector

    def mk(pattern, repl, label, decision=Decision.REDACT):
        rx = re.compile(pattern)

        def detect(self, text, *a, **k):
            ms = list(rx.finditer(text))
            if not ms:
                return DetectorResult(decision=Decision.ALLOW, risk_score=0)
            spans = [(m.start(), m.end(), label) for m in ms] if with_spans else []
            return DetectorResult(decision=decision, risk_score=40, sanitized_text=rx.sub(repl, text),
                                  redaction_spans=spans)
        return detect

    monkeypatch.setattr(CompetitorDetector, "detect", mk(r"\bINS\b", "INS <ins>", "X", Decision.TRANSFORM))
    monkeypatch.setattr(PIIDetector, "detect", mk(r"SEC \d{2} \d{2}", "[PII]", "PII"))
    monkeypatch.setattr(MaliciousURLDetector, "detect", mk(r"\bDEL\b ?", "", "Y"))
    monkeypatch.setattr(SensitiveBusinessDataDetector, "detect", mk(r"\[PII\] ?\[PII\]", "[PAIR]", "PAIR"))
    return {"competitor": {"enabled": True, "competitors": ["x"]}, "pii": {"enabled": True},
            "malicious_url": {"enabled": True}, "sensitive_business_data": {"enabled": True}}


@pytest.mark.parametrize("with_spans", [True, False])
def test_composed_rewrites_match_full_text(monkeypatch, with_spans):
    policy = _fake_rewriters(monkeypatch, with_spans)
    rng = random.Random(3)
    words = "alpha beta gamma delta eps zeta".split()
    for _ in range(120):
        parts = []
        for _ in range(rng.randint(10, 90)):
            r = rng.random()
            parts.append("SEC %02d %02d" % (rng.randint(0, 99), rng.randint(0, 99)) if r < .08
                         else "INS" if r < .13 else "DEL" if r < .18 else rng.choice(words))
        text = " ".join(parts)
        full = DetectorOrchestrator(default_registry).run_output_detectors(
            text, policy, EvaluationRequest(request_id="f", tenant_id="t", app_id="a", text=text)).current_text
        window, overlap, chunk = rng.choice([50, 96, 200, 400]), rng.choice([0, 60, 160]), rng.choice([1, 5, 33, 1000])
        ev, _ = _stream_with(policy, text, window=window, overlap=overlap, chunk=chunk)
        assert ev.released_text == full, f"window={window}/{overlap} chunk={chunk} text={text!r}"


def test_buffered_mode_matches_full_text_exactly():
    """Whole-response checks (here a score that needs two categories anywhere in
    the text) can only match the non-streaming path when nothing is released early."""
    policy = {"pii": {"enabled": True}, "sensitive_business_data": {"enabled": True, "action": "REDACT"}}
    filler = "the quarterly planning notes cover several unrelated items for the team today " * 6
    text = "We attached the term sheet for review. " + filler + "Also the customer list is in the drive."
    full = DetectorOrchestrator(default_registry).run_output_detectors(
        text, policy, EvaluationRequest(request_id="f", tenant_id="t", app_id="a", text=text)).current_text
    ev = StreamingEvaluator(policy, mode="buffered",
                            request=EvaluationRequest(request_id="s", tenant_id="t", app_id="a", text=""))
    for i in range(0, len(text), 7):
        assert not any(e["event"] == "chunk" for e in ev.push(text[i:i + 7]))
    done = ev.flush()[-1]["data"]
    assert ev.released_text == full
    assert done["mode"] == "buffered"


def test_policy_can_force_buffered_mode():
    ev = StreamingEvaluator({"streaming": {"mode": "buffered"}, "pii": {"enabled": True}},
                            mode="incremental")
    assert ev.mode == "buffered"


def test_full_response_check_appends_a_disclaimer_only_the_whole_text_triggers():
    policy = {"pii": {"enabled": True},
              "compliance": {"enabled": True, "action": "APPEND", "industry": "healthcare"}}
    filler = "the weather report says it will be sunny with light winds across the region later this week "
    text = "For your diagnosis the treatment and medication dosage depend on symptoms. " + filler * 6
    full = DetectorOrchestrator(default_registry).run_output_detectors(
        text, policy, EvaluationRequest(request_id="f", tenant_id="t", app_id="a", text=text)).current_text
    assert "Disclaimer" in full
    ev, _ = _stream_with(policy, text, chunk=9)
    assert "Disclaimer" in ev.released_text
    assert ev.released_text.count("Disclaimer") == full.count("Disclaimer")


def test_whole_response_checks_buffer_automatically():
    """Each window sees one sensitive-business category; only the whole text has
    two, which is what crosses the redaction threshold. Such a policy is buffered
    automatically, so the stream matches the non-streaming result exactly."""
    policy = {"sensitive_business_data": {"enabled": True, "action": "REDACT"}}
    filler = "the quarterly planning notes cover several unrelated items for the team today " * 6
    text = "We attached the term sheet for review. " + filler + "Also the customer list is in the drive."
    full = DetectorOrchestrator(default_registry).run_output_detectors(
        text, policy, EvaluationRequest(request_id="f", tenant_id="t", app_id="a", text=text)).current_text
    assert "term sheet" not in full
    ev, events = _stream_with(policy, text, chunk=7)
    done = events[-1]["data"]
    assert done["mode"] == "buffered"
    assert "sensitive_business_data" in done["mode_reason"]
    assert ev.released_text == full


def test_explicit_incremental_reports_what_streaming_could_not_know():
    """An admin can choose incremental anyway; then the stream cannot recall what it
    already released, and must report the difference."""
    policy = {"streaming": {"mode": "Incremental"},
              "sensitive_business_data": {"enabled": True, "action": "REDACT"}}
    filler = "the quarterly planning notes cover several unrelated items for the team today " * 6
    text = "We attached the term sheet for review. " + filler + "Also the customer list is in the drive."
    ev, events = _stream_with(policy, text, chunk=7)
    done = events[-1]["data"]
    assert done["mode"] == "incremental"
    assert "term sheet" in ev.released_text   # windows each saw only one category
    assert done["full_text_differs"] is True
    assert done["final_decision"] == "REDACT"


# ---------------------------------------------------------------------------
# Regression cover from adversarial review round 3 (2026-10-20).
# ---------------------------------------------------------------------------

def test_deferral_cannot_loop_forever():
    """A spaced value across the release point followed by a long unbroken token
    used to make the window shrink back to itself after every deferral, evaluating
    the same window forever and pinning a detector worker."""
    words = ("Thanks for reaching out about your account. Our team reviewed the request "
             "and will follow up soon with details. ")
    url = ("https://storage.example.com/bucket/q3-report.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256"
           "&X-Amz-Signature=" + "9f8e7d6c5b4a" * 30)
    text = (words * 6)[:231] + " (415) 555-0199 " + (words * 3)[:150] + " " + url + " done."
    ev = StreamingEvaluator({"pii": {"enabled": True}})
    for i in range(0, len(text), 16):
        ev.push(text[i:i + 16])
    done = ev.flush()[-1]["data"]
    assert done["windows_evaluated"] < 50
    assert "555-0199" not in ev.released_text


def test_secrets_redaction_survives_a_later_pii_rewrite_in_the_main_path():
    """/v1/evaluate/output returned a REDACT-action secret raw whenever PII also
    redacted: secrets did not feed the chained text, so PII's rewrite replaced it."""
    import asyncio
    from znyx_core.engine.evaluator import GuardrailsEvaluator
    aws = "AKIA" + "IOSFODNN7EXAMPLQ"
    text = "key " + aws + " and mail jane.doe" + "@example.com today please."
    policy = {"secrets": {"enabled": True, "action": "REDACT"}, "pii": {"enabled": True}}
    response = asyncio.run(GuardrailsEvaluator().evaluate(
        EvaluationRequest(request_id="r", tenant_id="t", app_id="a", text=text),
        context="output", policy=policy))
    assert aws not in (response.sanitized_text or "")
    assert "jane.doe" not in (response.sanitized_text or "")
    ev, _ = _stream_with(policy, text + " " + _FILL * 3, chunk=9)
    assert aws not in ev.released_text


def test_overlapping_secret_matches_never_splice_the_secret_back():
    from znyx_core.detectors.secrets import SecretsDetector
    text = "prefix " + "S" * 40 + " suffix"
    spans = [{"start": 7, "end": 47}, {"start": 12, "end": 30}, {"start": 40, "end": 52}]
    out = SecretsDetector({"enabled": True, "action": "REDACT"})._redact_text(text, spans)
    assert "S" not in out
    assert out.count("[REDACTED]") == 1


def test_control_character_sanitising_is_released():
    policy = {"output_control_char_sanitizer": {"enabled": True}}
    text = _FILL * 3 + "status \x1b]52;c;ZXZpbA==\x07 done " + _FILL * 3
    ev, _ = _stream_with(policy, text, chunk=11)
    assert "\x1b" not in ev.released_text


def test_keyword_separated_by_separator_characters_is_blocked():
    value = "Zq7bY2kLm9Xp4RtW" + "8sVn3HgJ5dFc6AeB1uKo0iPy"
    text = _FILL * 2 + "aws_secret_access_key" + " =" * 300 + " " + value + " " + _FILL
    assert _full_text_verdict(text)[0] == Decision.BLOCK
    ev, _ = _stream_with(POLICY, text, chunk=16)
    assert ev.is_blocked
    assert value not in ev.released_text


def test_context_starts_on_a_token_boundary():
    policy = {"pii": {"enabled": True, "types": {"address": {"enabled": True}}}}
    filler = "the quarterly planning notes cover several unrelated items for the team today "
    core = ("https://files.example.com/" + "q7Zk" * 18 +
            "/555 123 4567 Lake View Heights Gardens Street, Springfield, IL 62704 is the office.")
    text = (filler * 6)[:74] + " " + core + " " + filler * 3
    full = DetectorOrchestrator(default_registry).run_output_detectors(
        text, policy, EvaluationRequest(request_id="f", tenant_id="t", app_id="a", text=text)).current_text
    ev, _ = _stream_with(policy, text, chunk=16)
    assert ev.released_text == full


def test_filter_field_buffers_automatically():
    policy = {"pii": {"enabled": True}, "on_fail": "filter_field",
              "on_fail_config": {"fields": ["internal_notes"]}}
    obj = {"answer": "Contact jane.doe" + "@example.com about the planning review. "
                     + "More details follow in the notes for the team. " * 10,
           "internal_notes": "SECRET-NOTE: staff salary bands, do not share",
           "footer": "Contact bob.smith" + "@example.com for questions."}
    ev, events = _stream_with(policy, json.dumps(obj), chunk=16)
    assert events[-1]["data"]["mode"] == "buffered"
    assert "SECRET-NOTE" not in ev.released_text
    # The stream filters the text it holds: it used to filter its own empty request
    # text and release nothing at all.
    assert ev.released_text == _main_path_output(json.dumps(obj), policy)
    assert "planning review" in ev.released_text


def _main_path_output(text, policy):
    import asyncio
    from znyx_core.engine.evaluator import GuardrailsEvaluator
    response = asyncio.run(GuardrailsEvaluator().evaluate(
        EvaluationRequest(request_id="r", tenant_id="t", app_id="a", text=text),
        context="output", policy=policy))
    return response.sanitized_text if response.sanitized_text is not None else text


def test_filter_field_without_a_text_rewrite_keeps_the_rest_of_the_response():
    """/v1/evaluate/output returned an empty response when filter_field fired on a
    detector that changes no text: it filtered sanitized_text, which was None."""
    policy = {"sensitive_business_data": {"enabled": True, "on_fail": "filter_field",
                                          "on_fail_config": {"fields": ["internal_notes"]}}}
    obj = {"answer": "Here is the summary of the quarterly planning review for the team. " * 6,
           "internal_notes": "SECRET-NOTE: the term sheet values the deal at 41M, do not share"}
    out = _main_path_output(json.dumps(obj), policy)
    assert "SECRET-NOTE" not in out
    assert json.loads(out)["answer"] == obj["answer"]
    ev, _ = _stream_with(policy, json.dumps(obj), chunk=16)
    assert ev.released_text == out


@pytest.mark.parametrize("pad", ["", " " * 70, "\n" * 3, " RSA"])
def test_padded_private_key_header_is_blocked_on_both_paths(pad):
    header = "-----BEGIN" + pad + " RSA PRIVATE KEY-----"
    text = _FILL * 2 + header + "\nMIIEowIBAAKCAQEAw1x2abc\n-----END RSA PRIVATE KEY-----\n" + _FILL
    assert _full_text_verdict(text)[0] == Decision.BLOCK
    ev, _ = _stream_with(POLICY, text, chunk=16)
    assert ev.is_blocked


@pytest.mark.parametrize("policy", [
    {"competitor": {"enabled": True, "action": "RESTRICT", "competitors": ["Globex"],
                    "allowlist_contexts": ["partnership"]}},
    {"malicious_url": {"enabled": True, "action": "REDACT"}},
])
def test_other_whole_response_rewrites_buffer_automatically(policy):
    ev = StreamingEvaluator(policy)
    assert ev.mode == "buffered"
    assert "whole-response" in ev.mode_reason


_JUDGE_STRATEGY = {"order": ["deterministic", "remote_llm"], "fallback": "fail_closed"}


@pytest.mark.parametrize("policy,reason", [
    ({"hallucination": {"enabled": True, "action": "BLOCK"}}, "groundedness:hallucination"),
    ({"citation_integrity": {"enabled": True, "action": "block"}}, "groundedness:citation_integrity"),
    ({"toxicity": {"enabled": True, "action": "WARN", "strategy": _JUDGE_STRATEGY}},
     "llm_judge:toxicity"),
    ({"jailbreak": {"enabled": True, "strategy": {"order": ["deterministic", "LOCAL_LLM"]}}},
     "llm_judge:jailbreak"),
])
def test_checks_that_judge_the_whole_answer_buffer_automatically(policy, reason):
    """A blocking judge or groundedness check sees the whole answer once, before any
    of it reaches the reader, instead of one window at a time."""
    ev = StreamingEvaluator(policy)
    assert ev.mode == "buffered"
    assert reason in ev.mode_reason


@pytest.mark.parametrize("policy", [
    {"hallucination": {"enabled": True, "action": "WARN"}},
    {"hallucination": {"enabled": False, "action": "BLOCK"}},
    {"toxicity": {"enabled": True, "strategy": {"order": ["deterministic", "remote_api"]}}},
    {"toxicity": {"enabled": False, "strategy": _JUDGE_STRATEGY}},
    {"toxicity": {"enabled": True, "strategy": _JUDGE_STRATEGY},
     "streaming": {"mode": "incremental"}},
])
def test_non_blocking_or_local_checks_keep_streaming(policy):
    assert StreamingEvaluator(policy).mode == "incremental"


def test_disclaimer_is_released_once():
    policy = {"compliance": {"enabled": True, "action": "APPEND", "industry": "healthcare",
                             "ai_disclosure": True}}
    filler = "the weather report says it will be sunny with light winds across the region later this week "
    text = "For your diagnosis the treatment and medication dosage depend on symptoms. " + filler * 8
    full = DetectorOrchestrator(default_registry).run_output_detectors(
        text, policy, EvaluationRequest(request_id="f", tenant_id="t", app_id="a", text=text)).current_text
    ev, _ = _stream_with(policy, text, chunk=9)
    assert ev.released_text == full


def test_session_budget_is_charged_once_per_stream():
    """Reported tokens were charged once per window: one 1,500-token stream charged
    the session about 21,000 and blocked it."""
    policy = {"unbounded_consumption": {"enabled": True, "max_session_tokens": 20000}}
    filler = "the quarterly planning notes cover several unrelated items for the team today " * 40
    for i in range(3):
        req = EvaluationRequest(request_id=f"s{i}", tenant_id="t", app_id="a", text="",
                                metadata={"session_id": "sess-42", "tokens": 1500})
        ev = StreamingEvaluator(policy, request=req)
        for j in range(0, len(filler), 50):
            ev.push(filler[j:j + 50])
        ev.flush()
        assert not ev.is_blocked, f"stream {i}"
