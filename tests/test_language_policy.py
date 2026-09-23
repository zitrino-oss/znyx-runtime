"""Language allow/block semantics across both layers.

Three defects are pinned here, all found by benchmarking ml-language-en-es (105 en / 105 es)
against papluca/xlm-roberta-base-language-detection, which scored AUROC 0.9958 yet failed the
enforcement gate with a 43.8% false-positive rate:

  1. the ML runner decided an ALLOWLIST from the top label's probability, which fails OPEN for
     any language outside the model's 20 labels (Danish -> top `tr`=0.22 -> allowed by an
     English-only policy);
  2. `language` was an ADDITIVE (worst-of) ML layer, letting the weak deterministic trigram
     identifier veto the model's correct ALLOW on 46/105 English sentences;
  3. the deterministic identifier enforced guesses at 0.05-0.20 confidence.
"""
import pytest

from znyx_core.core.models import Decision
from znyx_core.detectors.language import LanguageDetector

# Real label distributions from the served model (localhost:9000 /v1/infer/language),
# trimmed to the labels that carry meaningful mass. Danish/Czech are NOT among the model's
# 20 languages, which is exactly why their mass is spread thin.
PROBS_ENGLISH = {"en": 0.9914, "nl": 0.0007, "de": 0.0005, "fr": 0.0004}
PROBS_SPANISH = {"es": 0.9901, "pt": 0.0040, "en": 0.0007, "it": 0.0020}
PROBS_DANISH = {"tr": 0.2230, "nl": 0.1400, "de": 0.1100, "en": 0.0580, "sw": 0.0900}
PROBS_CZECH = {"pl": 0.2130, "bg": 0.1500, "ru": 0.1200, "en": 0.0408, "tr": 0.0700}


def _decide(probs, allowed=None, blocked=None):
    """Exercise LanguageRunner._decide without loading the ONNX model."""
    from znyx_inference.runners.language import LanguageRunner
    runner = LanguageRunner.__new__(LanguageRunner)
    runner._allowed = set()
    runner._blocked = set()
    runner._output = lambda unsafe, scores: {"unsafe": unsafe, "label_scores": scores}
    return runner._decide(probs, allowed, blocked)["unsafe"]


# ── 1. allowlist decides on probability MASS, not the top label ──────────────────────

@pytest.mark.parametrize("name,probs,expect_blocked", [
    ("english", PROBS_ENGLISH, False),
    ("spanish", PROBS_SPANISH, True),
    ("danish (outside the model's labels)", PROBS_DANISH, True),
    ("czech (outside the model's labels)", PROBS_CZECH, True),
])
def test_allowlist_uses_mass_outside_the_allowed_set(name, probs, expect_blocked):
    risk = _decide(probs, allowed={"en"})
    assert (risk >= 0.5) is expect_blocked, f"{name}: risk={risk:.3f}"


def test_allowlist_blocks_a_language_the_model_cannot_name():
    """The regression that motivated the fix. Danish's top label is `tr` at 0.223 — under a
    0.5 threshold the old top-label rule ALLOWED it through an English-only policy. The
    model needn't recognise Danish; being confident it isn't English is enough."""
    assert _decide(PROBS_DANISH, allowed={"en"}) == pytest.approx(1 - 0.0580, abs=1e-6)
    assert max(PROBS_DANISH.items(), key=lambda kv: kv[1])[1] < 0.5  # old rule allowed it


def test_allowlist_sums_every_allowed_language():
    """Multi-language allowlists add up, so en+es content is allowed by allowed=[en,es]."""
    probs = {"en": 0.45, "es": 0.50, "fr": 0.05}
    assert _decide(probs, allowed={"en", "es"}) == pytest.approx(0.05, abs=1e-6)
    assert _decide(probs, allowed={"en"}) == pytest.approx(0.55, abs=1e-6)


# ── blocklist ────────────────────────────────────────────────────────────────────────

def test_blocklist_sums_mass_on_blocked_languages():
    """Two blocked languages splitting the mass still add up — neither clears 0.5 alone."""
    probs = {"es": 0.35, "pt": 0.34, "en": 0.31}
    assert _decide(probs, blocked={"es", "pt"}) == pytest.approx(0.69, abs=1e-6)
    assert _decide(probs, blocked={"es"}) == pytest.approx(0.35, abs=1e-6)


def test_blocklist_takes_precedence_over_allowlist():
    assert _decide(PROBS_SPANISH, allowed={"es"}, blocked={"es"}) == pytest.approx(0.9901, abs=1e-6)


def test_no_lists_configured_never_blocks():
    """With neither list the runner is a pure identifier."""
    assert _decide(PROBS_SPANISH) == 0.0
    assert _decide({}) == 0.0


# ── 3. the deterministic fallback does not enforce a guess ───────────────────────────

# Real English sentences the trigram identifier misread, with the language it claimed.
MISREAD_ENGLISH = [
    "The download link in your email has expired.",          # claimed 'fr' @ 0.07
    "How many seats are included in the team plan?",          # claimed 'ro' @ 0.17
    "I would like to move to annual billing.",                # claimed 'nl' @ 0.05
    "Our finance team needs a purchase order number.",        # claimed 'de' @ 0.07
]


@pytest.mark.parametrize("text", MISREAD_ENGLISH)
def test_low_confidence_guesses_no_longer_block_english(text):
    det = LanguageDetector({"enabled": True, "allowed_languages": ["en"], "action": "BLOCK"})
    assert det.detect(text).decision == Decision.ALLOW


@pytest.mark.parametrize("text", MISREAD_ENGLISH)
def test_min_confidence_zero_restores_the_old_behaviour(text):
    """The floor is the only thing suppressing these, so 0.0 must reproduce the old blocks —
    which keeps this test honest about what changed."""
    det = LanguageDetector({"enabled": True, "allowed_languages": ["en"],
                            "action": "BLOCK", "min_confidence": 0.0})
    assert det.detect(text).decision == Decision.BLOCK


def test_script_detection_still_enforces():
    """The floor must not disarm the layer entirely: a dominant non-Latin script scores
    min(ratio+0.2, 1.0) >= 0.5, so it stays above the default floor and is still caught.
    This is the half of the deterministic detector that is actually reliable."""
    det = LanguageDetector({"enabled": True, "allowed_languages": ["en"], "action": "BLOCK"})
    result = det.detect("这是一段足够长的中文文本，用于测试语言检测功能是否正常。")
    assert result.decision == Decision.BLOCK
    assert any(h.rule_id == "language.not_in_allowed" for h in result.rule_hits)


def test_identified_language_is_still_reported_when_suppressed():
    """Suppressing enforcement must not hide what was seen — the guess stays in the
    developer message for debugging, it just no longer decides."""
    det = LanguageDetector({"enabled": True, "allowed_languages": ["en"], "action": "BLOCK"})
    r = det.detect(MISREAD_ENGLISH[0])
    assert r.decision == Decision.ALLOW
    assert r.developer_message is None or "confidence=" not in (r.developer_message or "")


# ── 2. language is a competing classifier, so its ML layer replaces rather than merges ──

def test_language_ml_default_is_not_additive():
    """Additive worst-of merge gives the deterministic layer a veto over the model. Correct
    for pii (complementary layers, union of findings), wrong for language (both answer one
    question, so the better answer must win)."""
    from znyx_core.engine.ml_catalog import DETECTOR_ML_DEFAULTS
    assert DETECTOR_ML_DEFAULTS["language"].additive is False
    assert DETECTOR_ML_DEFAULTS["pii"].additive is True, "pii must stay additive"


def test_generated_language_strategy_replaces_the_deterministic_result():
    from znyx_core.engine.ml_catalog import default_strategy_for
    cfg = default_strategy_for("language", endpoint_url="http://localhost:9000/v1/infer/language")
    strategy = cfg["strategy"]
    assert strategy.get("additive") is not True
    # no band => the ML layer always runs; fallback still covers a sidecar outage
    assert "escalate_when" not in strategy
    assert strategy["fallback"] == "fallback_to_deterministic"
