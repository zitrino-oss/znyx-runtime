"""Two guards against a fabricated answer looking like a real one.

1. UNPINNED REQUESTS. Real models arrive as *variants*: the reconciler loads a pin beside
   the active slot rather than replacing it, so a deployment can hold working weights while
   the active slot is still the StubRunner. An unpinned request then routed to the stub,
   whose verdict is a keyword heuristic — and which answers a prompt-injection probe with
   BLOCK at risk 100, indistinguishable from a real inference to any caller reading
   `decision`/`risk_score`. ``_resolve_batcher`` already refused to serve the wrong REAL
   model (409 on a pin mismatch, "the caller must never be silently scored by the wrong
   model"); the same principle now covers the stub.

2. UNPINNED MODELS. ``verify_pinned`` only checks a digest when the spec carries one, so a
   model loaded without a sha256 is trusted as-is. Every model in the reference deployment
   was in that state while the service reported itself as serving sha256-pinned artifacts.
"""
import pytest

from znyx_inference.config import InferenceConfig
from znyx_inference.contract import ModelInfo
from znyx_inference.registry import RunnerRegistry
from znyx_inference.runners.base import RunnerUnavailable


def _registry(tasks=None):
    cfg = InferenceConfig.from_env()
    cfg.task_specs = dict(tasks or {})
    return RunnerRegistry(cfg)


# ── registry helpers ────────────────────────────────────────────────────────────────

def test_active_is_stub_reports_the_active_slots_runner():
    reg = _registry({"toxicity": {}})          # no runner => stub
    assert reg.active_is_stub("toxicity") is True
    assert reg.active_is_stub("nonexistent") is False


def test_real_variants_lists_only_available_non_stub_variants():
    reg = _registry({"toxicity": {}})
    reg._variant_models[("toxicity", "a")] = ModelInfo(
        task="toxicity", model_version="unitary/toxic-bert@main", runner="classifier",
        available=True, active=False)
    reg._variant_models[("toxicity", "b")] = ModelInfo(
        task="toxicity", model_version="broken@main", runner="classifier",
        available=False, active=False)          # failed to load
    reg._variant_models[("toxicity", "c")] = ModelInfo(
        task="toxicity", model_version="stub@v1", runner="stub",
        available=True, active=False)
    reg._variant_models[("other", "d")] = ModelInfo(
        task="other", model_version="x@main", runner="ner",
        available=True, active=False)
    assert reg.real_variants("toxicity") == ["unitary/toxic-bert@main"]


# ── guard 1: an unpinned request must not be answered by the stub ───────────────────

class _Req:
    def __init__(self, model_id=None, revision=None):
        self.model_id, self.revision = model_id, revision


def test_unpinned_request_refused_when_real_weights_are_loaded(monkeypatch):
    from fastapi import HTTPException
    from znyx_inference import main

    monkeypatch.setattr(main, "_ALLOW_STUB_WHEN_REAL_LOADED", False)
    reg = _registry({"toxicity": {}})
    reg._variant_models[("toxicity", "a")] = ModelInfo(
        task="toxicity", model_version="unitary/toxic-bert@main", runner="classifier",
        available=True, active=False)

    with pytest.raises(HTTPException) as ei:
        main._resolve_batcher(reg, "toxicity", _Req())
    assert ei.value.status_code == 409
    # the error must name what to pin, or it isn't actionable
    assert "unitary/toxic-bert@main" in ei.value.detail


def test_unpinned_request_still_served_when_only_the_stub_exists(monkeypatch):
    """A stub-only deployment is the honest case — the stub IS the answer there, and this
    guard must not break it."""
    from znyx_inference import main
    monkeypatch.setattr(main, "_ALLOW_STUB_WHEN_REAL_LOADED", False)
    reg = _registry({"toxicity": {}})
    batcher, version = main._resolve_batcher(reg, "toxicity", _Req())
    assert batcher is not None
    assert version == "stub@v1"


def test_the_escape_hatch_restores_stub_service(monkeypatch):
    from znyx_inference import main
    monkeypatch.setattr(main, "_ALLOW_STUB_WHEN_REAL_LOADED", True)
    reg = _registry({"toxicity": {}})
    reg._variant_models[("toxicity", "a")] = ModelInfo(
        task="toxicity", model_version="unitary/toxic-bert@main", runner="classifier",
        available=True, active=False)
    batcher, _ = main._resolve_batcher(reg, "toxicity", _Req())
    assert batcher is not None


def test_a_pinned_request_is_unaffected(monkeypatch):
    """The guard is only about UNPINNED requests; pinned routing keeps its own semantics."""
    from fastapi import HTTPException
    from znyx_inference import main
    monkeypatch.setattr(main, "_ALLOW_STUB_WHEN_REAL_LOADED", False)
    reg = _registry({"toxicity": {}})
    with pytest.raises(HTTPException) as ei:
        main._resolve_batcher(reg, "toxicity", _Req(model_id="unitary/toxic-bert"))
    assert ei.value.status_code == 409
    assert "pin mismatch" in ei.value.detail


# ── guard 2: an unpinned MODEL is surfaced, and refusable ───────────────────────────

class _FakeRunner:
    """A non-stub runner that loads successfully, so ``_make`` reaches its pin check."""
    runner_kind = "classifier"
    model_version = "fake@v0"

    def __init__(self, task, spec):
        self.task, self.spec = task, spec

    def load(self):
        return None

    def infer_batch(self, texts, params=None):
        return []


@pytest.fixture
def fake_classifier(monkeypatch):
    """Serve a dependency-free fake for 'classifier' and delegate every other kind to the
    real lookup, so the stub path still works inside these tests."""
    import znyx_inference.registry as reg_mod
    real = reg_mod._factory_for
    monkeypatch.setattr(reg_mod, "_factory_for",
                        lambda kind: ((lambda t, s: _FakeRunner(t, s))
                                      if kind == "classifier" else real(kind)))
    return reg_mod


PINNED = {"runner": "classifier", "model_id": "acme/model", "revision": "main",
          "sha256": "a" * 64}
UNPINNED = {k: v for k, v in PINNED.items() if k != "sha256"}


def test_unpinned_model_load_warns(fake_classifier, caplog):
    """A real model with no sha256 must leave a trace. Silence is what let the reference
    deployment report sha256-pinned serving while every model was unpinned."""
    import logging
    caplog.set_level(logging.WARNING, logger=fake_classifier.__name__)
    reg = _registry()
    _batcher, info = reg._make("toxicity", UNPINNED, active=True)
    assert info.available is True                     # it still loads
    assert info.sha256 is None
    assert any("UNPINNED" in r.message for r in caplog.records), \
        "an unpinned real model must warn"


def test_a_pinned_model_does_not_warn(fake_classifier, caplog):
    import logging
    caplog.set_level(logging.WARNING, logger=fake_classifier.__name__)
    reg = _registry()
    _batcher, info = reg._make("toxicity", PINNED, active=True)
    assert info.sha256 == "a" * 64
    assert not [r for r in caplog.records if "UNPINNED" in r.message]


def test_require_pinned_refuses_an_unpinned_model(fake_classifier, monkeypatch):
    """With the flag on, an unpinned non-stub model must NOT load. ``_make`` contains the
    failure per task, so the task reports unavailable rather than crashing the service."""
    monkeypatch.setattr(fake_classifier, "_REQUIRE_PINNED", True)
    reg = _registry()
    batcher, info = reg._make("toxicity", UNPINNED, active=True)
    assert batcher is None
    assert info.available is False
    assert "sha256" in (info.detail or "").lower()


def test_require_pinned_still_loads_a_pinned_model(fake_classifier, monkeypatch):
    monkeypatch.setattr(fake_classifier, "_REQUIRE_PINNED", True)
    reg = _registry()
    batcher, info = reg._make("toxicity", PINNED, active=True)
    assert batcher is not None and info.available is True


def test_require_pinned_does_not_affect_the_stub(fake_classifier, monkeypatch):
    """The stub has no artifact to pin, so the flag must not disarm a stub-only deployment."""
    monkeypatch.setattr(fake_classifier, "_REQUIRE_PINNED", True)
    reg = _registry()
    batcher, info = reg._make("toxicity", {}, active=True)
    assert batcher is not None and info.available is True
    assert info.runner == "stub"


def test_require_pinned_defaults_off():
    """Default must stay off so existing unpinned deployments keep serving — the warning
    carries the message instead."""
    import importlib

    import znyx_inference.registry as reg_mod
    importlib.reload(reg_mod)
    assert reg_mod._REQUIRE_PINNED is False


def test_artifact_sha256_is_stable_and_content_sensitive(tmp_path):
    """The digest the pins need: a whole model directory reduces to one stable value, and
    any content change moves it."""
    from znyx_inference.runners._artifacts import artifact_sha256

    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text('{"a":1}')
    (d / "weights.bin").write_bytes(b"\x00\x01\x02")
    first = artifact_sha256(str(d))
    assert first == artifact_sha256(str(d))           # stable across calls
    (d / "weights.bin").write_bytes(b"\x00\x01\x03")
    assert artifact_sha256(str(d)) != first           # content-sensitive
