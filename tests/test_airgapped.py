"""Air-gapped installs (``ZNYX_AIRGAPPED``): no boundary-crossing backend calls, no
model downloads, whatever a policy bundle says."""
import pytest
from fastapi.testclient import TestClient

from znyx_core.engine.egress import evaluate_egress
from znyx_core.utils.env import is_airgapped
from znyx_inference.install import InstallManager
from znyx_inference.runners._fetch import (
    ModelDownloadDisabled,
    ensure_downloads_allowed,
    fetch_model,
)


@pytest.fixture(autouse=True)
def _connected_by_default(monkeypatch):
    """Each test starts connected; air-gapped ones opt in."""
    monkeypatch.delenv("ZNYX_AIRGAPPED", raising=False)


@pytest.fixture()
def airgapped(monkeypatch):
    monkeypatch.setenv("ZNYX_AIRGAPPED", "true")


# ---------------------------------------------------------------------------
# is_airgapped
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["true", "1", "on", "yes", "TRUE", " True "])
def test_truthy_values_mean_airgapped(monkeypatch, value):
    monkeypatch.setenv("ZNYX_AIRGAPPED", value)
    assert is_airgapped() is True


@pytest.mark.parametrize("value", ["false", "0", "off", "no", "", "maybe"])
def test_other_values_mean_connected(monkeypatch, value):
    monkeypatch.setenv("ZNYX_AIRGAPPED", value)
    assert is_airgapped() is False


def test_unset_means_connected():
    assert is_airgapped() is False


# ---------------------------------------------------------------------------
# The egress gate
# ---------------------------------------------------------------------------

def _egress(mode="remote_llm", **overrides):
    args = dict(endpoint_url="https://api.llm.example.com/v1", region="us", in_boundary=None,
                no_external_calls=False, egress_allowlist=["api.llm.example.com"],
                allowed_regions=None, redact_pii=False, redact_secrets=False)
    args.update(overrides)
    return evaluate_egress(mode, **args)


def test_a_connected_install_allows_an_allowlisted_remote_call():
    decision = _egress()
    assert decision.is_egress is True and decision.allowed is True and decision.reason is None


def test_airgapped_denies_a_remote_call_even_when_the_policy_allows_it(airgapped):
    decision = _egress(no_external_calls=False, egress_allowlist=["api.llm.example.com"])
    assert decision.is_egress is True
    assert decision.allowed is False
    assert decision.reason == "no_external_calls"
    assert decision.host == "api.llm.example.com"


@pytest.mark.parametrize("mode", ["remote_llm", "remote_api"])
def test_airgapped_denies_every_remote_mode(airgapped, mode):
    assert _egress(mode).reason == "no_external_calls"


def test_airgapped_denies_a_hosted_model_server_that_declares_itself_outside(airgapped):
    decision = _egress("local_ml", in_boundary=False)
    assert decision.allowed is False and decision.reason == "no_external_calls"


def test_airgapped_denies_even_without_an_allowlist(airgapped):
    assert _egress(egress_allowlist=None).reason == "no_external_calls"


def test_airgapped_leaves_local_deterministic_detectors_alone(airgapped):
    decision = _egress("local_deterministic", endpoint_url=None, egress_allowlist=None)
    assert decision.is_egress is False and decision.allowed is True and decision.reason is None


def test_airgapped_leaves_an_in_boundary_model_server_alone(airgapped):
    decision = _egress("local_ml", endpoint_url="http://127.0.0.1:9000", in_boundary=True)
    assert decision.is_egress is False and decision.allowed is True


def test_the_policy_switch_still_works_on_its_own():
    assert _egress(no_external_calls=True).reason == "no_external_calls"


# ---------------------------------------------------------------------------
# Model downloads
# ---------------------------------------------------------------------------

def test_ensure_downloads_allowed_raises_only_when_airgapped(airgapped):
    with pytest.raises(ModelDownloadDisabled, match="air-gapped"):
        ensure_downloads_allowed()


def test_ensure_downloads_allowed_is_quiet_when_connected():
    ensure_downloads_allowed()


def test_model_download_disabled_is_a_runtime_error():
    """Callers that already catch RuntimeError keep working."""
    assert issubclass(ModelDownloadDisabled, RuntimeError)


def test_fetch_model_refuses_when_airgapped_and_never_downloads(airgapped, tmp_path):
    calls = []
    dest = tmp_path / "weights"
    with pytest.raises(ModelDownloadDisabled):
        fetch_model("org/model", "rev", str(dest), downloader=lambda *a: calls.append(a))
    assert calls == [], "the injected downloader must not run"
    assert not dest.exists(), "nothing is created on disk either"


def test_fetch_model_still_works_when_connected(tmp_path):
    def downloader(model_id, revision, dest_dir):
        (tmp_path / "weights" / "model.bin").write_bytes(b"weights")

    digest = fetch_model("org/model", "rev", str(tmp_path / "weights"), downloader=downloader)
    assert len(digest) == 64


def test_install_manager_refuses_when_airgapped(airgapped):
    manager = InstallManager()
    with pytest.raises(ModelDownloadDisabled):
        manager.start_install("prompt_injection")
    assert manager.list_jobs() == [], "no job is queued"


@pytest.fixture(scope="module")
def inference():
    from znyx_inference.main import app
    with TestClient(app) as client:
        yield client


def test_the_install_endpoint_returns_403_when_airgapped(inference, airgapped):
    response = inference.post("/v1/models/install", json={"task": "prompt_injection"})
    assert response.status_code == 403
    assert "air-gapped" in response.json()["detail"]
    assert inference.app.state.install_manager.list_jobs() == []


def test_the_install_endpoint_refuses_before_validating_the_task(inference, airgapped):
    response = inference.post("/v1/models/install", json={"task": "no_such_task"})
    assert response.status_code == 403


def test_the_rest_of_the_model_server_keeps_working_when_airgapped(inference, airgapped):
    assert inference.get("/healthz").status_code == 200
    assert inference.get("/v1/models").status_code == 200
