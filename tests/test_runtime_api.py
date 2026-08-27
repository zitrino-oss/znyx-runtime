"""Evaluator API surface: the runtime FastAPI app end to end, in local YAML
mode with an inline policy file - health, readiness, auth on/off, and
evaluate round-trips."""
import os

import pytest
from fastapi.testclient import TestClient

from conftest import MINIMAL_POLICY_YAML

_RUNTIME_ENV = {
    "ZNYX_MODE": "local",
    "RUNTIME_REQUIRE_AUTH": "false",
    "ZNYX_TELEMETRY": "false",
}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """The runtime app under TestClient, serving a temp policies.yaml.

    Module-scoped: the app's lifespan (bundle manager, spool sinks) starts
    once. Auth env vars are read per request, so individual tests can still
    flip them with monkeypatch.
    """
    tmp = tmp_path_factory.mktemp("runtime")
    policy_path = tmp / "policies.yaml"
    policy_path.write_text(MINIMAL_POLICY_YAML)

    env = dict(_RUNTIME_ENV)
    env["ZNYX_POLICY_PATH"] = str(policy_path)
    env["ZNYX_AUDIT_SPOOL_PATH"] = str(tmp / "egress-audit.spool")
    env["ZNYX_JUDGE_AUDIT_SPOOL_PATH"] = str(tmp / "judge-audit.spool")

    saved = {key: os.environ.get(key) for key in env}
    os.environ.update(env)

    # Keep the install-state file out of the developer's real home directory.
    import znyx_runtime.install_state as install_state
    saved_state = (install_state.STATE_DIR, install_state.STATE_FILE)
    install_state.STATE_DIR = tmp / ".znyx"
    install_state.STATE_FILE = install_state.STATE_DIR / "state.json"

    try:
        from znyx_runtime.main import app
        with TestClient(app) as test_client:
            yield test_client
    finally:
        install_state.STATE_DIR, install_state.STATE_FILE = saved_state
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _request_body(text, request_id="req-1"):
    return {
        "request_id": request_id,
        "tenant_id": "default",
        "app_id": "default",
        "text": text,
    }


class TestHealthAndStatus:
    def test_healthz(self, client):
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_readyz_ready_with_policy_loaded(self, client):
        response = client.get("/readyz")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    def test_root_reports_local_mode(self, client):
        body = client.get("/").json()
        assert body["service"] == "ZNYX Runtime"
        assert body["mode"] == "local"

    def test_bundle_status_yaml_mode(self, client):
        body = client.get("/v1/bundle/status").json()
        assert body["ready"] is True
        assert body["mode"] == "yaml"


class TestEvaluateRoundTrip:
    def test_benign_input_allows(self, client):
        response = client.post("/v1/evaluate/input", json=_request_body(
            "How do I bake bread?", request_id="rt-allow"))
        assert response.status_code == 200
        body = response.json()
        assert body["decision"] == "ALLOW"
        assert body["risk_score"] == 0
        assert body["request_id"] == "rt-allow"
        assert body["rule_hits"] == []

    def test_secret_in_input_blocks(self, client, fake_pat):
        response = client.post("/v1/evaluate/input", json=_request_body(
            "here is the deploy token " + fake_pat, request_id="rt-block"))
        assert response.status_code == 200
        body = response.json()
        assert body["decision"] == "BLOCK"
        assert body["risk_score"] == 100
        assert any(h["rule_id"] == "secrets.github_pat_classic"
                   for h in body["rule_hits"])

    def test_pii_in_output_redacts(self, client):
        email = "jane.doe@example.com"
        response = client.post("/v1/evaluate/output", json=_request_body(
            "email me at " + email, request_id="rt-redact"))
        assert response.status_code == 200
        body = response.json()
        assert body["decision"] == "REDACT"
        assert email not in (body["sanitized_text"] or "")
        assert any(h["rule_id"] == "pii.email" for h in body["rule_hits"])

    def test_validation_error_on_missing_fields(self, client):
        response = client.post("/v1/evaluate/input", json={"text": "hi"})
        assert response.status_code == 422


class TestAuth:
    @pytest.fixture()
    def auth_on(self, monkeypatch):
        monkeypatch.setenv("RUNTIME_REQUIRE_AUTH", "true")
        monkeypatch.setenv("RUNTIME_API_KEY", "test-key-123")

    def test_auth_off_allows_anonymous_calls(self, client):
        response = client.post("/v1/evaluate/input", json=_request_body("hi"))
        assert response.status_code == 200

    def test_missing_key_rejected(self, client, auth_on):
        response = client.post("/v1/evaluate/input", json=_request_body("hi"))
        assert response.status_code == 401

    def test_wrong_key_rejected(self, client, auth_on):
        response = client.post("/v1/evaluate/input", json=_request_body("hi"),
                               headers={"X-API-Key": "wrong"})
        assert response.status_code == 401

    def test_x_api_key_accepted(self, client, auth_on):
        response = client.post("/v1/evaluate/input", json=_request_body("hi"),
                               headers={"X-API-Key": "test-key-123"})
        assert response.status_code == 200

    def test_bearer_token_accepted(self, client, auth_on):
        response = client.post("/v1/evaluate/input", json=_request_body("hi"),
                               headers={"Authorization": "Bearer test-key-123"})
        assert response.status_code == 200

    def test_auth_required_without_key_is_a_server_error(self, client, monkeypatch):
        # Auth on but no key configured must fail loudly, not fail open.
        monkeypatch.setenv("RUNTIME_REQUIRE_AUTH", "true")
        monkeypatch.setenv("RUNTIME_API_KEY", "")
        response = client.post("/v1/evaluate/input", json=_request_body("hi"))
        assert response.status_code == 500

    def test_healthz_needs_no_auth(self, client, auth_on):
        assert client.get("/healthz").status_code == 200
