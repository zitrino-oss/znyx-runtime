"""Gateway configuration, secret references and the price table, without the app."""
import hashlib
import logging

import pytest

from znyx_core import secret_refs
from znyx_runtime.gateway.config import (
    GatewayConfigError, from_bundle_policy, from_document, from_environment,
)
from znyx_runtime.gateway.formats import Usage
from znyx_runtime.gateway.pricing import PriceTable

DIGEST = hashlib.sha256(b"k").hexdigest()


def _doc(**providers):
    return {"providers": providers or {"openai": {"format": "openai", "api_key": "env:K"}}}


# -- secret references -----------------------------------------------------------

def test_references_resolve_inside_the_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("PROVIDER_KEY", "sk-env")
    secret = tmp_path / "key"
    secret.write_text("sk-file\n")
    assert secret_refs.resolve("env:PROVIDER_KEY") == "sk-env"
    assert secret_refs.resolve(f"file:{secret}") == "sk-file"
    assert secret_refs.resolve("none:") is None and secret_refs.resolve(None) is None


@pytest.mark.parametrize("ref,message", [
    ("sk-plaintext", "reference"), ("env:", "variable name"), ("env:NOT_SET_ANYWHERE", "not set"),
    ("file:/nonexistent/x", "cannot read"), ("vault:x", "unknown reference scheme"),
])
def test_bad_references_are_refused_without_echoing_secrets(ref, message):
    with pytest.raises(secret_refs.SecretRefError, match=message) as exc:
        secret_refs.resolve(ref)
    assert "sk-plaintext" not in str(exc.value)


def test_maybe_resolve_keeps_old_plaintext_bundles_working(monkeypatch, caplog):
    monkeypatch.setenv("JUDGE_KEY", "sk-judge")
    assert secret_refs.maybe_resolve("env:JUDGE_KEY") == "sk-judge"
    assert secret_refs.maybe_resolve("sk-inline-from-an-old-control-plane") == "sk-inline-from-an-old-control-plane"
    with caplog.at_level(logging.ERROR):
        assert secret_refs.maybe_resolve("env:MISSING_JUDGE_KEY_X") == ""
        assert secret_refs.maybe_resolve("env:MISSING_JUDGE_KEY_X") == ""
    assert sum("MISSING_JUDGE_KEY_X" in r.message for r in caplog.records) == 1


def test_a_remote_detector_uses_the_resolved_credential(monkeypatch):
    from znyx_core.detectors.remote import RemoteDetector
    monkeypatch.setenv("REMOTE_DETECTOR_KEY", "sk-remote")
    detector = RemoteDetector({"endpoint_url": "https://detector.example.com/v1",
                               "auth_type": "bearer", "auth_value": "env:REMOTE_DETECTOR_KEY"})
    assert detector.auth_value == "sk-remote"


def test_custom_resolvers_plug_in(monkeypatch):
    monkeypatch.setitem(secret_refs._RESOLVERS, "vault", lambda path: {"llm/openai": "sk-vault"}.get(path))
    assert secret_refs.resolve("vault:llm/openai") == "sk-vault"
    with pytest.raises(secret_refs.SecretRefError, match="resolved to nothing"):
        secret_refs.resolve("vault:missing")


# -- configuration ------------------------------------------------------------------

@pytest.mark.parametrize("provider,message", [
    ({"format": "openai", "api_key": "sk-plaintext"}, "reference"),
    ({"format": "openai", "base_url": "http://api.example.com/v1", "api_key": "env:K"}, "inside your network"),
    ({"format": "azure_openai", "base_url": "https://x.openai.azure.com", "api_key": "env:K"}, "api_version"),
    ({"format": "openai", "api_key": "env:K", "headers": {"Authorization": "Bearer x"}}, "credentials"),
    ({"format": "cohere", "api_key": "env:K"}, "format"),
    ({"format": "openai", "api_key": "env:K", "tiemout_seconds": 3}, "unknown"),
])
def test_invalid_providers_are_refused(provider, message):
    with pytest.raises(GatewayConfigError, match=message):
        from_document({"providers": {"p": provider}})


def test_in_network_http_providers_are_allowed():
    cfg = from_document({"providers": {"vllm": {"format": "openai", "base_url": "http://vllm.internal:8000/v1",
                                                "api_key": "none:"}}})
    assert cfg.providers["vllm"].base_url == "http://vllm.internal:8000/v1"


@pytest.mark.parametrize("key,message", [
    ({"id": "a", "key_sha256": "not-hex"}, "64-character"),
    ({"id": "a", "key_sha256": DIGEST, "providers": ["nope"]}, "unknown provider"),
    ({"id": "a", "key_sha256": DIGEST, "limits": {"tokens_per_day": -1}}, "positive"),
    ({"key_sha256": DIGEST}, "needs an id"),
])
def test_invalid_keys_are_refused(key, message):
    with pytest.raises(GatewayConfigError, match=message):
        from_document({**_doc(), "keys": [key]})


def test_duplicate_keys_are_refused():
    with pytest.raises(GatewayConfigError, match="twice"):
        from_document({**_doc(), "keys": [{"id": "a", "key_sha256": DIGEST}, {"id": "b", "key_sha256": DIGEST}]})


def test_a_key_given_as_a_reference_is_stored_as_a_digest(monkeypatch):
    monkeypatch.setenv("APP_KEY", "zxg_secret")
    cfg = from_document({**_doc(), "keys": [{"id": "a", "key": "env:APP_KEY"}]})
    assert list(cfg.keys_by_digest) == [hashlib.sha256(b"zxg_secret").hexdigest()]


def test_provider_selection_never_guesses():
    cfg = from_document({"providers": {
        "openai": {"format": "openai", "api_key": "env:K"},
        "vllm": {"format": "openai", "base_url": "http://127.0.0.1:8000/v1"},
        "claude": {"format": "anthropic", "api_key": "env:K"}},
        "keys": [{"id": "a", "key_sha256": DIGEST}]})
    key = cfg.keys_by_digest[DIGEST]
    with pytest.raises(LookupError, match="several providers"):
        cfg.provider_for(key, "openai")
    assert cfg.provider_for(key, "openai", "vllm").name == "vllm"
    assert cfg.provider_for(key, "anthropic").name == "claude"
    with pytest.raises(LookupError, match="does not speak"):
        cfg.provider_for(key, "anthropic", "openai")


def test_zero_configuration_providers_come_from_the_usual_variables():
    cfg = from_environment({"OPENAI_API_KEY": "sk", "ANTHROPIC_API_KEY": "sk-ant"})
    assert set(cfg.providers) == {"openai", "anthropic"}
    assert cfg.providers["openai"].api_key == "env:OPENAI_API_KEY"
    assert not from_environment({}).enabled


def test_managed_mode_reads_the_bundle_section():
    assert not from_bundle_policy({"pii": {"enabled": True}}).enabled
    cfg = from_bundle_policy({"gateway": _doc()})
    assert cfg.enabled and cfg.source == "bundle"
    with pytest.raises(GatewayConfigError):
        from_bundle_policy({"gateway": {"providers": {"p": {"api_key": "sk-plain"}}}})


# -- prices -------------------------------------------------------------------------

@pytest.mark.parametrize("model,expected", [
    ("gpt-4o-mini", "gpt-4o-mini"), ("gpt-4o-mini-2024-07-18", "gpt-4o-mini"),
    ("gpt-4o-2024-08-06", "gpt-4o"), ("claude-3-5-haiku-20241022", "claude-3-5-haiku"),
    ("claude-sonnet-4-5-20250929", "claude-sonnet-4-5"), ("openai/gpt-4.1", "gpt-4.1"),
    ("gpt-4o-mini-search-preview", None), ("o3-pro", None), ("my-finetune", None),
])
def test_price_lookup(model, expected):
    table = PriceTable()
    found = table.lookup(model)
    assert (found is None) == (expected is None)
    if expected:
        assert found == table.table[expected]


def test_cost_uses_cached_and_override_prices():
    table = PriceTable({"house-model": {"input_per_mtok": 1.0, "output_per_mtok": 2.0}})
    assert table.cost("house-model", Usage(1_000_000, 500_000)) == pytest.approx(2.0)
    assert table.cost("gpt-4o-mini", Usage(1000, 0, 400)) == pytest.approx((600 * 0.15 + 400 * 0.075) / 1e6)
    assert table.cost("unknown-model", Usage(10, 10)) is None


# -- what a bundle may name (managed mode) --------------------------------------------

@pytest.fixture
def restricted():
    secret_refs.restrict_references(True)
    yield
    secret_refs.restrict_references(False)


def test_a_bundle_cannot_name_arbitrary_host_variables(restricted, monkeypatch):
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "host-cloud-credential")
    monkeypatch.setenv("ZNYX_SECRET_OPENAI", "sk-allowed")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-standard")
    with pytest.raises(secret_refs.SecretRefError, match="does not allow"):
        secret_refs.resolve("env:AWS_SECRET_ACCESS_KEY")
    assert secret_refs.resolve("env:ZNYX_SECRET_OPENAI") == "sk-allowed"
    assert secret_refs.resolve("env:OPENAI_API_KEY") == "sk-standard"
    assert secret_refs.maybe_resolve("env:AWS_SECRET_ACCESS_KEY") == ""


def test_the_operator_sets_what_a_bundle_may_name(restricted, monkeypatch):
    monkeypatch.setenv("ZNYX_SECRET_ENV_ALLOW", "TEAM_LLM_*")
    monkeypatch.setenv("TEAM_LLM_KEY", "sk-team")
    monkeypatch.setenv("ZNYX_SECRET_OPENAI", "sk-default-prefix")
    assert secret_refs.resolve("env:TEAM_LLM_KEY") == "sk-team"
    with pytest.raises(secret_refs.SecretRefError):
        secret_refs.resolve("env:ZNYX_SECRET_OPENAI")


def test_a_bundle_cannot_read_files_outside_the_secrets_directory(restricted, monkeypatch, tmp_path):
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir()
    (secrets_dir / "openai").write_text("sk-mounted")
    outside = tmp_path / "id_rsa"
    outside.write_text("PRIVATE")
    monkeypatch.setenv("ZNYX_SECRETS_DIR", str(secrets_dir))
    assert secret_refs.resolve(f"file:{secrets_dir}/openai") == "sk-mounted"
    for ref in (f"file:{outside}", f"file:{secrets_dir}/../id_rsa", "file:/etc/passwd"):
        with pytest.raises(secret_refs.SecretRefError, match="outside"):
            secret_refs.resolve(ref)
    link = secrets_dir / "sneaky"
    link.symlink_to(outside)
    with pytest.raises(secret_refs.SecretRefError, match="outside"):
        secret_refs.resolve(f"file:{link}")


def test_local_configuration_is_unrestricted(monkeypatch):
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "operator-chose-this")
    assert secret_refs.resolve("env:AWS_SECRET_ACCESS_KEY") == "operator-chose-this"
