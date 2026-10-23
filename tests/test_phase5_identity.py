"""Phase 5, part 3: agent identity. A person's identity-provider token (JWT) is verified and
exchanged for a short-lived delegated token that acts for that person with the tools the
agent's rules grant to their groups; it is accepted wherever a gateway key is.

Keys are generated here (RSA, EC P-256, Ed25519) and tokens signed with them, so verification
runs on real signatures. The runtime app under TestClient with the fake provider, no console:
identity providers and agent rules come from the gateway file.
"""
import base64
import hashlib
import json
import os
import time

import httpx
import pytest
import yaml
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from fastapi.testclient import TestClient

import gateway_fake_provider as fake
from znyx_core import identity, usage_events

ISSUER = "https://login.corp.example"
AUDIENCE = "znyx-gateway"


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _int(value: int) -> str:
    return _b64(value.to_bytes((value.bit_length() + 7) // 8, "big"))


RSA_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
EC_KEY = ec.generate_private_key(ec.SECP256R1())
ED_KEY = ed25519.Ed25519PrivateKey.generate()
OTHER_RSA = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _jwk_rsa(key, kid):
    n = key.public_key().public_numbers()
    return {"kty": "RSA", "kid": kid, "use": "sig", "n": _int(n.n), "e": _int(n.e)}


def _jwk_ec(key, kid):
    n = key.public_key().public_numbers()
    return {"kty": "EC", "kid": kid, "crv": "P-256", "x": _b64(n.x.to_bytes(32, "big")), "y": _b64(n.y.to_bytes(32, "big"))}


def _jwk_ed(key, kid):
    raw = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return {"kty": "OKP", "kid": kid, "crv": "Ed25519", "x": _b64(raw)}


JWKS = {"keys": [_jwk_rsa(RSA_KEY, "r1"), _jwk_ec(EC_KEY, "e1"), _jwk_ed(ED_KEY, "d1")]}


def mint(claims=None, alg="RS256", kid="r1", key=None, header=None):
    now = int(time.time())
    body = {"iss": ISSUER, "aud": AUDIENCE, "sub": "u-123", "email": "dana@corp.example", "groups": ["support"],
            "iat": now, "exp": now + 600, **(claims or {})}
    head = {"alg": alg, "typ": "JWT", **({"kid": kid} if kid else {}), **(header or {})}
    signing = f"{_b64(json.dumps(head).encode())}.{_b64(json.dumps(body).encode())}".encode()
    if alg == "RS256":
        sig = (key or RSA_KEY).sign(signing, padding.PKCS1v15(), hashes.SHA256())
    elif alg == "PS256":
        sig = (key or RSA_KEY).sign(signing, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
                                    hashes.SHA256())
    elif alg == "ES256":
        r, s = decode_dss_signature((key or EC_KEY).sign(signing, ec.ECDSA(hashes.SHA256())))
        sig = r.to_bytes(32, "big") + s.to_bytes(32, "big")
    elif alg == "EdDSA":
        sig = (key or ED_KEY).sign(signing)
    else:
        sig = b"x" * 32
    return f"{signing.decode()}.{_b64(sig)}"


def _provider(**overrides):
    provider, problem = identity.clean_provider({"id": "corp", "issuer": ISSUER, "audiences": [AUDIENCE], "jwks": JWKS,
                                                 **overrides})
    assert provider is not None, problem
    return provider


# -- verification ------------------------------------------------------------------------------

@pytest.mark.parametrize("alg,kid", [("RS256", "r1"), ("PS256", "r1"), ("ES256", "e1"), ("EdDSA", "d1")])
def test_tokens_signed_with_the_providers_keys_are_accepted(alg, kid):
    provider, claims = identity.verify_token(mint(alg=alg, kid=kid), [_provider()])
    assert provider.id == "corp" and identity.person_of(provider, claims) == ("dana@corp.example", ["support"])


@pytest.mark.parametrize("token,why", [
    (lambda: mint(key=OTHER_RSA), "signature"),
    (lambda: mint(alg="none"), "not accepted"),
    (lambda: mint(alg="HS256"), "not accepted"),
    (lambda: mint({"iss": "https://evil.example"}), "issuer"),
    (lambda: mint({"aud": "someone-else"}), "audience"),
    (lambda: mint({"exp": int(time.time()) - 3600}), "expired"),
    (lambda: mint({"nbf": int(time.time()) + 3600}), "not valid yet"),
    (lambda: mint({"iat": int(time.time()) + 3600}), "future"),
    (lambda: mint({"exp": "soon"}), "not a time"),
    (lambda: mint(kid="nope"), "signature"),
    (lambda: mint(kid=None), "several"),
    (lambda: mint(header={"crit": ["x"]}), "extensions"),
    (lambda: mint(alg="ES256", kid="r1"), "signature"),
    (lambda: "a.b", "signed JWT"),
    (lambda: mint() + "x" * 20000, "signed JWT"),
])
def test_tokens_that_must_not_pass(token, why):
    with pytest.raises(identity.TokenError, match=why):
        identity.verify_token(token(), [_provider()])


def test_a_provider_can_narrow_the_algorithms():
    with pytest.raises(identity.TokenError, match="not accepted for this provider"):
        identity.verify_token(mint(alg="ES256", kid="e1"), [_provider(algorithms=["RS256"])])


@pytest.mark.parametrize("raw,problem", [
    ({"issuer": "http://login.corp.example"}, "https"),
    ({"audiences": []}, "audience"),
    ({"algorithms": ["HS256"]}, "algorithms"),
    ({"jwks": {"keys": [{**_jwk_rsa(RSA_KEY, "x"), "d": "secret"}]}}, "private"),
    ({"jwks": {"keys": [_jwk_rsa(rsa.generate_private_key(public_exponent=65537, key_size=1024), "s")]}}, "2048"),
    ({"jwks": {"keys": [{"kty": "oct", "k": "abc"}]}}, "private"),
    ({"user_claim": "a b"}, "claim names"),
])
def test_a_provider_that_cannot_work_is_refused(raw, problem):
    provider, why = identity.clean_provider({"id": "corp", "issuer": ISSUER, "audiences": [AUDIENCE], "jwks": JWKS,
                                             **raw})
    assert provider is None and problem in why


def test_what_an_agent_may_do_for_a_person():
    rules, _ = identity.clean_agent({"agent_id": "a", "groups": {"support": ["search", "lookup_*"],
                                                                  "admins": ["delete_*"]}, "all_users": ["ping"]})
    assert rules.grant(["support"]) == ("ping", "search", "lookup_*")
    assert rules.grant(["support", "admins"], ["delete_rows", "search", "nope"]) == ("delete_rows", "search")
    assert rules.grant(["support"], ["*"]) == () and rules.grant(["support"], ["lookup_*"]) == ("lookup_*",)
    assert rules.grant(["support"], ["lookup_[a-z]*"]) == ()               # a pattern counts only as itself
    assert rules.grant([]) == ("ping",)
    assert identity.clean_agent({"agent_id": "a", "max_ttl_seconds": 7200})[0] is None


# -- the runtime ---------------------------------------------------------------------------------

def _sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


KEYS = {"assistant": "zxg_id_assistant", "strict": "zxg_id_strict", "loner": "zxg_id_loner"}
INSTANCE_KEY = "zx_phase5_identity_instance"  # gitleaks:allow (test value)

GATEWAY = {
    "providers": {"openai": {"format": "openai", "base_url": "https://api.openai.com/v1",
                             "api_key": "env:FAKE_OPENAI_KEY", "region": "us"}},
    "mcp_servers": {"crm": {"url": "https://mcp.crm.example/mcp", "region": "us"}},
    "keys": [{"id": f"k-{n}", "key_sha256": _sha(v), "tenant_id": "acme", "app_id": f"app-{n}", "agent_id": f"agent-{n}",
              "providers": ["openai"], "default_provider": "openai"} for n, v in KEYS.items()],
    "identity_providers": [{"id": "corp", "issuer": ISSUER, "audiences": [AUDIENCE], "jwks": JWKS}],
    "agents": [{"agent_id": "agent-assistant", "groups": {"support": ["search", "lookup_*"], "admins": ["delete_*"]},
                "all_users": ["ping"], "max_ttl_seconds": 600},
               {"agent_id": "agent-strict", "groups": {"support": ["search"]}, "require_delegation": True}],
}
POLICY = """
default:
  pii: {enabled: true, action: REDACT}
  runtime_policy: {allowed_regions: [us, eu]}
"""


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("phase5id")
    (tmp / "policies.yaml").write_text(POLICY)
    (tmp / "gateway.yaml").write_text(yaml.safe_dump(GATEWAY))
    env = {"ZNYX_MODE": "local", "ZNYX_POLICY_PATH": str(tmp / "policies.yaml"),
           "ZNYX_GATEWAY_CONFIG": str(tmp / "gateway.yaml"), "RUNTIME_API_KEY": INSTANCE_KEY,
           "RUNTIME_REQUIRE_AUTH": "true", "ZNYX_TELEMETRY": "false",
           "ZNYX_AUDIT_SPOOL_PATH": str(tmp / "egress-audit.spool"),
           "ZNYX_JUDGE_AUDIT_SPOOL_PATH": str(tmp / "judge-audit.spool"),
           "ZNYX_USAGE_SPOOL_PATH": str(tmp / "llm-calls.spool"), "FAKE_OPENAI_KEY": "sk-upstream-openai"}
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    import znyx_runtime.install_state as install_state
    saved_state = (install_state.STATE_DIR, install_state.STATE_FILE)
    install_state.STATE_DIR = tmp / ".znyx"
    install_state.STATE_FILE = install_state.STATE_DIR / "state.json"
    from znyx_runtime.gateway.upstream import Upstream
    provider_app = fake.build_app()
    original = Upstream.client

    async def fake_client(self, provider):
        c = self._clients.get(provider.name)
        if c is None:
            c = httpx.AsyncClient(transport=httpx.ASGITransport(app=provider_app))
            self._clients[provider.name] = c
        return c

    Upstream.client = fake_client
    try:
        from znyx_runtime.main import app
        with TestClient(app) as c:
            gateway = app.state.gateway
            c.records = []
            original_on_call = gateway.on_call

            def capture(record):
                c.records.append(record)
                if original_on_call is not None:
                    original_on_call(record)

            gateway.on_call = capture
            c.gateway = gateway
            c.sink = original_on_call
            yield c
    finally:
        Upstream.client = original
        install_state.STATE_DIR, install_state.STATE_FILE = saved_state
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


@pytest.fixture(autouse=True)
def _fresh(request):
    fake.reset()
    if "client" in request.fixturenames:
        request.getfixturevalue("client").records.clear()
    yield


@pytest.fixture
def controls(client, monkeypatch):
    state = {"controls": {}}
    monkeypatch.setattr(client.gateway, "_controls", lambda: state["controls"])

    def put(**values):
        state["controls"] = dict(values)
    return put


def exchange(client, key="assistant", token=None, **extra):
    return client.post("/v1/znyx/token", json={"subject_token": token or mint(), **extra},
                       headers={"Authorization": f"Bearer {KEYS[key]}"})


TOOLS = [{"type": "function", "function": {"name": n, "parameters": {"type": "object"}}}
         for n in ("search", "lookup_order", "delete_rows", "ping")]


def chat(client, bearer, content="SAY:hi", tools=None):
    body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": content}]}
    if tools:
        body["tools"] = tools
    return client.post("/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {bearer}"})


def test_an_agent_exchanges_a_persons_token_for_a_short_lived_one(client):
    resp = exchange(client, token=mint({"exp": int(time.time()) + 3600}), expires_in=99999)
    assert resp.status_code == 200 and resp.headers["cache-control"] == "no-store"
    out = resp.json()
    assert out["access_token"].startswith("zat_") and out["token_type"] == "Bearer"
    assert out["expires_in"] == 600 and set(out["scope"].split()) == {"ping", "search", "lookup_*"}
    assert exchange(client, scope="search delete_rows").json()["scope"] == "search"
    form = client.post("/v1/znyx/token", data={"subject_token": mint(alg="ES256", kid="e1")},
                       headers={"Authorization": f"Bearer {KEYS['assistant']}"})
    assert form.status_code == 200
    short = exchange(client, token=mint({"exp": int(time.time()) + 100}))
    assert short.json()["expires_in"] <= 100                               # never outlives the person's token


@pytest.mark.parametrize("key,token,status,error", [
    ("loner", None, 403, "unauthorized_client"),                          # no rules for this agent
    ("assistant", lambda: mint(key=OTHER_RSA), 400, "invalid_grant"),
    ("assistant", lambda: mint({"groups": ["marketing"]}, kid="r1"), 200, None),    # all_users still applies
    ("assistant", lambda: mint({"exp": int(time.time()) + 10}), 400, "invalid_grant"),
])
def test_exchanges_that_are_refused(client, key, token, status, error):
    resp = exchange(client, key=key, token=token() if token else None)
    assert resp.status_code == status and (error is None or resp.json()["error"] == error)


def test_only_an_agents_own_key_may_exchange(client):
    assert client.post("/v1/znyx/token", json={"subject_token": mint()},
                       headers={"Authorization": f"Bearer {INSTANCE_KEY}"}).json()["error"] == "invalid_client"
    token = exchange(client).json()["access_token"]
    assert client.post("/v1/znyx/token", json={"subject_token": mint()},
                       headers={"Authorization": f"Bearer {token}"}).json()["error"] == "invalid_client"
    assert client.post("/v1/znyx/token", json={"subject_token": mint()}).status_code == 401
    resp = client.post("/v1/znyx/token", json={"subject_token": mint({"groups": []})},
                       headers={"Authorization": f"Bearer {KEYS['strict']}"})
    assert resp.status_code == 403 and resp.json()["error"] == "access_denied"   # nothing granted to no groups


def test_a_delegated_call_runs_for_the_person_with_only_the_granted_tools(client):
    token = exchange(client).json()["access_token"]
    resp = chat(client, token, tools=TOOLS)
    assert resp.status_code == 200
    sent = [t["function"]["name"] for t in fake.received[-1]["body"]["tools"]]
    assert sent == ["search", "lookup_order", "ping"]
    record = client.records[-1]
    assert record.delegated and record.identity_provider == "corp" and record.user.startswith("u_")
    assert record.user != "dana@corp.example"                              # pseudonymised like any user id
    event = usage_events.validate(json.loads(json.dumps(client.sink.event_for(record), default=str)))
    assert event["delegated"] is True and event["identity_provider"] == "corp"
    fake_answer = 'TOOL:delete_rows:{"all": true}'
    resp = chat(client, token, content=fake_answer, tools=TOOLS[:1])
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "content_filter" and identity.RULE_NOT_DELEGATED in client.records[-1].rule_ids


def test_an_agent_that_acts_only_for_people_refuses_its_key_alone(client):
    resp = chat(client, KEYS["strict"])
    assert resp.status_code == 403 and resp.json()["error"]["code"] == "delegation_required"
    token = exchange(client, key="strict").json()["access_token"]
    assert chat(client, token).status_code == 200
    out = client.post("/v1/evaluate/input", json={"request_id": "r", "tenant_id": "acme", "app_id": "app-strict",
                                                  "text": "hi"},
                      headers={"Authorization": f"Bearer {KEYS['strict']}"}).json()
    assert out["decision"] == "BLOCK"
    resp = client.post("/v1/mcp/crm", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                       headers={"Authorization": f"Bearer {KEYS['strict']}"})
    assert resp.status_code == 403


def test_the_evaluate_api_and_mcp_honour_the_grant(client):
    token = exchange(client).json()["access_token"]
    body = {"request_id": "r", "tenant_id": "acme", "app_id": "app-assistant", "tool_args": {}}
    headers = {"Authorization": f"Bearer {token}"}
    assert client.post("/v1/evaluate/tool", json={**body, "tool_name": "delete_rows"},
                       headers=headers).json()["rule_hits"][0]["rule_id"] == identity.RULE_NOT_DELEGATED
    assert client.post("/v1/evaluate/tool", json={**body, "tool_name": "search"}, headers=headers).json()["decision"] == "ALLOW"
    resp = client.post("/v1/mcp/crm", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, headers=headers)
    assert [t["name"] for t in resp.json()["result"]["tools"]] == ["search"]
    resp = client.post("/v1/mcp/crm", json={"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                            "params": {"name": "delete_record", "arguments": {}}}, headers=headers)
    assert resp.json()["result"]["isError"]


def test_a_person_can_be_stopped(client, controls):
    token = exchange(client).json()["access_token"]
    controls(stops=[{"target": "user", "match": "*@corp.example", "reason": "account compromised"}])
    resp = chat(client, token)
    assert resp.status_code == 403 and "account compromised" in resp.text
    assert chat(client, KEYS["assistant"]).status_code == 200                # the key alone: no person, not stopped
    assert exchange(client).json()["error"] == "access_denied"


def test_a_token_can_be_revoked_and_expires(client):
    token = exchange(client).json()["access_token"]
    assert client.request("DELETE", "/v1/znyx/token", headers={"Authorization": f"Bearer {token}"}).json()["revoked"]
    assert chat(client, token).status_code == 401
    token = exchange(client).json()["access_token"]
    from znyx_core.state_store import get_state_store
    store = get_state_store()
    key = "pin:gw:agent-token:" + hashlib.sha256(token.encode()).hexdigest()
    entry = json.loads(store.get_value(key))
    store.set_value(key, json.dumps({**entry, "exp": time.time() - 1}), 60)
    assert chat(client, token).status_code == 401


def test_a_change_in_the_rules_applies_to_tokens_already_issued(client, monkeypatch):
    import dataclasses
    token = exchange(client, token=mint({"groups": ["support", "admins"]})).json()["access_token"]
    chat(client, token, tools=TOOLS)
    assert "delete_rows" in [t["function"]["name"] for t in fake.received[-1]["body"]["tools"]]
    cfg = client.gateway.config
    rules, _ = identity.clean_agent({"agent_id": "agent-assistant", "groups": {"support": ["search"]}})
    monkeypatch.setattr(client.gateway, "config", dataclasses.replace(cfg, agents=(rules,)))
    chat(client, token, tools=TOOLS)
    assert [t["function"]["name"] for t in fake.received[-1]["body"]["tools"]] == ["search"]
    monkeypatch.setattr(client.gateway, "config", dataclasses.replace(cfg, agents=()))
    assert chat(client, token).status_code == 401                          # no rules any more
    monkeypatch.setattr(client.gateway, "config", dataclasses.replace(cfg, identity_providers=()))
    assert chat(client, token).status_code == 401                          # provider removed


def test_bad_identity_settings_in_the_gateway_file_stop_the_start():
    from znyx_runtime.gateway import config as gw_config
    with pytest.raises(gw_config.GatewayConfigError, match="identity_providers"):
        gw_config.from_document({**GATEWAY, "identity_providers": [{"id": "x", "issuer": "http://x", "audiences": ["a"],
                                                                    "jwks": JWKS}]}, source="test")
    with pytest.raises(gw_config.GatewayConfigError, match="agents"):
        gw_config.from_document({**GATEWAY, "agents": [{"agent_id": "a", "max_ttl_seconds": 1}]}, source="test")
    with pytest.raises(gw_config.GatewayConfigError, match="unknown section"):
        gw_config.from_document({"providers": {}, "agents": []}, source="bundle", managed=True)
