"""AI gateway Phase 5 part 3: the adversarial review's findings (2026-10-23), kept as
regression tests. Each test failed before its fix."""
import dataclasses
import json
import math
import os

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

import gateway_fake_provider as fake
from test_phase5_identity import AUDIENCE, ISSUER, JWKS, _sha, mint
from znyx_core import identity

KEYS = {"assistant": "zxg_rv_assistant", "batchy": "zxg_rv_batchy", "approvey": "zxg_rv_approvey"}
INSTANCE_KEY = "zx_review_p5c_instance"  # gitleaks:allow (test value)


def _key(name, **extra):
    return {"id": f"k-{name}", "key_sha256": _sha(KEYS[name]), "tenant_id": "acme", "app_id": f"app-{name}",
            "agent_id": f"agent-{name}", "providers": ["openai"], "default_provider": "openai", **extra}


GATEWAY = {
    "providers": {"openai": {"format": "openai", "base_url": "https://api.openai.com/v1",
                             "api_key": "env:FAKE_OPENAI_KEY", "region": "us"}},
    "keys": [_key("assistant"), _key("batchy", optimisation={"enable": ["batch"]}), _key("approvey")],
    "identity_providers": [{"id": "corp", "issuer": ISSUER, "audiences": [AUDIENCE], "jwks": JWKS}],
    "agents": [{"agent_id": "agent-assistant", "groups": {"support": ["search", "lookup_*"], "admins": ["delete_*"]},
                "max_ttl_seconds": 600},
               {"agent_id": "agent-batchy", "groups": {"support": ["search"]}},
               {"agent_id": "agent-approvey", "groups": {"support": ["search"]}}],
}
POLICY = """
default:
  pii: {enabled: true, action: REDACT}
  runtime_policy: {allowed_regions: [us, eu]}
tenants:
  acme:
    apps:
      app-approvey:
        approval:
          tools: ["search"]
          timeout_seconds: 10
      app-jb:
        jailbreak:
          enabled: true
"""

SEARCH = [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("reviewp5c")
    (tmp / "policies.yaml").write_text(POLICY)
    (tmp / "gateway.yaml").write_text(yaml.safe_dump(GATEWAY))
    env = {"ZNYX_MODE": "local", "ZNYX_POLICY_PATH": str(tmp / "policies.yaml"),
           "ZNYX_GATEWAY_CONFIG": str(tmp / "gateway.yaml"), "RUNTIME_API_KEY": INSTANCE_KEY,
           "RUNTIME_REQUIRE_AUTH": "true", "ZNYX_TELEMETRY": "false",
           "ZNYX_AUDIT_SPOOL_PATH": str(tmp / "egress-audit.spool"),
           "ZNYX_JUDGE_AUDIT_SPOOL_PATH": str(tmp / "judge-audit.spool"),
           "ZNYX_USAGE_SPOOL_PATH": str(tmp / "llm-calls.spool"), "ZNYX_BATCH_DB": str(tmp / "batches.sqlite"),
           "FAKE_OPENAI_KEY": "sk-upstream-openai"}
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
            c.gateway = app.state.gateway
            yield c
    finally:
        Upstream.client = original
        install_state.STATE_DIR, install_state.STATE_FILE = saved_state
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


@pytest.fixture(autouse=True)
def _fresh():
    fake.reset()
    yield
    fake.reset()


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


def chat(client, bearer, content="SAY:hi", tools=None, headers=None):
    body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": content}]}
    if tools:
        body["tools"] = tools
    return client.post("/v1/chat/completions", json=body,
                       headers={"Authorization": f"Bearer {bearer}", **(headers or {})})


def _status_or_exception(fn):
    try:
        return fn().status_code
    except Exception as exc:  # noqa: BLE001 - TestClient re-raises server errors
        return f"server raised {type(exc).__name__}: {exc}"


# -- F1: a `user` stop is matched case-sensitively ------------------------------------------------

def test_user_stop_matches_the_person_whatever_the_case_of_their_email(client, controls):
    """The console tells the admin to type 'sam@example.com'. Identity providers (Entra ID, Okta)
    emit the email as stored, often 'Sam@Example.com'. The stop must still apply."""
    controls(stops=[{"target": "user", "match": "dana@corp.example", "reason": "account compromised"}])
    resp = exchange(client, token=mint({"email": "Dana@Corp.example"}))
    assert resp.status_code == 403, f"stopped person got a delegated token: {resp.status_code} {resp.text}"


def test_user_stop_applies_when_the_token_lacks_the_user_claim(client, controls):
    """Same person, but the agent exchanges a token without `email` (an access token, say): the
    person silently becomes `sub` and the stop on their email no longer matches."""
    controls(stops=[{"target": "user", "match": "dana@corp.example", "reason": "account compromised"}])
    claims = {"email": None, "sub": "00u1dana"}
    resp = exchange(client, token=mint(claims))
    assert resp.status_code in (400, 403), f"stopped person got a delegated token: {resp.status_code} {resp.text}"


def test_unverified_email_is_not_taken_as_the_person(client):
    """The default user claim is `email`; `email_verified: false` is not looked at, so the
    person is whatever address the account claims (stops and attribution key on it)."""
    resp = exchange(client, token=mint({"email": "ceo@corp.example", "email_verified": False}))
    assert resp.status_code == 400, f"unverified email accepted as the person: {resp.status_code}"


def test_control_user_stop_with_the_exact_case_applies(client, controls):
    controls(stops=[{"target": "user", "match": "dana@corp.example", "reason": "account compromised"}])
    assert exchange(client).status_code == 403


# -- F2: expires_in = NaN ----------------------------------------------------------------------------

def _nan_exchange(client):
    body = '{"subject_token": "%s", "expires_in": NaN}' % mint()
    raw = TestClient(client.app, raise_server_exceptions=False)
    return raw.post("/v1/znyx/token", content=body,
                    headers={"Authorization": f"Bearer {KEYS['assistant']}", "content-type": "application/json"})


def test_expires_in_nan_is_refused_as_invalid_request(client):
    from znyx_core.state_store import get_state_store
    resp = _nan_exchange(client)
    leaked = [k for k, v in get_state_store()._data.items()
              if k.startswith("pin:gw:agent-token:") and isinstance(v.get("expires"), float) and math.isnan(v["expires"])]
    assert resp.status_code == 400 and not leaked, (
        f"status {resp.status_code}; {len(leaked)} token record(s) stored with a NaN expiry (never expire)")


def test_expires_in_nan_takes_the_shared_state_store_down(client):
    """With Redis (fail mode closed), one exchange with expires_in NaN makes set_value raise
    ValueError (int(nan * 1000)) inside its try, which the store reads as an outage: Redis is
    skipped for 5 s and every call on this replica that needs the store fails, including
    other people's valid delegated tokens."""
    redis = pytest.importorskip("redis")
    client_r = redis.Redis.from_url("redis://127.0.0.1:6399/14", socket_timeout=1)
    try:
        client_r.ping()
    except Exception:  # noqa: BLE001
        pytest.skip("test Redis on 127.0.0.1:6399 not reachable")
    from znyx_core.state_store import RedisStateStore, set_state_store
    set_state_store(RedisStateStore(client_r, fail_mode="closed"))
    try:
        token = exchange(client).json()["access_token"]
        assert chat(client, token).status_code == 200
        _nan_exchange(client)
        after = _status_or_exception(lambda: chat(client, token))
        assert after == 200, f"a valid delegated token stopped working after one NaN exchange: {after}"
    finally:
        for k in client_r.scan_iter("znyx:pin:gw:agent-token:*"):
            client_r.delete(k)


# -- F3: expired delegated-token records are never purged from the in-memory store -----------------

def test_expired_token_records_do_not_accumulate_in_the_memory_store(client, monkeypatch):
    from znyx_core.state_store import MemoryStateStore, set_state_store
    clock = [1000.0]
    store = MemoryStateStore(max_keys=50, clock=lambda: clock[0])
    set_state_store(store)
    from znyx_runtime.gateway import identity as delegations
    from znyx_runtime.gateway.identity import MAX_EXCHANGES_PER_MINUTE
    monkeypatch.setattr(delegations, "_minute", lambda: 1)            # one window, whatever the clock
    for _ in range(MAX_EXCHANGES_PER_MINUTE):
        assert exchange(client, expires_in=60).status_code == 200
    clock[0] += 10_000                                    # every token expired long ago
    for i in range(20):                                   # ordinary pinned traffic carries on
        store.consume_token(f"pin:gw:rpm:k{i}", 1.0, 10.0)
    leftover = [k for k in store._data if k.startswith("pin:gw:agent-token:")]
    assert not leftover, (f"{len(leftover)} expired token records still held; store holds {len(store._data)} "
                          f"keys with max_keys=50 (pinned keys are never evicted, only expired on access)")


def test_many_tokens_do_not_break_the_memory_store_for_everything_else(client):
    """Past max_keys pinned entries, creating any unpinned key (jailbreak conversation history,
    abuse flood counters, unbounded-consumption history, loop counters) raises KeyError:
    _entry() inserts it, _evict() deletes it as the only evictable key, move_to_end() fails."""
    from znyx_core.state_store import MemoryStateStore, set_state_store
    store = MemoryStateStore(max_keys=50)
    set_state_store(store)
    for _ in range(60):
        assert exchange(client, expires_in=60).status_code == 200
    try:
        store.push_capped("jailbreak:history:0123456789abcdef", "{}", 10, 600)
        outcome = "ok"
    except Exception as exc:  # noqa: BLE001
        outcome = f"{type(exc).__name__}: {exc}"
    assert outcome == "ok", f"the state store refuses new unpinned keys after 60 delegated tokens: {outcome}"



# -- F2: batch calls skipped the delegated tool limit and the person's stops -----------------------

def test_a_delegated_call_marked_not_urgent_runs_live_with_the_persons_limits(client, controls):
    """A call for a person never goes to the batch queue, where its tools and the person's stops
    could not be checked when it runs: it runs live, the person's limits applied."""
    fake.scripted = lambda body: {
        "id": "chatcmpl-b", "object": "chat.completion", "model": body.get("model"),
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "delete_rows", "arguments": "{\"all\": true}"}}]}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    token = exchange(client, key="batchy").json()["access_token"]     # grants: search only
    resp = chat(client, token, content="SAY:clean up", tools=SEARCH, headers={"x-znyx-urgency": "low"})
    assert resp.status_code == 200 and resp.json()["choices"][0]["finish_reason"] == "content_filter"
    fake.scripted = None
    controls(stops=[{"target": "user", "match": "dana@corp.example", "reason": "account compromised"}])
    sent = len(fake.received)
    resp = chat(client, token, content="SAY:summarise my mail", headers={"x-znyx-urgency": "low"})
    assert resp.status_code == 403 and len(fake.received) == sent
    controls()
    queued = chat(client, KEYS["batchy"], content="SAY:the key alone", headers={"x-znyx-urgency": "low"})
    assert queued.status_code == 202                                   # the key alone still queues


def test_control_live_answer_blocks_the_same_tool_call(client):
    token = exchange(client, key="batchy").json()["access_token"]
    resp = chat(client, token, content='TOOL:delete_rows:{"all": true}', tools=SEARCH)
    assert resp.json()["choices"][0]["finish_reason"] == "content_filter"


# -- F5: a `user` stop that arrives while a tool call waits for approval is ignored -----------------

def test_user_stop_during_approval_withholds_the_tool_call(client, controls, monkeypatch):
    token = exchange(client, key="approvey").json()["access_token"]

    async def ask(**_kw):
        controls(stops=[{"target": "user", "match": "dana@corp.example", "reason": "account compromised"}])
        return True, "approved"

    monkeypatch.setattr(client.gateway.approvals, "ask", ask)
    resp = chat(client, token, content='TOOL:search:{"q": "x"}', tools=SEARCH)
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "content_filter", f"tool call released for a stopped person: {choice}"


def test_control_agent_stop_during_approval_withholds_the_tool_call(client, controls, monkeypatch):
    token = exchange(client, key="approvey").json()["access_token"]

    async def ask(**_kw):
        controls(stops=[{"target": "agent", "match": "agent-approvey", "reason": "rogue"}])
        return True, "approved"

    monkeypatch.setattr(client.gateway.approvals, "ask", ask)
    resp = chat(client, token, content='TOOL:search:{"q": "x"}', tools=SEARCH)
    assert resp.json()["choices"][0]["finish_reason"] == "content_filter"


# -- F6: a token keeps acting for the person after the rules stop granting them anything ----------

def test_token_refused_once_the_rules_grant_the_person_nothing(client, monkeypatch):
    token = exchange(client).json()["access_token"]                   # support: search, lookup_*
    cfg = client.gateway.config
    rules, _ = identity.clean_agent({"agent_id": "agent-assistant", "groups": {"admins": ["delete_*"]}})
    monkeypatch.setattr(client.gateway, "config", dataclasses.replace(cfg, agents=(rules,)))
    assert exchange(client).json()["error"] == "access_denied"       # a new token is refused...
    assert chat(client, token).status_code == 401, "...but the old one still acts for the person"


# -- F7: malformed person tokens give 500, not invalid_grant -----------------------------------------

@pytest.mark.parametrize("make", [
    lambda: ".".join(mint().split(".")[:2] + ["A"]),                                     # bad base64 length
    lambda: "%s.%s.%s" % (__import__("base64").urlsafe_b64encode(b"[" * 3000 + b"]" * 3000).rstrip(b"=").decode(),
                          *mint().split(".")[1:]),                                       # deep JSON nesting
])
def test_malformed_subject_tokens_are_invalid_grant(client, make):
    raw = TestClient(client.app, raise_server_exceptions=False)
    resp = raw.post("/v1/znyx/token", json={"subject_token": make()},
                    headers={"Authorization": f"Bearer {KEYS['assistant']}"})
    assert resp.status_code == 400, f"{resp.status_code} {resp.text[:80]}"


# -- F8: a scope sent as a JSON list is ignored (full grant) -----------------------------------------

def test_scope_as_a_list_narrows_or_is_refused(client):
    out = exchange(client, scope=["search"]).json()
    assert out.get("scope") == "search", out
    assert exchange(client, scope={"search": True}).json()["error"] == "invalid_request"
    assert exchange(client, scope=["search", 3]).json()["error"] == "invalid_request"


# -- F9: a form-encoded exchange can never set expires_in ---------------------------------------------

def test_form_encoded_expires_in_is_honoured(client):
    resp = client.post("/v1/znyx/token", data={"subject_token": mint(), "expires_in": "120"},
                       headers={"Authorization": f"Bearer {KEYS['assistant']}"})
    assert resp.status_code == 200 and resp.json()["expires_in"] <= 120, f"{resp.status_code} {resp.text}"


# -- F4: exchanges are rate limited per agent key ------------------------------------------------------

def test_exchanges_are_limited_per_agent_key(client, monkeypatch):
    from znyx_runtime.gateway import identity as delegations
    from znyx_runtime.gateway.identity import MAX_EXCHANGES_PER_MINUTE
    monkeypatch.setattr(delegations, "_minute", lambda: 1)            # one window, whatever the clock
    codes = [exchange(client, expires_in=60).status_code for _ in range(MAX_EXCHANGES_PER_MINUTE + 3)]
    assert codes[:MAX_EXCHANGES_PER_MINUTE] == [200] * MAX_EXCHANGES_PER_MINUTE
    last = exchange(client)
    assert last.status_code == 429 and last.json()["error"] == "temporarily_unavailable"
    assert last.headers["retry-after"] == "60"
    assert exchange(client, key="approvey").status_code == 200        # other agents are not affected


def test_a_lowered_maximum_lifetime_applies_to_tokens_already_issued(client, monkeypatch):
    token = exchange(client).json()["access_token"]
    from znyx_core.state_store import get_state_store
    import hashlib
    import time
    store = get_state_store()
    name = "pin:gw:agent-token:" + hashlib.sha256(token.encode()).hexdigest()
    entry = json.loads(store.get_value(name))
    store.set_value(name, json.dumps({**entry, "iat": time.time() - 300}), 600)
    assert chat(client, token).status_code == 200
    cfg = client.gateway.config
    rules, _ = identity.clean_agent({"agent_id": "agent-assistant", "groups": {"support": ["search"]},
                                     "max_ttl_seconds": 120})
    monkeypatch.setattr(client.gateway, "config", dataclasses.replace(cfg, agents=(rules,)))
    assert chat(client, token).status_code == 401


def test_infinite_or_huge_times_in_a_person_token_are_refused():
    from test_phase5_identity import _provider
    provider = _provider()
    for exp in (10 ** 30, 1e300):
        with pytest.raises(identity.TokenError):
            identity.verify_token(mint({"exp": exp}), [provider])
    bad = mint().split(".")
    import base64
    claims = json.loads(base64.urlsafe_b64decode(bad[1] + "=" * (-len(bad[1]) % 4)))
    raw = json.dumps(claims).replace(str(claims["exp"]), "Infinity").encode()
    with pytest.raises(identity.TokenError):
        identity.verify_token(".".join([bad[0], base64.urlsafe_b64encode(raw).rstrip(b"=").decode(), bad[2]]),
                              [provider])


def test_the_governance_records_never_hold_the_raw_person_when_ids_are_hashed(client, monkeypatch):
    from pathlib import Path
    settings = dataclasses.replace(client.gateway.config.settings, user_ids="hash")
    monkeypatch.setattr(client.gateway, "config", dataclasses.replace(client.gateway.config, settings=settings))
    path = Path(client.gateway.governance.path)
    before = len(path.read_text().splitlines()) if path.exists() else 0
    token = exchange(client).json()["access_token"]
    client.request("DELETE", "/v1/znyx/token", headers={"Authorization": f"Bearer {token}"})
    import time
    deadline = time.time() + 3
    while time.time() < deadline:
        records = [json.loads(x) for x in path.read_text().splitlines()[before:]] if path.exists() else []
        if {"token.issued", "token.revoked"} <= {r["type"] for r in records}:
            break
        time.sleep(0.05)
    people = {r.get("person") for r in records if r["type"] in ("token.issued", "token.revoked")}
    assert len(people) == 1 and next(iter(people)).startswith("u_")
    assert "dana@corp.example" not in path.read_text()[sum(len(x) + 1 for x in path.read_text().splitlines()[:before]):]
