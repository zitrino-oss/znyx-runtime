"""Phase 5, part 1: the kill switch (stops and quarantines in the signed controls or the
gateway file, usable keys without a publish) and the tool names usage events carry for the
AI inventory.

The runtime app under TestClient with the fake provider, as in test_phase4_optimise.py. The
console's controls are put in place by replacing the gateway's view of them.
"""
import hashlib
import json
import logging
import os
from datetime import datetime, timedelta, timezone

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

import gateway_fake_provider as fake
from znyx_core import governance, usage_events

POLICY = """
default:
  pii:
    enabled: true
    action: REDACT
  secrets:
    enabled: true
  jailbreak:
    enabled: true
    threshold: 60
  runtime_policy:
    allowed_regions: [us, eu]
"""

INSTANCE_KEY = "zx_phase5_instance_key"  # gitleaks:allow (test value)


def _sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


KEYS = {"alpha": "zxg_alpha_key", "beta": "zxg_beta_key", "batcher": "zxg_p5batch_key"}


def _key_entry(name):
    entry = {"id": f"k-{name}", "key_sha256": _sha(KEYS[name]), "tenant_id": "acme", "app_id": f"app-{name}",
             "agent_id": f"agent-{name}", "providers": ["openai", "anthropic", "azure"], "default_provider": "openai"}
    if name == "batcher":
        entry["optimisation"] = {"enable": ["batch"]}
    return entry


GATEWAY = {
    "providers": {
        "openai": {"format": "openai", "base_url": "https://api.openai.com/v1",
                   "api_key": "env:FAKE_OPENAI_KEY", "region": "us"},
        "anthropic": {"format": "anthropic", "base_url": "https://api.anthropic.com",
                      "api_key": "env:FAKE_ANTHROPIC_KEY", "region": "us"},
        "azure": {"format": "azure_openai", "base_url": "https://acme.openai.azure.com",
                  "api_key": "env:FAKE_OPENAI_KEY", "api_version": "2024-10-21", "region": "us"},
    },
    "keys": [_key_entry(name) for name in KEYS],
}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("phase5")
    (tmp / "policies.yaml").write_text(POLICY)
    (tmp / "gateway.yaml").write_text(yaml.safe_dump(GATEWAY))
    env = {"ZNYX_MODE": "local", "ZNYX_POLICY_PATH": str(tmp / "policies.yaml"),
           "ZNYX_GATEWAY_CONFIG": str(tmp / "gateway.yaml"), "RUNTIME_API_KEY": INSTANCE_KEY,
           "RUNTIME_REQUIRE_AUTH": "true", "ZNYX_TELEMETRY": "false",
           "ZNYX_AUDIT_SPOOL_PATH": str(tmp / "egress-audit.spool"),
           "ZNYX_JUDGE_AUDIT_SPOOL_PATH": str(tmp / "judge-audit.spool"),
           "ZNYX_USAGE_SPOOL_PATH": str(tmp / "llm-calls.spool"),
           "ZNYX_BATCH_DIR": str(tmp / "batches"),
           "FAKE_OPENAI_KEY": "sk-upstream-openai", "FAKE_ANTHROPIC_KEY": "sk-ant-upstream"}
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
    """Set the console's signed controls for one test."""
    state = {"controls": {}}
    monkeypatch.setattr(client.gateway, "_controls", lambda: state["controls"])

    def put(**values):
        state["controls"] = dict(values)
    return put


def openai_call(client, messages, key="alpha", model="gpt-4o-mini", headers=None, **extra):
    body = {"model": model, "messages": messages, **extra}
    return client.post("/v1/chat/completions", json=body,
                       headers={"Authorization": f"Bearer {KEYS[key]}", **(headers or {})})


def anthropic_call(client, messages, key="alpha", model="claude-sonnet-4-5", **extra):
    body = {"model": model, "max_tokens": 100, "messages": messages, **extra}
    headers = {"x-api-key": KEYS[key], "anthropic-version": "2023-06-01", "x-znyx-provider": "anthropic"}
    return client.post("/v1/messages", json=body, headers=headers)


def user(text):
    return {"role": "user", "content": text}


TOOLS = [{"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
         for name in ("search", "delete_records", "send_email")]
ANTHROPIC_TOOLS = [{"name": name, "input_schema": {"type": "object"}} for name in ("search", "delete_records")]


def sent_tools(index=-1):
    body = fake.received[index]["body"]
    tools = body.get("tools") or []
    return [t.get("name") or (t.get("function") or {}).get("name") for t in tools]


def evaluate(client, path, body, key="alpha"):
    body = {"tenant_id": "acme", "app_id": f"app-{key}", **body}
    return client.post(path, json=body, headers={"Authorization": f"Bearer {KEYS[key]}"})


# -- the shared stop logic ------------------------------------------------------------------

@pytest.mark.parametrize("raw,problem", [
    ({"target": "agent", "match": "a1"}, None),
    ({"target": "agent", "match": "a1", "mode": "quarantine", "until": "2030-01-01T00:00:00Z"}, None),
    ({"target": "everything"}, None),
    ({"target": "everything", "match": "x"}, "takes no match"),
    ({"target": "model", "match": "gpt-4o*", "mode": "quarantine"}, "cannot be quarantined"),
    ({"target": "agent"}, "needs a match"),
    ({"target": "agent", "match": "a\x00b"}, "needs a match"),
    ({"target": "agent", "match": "x" * 201}, "needs a match"),
    ({"target": "robot", "match": "r"}, "unknown target"),
    ({"target": "agent", "match": "a", "mode": "pause"}, "unknown mode"),
    ({"target": "agent", "match": "a", "until": "soon"}, "not a time"),
    ({"target": "agent", "match": "a", "reason": 7}, "must be text"),
    ("agent", "not a mapping"),
])
def test_a_stop_is_read_the_same_everywhere(raw, problem):
    stop, why = governance.clean_stop(raw)
    if problem is None:
        assert stop is not None and why is None
    else:
        assert stop is None and problem in why


def test_bad_stops_from_the_console_are_dropped_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING):
        stops = governance.clean_stops([{"target": "agent", "match": "a"}, {"target": "robot"}, 5], "test")
    assert [s.match for s in stops] == ["a"] and "unknown target" in caplog.text
    assert governance.clean_stops("nope") == () and governance.clean_stops(None) == ()


def test_who_a_stop_concerns():
    past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    stops = governance.StopSet(governance.clean_stops([
        {"id": "s1", "target": "agent", "match": "agent-a", "reason": "runaway"},
        {"id": "s2", "target": "app", "match": "app-b", "mode": "quarantine"},
        {"id": "s3", "target": "key", "match": "k1", "project_id": "p1"},
        {"id": "s4", "target": "tool", "match": "delete_*"},
        {"id": "s5", "target": "model", "match": "gpt-4o*", "environment_id": "e1"},
        {"id": "s6", "target": "agent", "match": "agent-old", "until": past},
        {"id": "s7", "target": "everything", "project_id": "p9"},
    ]))
    def caller(**kw):
        args = {"project_id": "p1", "environment_id": "e1", "app_id": "app-x", "agent_id": "agent-x", "key_id": "k0"}
        return stops.for_caller(**{**args, **kw})
    assert caller(agent_id="agent-a").stopped.id == "s1"
    assert caller(app_id="app-b").quarantined.id == "s2" and caller(app_id="app-b").stopped is None
    assert caller(key_id="k1").stopped.id == "s3" and caller(key_id="k1", project_id="p2").stopped is None
    assert caller(agent_id="agent-old").stopped is None                      # over
    assert caller(project_id="p9").stopped.id == "s7" and caller().stopped is None
    assert caller().tool_stop("delete_records").id == "s4" and caller().tool_stop("search") is None
    assert stops.model("gpt-4o-mini", "p1", "e1").id == "s5" and stops.model("gpt-4o-mini", "p1", "e2") is None
    assert "runaway" in caller(agent_id="agent-a").stopped.message()
    assert governance.StopSet().for_caller(project_id="", environment_id="", app_id="a", agent_id="b",
                                           key_id="c") is governance.NO_STOPS


# -- stops on gateway calls -----------------------------------------------------------------

def test_a_stopped_agent_is_refused_before_anything_is_sent(client, controls):
    controls(stops=[{"id": "s1", "target": "agent", "match": "agent-alpha", "reason": "looping on refunds"}])
    resp = openai_call(client, [user("SAY:hello")])
    assert resp.status_code == 403 and resp.json()["error"]["code"] == "stopped"
    assert "looping on refunds" in resp.json()["error"]["message"] and fake.received == []
    assert client.records[-1].outcome == "denied" and client.records[-1].reason == "stopped"
    assert openai_call(client, [user("SAY:hello")], key="beta").status_code == 200        # another agent
    resp = anthropic_call(client, [user("SAY:hi")])
    assert resp.status_code == 403 and resp.json()["error"]["type"] == "permission_error"


def test_stop_everything_and_its_end(client, controls):
    controls(stops=[{"target": "everything"}])
    assert openai_call(client, [user("SAY:x")], key="beta").status_code == 403
    controls(stops=[{"target": "everything", "until": "2001-01-01T00:00:00Z"}])
    assert openai_call(client, [user("SAY:x")], key="beta").status_code == 200
    controls(stops=[{"target": "everything", "project_id": "another-project"}])
    assert openai_call(client, [user("SAY:x")], key="beta").status_code == 200           # not this one


def test_a_stopped_model_or_provider_is_refused(client, controls):
    controls(stops=[{"target": "model", "match": "gpt-4o*"}])
    assert openai_call(client, [user("SAY:x")], model="gpt-4o-mini").status_code == 403
    assert openai_call(client, [user("SAY:x")], model="o3-mini").status_code == 200
    controls(stops=[{"target": "provider", "match": "anthropic"}])
    assert anthropic_call(client, [user("SAY:x")]).status_code == 403
    assert openai_call(client, [user("SAY:x")]).status_code == 200


def test_a_quarantined_agent_can_talk_but_not_act(client, controls):
    controls(stops=[{"target": "agent", "match": "agent-alpha", "mode": "quarantine"}])
    resp = openai_call(client, [user("SAY:just words")], tools=TOOLS, tool_choice={
        "type": "function", "function": {"name": "search"}})
    assert resp.status_code == 200 and resp.json()["choices"][0]["message"]["content"] == "just words"
    body = fake.received[-1]["body"]
    assert "tools" not in body and "tool_choice" not in body
    assert client.records[-1].techniques.get("quarantined") == 1
    resp = openai_call(client, [user('TOOL:search:{"q": "x"}')], tools=TOOLS)
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "content_filter" and not choice["message"].get("tool_calls")
    assert governance.RULE_QUARANTINED in client.records[-1].rule_ids


def test_a_quarantined_agent_on_the_anthropic_api(client, controls):
    controls(stops=[{"target": "app", "match": "app-alpha", "mode": "quarantine"}])
    resp = anthropic_call(client, [user('TOOL:search:{"q": "x"}')], tools=ANTHROPIC_TOOLS)
    assert resp.status_code == 200 and "tools" not in fake.received[-1]["body"]
    assert not any(b.get("type") == "tool_use" for b in resp.json()["content"])


def test_a_stopped_tool_is_never_offered_nor_called(client, controls):
    controls(stops=[{"target": "tool", "match": "delete_*", "reason": "incident 42"}])
    resp = openai_call(client, [user("SAY:fine")], tools=TOOLS)
    assert resp.status_code == 200 and sent_tools() == ["search", "send_email"]
    resp = openai_call(client, [user('TOOL:delete_records:{"all": true}')], tools=TOOLS[:1])
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "content_filter" and not choice["message"].get("tool_calls")
    assert governance.RULE_TOOL_STOPPED in client.records[-1].rule_ids
    resp = openai_call(client, [user('TOOL:search:{"q": "x"}')], tools=TOOLS)
    assert resp.json()["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "search"


def test_a_stopped_tool_is_withheld_from_a_stream(client, controls):
    controls(stops=[{"target": "tool", "match": "delete_records"}])
    with client.stream("POST", "/v1/chat/completions", headers={"Authorization": f"Bearer {KEYS['alpha']}"},
                       json={"model": "gpt-4o-mini", "stream": True, "tools": TOOLS[:1],
                             "messages": [user('TOOL:delete_records:{"all": true}')]}) as resp:
        text = resp.read().decode()
    assert resp.status_code == 200 and "delete_records" not in text


def test_an_answer_in_a_conversation_with_earlier_tool_calls_is_not_blocked_by_quarantine(client, controls):
    controls(stops=[{"target": "agent", "match": "agent-alpha", "mode": "quarantine"}])
    history = [user("SAY:look"), {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_0", "type": "function", "function": {"name": "search", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_0", "content": "result"}, user("SAY:summary done")]
    resp = openai_call(client, history)
    assert resp.status_code == 200 and resp.json()["choices"][0]["message"]["content"] == "summary done"


def test_a_stopped_model_is_never_used_for_another_call(client, controls):
    from znyx_runtime.gateway.pipeline import CallRecord
    controls(stops=[{"target": "model", "match": "gpt-4o-mini"}])
    key = next(k for k in client.gateway.config.keys_by_digest.values() if k.id == "k-alpha")
    record = CallRecord(request_id="r", key_id="k-alpha", tenant_id="acme", app_id="app-alpha",
                        agent_id="agent-alpha", api="openai")
    assert not client.gateway._model_gates_pass(key, "gpt-4o-mini", record, "escalation_refused")
    assert record.techniques == {"escalation_refused": 1}
    assert client.gateway._model_gates_pass(key, "gpt-4o", record, "escalation_refused")


def test_new_controls_take_effect_at_once(client, controls):
    controls(stops=[{"target": "agent", "match": "agent-alpha"}])
    assert openai_call(client, [user("SAY:x")]).status_code == 403
    controls(stops=[])
    assert openai_call(client, [user("SAY:x")]).status_code == 200


# -- stops on the evaluate API --------------------------------------------------------------

def test_the_evaluate_api_answers_a_stop_with_a_block_decision(client, controls):
    controls(stops=[{"id": "s9", "target": "agent", "match": "agent-alpha", "reason": "incident"}])
    resp = evaluate(client, "/v1/evaluate/input", {"request_id": "r1", "text": "hello"})
    assert resp.status_code == 200
    out = resp.json()
    assert out["decision"] == "BLOCK" and out["rule_hits"][0]["rule_id"] == governance.RULE_STOPPED
    assert "incident" in out["user_message"]
    assert evaluate(client, "/v1/evaluate/input", {"request_id": "r2", "text": "hello"},
                    key="beta").json()["decision"] == "ALLOW"


def test_a_quarantined_agent_may_check_words_but_not_actions(client, controls):
    controls(stops=[{"target": "agent", "match": "agent-alpha", "mode": "quarantine"}])
    assert evaluate(client, "/v1/evaluate/input", {"request_id": "r", "text": "hello"}).json()["decision"] == "ALLOW"
    tool = evaluate(client, "/v1/evaluate/tool", {"request_id": "r", "tool_name": "search", "tool_args": {}}).json()
    assert tool["decision"] == "BLOCK" and tool["rule_hits"][0]["rule_id"] == governance.RULE_QUARANTINED
    step = evaluate(client, "/v1/evaluate/agent-step", {"request_id": "r", "action": "call search",
                                                        "iteration": 1}).json()
    assert step["decision"] == "BLOCK"


def test_the_evaluate_api_blocks_a_stopped_tool(client, controls):
    controls(stops=[{"target": "tool", "match": "delete_records"}])
    blocked = evaluate(client, "/v1/evaluate/tool", {"request_id": "r", "tool_name": "delete_records",
                                                     "tool_args": {}}).json()
    assert blocked["decision"] == "BLOCK" and blocked["rule_hits"][0]["rule_id"] == governance.RULE_TOOL_STOPPED
    assert evaluate(client, "/v1/evaluate/tool", {"request_id": "r", "tool_name": "search",
                                                  "tool_args": {}}).json()["decision"] == "ALLOW"


def test_an_instance_key_caller_naming_a_stopped_agent_is_blocked(client, controls):
    controls(stops=[{"target": "agent", "match": "agent-alpha"}])
    resp = client.post("/v1/evaluate/input", json={"request_id": "r", "text": "hi", "tenant_id": "acme",
                                                   "app_id": "app-alpha", "agent_id": "agent-alpha"},
                       headers={"Authorization": f"Bearer {INSTANCE_KEY}"})
    assert resp.json()["decision"] == "BLOCK"


# -- usable keys without a publish ----------------------------------------------------------

def test_a_key_the_console_no_longer_lists_is_refused(client, controls):
    controls(live_keys=[_sha(KEYS["beta"])])
    resp = openai_call(client, [user("SAY:x")])
    assert resp.status_code == 401 and resp.json()["error"]["code"] == "invalid_api_key"
    assert evaluate(client, "/v1/evaluate/input", {"request_id": "r", "text": "x"}).status_code == 401
    assert openai_call(client, [user("SAY:x")], key="beta").status_code == 200
    controls(live_keys=[])
    assert openai_call(client, [user("SAY:x")], key="beta").status_code == 401          # every key revoked
    resp = client.post("/v1/evaluate/input", json={"request_id": "r", "text": "x", "tenant_id": "acme",
                                                   "app_id": "app-alpha"},
                       headers={"Authorization": f"Bearer {INSTANCE_KEY}"})
    assert resp.status_code == 200                                                    # never the instance key
    controls()
    assert openai_call(client, [user("SAY:x")]).status_code == 200                   # no list: not filtered


# -- stops in the gateway file (no console) -------------------------------------------------

def test_stops_from_the_gateway_file():
    from znyx_runtime.gateway import config as gw_config
    doc = {**GATEWAY, "stops": [{"target": "tool", "match": "delete_*"}]}
    cfg = gw_config.from_document(doc, source="test")
    assert [s.match for s in cfg.stops] == ["delete_*"]
    with pytest.raises(gw_config.GatewayConfigError, match="stops"):
        gw_config.from_document({**GATEWAY, "stops": [{"target": "robot"}]}, source="test")
    with pytest.raises(gw_config.GatewayConfigError, match="unknown section"):
        gw_config.from_document({**GATEWAY, "stops": []}, source="bundle", managed=True)


def test_file_stops_are_enforced(client, monkeypatch):
    import dataclasses
    stop, _ = governance.clean_stop({"target": "agent", "match": "agent-beta"})
    monkeypatch.setattr(client.gateway, "config", dataclasses.replace(client.gateway.config, stops=(stop,)))
    assert openai_call(client, [user("SAY:x")], key="beta").status_code == 403
    assert openai_call(client, [user("SAY:x")]).status_code == 200


# -- batch ----------------------------------------------------------------------------------

def _tick(client):
    client.portal.call(client.app.state.batch_runner.tick)


def test_a_queued_call_of_a_quarantined_agent_offering_tools_is_not_sent(client, controls):
    if client.app.state.batch_runner is None:
        pytest.skip("no batch runner in this process")
    resp = client.post("/v1/chat/completions", json={"model": "gpt-4o-mini", "messages": [user("SAY:later")],
                                                     "tools": TOOLS[:1]},
                       headers={"Authorization": f"Bearer {KEYS['batcher']}", "x-znyx-urgency": "low"})
    assert resp.status_code == 202
    job = resp.json()["id"]
    controls(stops=[{"target": "agent", "match": "agent-batcher", "mode": "quarantine"}])
    _tick(client)
    status = client.get(f"/v1/znyx/batches/{job}", headers={"Authorization": f"Bearer {KEYS['batcher']}"}).json()
    assert status["status"] == "failed" and "quarantined" in status["error"] and fake.received == []
    controls(stops=[{"target": "agent", "match": "agent-batcher"}])
    resp = client.get(f"/v1/znyx/batches/{job}", headers={"Authorization": f"Bearer {KEYS['batcher']}"})
    assert resp.status_code == 403                                                   # a stopped agent reads nothing


# -- tool names for the inventory -----------------------------------------------------------

def test_usage_events_carry_the_tools_offered_and_called(client):
    resp = openai_call(client, [user('TOOL:search:{"q": "x"}')], tools=TOOLS)
    assert resp.status_code == 200
    record = client.records[-1]
    assert record.tools_offered == ["search", "delete_records", "send_email"] and record.tools_called == ["search"]
    event = client.sink.event_for(record)
    stored = usage_events.validate(json.loads(json.dumps(event, default=str)))
    assert stored["tools_offered"] == ["search", "delete_records", "send_email"]
    assert stored["tools_called"] == ["search"]
    assert "_stops" not in record.to_event()
    openai_call(client, [user("SAY:no tools")])
    assert client.records[-1].tools_offered == [] and "tools_offered" not in client.sink.event_for(client.records[-1])


# -- review fixes (2026-10-22) ----------------------------------------------------------------

def test_the_stream_evaluate_route_honours_a_stop(client, controls):
    controls(stops=[{"target": "agent", "match": "agent-alpha", "reason": "incident"}])
    resp = client.post("/v1/evaluate/stream", json={"request_id": "r", "tenant_id": "acme", "app_id": "app-alpha",
                                                   "chunks": ["secret plans ", "for everyone"]},
                       headers={"Authorization": f"Bearer {KEYS['alpha']}"})
    assert resp.status_code == 200
    assert "event: chunk" not in resp.text and "secret plans" not in resp.text
    assert '"final_decision": "BLOCK"' in resp.text and governance.RULE_STOPPED in resp.text


def test_an_azure_deployment_is_checked_against_model_stops(client, controls):
    controls(stops=[{"target": "model", "match": "gpt-4o*"}])
    for body in ({}, {"model": "anything-else"}):
        resp = client.post("/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21",
                           json={"messages": [user("SAY:hi")], **body},
                           headers={"api-key": KEYS["alpha"], "x-znyx-provider": "azure"})
        assert resp.status_code == 403 and resp.json()["error"]["code"] == "stopped", body
    assert fake.received == []


CUSTOM_TOOLS = [{"type": "custom", "custom": {"name": "shell", "description": "Run a command"}},
                {"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}]


def _custom_answer(body):
    return {"id": "c", "object": "chat.completion", "created": 1, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [
                    {"id": "call_1", "type": "custom", "custom": {"name": "shell", "input": "rm -rf /"}}]}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5}}


def test_custom_tools_are_filtered_and_their_calls_checked(client, controls):
    controls(stops=[{"target": "tool", "match": "shell"}])
    fake.scripted = _custom_answer
    resp = openai_call(client, [user("SAY:x")], tools=CUSTOM_TOOLS)
    assert sent_tools() == ["search"]                                   # the custom tool is not offered
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "content_filter" and not choice["message"].get("tool_calls")
    assert client.records[-1].tools_called == ["shell"]
    controls(stops=[{"target": "tool", "match": "delete_*"}])
    resp = openai_call(client, [user("SAY:x")], tools=[{"type": "mystery"}, CUSTOM_TOOLS[1]])
    assert sent_tools() == ["search"]                                   # unnamed: removed under a stop


def test_a_custom_tool_call_in_a_stream_is_withheld(client, controls):
    controls(stops=[{"target": "tool", "match": "shell"}])
    env = {"id": "s", "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini"}
    fake.scripted = lambda body: [
        f"data: {json.dumps({**env, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'tool_calls': [{'index': 0, 'id': 'c1', 'type': 'custom', 'custom': {'name': 'shell', 'input': 'rm -rf'}}]}, 'finish_reason': None}]})}\n\n",
        f"data: {json.dumps({**env, 'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 0, 'custom': {'input': ' /'}}]}, 'finish_reason': None}]})}\n\n",
        f"data: {json.dumps({**env, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls'}]})}\n\n",
        "data: [DONE]\n\n"]
    with client.stream("POST", "/v1/chat/completions", headers={"Authorization": f"Bearer {KEYS['alpha']}"},
                       json={"model": "gpt-4o-mini", "stream": True, "messages": [user("SAY:x")]}) as resp:
        text = resp.read().decode()
    assert "rm -rf" not in text and "content_filter" in text
    controls()
    with client.stream("POST", "/v1/chat/completions", headers={"Authorization": f"Bearer {KEYS['alpha']}"},
                       json={"model": "gpt-4o-mini", "stream": True, "messages": [user("SAY:x")]}) as resp:
        text = resp.read().decode()
    assert '"custom":{"name":"shell","input":"rm -rf /"}' in text                  # released whole when allowed


def test_a_quarantined_request_stays_valid_for_the_provider(client, controls):
    controls(stops=[{"target": "agent", "match": "agent-alpha", "mode": "quarantine"}])
    openai_call(client, [user("SAY:x")], tools=TOOLS, parallel_tool_calls=False)
    assert not {"tools", "tool_choice", "parallel_tool_calls"} & set(fake.received[-1]["body"])
    history = [user("SAY:look"), {"role": "assistant", "content": [
        {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "found"},
                                     {"type": "text", "text": "SAY:summary"}]}]
    resp = anthropic_call(client, history, tools=ANTHROPIC_TOOLS)
    body = fake.received[-1]["body"]
    assert resp.status_code == 200 and [t["name"] for t in body["tools"]] == ["znyx_no_tools"]
    assert body["tool_choice"] == {"type": "none"}


def test_quarantine_removes_provider_side_mcp(client, controls, monkeypatch):
    import dataclasses
    cfg = client.gateway.config
    monkeypatch.setattr(client.gateway, "config", dataclasses.replace(
        cfg, settings=dataclasses.replace(cfg.settings, allow_provider_mcp=True)))
    controls(stops=[{"target": "agent", "match": "agent-alpha", "mode": "quarantine"}])
    anthropic_call(client, [user("SAY:x")], mcp_servers=[{"type": "url", "url": "https://mcp.example", "name": "m"}])
    assert "mcp_servers" not in fake.received[-1]["body"]
    controls()
    anthropic_call(client, [user("SAY:x")], mcp_servers=[{"type": "url", "url": "https://mcp.example", "name": "m"}])
    assert "mcp_servers" in fake.received[-1]["body"]                        # allowed and not stopped


def test_a_finished_batch_answer_read_after_a_quarantine_loses_its_tool_calls(client, controls):
    fake.scripted = lambda body: {"id": "b", "object": "chat.completion", "created": 1, "model": body["model"],
                                  "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                                      "role": "assistant", "content": None, "tool_calls": [
                                          {"id": "c1", "type": "function",
                                           "function": {"name": "search", "arguments": "{}"}}]}}],
                                  "usage": {"prompt_tokens": 5, "completion_tokens": 5}}
    resp = client.post("/v1/chat/completions", json={"model": "gpt-4o-mini", "tools": TOOLS[:1],
                                                     "messages": [user('TOOL:search:{"q": 1}')]},
                       headers={"Authorization": f"Bearer {KEYS['batcher']}", "x-znyx-urgency": "low"})
    job = resp.json()["id"]
    _tick(client)
    fake.BATCH_DONE = True
    _tick(client)
    read = lambda: client.get(f"/v1/znyx/batches/{job}", headers={"Authorization": f"Bearer {KEYS['batcher']}"}).json()
    assert read()["status"] == "done" and read()["result"]["choices"][0]["message"].get("tool_calls")
    controls(stops=[{"target": "agent", "match": "agent-batcher", "mode": "quarantine"}])
    choice = read()["result"]["choices"][0]
    assert choice["finish_reason"] == "content_filter" and not choice["message"].get("tool_calls")
