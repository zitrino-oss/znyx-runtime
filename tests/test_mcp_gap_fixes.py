"""MCP content coverage and authorization after approval waits."""
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import httpx
import pytest

import gateway_fake_provider as fake
from test_phase5_mcp import KEYS, _decide, _fresh, _pending, client, controls, rpc, upstream_calls

SECRET = "AKIA" + "IOSFODNN7EXAMPLQ"


def test_oversized_text_resource_is_refused_without_releasing_its_unchecked_tail(client):
    from znyx_runtime.gateway.mcp import MAX_BLOB_CHECK
    content = b"ordinary text " * (MAX_BLOB_CHECK // 10) + SECRET.encode()
    fake.mcp_results["search"] = {"content": [{"type": "resource", "resource": {
        "uri": "file:///report.txt", "mimeType": "text/plain", "blob": base64.b64encode(content).decode()}}]}
    response = rpc(client, "crm", "tools/call", {"name": "search", "arguments": {}}, accept="application/json")
    assert response.json()["result"]["isError"]
    assert "blob" not in response.text


@pytest.mark.parametrize("parts", [
    [{"type": "text", "text": SECRET[:10]}, {"type": "text", "text": SECRET[10:]}],
    [{"type": "text", "text": SECRET[:10]},
     {"type": "resource", "resource": {"uri": "file:///tail", "text": SECRET[10:]}}],
])
def test_secret_spanning_mcp_content_items_is_withheld(client, parts):
    fake.mcp_results["search"] = {"content": parts}
    response = rpc(client, "crm", "tools/call", {"name": "search", "arguments": {}}, accept="application/json")
    assert response.json()["result"]["isError"]


def test_pii_spanning_mcp_text_items_is_redacted_as_one_value(client):
    fake.mcp_results["search"] = {"content": [{"type": "text", "text": "contact jane.doe@"},
                                              {"type": "text", "text": "example.com"}]}
    response = rpc(client, "crm", "tools/call", {"name": "search", "arguments": {}}, accept="application/json")
    result = response.json()["result"]
    assert not result.get("isError")
    text = "".join(i.get("text", "") for i in result["content"])
    assert "jane.doe" not in text and "[EMAIL]" in text


@pytest.mark.parametrize("result", [
    {"content": [{"type": "text", "text": "safe", "annotations": {"title": SECRET}}]},
    {"content": [{"type": "text", "text": "safe"}], "_meta": {"debug": SECRET}},
])
def test_tool_result_metadata_is_checked(client, result):
    fake.mcp_results["search"] = result
    response = rpc(client, "crm", "tools/call", {"name": "search", "arguments": {}}, accept="application/json")
    assert response.json()["result"]["isError"] and SECRET not in response.text


def test_prompt_messages_are_checked_together(client, monkeypatch):
    async def send(request, ctx, message):
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": {"messages": [
            {"role": "user", "content": {"type": "text", "text": SECRET[:10]}},
            {"role": "user", "content": {"type": "text", "text": SECRET[10:]}}]}})
    monkeypatch.setattr(client.gateway.mcp, "_send", send)
    response = rpc(client, "crm", "prompts/get", {"name": "report"}, accept="application/json")
    assert "error" in response.json() and SECRET not in response.text


@pytest.mark.parametrize("method", ["completion/complete", "prompts/list", "resources/list", "tasks/get"])
def test_protocol_success_results_receive_content_checks(client, monkeypatch, method):
    async def send(request, ctx, message):
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"],
                                        "result": {"completion": {"values": [SECRET]}}})
    monkeypatch.setattr(client.gateway.mcp, "_send", send)
    response = rpc(client, "crm", method, {}, accept="application/json")
    assert "error" in response.json() and SECRET not in response.text


def test_revoked_key_cannot_execute_an_approved_pending_call(client, controls):
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(rpc, client, "crm", "tools/call", {"name": "delete_record", "arguments": {}},
                             "approver", None, 1, None, "application/json")
        pending = _pending(client)
        controls(live_keys=[])
        _decide(client, pending["id"], "approve")
        response = future.result(timeout=15)
    assert response.json()["result"]["isError"]
    assert not upstream_calls("tools/call")


@pytest.mark.parametrize("change", ["revoke", "expire", "permissions"])
def test_delegated_mcp_identity_is_revalidated_after_approval(client, monkeypatch, change):
    from test_phase5_identity import _provider, mint
    from znyx_core import identity
    from znyx_runtime.gateway.identity import _STORE_PREFIX, _digest
    rules, _ = identity.clean_agent({"agent_id": "agent-approver",
                                     "groups": {"support": ["delete_record", "search"]}})
    monkeypatch.setattr(client.gateway, "config", replace(client.gateway.config,
                                                         identity_providers=(_provider(),), agents=(rules,)))
    response = client.post("/v1/znyx/token", json={"subject_token": mint()},
                           headers={"authorization": f"Bearer {KEYS['approver']}"})
    assert response.status_code == 200, response.text
    token = response.json()["access_token"]
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(rpc, client, "crm", "tools/call", {"name": "delete_record", "arguments": {}},
                             "approver", None, 1, {"Authorization": f"Bearer {token}"}, "application/json")
        pending = _pending(client)
        if change == "revoke":
            assert client.gateway.delegations.revoke(token)
        elif change == "expire":
            entry = client.gateway.delegations.resolve(token)
            entry["exp"] = time.time() - 1
            client.gateway.delegations._store().set_value(_STORE_PREFIX + _digest(token), json.dumps(entry), 60)
        else:
            narrower, _ = identity.clean_agent({"agent_id": "agent-approver", "groups": {"support": ["search"]}})
            monkeypatch.setattr(client.gateway, "config", replace(client.gateway.config, agents=(narrower,)))
        _decide(client, pending["id"], "approve")
        response = future.result(timeout=15)
    assert response.json()["result"]["isError"]
    assert not upstream_calls("tools/call")
