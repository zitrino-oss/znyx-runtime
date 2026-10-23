"""Regressions for the phase 1-5 roadmap review: policy parity and budget admission."""
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

import gateway_fake_provider as fake
from test_phase4_optimise import (
    KEYS, _day_usage, _fresh, anthropic_call, client, openai_call, user,
)
from znyx_runtime.gateway import formats as fmts

SECRET = "AKIA" + "IOSFODNN7EXAMPLQ"


@pytest.mark.parametrize("opposite", ["The door is not open", "The door is never open", "No, the door is open"])
def test_lexical_diagnostics_preserve_negation(opposite):
    from znyx_runtime.gateway.optimise.shadow import lexical_agreement
    assert lexical_agreement("The door is open", opposite) == 0


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -1, 2])
def test_invalid_nli_scores_cannot_be_reported_as_verified_evidence(score):
    from znyx_runtime.gateway.optimise.shadow import compare
    assert compare("same answer", "same answer", lambda *_: [score])[2] == "lexical"


@pytest.mark.parametrize("api", ["openai", "anthropic"])
def test_blocking_contract_refuses_streaming_before_provider_egress(client, api):
    call = openai_call if api == "openai" else anthropic_call
    response = call(client, [user("SAY:plain answer")], key="strictplain", stream=True)
    assert response.status_code == 400
    assert "streaming_contract_not_supported" in response.text
    assert not fake.received


def test_custom_tool_history_is_checked_and_redacted_like_function_history(client):
    messages = [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "custom", "custom": {"name": "run_code", "input": SECRET}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "done"}, user("hello")]
    response = openai_call(client, messages, key="nothing")
    assert response.status_code == 400 and not fake.received
    messages[0]["tool_calls"][0]["custom"]["input"] = "contact jane.doe@example.com"
    response = openai_call(client, messages, key="nothing")
    assert response.status_code == 200
    assert "jane.doe@example.com" not in json.dumps(fake.received[-1]["body"])


def test_secret_spanning_anthropic_text_blocks_is_withheld(client):
    fake.scripted = lambda body: {
        "id": "msg-test", "type": "message", "role": "assistant", "model": body["model"],
        "content": [{"type": "text", "text": SECRET[:10]}, {"type": "text", "text": SECRET[10:]}],
        "stop_reason": "end_turn", "usage": {"input_tokens": 5, "output_tokens": 5}}
    response = anthropic_call(client, [user("hello")], key="nothing")
    assert response.status_code == 200 and response.json()["stop_reason"] == "refusal"
    assert SECRET not in "".join(p.get("text", "") for p in response.json()["content"])


def _stream_blocks(body, texts):
    events = [fmts.sse({"type": "message_start", "message": {
        "id": "msg-test", "type": "message", "role": "assistant", "model": body["model"],
        "content": [], "usage": {"input_tokens": 5, "output_tokens": 0}}}, "message_start")]
    for i, text in enumerate(texts):
        events.extend([
            fmts.sse({"type": "content_block_start", "index": i,
                      "content_block": {"type": "text", "text": ""}}, "content_block_start"),
            fmts.sse({"type": "content_block_delta", "index": i,
                      "delta": {"type": "text_delta", "text": text}}, "content_block_delta"),
            fmts.sse({"type": "content_block_stop", "index": i}, "content_block_stop")])
    events.extend([
        fmts.sse({"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                  "usage": {"output_tokens": 5}}, "message_delta"),
        fmts.sse({"type": "message_stop"}, "message_stop")])
    return events


def test_stream_guard_holdback_survives_anthropic_block_boundaries(client):
    fake.scripted = lambda body: _stream_blocks(body, [SECRET[:10], SECRET[10:]])
    response = anthropic_call(client, [user("hello")], key="nothing", stream=True)
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    text = "".join(e.get("delta", {}).get("text", "") for e in events)
    assert SECRET not in text
    assert any(e.get("delta", {}).get("stop_reason") == "refusal" for e in events)


def test_streamed_adjacent_clean_blocks_keep_text_and_valid_indices(client):
    fake.scripted = lambda body: _stream_blocks(body, ["hello ", "world"])
    response = anthropic_call(client, [user("hello")], key="nothing", stream=True)
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert "".join(e.get("delta", {}).get("text", "") for e in events) == "hello world"
    assert [e["index"] for e in events if e["type"] == "content_block_start"] == [0]
    assert [e["index"] for e in events if e["type"] == "content_block_stop"] == [0]


@pytest.mark.parametrize("secret", [False, True])
def test_anthropic_text_keeps_context_and_dense_indices_across_a_tool(client, secret):
    def script(body):
        texts = [SECRET[:5], SECRET[5:10], SECRET[10:]] if secret else ["hello ", "world ", "done"]
        events = _stream_blocks(body, texts)
        # Two adjacent text blocks, a tool, then another text block.
        for pos in range(7, 10):
            data = json.loads(events[pos].decode().split("data: ", 1)[1])
            data["index"] = 3
            events[pos] = fmts.sse(data, data["type"])
        tool = [
            {"type": "content_block_start", "index": 2, "content_block": {
                "type": "tool_use", "id": "t1", "name": "lookup_order", "input": {}}},
            {"type": "content_block_delta", "index": 2, "delta": {
                "type": "input_json_delta", "partial_json": "{}"}},
            {"type": "content_block_stop", "index": 2}]
        events[7:7] = [fmts.sse(e, e["type"]) for e in tool]
        return events

    fake.scripted = script
    response = anthropic_call(client, [user("hello")], key="nothing", stream=True)
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    text = "".join(e.get("delta", {}).get("text", "") for e in events)
    if secret:
        assert SECRET not in text
        assert any(e.get("delta", {}).get("stop_reason") == "refusal" for e in events)
    else:
        assert text == "hello world done"
        assert [e["index"] for e in events if e["type"] == "content_block_start"] == [0, 1, 2]
        assert [e["index"] for e in events if e["type"] == "content_block_stop"] == [0, 1, 2]


@pytest.mark.parametrize("budget", ["tokens", "cost"])
def test_concurrent_calls_reserve_daily_capacity_and_settle_actual_usage(client, monkeypatch, budget):
    from znyx_runtime.gateway.config import Limits
    key = next(k for k in client.gateway.config.keys_by_digest.values() if k.id == "k-budgeted")
    limits = Limits(tokens_per_day=1000) if budget == "tokens" else Limits(spend_usd_per_day=0.0003)
    monkeypatch.setitem(client.gateway.config.keys_by_digest, key.key_sha256, replace(key, limits=limits))
    fake.RESPONSE_DELAY = 0.4
    with ThreadPoolExecutor(12) as pool:
        responses = list(pool.map(lambda i: openai_call(client, [user(f"hello {i}")], key="budgeted",
                                                       max_tokens=100), range(12)))
    assert 0 < sum(r.status_code == 200 for r in responses) < 12
    assert all(r.status_code in (200, 429) for r in responses)
    used = _day_usage(client, "budgeted")
    records = [r for r in client.records if r.status == 200]
    assert used["tokens"] == sum((r.input_tokens or 0) + (r.output_tokens or 0) for r in records)
    assert used["cost"] == pytest.approx(sum(r.cost_usd or 0 for r in records))
    assert used[budget] <= (1000 if budget == "tokens" else 0.0003)


def test_provider_rejection_refunds_reservation(client):
    response = openai_call(client, [user("FAIL:400")], key="budgeted", max_tokens=100)
    assert response.status_code == 400
    assert _day_usage(client, "budgeted").get("tokens", 0) == 0
    assert openai_call(client, [user("hello")], key="budgeted", max_tokens=100).status_code == 200
