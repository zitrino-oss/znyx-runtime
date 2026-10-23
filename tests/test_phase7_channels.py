"""Real gateway routes with provider-shaped mock transports; no paid provider calls."""
import asyncio
import base64
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from starlette.requests import Request

from test_phase4_optimise import KEYS, _fresh, client, openai_call, user
from znyx_runtime.gateway import media
from znyx_runtime.gateway.pipeline import CallRecord, Denied

SECRET = "AKIA" + "IOSFODNN7EXAMPLQ"
MODEL = "gpt-4o-mini"


@pytest.fixture
def provider(client, monkeypatch):
    state = SimpleNamespace(calls=[], response={}, binary=None, status=200)
    def send(request):
        state.calls.append(request)
        if state.binary is not None:
            return httpx.Response(state.status, content=state.binary, headers={"content-type": "audio/wav"})
        return httpx.Response(state.status, json=state.response)
    transport_client = httpx.AsyncClient(transport=httpx.MockTransport(send))
    monkeypatch.setattr(client.gateway.upstream, "client", AsyncMock(return_value=transport_client))
    yield state
    asyncio.run(transport_client.aclose())


def call(client, path, body=None, **kwargs):
    return client.post("/v1/" + path, json=body, headers={"Authorization": f"Bearer {KEYS['nothing']}"}, **kwargs)


def answer(text):
    return {"id": "resp_1", "object": "response", "status": "completed", "model": MODEL,
            "output": [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": text, "annotations": []}]}],
            "usage": {"input_tokens": 5, "output_tokens": 3}}


def test_embedding_input_is_guarded_and_redacted_before_egress(client, provider):
    provider.response = {"object": "list", "data": [{"index": 0, "embedding": [0.1, 0.2]}],
                         "model": MODEL, "usage": {"prompt_tokens": 5, "total_tokens": 5}}
    assert call(client, "embeddings", {"model": MODEL, "input": SECRET}).status_code == 400
    assert not provider.calls
    response = call(client, "embeddings", {"model": MODEL, "input": ["jane.doe@example.com", "safe"]})
    assert response.status_code == 200, response.text
    assert "jane.doe" not in provider.calls[-1].content.decode()
    assert response.json()["data"][0]["embedding"] == [0.1, 0.2]
    assert call(client, "embeddings", {"model": MODEL, "input": [-1, 456]}).status_code == 400


@pytest.mark.parametrize("stream", [False, True])
def test_responses_preserves_the_sdk_shape_and_guards_split_output(client, provider, stream):
    provider.response = answer("hello")
    response = call(client, "responses", {"model": MODEL, "input": "hello", "stream": stream})
    assert response.status_code == 200, response.text
    if stream:
        assert "response.output_text.delta" in response.text and "response.completed" in response.text
    else:
        assert response.json()["output"][0]["content"][0]["text"] == "hello"
    sent = json.loads(provider.calls[-1].content)
    assert sent["store"] is False and "stream" not in sent
    provider.response = answer(SECRET[:10])
    provider.response["output"][0]["content"].append({"type": "output_text", "text": SECRET[10:]})
    response = call(client, "responses", {"model": MODEL, "input": "hello", "stream": stream})
    assert response.status_code == 400 and SECRET not in response.text


@pytest.mark.parametrize("fields", [{"previous_response_id": "resp_other"}, {"background": True},
    {"conversation": "conv_other"}, {"input": [{"type": "item_reference", "id": "stored"}]},
    {"tools": [{"type": "web_search"}]}])
def test_responses_cannot_escape_inspection_via_stored_context_or_hosted_tools(client, provider, fields):
    response = call(client, "responses", {"model": MODEL, "input": "hello", **fields})
    assert response.status_code == 400 and not provider.calls


def test_image_output_is_inspected_from_bytes_and_withheld_when_ocr_requires_redaction(client, provider, monkeypatch):
    provider.response = {"created": 1, "data": [{"b64_json": base64.b64encode(b"image").decode()}]}
    extractor = AsyncMock(return_value="a green field")
    monkeypatch.setattr(media, "extract", extractor)
    response = call(client, "images/generations", {"model": "image-test", "prompt": "a field"})
    assert response.status_code == 200, response.text
    extractor.assert_awaited()
    extractor.return_value = "jane.doe@example.com"
    response = call(client, "images/generations", {"model": "image-test", "prompt": "a field"})
    assert response.status_code == 400 and "b64_json" not in response.text


def test_speech_is_checked_before_sending_and_generated_audio_is_checked_before_returning(client, provider, monkeypatch):
    provider.binary = b"wav-bytes"
    extractor = AsyncMock(return_value="safe speech")
    monkeypatch.setattr(media, "extract", extractor)
    assert call(client, "audio/speech", {"model": MODEL, "input": SECRET}).status_code == 400
    assert not provider.calls
    response = call(client, "audio/speech", {"model": MODEL, "input": "hello", "voice": "alloy"})
    assert response.status_code == 200 and response.content == b"wav-bytes"
    extractor.return_value = SECRET
    assert call(client, "audio/speech", {"model": MODEL, "input": "hello", "voice": "alloy"}).status_code == 400


def test_transcription_redaction_drops_duplicate_word_timings(client, provider, monkeypatch):
    monkeypatch.setattr(media, "extract", AsyncMock(return_value="safe input speech"))
    provider.response = {"text": "mail jane.doe@example.com", "segments": [{"text": "mail jane.doe@example.com"}],
                         "words": [{"word": "jane.doe@"}, {"word": "example.com"}]}
    response = client.post("/v1/audio/transcriptions", data={"model": MODEL, "response_format": "verbose_json"},
        files={"file": ("voice.wav", b"wav", "audio/wav")}, headers={"Authorization": f"Bearer {KEYS['nothing']}"})
    assert response.status_code == 200, response.text
    assert "jane.doe" not in response.text and "words" not in response.json() and "segments" not in response.json()


def test_inline_chat_media_uses_extractor_not_caller_supplied_text(client, provider, monkeypatch):
    monkeypatch.setattr(media, "extract", AsyncMock(return_value=SECRET))
    response = openai_call(client, [{"role": "user", "content": [{"type": "image_url", "extracted_text": "safe",
        "image_url": {"url": "data:image/png;base64,aW1hZ2U="}}]}], key="nothing")
    assert response.status_code == 400 and not provider.calls


def test_parallel_input_cannot_send_media_before_inspection(client, provider, monkeypatch):
    monkeypatch.setattr(media, "extract", AsyncMock(return_value=SECRET))
    response = openai_call(client, [{"role": "user", "content": [{"type": "image_url",
        "image_url": {"url": "data:image/png;base64,aW1hZ2U="}}]}], key="racer")
    assert response.status_code == 400 and not provider.calls


def test_media_fails_closed_when_not_configured_or_malformed(client, provider, monkeypatch):
    monkeypatch.delenv("ZNYX_INFERENCE_URL", raising=False)
    monkeypatch.delenv("ZNYX_MEDIA_EXTRACT_URL", raising=False)
    response = call(client, "responses", {"model": MODEL, "input": [{"role": "user", "content": [
        {"type": "input_image", "image_url": "data:image/png;base64,aW1hZ2U="}]}]})
    assert response.status_code == 503 and not provider.calls
    with pytest.raises(Denied):
        media.decode("bad-base64!")


@pytest.fixture
def realtime_session(client):
    from znyx_runtime.gateway.realtime import Session
    from znyx_runtime.gateway.channels import context
    request = Request({"type": "http", "method": "GET", "path": "/v1/realtime", "scheme": "http",
        "query_string": b"", "server": ("test", 80), "headers": [
            (b"authorization", f"Bearer {KEYS['nothing']}".encode())]})
    record = CallRecord(request_id="rt-test", key_id="", tenant_id="", app_id="", agent_id="", api="realtime")
    key, credential = client.gateway._identify(request)
    async def setup():
        provider, scope, policy = await context(client.gateway, request, key, credential, {"model": MODEL}, record)
        return Session(client.gateway, request, key, credential, provider, scope, policy, record, SimpleNamespace(send=AsyncMock()))
    return asyncio.run(setup())


def test_realtime_audio_does_not_cross_until_the_complete_turn_passes(realtime_session, monkeypatch):
    session = realtime_session
    monkeypatch.setattr(media, "extract", AsyncMock(return_value=SECRET))
    async def run():
        await session.client_event({"type": "input_audio_buffer.append", "audio": base64.b64encode(b"\0" * 400).decode()})
        session.upstream.send.assert_not_awaited()
        with pytest.raises(Denied):
            await session.client_event({"type": "input_audio_buffer.commit"})
        session.upstream.send.assert_not_awaited()
    asyncio.run(run())


def test_realtime_holds_split_text_and_refuses_before_any_delta_is_returned(realtime_session):
    session = realtime_session
    async def run():
        await session.client_event({"type": "response.create", "response": {"output_modalities": ["text"]}})
        assert await session.server_event({"type": "response.output_text.delta", "delta": SECRET[:10]}) == []
        assert await session.server_event({"type": "response.output_text.delta", "delta": SECRET[10:]}) == []
        with pytest.raises(Denied):
            await session.server_event({"type": "response.done", "response": {"output": [], "usage": {
                "input_tokens": 5, "output_tokens": 5}}})
    asyncio.run(run())


def test_realtime_revalidates_a_revoked_key_on_the_next_event(realtime_session, monkeypatch):
    session = realtime_session
    monkeypatch.setitem(session.gw.config.keys_by_digest, session.key.key_sha256, replace(session.key, disabled=True))
    with pytest.raises(Denied):
        asyncio.run(session.client_event({"type": "response.create"}))


def test_realtime_requires_reconnect_when_shared_defence_changes(realtime_session, monkeypatch):
    from test_phase7_defence import feed
    session = realtime_session
    monkeypatch.setattr(session.gw, "_controls", lambda: {"shared_defence": feed("new attack")})
    with pytest.raises(Denied) as result:
        asyncio.run(session.client_event({"type": "response.create"}))
    assert result.value.code == "defence_changed"
    session.upstream.send.assert_not_awaited()


def test_realtime_clean_turn_keeps_provider_events_and_settles(realtime_session):
    session = realtime_session
    async def run():
        await session.client_event({"type": "response.create", "response": {"output_modalities": ["text"]}})
        delta = {"type": "response.output_text.delta", "delta": "hello"}
        assert await session.server_event(delta) == []
        done = {"type": "response.done", "response": {"output": [], "usage": {"input_tokens": 5, "output_tokens": 2}}}
        assert await session.server_event(done) == [delta, done]
        assert not session.active
    asyncio.run(run())


def test_realtime_per_response_input_cannot_bypass_guards(realtime_session):
    with pytest.raises(Denied):
        asyncio.run(realtime_session.client_event({"type": "response.create", "response": {"input": [
            {"role": "user", "content": [{"type": "input_text", "text": SECRET}]}]}}))


def test_realtime_done_events_cannot_smuggle_content_absent_from_deltas(realtime_session):
    session = realtime_session
    async def run():
        await session.client_event({"type": "response.create"})
        assert await session.server_event({"type": "response.output_text.done", "text": SECRET}) == []
        with pytest.raises(Denied):
            await session.server_event({"type": "response.done", "response": {"output": []}})
    asyncio.run(run())


def test_official_sdk_embeddings_and_buffered_responses(client, provider):
    openai = pytest.importorskip("openai")
    api = openai.OpenAI(base_url="http://testserver/v1", api_key=KEYS["nothing"], http_client=client, max_retries=0)
    provider.response = {"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
                         "model": MODEL, "usage": {"prompt_tokens": 2, "total_tokens": 2}}
    assert api.embeddings.create(model=MODEL, input="safe", encoding_format="float").data[0].embedding == [0.1]
    provider.response = answer("hello")
    assert api.responses.create(model=MODEL, input="safe").output_text == "hello"
    with api.responses.stream(model=MODEL, input="safe") as stream:
        events = list(stream)
        assert any(event.type == "response.output_text.delta" for event in events)
        assert stream.get_final_response().output_text == "hello"


def test_remote_media_is_frozen_to_the_inspected_bytes_before_forwarding(client, provider, monkeypatch):
    monkeypatch.setattr(media, "fetch", AsyncMock(return_value=(b"image", "image/png")))
    monkeypatch.setattr(media, "extract", AsyncMock(return_value="safe"))
    provider.response = answer("hello")
    response = call(client, "responses", {"model": MODEL, "input": [{"role": "user", "content": [
        {"type": "input_image", "image_url": "https://approved.test/mutable.png"}]}]})
    assert response.status_code == 200
    sent = json.loads(provider.calls[-1].content)
    assert sent["input"][0]["content"][0]["image_url"] == "data:image/png;base64,aW1hZ2U="


def test_media_spend_budget_reserves_unit_prices_and_blocks_the_next_image(client, provider, monkeypatch):
    from znyx_runtime.gateway.config import Limits
    from znyx_runtime.gateway.pricing import PriceTable
    key = next(k for k in client.gateway.config.keys_by_digest.values() if k.id == "k-nothing")
    monkeypatch.setitem(client.gateway.config.keys_by_digest, key.key_sha256, replace(key, limits=Limits(spend_usd_per_day=0.05)))
    monkeypatch.setattr(client.gateway, "prices_for", lambda _: PriceTable({"image-test": {
        "input_per_mtok": 0, "output_per_mtok": 0, "per_image": 0.04}}))
    monkeypatch.setattr(media, "extract", AsyncMock(return_value="safe"))
    provider.response = {"data": [{"b64_json": base64.b64encode(b"image").decode()}]}
    assert call(client, "images/generations", {"model": "image-test", "prompt": "safe"}).status_code == 200
    assert call(client, "images/generations", {"model": "image-test", "prompt": "safe"}).status_code == 429
    assert len(provider.calls) == 1


def test_remote_download_pins_the_public_address_and_preserves_tls_identity(realtime_session, monkeypatch):
    import socket
    session = realtime_session
    monkeypatch.setenv("ZNYX_MEDIA_URL_ALLOWLIST", "assets.example.com")
    monkeypatch.setenv("ZNYX_MEDIA_URL_REGION", "us")
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))])
    requests = []
    def send(request):
        requests.append(request)
        return httpx.Response(200, content=b"image", headers={"content-type": "image/png"})
    remote = httpx.AsyncClient(transport=httpx.MockTransport(send))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: remote)
    result = asyncio.run(media.fetch(session.gw, "https://assets.example.com/image.png", session.scope, session.policy, session.record))
    assert result == (b"image", "image/png")
    assert requests[0].url.host == "93.184.216.34"
    assert requests[0].headers["host"] == "assets.example.com"
    assert requests[0].extensions["sni_hostname"] == "assets.example.com"


@pytest.mark.parametrize("ip", ["127.0.0.1", "169.254.169.254", "::1", "10.0.0.1"])
def test_remote_media_cannot_reach_private_or_metadata_addresses(realtime_session, monkeypatch, ip):
    import socket
    session = realtime_session
    monkeypatch.setenv("ZNYX_MEDIA_URL_ALLOWLIST", "assets.example.com")
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))])
    with pytest.raises(Denied):
        asyncio.run(media.fetch(session.gw, "https://assets.example.com/image.png", session.scope, session.policy, session.record))


def test_responses_unknown_content_is_refused_and_inline_tool_output_keeps_its_shape(client, provider):
    bad = {"model": MODEL, "input": [{"role": "user", "content": [{"type": "future_video", "data": "opaque"}]}]}
    assert call(client, "responses", bad).status_code == 400 and not provider.calls
    provider.response = answer("safe")
    body = {"model": MODEL, "input": [{"type": "function_call_output", "call_id": "call_1",
        "output": [{"type": "input_text", "text": "jane.doe@example.com"}]}]}
    assert call(client, "responses", body).status_code == 200
    sent = json.loads(provider.calls[-1].content)
    assert isinstance(sent["input"][0]["output"], list) and "jane.doe" not in provider.calls[-1].content.decode()


def test_responses_annotations_are_inspected(client, provider):
    provider.response = answer("safe")
    provider.response["output"][0]["content"][0]["annotations"] = [{"type": "url_citation", "title": SECRET}]
    response = call(client, "responses", {"model": MODEL, "input": "safe"})
    assert response.status_code == 400 and SECRET not in response.text


def test_revocation_during_media_inspection_prevents_provider_egress(client, provider, monkeypatch):
    async def revoke(*args, **kwargs):
        key = next(k for k in client.gateway.config.keys_by_digest.values() if k.id == "k-nothing")
        monkeypatch.setitem(client.gateway.config.keys_by_digest, key.key_sha256, replace(key, disabled=True))
        return "safe"
    monkeypatch.setattr(media, "extract", revoke)
    response = call(client, "responses", {"model": MODEL, "input": [{"role": "user", "content": [
        {"type": "input_image", "image_url": "data:image/png;base64,aW1hZ2U="}]}]})
    assert response.status_code == 401 and not provider.calls


def test_realtime_uses_ga_audio_configuration_without_automatic_vad_or_extra_asr(realtime_session):
    session = realtime_session
    async def run():
        await session.client_event({"type": "session.update", "session": {"audio": {"input": {
            "turn_detection": {"type": "server_vad"}, "transcription": {"model": "whisper-1"}}}}})
        config = json.loads(session.upstream.send.call_args.args[0])["session"]
        assert config["type"] == "realtime" and "input_audio_format" not in config
        assert config["audio"]["input"] == {"format": {"type": "audio/pcm", "rate": 24000},
            "turn_detection": None, "transcription": None}
        await session.client_event({"type": "response.create", "response": {"output_modalities": ["text"]}})
        config = json.loads(session.upstream.send.call_args.args[0])["response"]
        assert "input" not in config["audio"] and "type" not in config
    asyncio.run(run())


def test_realtime_prices_native_audio_tokens_separately():
    from znyx_runtime.gateway.realtime import usage_cost
    price = {"input_per_mtok": 2, "output_per_mtok": 4, "cached_input_per_mtok": 1,
        "audio_input_per_mtok": 40, "audio_output_per_mtok": 80, "cached_audio_input_per_mtok": 3}
    usage = {"input_tokens": 100, "output_tokens": 50, "input_token_details": {
        "audio_tokens": 20, "cached_tokens": 10, "cached_tokens_details": {"audio_tokens": 5}},
        "output_token_details": {"audio_tokens": 30}}
    record = CallRecord(request_id="rt", key_id="k", tenant_id="", app_id="", agent_id="", api="realtime")
    assert usage_cost(price, usage, record) == (75 * 2 + 5 + 20 * 4 + 15 * 40 + 5 * 3 + 30 * 80) / 1_000_000
    assert usage_cost({"input_per_mtok": 2, "output_per_mtok": 4}, usage, record) is None


def test_realtime_disconnect_keeps_the_reservation_when_provider_usage_is_unknown(realtime_session, monkeypatch):
    from znyx_runtime.gateway.config import Limits
    from znyx_runtime.gateway.pricing import PriceTable
    from znyx_core.state_store import get_state_store
    session = realtime_session
    key = replace(session.key, limits=Limits(spend_usd_per_day=1))
    monkeypatch.setitem(session.gw.config.keys_by_digest, key.key_sha256, key)
    monkeypatch.setattr(session.gw, "prices_for", lambda _: PriceTable({MODEL: {"input_per_mtok": 2, "output_per_mtok": 4}}))
    asyncio.run(session.client_event({"type": "response.create", "response": {"output_modalities": ["text"]}}))
    day, tokens, cost = session.record._reservation
    session.close()
    assert session.record.usage_estimated and session.record.cost_usd == cost
    assert get_state_store().get(day)["cost"] == cost
    assert not session.active and session.record.reason == "realtime_incomplete"


@pytest.mark.parametrize("text", ["hello", SECRET])
def test_realtime_websocket_route_replaces_credentials_and_releases_only_checked_events(client, monkeypatch, text):
    from znyx_runtime.gateway import realtime
    class Upstream:
        def __init__(self):
            self.queue, self.sent = asyncio.Queue(), []
        async def send(self, raw):
            event = json.loads(raw)
            self.sent.append(event)
            if event["type"] == "session.update":
                await self.queue.put({"type": "session.updated", "session": {"id": "sess_test"}})
            elif event["type"] == "response.create":
                await self.queue.put({"type": "response.output_text.delta", "delta": text})
                await self.queue.put({"type": "response.done", "response": {"output": [],
                    "usage": {"input_tokens": 2, "output_tokens": 3}}})
        def __aiter__(self):
            return self
        async def __anext__(self):
            return json.dumps(await self.queue.get())
    upstream, connection = Upstream(), {}
    class Connect:
        def __init__(self, url, **kwargs):
            connection.update(url=url, **kwargs)
        async def __aenter__(self):
            return upstream
        async def __aexit__(self, *args):
            return False
    monkeypatch.setattr(realtime, "connect", Connect)
    with client.websocket_connect("/v1/realtime?model=" + MODEL,
            headers={"authorization": "Bearer " + KEYS["nothing"]}, subprotocols=["realtime"]) as ws:
        assert ws.receive_json()["type"] == "session.updated"
        ws.send_json({"type": "response.create", "response": {"output_modalities": ["text"]}})
        first = ws.receive_json()
        if text == SECRET:
            assert first["type"] == "error" and SECRET not in json.dumps(first)
        else:
            assert first == {"type": "response.output_text.delta", "delta": "hello"}
            assert ws.receive_json()["type"] == "response.done"
    assert KEYS["nothing"] not in json.dumps(connection)
    assert upstream.sent[0]["session"]["audio"]["input"]["turn_detection"] is None
