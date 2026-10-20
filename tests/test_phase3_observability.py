"""Phase 3 on the runtime: usage events, trace context, OTLP spans, prices of prompt-cache
writes, spend controls from the signed bundle set, bounded metric labels.
"""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from znyx_core import usage_events


# -- the usage event contract -----------------------------------------------------------------

def test_an_event_round_trips_and_carries_no_unknown_fields():
    event = usage_events.build(request_id="r1", model="gpt-4o-mini", input_tokens=10, output_tokens=5,
                               cost_usd=0.0001, rule_ids=["pii.email"], techniques={"canary": 1},
                               prompt="never shipped", outcome="ok")
    assert "prompt" not in event and event["schema"] == usage_events.SCHEMA_VERSION
    stored = usage_events.validate(json.loads(json.dumps(event)))
    assert stored["model"] == "gpt-4o-mini" and stored["input_tokens"] == 10
    assert stored["occurred_at"].tzinfo is not None
    assert usage_events.total_tokens(stored) == 15


@pytest.mark.parametrize("bad", [None, [], {"schema": 99}, {"schema": 1}, "x"])
def test_unusable_events_are_refused(bad):
    assert usage_events.validate(bad) is None


def test_validation_bounds_what_a_sender_controls():
    event = usage_events.build(app_id="a" * 5000, rule_ids=[str(i) for i in range(500)],
                               techniques={str(i): i for i in range(100)}, input_tokens=-5,
                               cost_usd=float("inf"), trace_id="not-hex", span_id="ABCDEF0123456789")
    stored = usage_events.validate(event)
    assert len(stored["app_id"]) == usage_events.MAX_STR
    assert len(stored["rule_ids"]) == usage_events.MAX_LIST
    assert len(stored["techniques"]) == usage_events.MAX_MAP
    assert stored["input_tokens"] == 0 and "cost_usd" not in stored and "trace_id" not in stored
    assert stored["span_id"] == "abcdef0123456789"


@pytest.mark.parametrize("header,expected", [
    ("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
     ("0af7651916cd43dd8448eb211c80319c", "b7ad6b7169203331")),
    ("00-00000000000000000000000000000000-b7ad6b7169203331-01", None),
    ("ff-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01", None),
    ("garbage", None), (None, None),
])
def test_traceparent(header, expected):
    assert usage_events.parse_traceparent(header) == expected


# -- prices: prompt-cache writes ---------------------------------------------------------------

def test_anthropic_cache_writes_cost_more_than_input():
    from znyx_runtime.gateway.formats import AnthropicMessages, Usage
    from znyx_runtime.gateway.pricing import PriceTable
    usage = AnthropicMessages.usage({"usage": {"input_tokens": 100, "cache_read_input_tokens": 1000,
                                               "cache_creation_input_tokens": 2000, "output_tokens": 50}})
    assert (usage.input_tokens, usage.cached_input_tokens, usage.cache_write_tokens) == (3100, 1000, 2000)
    cost = PriceTable().cost("claude-sonnet-4-5", usage)
    expected = (100 * 3.00 + 1000 * 0.30 + 2000 * 3.75 + 50 * 15.00) / 1_000_000
    assert cost == pytest.approx(expected)
    assert PriceTable().cost("claude-sonnet-4-5", Usage(3100, 50, 1000, None)) < cost


# -- trace context, users, usage events through the gateway -------------------------------------

def _chat(client, key, headers=None, **body):
    return client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {key}", **(headers or {})},
                       json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "SAY:ok"}], **body})


def test_calls_join_the_callers_trace_and_say_so(runtime):
    from test_bundle_sets import KEY_ALPHA
    parent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    resp = _chat(runtime, KEY_ALPHA, headers={"traceparent": parent})
    assert resp.status_code == 200
    trace_id, span_id = usage_events.parse_traceparent(resp.headers["traceparent"])
    assert trace_id == "0af7651916cd43dd8448eb211c80319c" and span_id != "b7ad6b7169203331"
    assert resp.headers["x-znyx-trace-id"] == trace_id
    alone = _chat(runtime, KEY_ALPHA)
    assert usage_events.parse_traceparent(alone.headers["traceparent"])[0] != trace_id


def test_each_call_is_spooled_as_a_usage_event_of_its_environment(runtime):
    from test_bundle_sets import ENV_ALPHA, ALPHA, KEY_ALPHA
    spool = runtime.usage_spool
    before = spool.read_text().count("\n") if spool.exists() else 0
    resp = _chat(runtime, KEY_ALPHA, headers={"x-znyx-user": "jane@example.com"})
    assert resp.status_code == 200
    lines = spool.read_text().splitlines()
    assert len(lines) == before + 1
    event = usage_events.validate(json.loads(lines[-1]))
    assert event["environment_id"] == ENV_ALPHA and event["project_id"] == ALPHA
    assert event["key_id"] == "alpha-app" and event["model"] == "gpt-4o-mini" and event["outcome"] == "ok"
    assert event["input_tokens"] and event["output_tokens"] and event["cost_usd"] > 0
    assert event["price_known"] is True and event["trace_id"] == resp.headers["x-znyx-trace-id"]
    # the end user is a pseudonym, and the same user always the same one
    assert event["user"].startswith("u_") and "jane" not in lines[-1]
    _chat(runtime, KEY_ALPHA, headers={"x-znyx-user": "jane@example.com"})
    again = usage_events.validate(json.loads(spool.read_text().splitlines()[-1]))
    assert again["user"] == event["user"]
    assert "SAY:ok" not in spool.read_text()


def test_a_refused_call_is_still_a_usage_event(runtime):
    from test_bundle_sets import INSTANCE_KEY
    spool = runtime.usage_spool
    before = spool.read_text().count("\n") if spool.exists() else 0
    assert _chat(runtime, INSTANCE_KEY).status_code == 403
    event = json.loads(spool.read_text().splitlines()[-1])
    assert spool.read_text().count("\n") == before + 1 and event["outcome"] == "denied"


# -- spend controls from the signed set ----------------------------------------------------------

def _gateway_with_controls(controls):
    from znyx_runtime.gateway.config import GatewayConfig
    from znyx_runtime.gateway.pipeline import Gateway
    bm = SimpleNamespace(controls=controls, members=[], effective_env=None)
    return Gateway(GatewayConfig(providers={}, keys_by_digest={}), evaluator=None, bundle_manager=bm)


@pytest.mark.parametrize("entry,blocked", [
    ({"scope": "org"}, True),
    ({"scope": "project", "project_id": "p1"}, True),
    ({"scope": "project", "project_id": "p2"}, False),
    ({"scope": "environment", "environment_id": "e1"}, True),
    ({"scope": "app", "app_id": "support", "project_id": "p1"}, True),
    ({"scope": "app", "app_id": "support", "project_id": "p2"}, False),
    ({"scope": "key", "key_id": "k1", "project_id": "p1"}, True),
    ({"scope": "key", "key_id": "k1", "project_id": "p2"}, False),
    ({"scope": "org", "until": "2001-01-01T00:00:00+00:00"}, False),
])
def test_spend_controls_block_calls_in_their_scope(entry, blocked):
    from znyx_runtime.gateway.config import GatewayKey
    from znyx_runtime.gateway.pipeline import Denied
    gw = _gateway_with_controls({"spend_blocked": [{"name": "October", **entry}]})
    key = GatewayKey(id="k1", key_sha256="", app_id="support", member="e1")
    member = SimpleNamespace(project_id="p1", environment_id="e1")
    if blocked:
        with pytest.raises(Denied) as exc:
            gw._check_spend_controls(key, member)
        assert exc.value.status == 429 and exc.value.code == "budget_exhausted"
    else:
        gw._check_spend_controls(key, member)


# -- OTLP --------------------------------------------------------------------------------------

def _record(**over):
    from znyx_runtime.gateway.pipeline import CallRecord
    record = CallRecord(request_id="gw-1", key_id="k1", tenant_id="t", app_id="support", agent_id="default",
                        api="openai", provider="openai", provider_format="openai", model="gpt-4o-mini",
                        response_model="gpt-4o-mini-2024-07-18", status=200, input_tokens=10, output_tokens=5,
                        cost_usd=0.0001, latency_ms=120, ttft_ms=40, guard_ms=8,
                        trace_id="0af7651916cd43dd8448eb211c80319c", span_id="00f067aa0ba902b7",
                        parent_span_id="b7ad6b7169203331", user="u_abc")
    for k, v in over.items():
        setattr(record, k, v)
    return record


def _attrs(span):
    out = {}
    for a in span["attributes"]:
        (kind, value), = a["value"].items()
        out[a["key"]] = value
    return out


def test_a_call_is_a_genai_client_span():
    from znyx_runtime.otel import SPAN_KIND_CLIENT, STATUS_OK, call_span
    span = call_span(_record(), {"project_id": "p1", "environment_id": "e1"})
    attrs = _attrs(span)
    assert span["name"] == "chat gpt-4o-mini" and span["kind"] == SPAN_KIND_CLIENT
    assert span["parentSpanId"] == "b7ad6b7169203331" and span["status"]["code"] == STATUS_OK
    assert attrs["gen_ai.operation.name"] == "chat" and attrs["gen_ai.provider.name"] == "openai"
    assert attrs["gen_ai.usage.input_tokens"] == "10" and attrs["gen_ai.response.model"] == "gpt-4o-mini-2024-07-18"
    assert attrs["enduser.id"] == "u_abc" and attrs["znyx.project_id"] == "p1"
    assert int(span["endTimeUnixNano"]) - int(span["startTimeUnixNano"]) == 120_000_000


def test_a_check_is_a_child_span_of_its_call():
    from znyx_runtime.otel import evaluation_span
    span = evaluation_span({"trace_id": "0af7651916cd43dd8448eb211c80319c", "span_id": "00f067aa0ba902b7",
                            "context": "input", "decision": "REDACT", "latency_ms": 9,
                            "timestamp": "2026-10-20T10:00:00+00:00", "rule_hit_ids": ["pii.email"]})
    assert span["traceId"] == "0af7651916cd43dd8448eb211c80319c" and span["parentSpanId"] == "00f067aa0ba902b7"
    assert span["name"] == "znyx.guard input" and _attrs(span)["znyx.decision"] == "REDACT"


def test_spans_are_exported_in_batches_and_retried():
    from znyx_runtime.otel import OtlpExporter, call_span
    received, fail = [], {"n": 1}

    def handler(request):
        if fail["n"]:
            fail["n"] -= 1
            return httpx.Response(503)
        received.append((request.headers.get("x-api-key"), json.loads(request.content)))
        return httpx.Response(200)

    exporter = OtlpExporter("https://collector.example/v1/traces", {"x-api-key": "secret"},
                            transport=httpx.MockTransport(handler))
    exporter.add(call_span(_record(), {}))
    assert asyncio.run(exporter.flush()) is False and exporter.failures == 1
    assert asyncio.run(exporter.flush()) is True and exporter.exported == 1
    key, payload = received[0]
    assert key == "secret"
    spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
    resource = {a["key"]: a["value"] for a in payload["resourceSpans"][0]["resource"]["attributes"]}
    assert len(spans) == 1 and resource["service.name"] == {"stringValue": "znyx"}


def test_the_exporter_reads_the_standard_variables(monkeypatch):
    from znyx_runtime.otel import OtlpExporter
    assert OtlpExporter.from_env({}) is None
    exporter = OtlpExporter.from_env({"OTEL_EXPORTER_OTLP_ENDPOINT": "https://otel.example.com:4318/",
                                      "OTEL_EXPORTER_OTLP_HEADERS": "Authorization=Basic%20abc,dd-api-key=1",
                                      "OTEL_SERVICE_NAME": "payments-gw"})
    assert exporter.endpoint == "https://otel.example.com:4318/v1/traces"
    assert exporter.headers["Authorization"] == "Basic abc" and exporter.service_name == "payments-gw"
    monkeypatch.setenv("ZNYX_AIRGAPPED", "true")
    assert OtlpExporter.from_env({"OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "https://otel.example.com/v1/traces"}) is None
    assert OtlpExporter.from_env({"OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "http://otel-collector:4318/v1/traces"})


def test_a_full_queue_drops_the_oldest():
    from znyx_runtime.otel import OtlpExporter
    exporter = OtlpExporter("https://c.example/v1/traces", max_queue=3)
    for i in range(5):
        exporter.add({"spanId": str(i)})
    assert [s["spanId"] for s in exporter._queue] == ["2", "3", "4"] and exporter.dropped == 2


# -- bounded labels ------------------------------------------------------------------------------

def test_the_app_label_comes_from_configuration(runtime):
    from test_bundle_sets import KEY_ALPHA
    _chat(runtime, KEY_ALPHA)
    metrics = runtime.get("/metrics", headers={"X-API-Key": "zx_set_instance_key"}).text
    lines = [l for l in metrics.splitlines() if l.startswith("znyx_gateway_requests_total{")]
    assert any('app="8d0c1f9e-0000-4000-8000-0000000000a1"' in l for l in lines)
    assert any('app="other"' in l for l in lines)            # the instance key's denied call



# -- review fixes (2026-10-20) -----------------------------------------------------------------

def test_strings_the_store_refuses_never_leave_the_runtime():
    """A NUL or a lone surrogate in a model name used to make the control plane refuse the
    whole batch, and the shipper retried it forever."""
    event = usage_events.build(model="gpt-4o-mini\x00evil", app_id="a\ud800b", user="u\x00",
                               rule_ids=["r\x00"], techniques={"t\x00": 1})
    assert event["model"] == "gpt-4o-minievil" and event["app_id"] == "ab"
    stored = usage_events.validate({**event, "model": "x\x00y"})
    assert stored["model"] == "xy" and stored["rule_ids"] == ["r"] and stored["techniques"] == {"t": 1}
    json.dumps(stored, default=str).encode("utf-8")      # encodable


def test_per_event_values_are_capped():
    stored = usage_events.validate(usage_events.build(input_tokens=10 ** 15, latency_ms=10 ** 12,
                                                      cost_usd=999_999_999.0, status=10 ** 6))
    assert stored["input_tokens"] == 10_000_000 and stored["latency_ms"] == 86_400_000
    assert "cost_usd" not in stored and stored["status"] == 999


def test_an_unpriced_model_is_refused_where_a_blocking_budget_applies():
    from znyx_runtime.gateway.config import GatewayKey
    from znyx_runtime.gateway.pipeline import Denied
    gw = _gateway_with_controls({"priced_only": [{"name": "Q4", "scope": "project", "project_id": "p1"}]})
    key = GatewayKey(id="k1", key_sha256="", app_id="support", member="e1")
    here = SimpleNamespace(project_id="p1", environment_id="e1")
    elsewhere = SimpleNamespace(project_id="p2", environment_id="e2")
    with pytest.raises(Denied) as exc:
        gw._check_spend_controls(key, here, "o1-pro")
    assert exc.value.status == 403 and exc.value.code == "price_unknown"
    gw._check_spend_controls(key, here, "gpt-4o-mini")       # priced: fine
    gw._check_spend_controls(key, elsewhere, "o1-pro")       # no budget there: fine


def test_a_block_without_a_timezone_is_read_as_utc():
    from znyx_runtime.gateway.config import GatewayKey
    gw = _gateway_with_controls({"spend_blocked": [{"scope": "org", "until": "2001-01-01T00:00:00"}]})
    gw._check_spend_controls(GatewayKey(id="k", key_sha256=""), None)    # long expired: not blocked


def test_pseudonyms_are_keyed_on_the_runtime_side(tmp_path, monkeypatch):
    import hashlib
    import os
    import znyx_runtime.install_state as install_state
    monkeypatch.setattr(install_state, "STATE_DIR", tmp_path)
    monkeypatch.setattr(install_state, "_user_id_key", None)
    monkeypatch.delenv("ZNYX_USER_ID_KEY", raising=False)
    gw = _gateway_with_controls({})
    gw.bundle_manager.members = []
    request = SimpleNamespace(headers={"x-znyx-user": "jane@example.com"})
    first = gw._user_of(request, {})
    assert first == gw._user_of(request, {}) and first.startswith("u_")
    guess = "u_" + hashlib.sha256("\x1fjane@example.com".encode()).hexdigest()[:16]
    assert first != guess
    assert oct(os.stat(tmp_path / "user-id.key").st_mode & 0o777) == "0o600"
    monkeypatch.setenv("ZNYX_USER_ID_KEY", "shared-by-every-replica-0123456789abcdef")
    assert gw._user_of(request, {}) != first        # the operator's key, the same on every replica


def test_an_estimated_call_knows_whether_its_model_has_a_price():
    gw = _gateway_with_controls({})
    record = _record(input_tokens=None, output_tokens=None, cost_usd=None, response_model=None,
                     price_known=False, estimated_input_tokens=100)
    gw.estimate_missing_usage(record, 400)
    assert record.usage_estimated and record.price_known and record.cost_usd > 0


def test_spans_lost_when_the_queue_is_full_are_all_counted():
    """A failed batch goes back to the front of the queue; what no longer fits (new spans
    arrived meanwhile) is lost and every lost span is counted."""
    from znyx_runtime.otel import OtlpExporter
    exporter = OtlpExporter("https://c.example/v1/traces", max_queue=4, batch_size=4,
                            transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    for i in range(4):
        exporter.add({"spanId": str(i)})

    async def flush_while_spans_arrive():
        task = asyncio.ensure_future(exporter.flush())
        await asyncio.sleep(0)                      # the batch is out of the queue now
        for i in range(4, 7):
            exporter.add({"spanId": str(i)})
        return await task

    assert asyncio.run(flush_while_spans_arrive()) is False
    assert len(exporter._queue) == 4 and exporter.dropped == 3


def test_model_labels_are_bounded_even_when_the_provider_echoes():
    from znyx_runtime.gateway import usage
    saved = set(usage._MODEL_LABELS)
    usage._MODEL_LABELS.clear()
    try:
        labels = {usage._model_label(SimpleNamespace(response_model=f"m{i}")) for i in range(usage.MAX_MODEL_LABELS + 20)}
        assert len(labels) == usage.MAX_MODEL_LABELS + 1 and "other" in labels
    finally:
        usage._MODEL_LABELS.clear()
        usage._MODEL_LABELS.update(saved)

# The managed runtime (two projects) from the bundle-set tests.
from test_bundle_sets import keys, runtime, _fresh  # noqa: E402,F401


def test_a_call_without_a_recognised_key_is_not_usage(runtime):
    spool = runtime.usage_spool
    before = spool.read_text().count("\n") if spool.exists() else 0
    assert _chat(runtime, "zxg_not_a_key").status_code == 401
    assert (spool.read_text().count("\n") if spool.exists() else 0) == before
