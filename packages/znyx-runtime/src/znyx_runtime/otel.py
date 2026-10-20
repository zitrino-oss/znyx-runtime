"""OpenTelemetry traces from the runtime, exported over OTLP/HTTP (JSON).

Every gateway call is a span following the OpenTelemetry GenAI conventions
(``chat {model}``, ``gen_ai.*`` attributes), and every guardrail check a child span of
the call it belongs to (or of the caller's span, for the evaluate API). A caller that
sends ``traceparent`` (W3C trace context) with its LLM calls and its tool checks gets
one trace per agent run. Metadata only: no prompt or answer text is ever put in a span.

Exported to the customer's own collector or tool, never to ZNYX: Datadog, Splunk,
Langfuse and SIEM tools accept OTLP/HTTP directly or through an OpenTelemetry
Collector. Configured with the standard variables: ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT``
(or ``OTEL_EXPORTER_OTLP_ENDPOINT``, to which ``/v1/traces`` is added),
``OTEL_EXPORTER_OTLP_HEADERS`` (``key=value,key=value``, for API keys) and
``OTEL_SERVICE_NAME``. Without an endpoint nothing is exported. Air-gapped, an endpoint
outside the network is refused.

No OpenTelemetry SDK dependency: spans are queued in memory (bounded; the oldest are
dropped when the collector is unreachable for long) and sent in batches by a
background task. Exporting never slows or fails a request.
"""
from __future__ import annotations

import asyncio
import collections
import contextvars
import logging
import os
import time
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse

logger = logging.getLogger(__name__)

SPAN_KIND_INTERNAL, SPAN_KIND_SERVER, SPAN_KIND_CLIENT = 1, 2, 3
STATUS_UNSET, STATUS_OK, STATUS_ERROR = 0, 1, 2

# The trace context a request arrived with (traceparent), for evaluate-API spans whose
# body carries no trace id. Set by TraceContextMiddleware.
incoming: contextvars.ContextVar[Optional[Tuple[str, str]]] = contextvars.ContextVar(
    "znyx_incoming_trace", default=None)

_GEN_AI_PROVIDER = {"openai": "openai", "azure_openai": "azure.ai.openai", "anthropic": "anthropic"}


def _attr(key: str, value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    if isinstance(value, (list, tuple)):
        return {"key": key, "value": {"arrayValue": {"values": [{"stringValue": str(v)} for v in value]}}}
    return {"key": key, "value": {"stringValue": str(value)}}


def _attrs(pairs: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [a for a in (_attr(k, v) for k, v in pairs.items()) if a is not None]


def _hex(value: Optional[str], length: int) -> Optional[str]:
    if not value:
        return None
    value = str(value).lower().replace("-", "")
    if len(value) != length or any(c not in "0123456789abcdef" for c in value):
        return None
    return value


def call_span(record, event: Dict[str, Any]) -> Dict[str, Any]:
    """The span of one gateway call (GenAI ``chat`` client span)."""
    model = record.model or "unknown"
    end_ns = record.started_ns + int(record.latency_ms) * 1_000_000
    failed = record.outcome in ("error", "denied") or (record.status or 0) >= 400
    attributes = _attrs({
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": _GEN_AI_PROVIDER.get(record.provider_format or "", record.provider_format),
        "gen_ai.system": _GEN_AI_PROVIDER.get(record.provider_format or "", record.provider_format),
        "gen_ai.request.model": record.model,
        "gen_ai.response.model": record.response_model,
        "gen_ai.usage.input_tokens": record.input_tokens,
        "gen_ai.usage.output_tokens": record.output_tokens,
        "enduser.id": record.user,
        "error.type": record.reason if failed else None,
        "http.response.status_code": record.status or None,
        "znyx.request_id": record.request_id,
        "znyx.provider": record.provider,
        "znyx.app_id": record.app_id,
        "znyx.agent_id": record.agent_id,
        "znyx.key_id": record.key_id,
        "znyx.project_id": event.get("project_id"),
        "znyx.environment_id": event.get("environment_id"),
        "znyx.outcome": record.outcome,
        "znyx.decision.input": record.input_decision,
        "znyx.decision.output": record.output_decision,
        "znyx.rule_ids": list(record.rule_ids) or None,
        "znyx.stream": record.stream,
        "znyx.cost_usd": float(record.cost_usd) if record.cost_usd is not None else None,
        "znyx.usage.cached_input_tokens": record.cached_input_tokens,
        "znyx.usage.cache_write_tokens": record.cache_write_tokens,
        "znyx.usage.estimated": record.usage_estimated or None,
        "znyx.ttft_ms": record.ttft_ms,
        "znyx.upstream_ms": record.upstream_ms,
        "znyx.guard_ms": record.guard_ms,
        "znyx.techniques": [f"{k}={v}" for k, v in sorted(record.techniques.items())] or None,
    })
    span = {"traceId": record.trace_id, "spanId": record.span_id, "name": f"chat {model}",
            "kind": SPAN_KIND_CLIENT, "startTimeUnixNano": str(record.started_ns),
            "endTimeUnixNano": str(end_ns), "attributes": attributes,
            "status": {"code": STATUS_ERROR if failed else STATUS_OK}}
    if record.parent_span_id:
        span["parentSpanId"] = record.parent_span_id
    return span


def evaluation_span(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The span of one guardrail check (an evaluation event), a child of the span it
    belongs to: the gateway call's, or the caller's for the evaluate API."""
    from znyx_core import usage_events

    trace_id = _hex(event.get("trace_id"), 32)
    parent = _hex(event.get("span_id"), 16)
    if trace_id is None:
        given = incoming.get()
        if given:
            trace_id, parent = given
        else:
            trace_id = usage_events.new_trace_id()
    try:
        end = datetime.fromisoformat(str(event.get("timestamp")))
        end_ns = int(end.timestamp() * 1_000_000_000)
    except (TypeError, ValueError):
        end_ns = time.time_ns()
    start_ns = end_ns - int(event.get("latency_ms") or 0) * 1_000_000
    context = event.get("context") or "input"
    decision = event.get("decision")
    span = {"traceId": trace_id, "spanId": usage_events.new_span_id(), "name": f"znyx.guard {context}",
            "kind": SPAN_KIND_INTERNAL, "startTimeUnixNano": str(start_ns), "endTimeUnixNano": str(end_ns),
            "attributes": _attrs({
                "znyx.context": context, "znyx.decision": decision,
                "znyx.risk_score": event.get("risk_score"),
                "znyx.rule_ids": list(event.get("rule_hit_ids") or []) or None,
                "znyx.policy_version": event.get("policy_version"),
                "znyx.request_id": event.get("request_id"),
                "znyx.app_id": event.get("app_id"), "znyx.agent_id": event.get("agent_id"),
                "session.id": event.get("session_id"),
            }),
            "status": {"code": STATUS_OK}}
    if parent:
        span["parentSpanId"] = parent
    return span


def _parse_headers(raw: str) -> Dict[str, str]:
    headers = {}
    for pair in (raw or "").split(","):
        if "=" in pair:
            key, _, value = pair.partition("=")
            if key.strip():
                headers[unquote(key.strip())] = unquote(value.strip())
    return headers


class OtlpExporter:
    def __init__(self, endpoint: str, headers: Optional[Dict[str, str]] = None,
                 service_name: str = "znyx", max_queue: int = 10_000, batch_size: int = 512,
                 interval: float = 2.0, transport=None):
        self.endpoint = endpoint
        self.headers = {"content-type": "application/json", **(headers or {})}
        self.service_name = service_name
        self.batch_size = batch_size
        self.interval = interval
        self._transport = transport
        self._queue: Deque[Dict[str, Any]] = collections.deque(maxlen=max_queue)
        self._task: Optional[asyncio.Task] = None
        self.exported = 0
        self.dropped = 0
        self.failures = 0

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> Optional["OtlpExporter"]:
        env = os.environ if env is None else env
        endpoint = (env.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or "").strip()
        if not endpoint:
            base = (env.get("OTEL_EXPORTER_OTLP_ENDPOINT") or "").strip().rstrip("/")
            endpoint = f"{base}/v1/traces" if base else ""
        if not endpoint:
            return None
        if not endpoint.startswith(("http://", "https://")):
            logger.error("OTLP endpoint must be an http(s) URL; traces are not exported")
            return None
        from znyx_core.engine.egress import in_network
        from znyx_core.utils.env import is_airgapped
        if is_airgapped() and not in_network(urlparse(endpoint).hostname):
            logger.error("Air-gapped: the OTLP endpoint is outside the network; traces are not exported")
            return None
        headers = _parse_headers(env.get("OTEL_EXPORTER_OTLP_TRACES_HEADERS")
                                 or env.get("OTEL_EXPORTER_OTLP_HEADERS") or "")
        return cls(endpoint, headers, service_name=env.get("OTEL_SERVICE_NAME") or "znyx")

    # -- queueing ---------------------------------------------------------------

    def add(self, span: Optional[Dict[str, Any]]) -> None:
        if span is None:
            return
        if len(self._queue) == self._queue.maxlen:
            self.dropped += 1
        self._queue.append(span)

    def export_call(self, record, event: Dict[str, Any]) -> None:
        try:
            self.add(call_span(record, event))
        except Exception:  # noqa: BLE001 - tracing must never affect a call
            logger.debug("otel: call span failed", exc_info=True)

    def export_evaluation(self, event: Dict[str, Any]) -> None:
        try:
            self.add(evaluation_span(event))
        except Exception:  # noqa: BLE001
            logger.debug("otel: evaluation span failed", exc_info=True)

    def payload(self, spans: List[Dict[str, Any]]) -> Dict[str, Any]:
        from znyx_runtime import __version__
        resource = _attrs({"service.name": self.service_name, "service.version": __version__,
                           "telemetry.sdk.name": "znyx", "telemetry.sdk.language": "python"})
        return {"resourceSpans": [{"resource": {"attributes": resource},
                                   "scopeSpans": [{"scope": {"name": "znyx.runtime", "version": __version__},
                                                   "spans": spans}]}]}

    # -- sending ------------------------------------------------------------------

    async def flush(self) -> bool:
        if not self._queue:
            return True
        import httpx
        spans = [self._queue.popleft() for _ in range(min(self.batch_size, len(self._queue)))]
        try:
            async with httpx.AsyncClient(timeout=10.0, transport=self._transport) as client:
                resp = await client.post(self.endpoint, json=self.payload(spans), headers=self.headers)
            if resp.status_code < 300:
                self.exported += len(spans)
                return True
            logger.warning("OTLP export rejected: HTTP %s", resp.status_code)
            if 400 <= resp.status_code < 500 and resp.status_code not in (408, 429):
                self.dropped += len(spans)       # the collector will not take these
                return True
        except Exception as exc:  # noqa: BLE001 - an unreachable collector is retried
            logger.debug("OTLP export failed: %s", exc)
        self.failures += 1
        room = self._queue.maxlen - len(self._queue)
        kept = spans[-room:] if room > 0 else []
        self.dropped += len(spans) - len(kept)    # what no longer fits is lost, and counted
        for span in reversed(kept):               # back to the front, for the next try
            self._queue.appendleft(span)
        return False

    async def _run(self) -> None:
        backoff = self.interval
        while True:
            await asyncio.sleep(backoff)
            ok = True
            while self._queue and ok:
                ok = await self.flush()
            backoff = self.interval if ok else min(backoff * 2, 60.0)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None
        for _ in range(3):                         # a last try, briefly
            if not self._queue or not await self.flush():
                break

    def status(self) -> Dict[str, Any]:
        return {"endpoint_host": urlparse(self.endpoint).hostname, "queued": len(self._queue),
                "exported": self.exported, "dropped": self.dropped, "failures": self.failures}


class TraceContextMiddleware:
    """Remember a request's ``traceparent`` (for evaluate-API spans)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        from znyx_core import usage_events
        header = dict(scope.get("headers") or []).get(b"traceparent")
        token = incoming.set(usage_events.parse_traceparent(header.decode("latin-1")) if header else None)
        try:
            await self.app(scope, receive, send)
        finally:
            incoming.reset(token)
