"""Usage capture per gateway call: Prometheus metrics and the local dashboard.

Metadata only: model, provider, tokens (from the provider's usage field), cost (from
the price table), latency, time to first token, guard overhead, decisions and rule
ids. Request and response text is never recorded here. With several worker
processes the metrics merge like every other counter (workers.instance_snapshot).
"""
from __future__ import annotations

import collections
import threading
import time
from typing import Any, Deque, Dict, List

_RECENT: Deque[Dict[str, Any]] = collections.deque(maxlen=100)
_LOCK = threading.Lock()
_REGISTERED = False

LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)
GUARD_BUCKETS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.25, 0.5, 1.0, 2.5)


def _collector():
    global _REGISTERED
    from znyx_core.middleware.metrics import MetricsCollector
    collector = MetricsCollector()
    if not _REGISTERED:
        collector.register("znyx_gateway_requests_total", "counter", "Gateway calls")
        collector.register("znyx_gateway_tokens_total", "counter", "Tokens reported by providers")
        collector.register("znyx_gateway_cost_usd_total", "counter", "Cost of gateway calls (price table)")
        collector.register("znyx_gateway_latency_seconds", "histogram",
                           "End-to-end gateway call latency", buckets=LATENCY_BUCKETS)
        collector.register("znyx_gateway_ttft_seconds", "histogram",
                           "Time to first token released to the client", buckets=LATENCY_BUCKETS)
        collector.register("znyx_gateway_guard_seconds", "histogram",
                           "Guardrail time added to a gateway call", buckets=GUARD_BUCKETS)
        _REGISTERED = True
    return collector


def _model_label(record) -> str:
    """The model a provider reported serving. A requested name is the caller's free
    text (unbounded, and on a denied call never served), so it is not a label."""
    model = record.response_model
    if not isinstance(model, str) or not model:
        return "other"
    return model[:100]


def record_call(record) -> None:
    collector = _collector()
    labels = {"provider": record.provider or "none", "model": _model_label(record)}
    collector.counter_inc("znyx_gateway_requests_total",
                          labels={**labels, "outcome": record.outcome, "status": str(record.status)})
    if record.input_tokens:
        collector.counter_inc("znyx_gateway_tokens_total", record.input_tokens, {**labels, "direction": "input"})
    if record.output_tokens:
        collector.counter_inc("znyx_gateway_tokens_total", record.output_tokens, {**labels, "direction": "output"})
    if record.cost_usd:
        collector.counter_inc("znyx_gateway_cost_usd_total", record.cost_usd, labels)
    collector.histogram_observe("znyx_gateway_latency_seconds", record.latency_ms / 1000.0,
                                {"provider": labels["provider"]})
    if record.ttft_ms is not None:
        collector.histogram_observe("znyx_gateway_ttft_seconds", record.ttft_ms / 1000.0,
                                    {"provider": labels["provider"]})
    collector.histogram_observe("znyx_gateway_guard_seconds", record.guard_ms / 1000.0,
                                {"provider": labels["provider"]})
    with _LOCK:
        entry = {k: getattr(record, k) for k in (
            "request_id", "key_id", "app_id", "agent_id", "api", "provider", "model", "response_model",
            "stream", "status", "outcome", "input_decision", "output_decision", "input_tokens",
            "output_tokens", "cost_usd", "latency_ms", "ttft_ms", "guard_ms", "reason")}
        entry["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _RECENT.append(entry)


def recent_calls() -> List[Dict[str, Any]]:
    with _LOCK:
        return list(_RECENT)
