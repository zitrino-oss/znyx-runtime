"""The local dashboard: one instance's activity, without the console.

``GET /dashboard`` serves a static page (no third-party assets, so it works
air-gapped). The page reads ``GET /v1/local/summary``, which needs the same API key
as the evaluate endpoints and returns metadata only: counts, rule ids, latencies
and the last decisions, never request or response text. With several worker
processes the numbers are the whole instance's (``workers.instance_snapshot``).

Switch it off with ``ZNYX_DASHBOARD=false`` (``dashboard.enabled`` in znyx.yaml).
"""
from __future__ import annotations

import asyncio
import os
import time
from importlib.resources import files
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from znyx_runtime.api.routes import verify_runtime_auth

router = APIRouter(include_in_schema=False)

_STARTED = time.time()
_ASSETS = {"": ("index.html", "text/html; charset=utf-8"),
           "app.js": ("app.js", "text/javascript; charset=utf-8"),
           "app.css": ("app.css", "text/css; charset=utf-8")}
# Same-origin only; nothing inline, nothing from elsewhere.
_CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
        "img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")


def dashboard_enabled() -> bool:
    return (os.getenv("ZNYX_DASHBOARD") or "true").strip().lower() not in ("0", "false", "no", "off")


def _asset(name: str) -> Response:
    if not dashboard_enabled() or name not in _ASSETS:
        raise HTTPException(status_code=404)
    filename, media_type = _ASSETS[name]
    body = (files("znyx_runtime") / "dashboard" / filename).read_bytes()
    return Response(content=body, media_type=media_type,
                    headers={"Content-Security-Policy": _CSP, "Cache-Control": "no-cache"})


@router.get("/dashboard")
async def dashboard_page() -> Response:
    return _asset("")


@router.get("/dashboard/{name}")
async def dashboard_asset(name: str) -> Response:
    return _asset(name)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _series(snapshot: Dict[str, Any], kind: str, name: str) -> List[Tuple[Dict[str, str], Any]]:
    return [(dict(k), v) for k, v in snapshot.get(kind, {}).get(name, [])]


def _percentile(buckets: List[List[float]], count: int, q: float) -> Optional[float]:
    """Upper bound of the bucket holding the q-th observation, in milliseconds."""
    if not count:
        return None
    target = q * count
    cumulative = 0
    for bound, n in sorted(buckets):
        cumulative += n
        if cumulative >= target:
            return round(bound * 1000, 1)
    return None  # beyond the largest bucket


def summarise_gateway(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Gateway totals from the merged metrics (metadata only)."""
    outcomes: Dict[str, float] = {}
    models: Dict[str, Dict[str, float]] = {}
    for labels, value in _series(snapshot, "counters", "znyx_gateway_requests_total"):
        outcomes[labels.get("outcome", "?")] = outcomes.get(labels.get("outcome", "?"), 0) + value
        m = models.setdefault(labels.get("model", "?"), {"calls": 0, "tokens": 0, "cost_usd": 0.0})
        m["calls"] += value
    tokens = {"input": 0.0, "output": 0.0}
    for labels, value in _series(snapshot, "counters", "znyx_gateway_tokens_total"):
        tokens[labels.get("direction", "input")] = tokens.get(labels.get("direction", "input"), 0) + value
        models.setdefault(labels.get("model", "?"), {"calls": 0, "tokens": 0, "cost_usd": 0.0})["tokens"] += value
    cost = 0.0
    for labels, value in _series(snapshot, "counters", "znyx_gateway_cost_usd_total"):
        cost += value
        models.setdefault(labels.get("model", "?"), {"calls": 0, "tokens": 0, "cost_usd": 0.0})["cost_usd"] += value

    def percentiles(name: str) -> Dict[str, Optional[float]]:
        merged: Dict[float, float] = {}
        count = 0
        for _labels, entry in _series(snapshot, "histograms", name):
            for bound, n in entry["buckets"]:
                merged[bound] = merged.get(bound, 0) + n
            count += entry["count"]
        buckets = [[b, n] for b, n in merged.items()]
        return {"p50": _percentile(buckets, count, 0.5), "p95": _percentile(buckets, count, 0.95)}

    calls = int(sum(outcomes.values()))
    return {
        "calls": calls,
        "outcomes": {k: int(v) for k, v in sorted(outcomes.items())},
        "tokens": {k: int(v) for k, v in tokens.items()},
        "cost_usd": round(cost, 6),
        "models": [{"model": k, "calls": int(v["calls"]), "tokens": int(v["tokens"]),
                    "cost_usd": round(v["cost_usd"], 6)}
                   for k, v in sorted(models.items(), key=lambda kv: -kv[1]["calls"])[:10]],
        "ttft_ms": percentiles("znyx_gateway_ttft_seconds"),
        "guard_ms": percentiles("znyx_gateway_guard_seconds"),
        "latency_ms": percentiles("znyx_gateway_latency_seconds"),
    }


def summarise(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    decisions: Dict[str, float] = {}
    contexts: Dict[str, float] = {}
    for labels, value in _series(snapshot, "counters", "guardrails_evaluations_total"):
        decisions[labels.get("decision", "?")] = decisions.get(labels.get("decision", "?"), 0) + value
        contexts[labels.get("context", "?")] = contexts.get(labels.get("context", "?"), 0) + value
    rules: Dict[str, float] = {}
    for labels, value in _series(snapshot, "counters", "guardrails_detector_hits_total"):
        rule = labels.get("detector_name", "?")
        rules[rule] = rules.get(rule, 0) + value
    merged: Dict[float, float] = {}
    count, total = 0, 0.0
    for _labels, entry in _series(snapshot, "histograms", "guardrails_evaluation_duration_seconds"):
        for bound, n in entry["buckets"]:
            merged[bound] = merged.get(bound, 0) + n
        count += entry["count"]
        total += entry["sum"]
    buckets = [[b, n] for b, n in merged.items()]
    evaluations = int(sum(decisions.values()))
    return {
        "evaluations": evaluations,
        "decisions": {k: int(v) for k, v in sorted(decisions.items())},
        "contexts": {k: int(v) for k, v in sorted(contexts.items())},
        "top_rules": [{"rule": r, "hits": int(n)}
                      for r, n in sorted(rules.items(), key=lambda kv: -kv[1])[:15]],
        "latency_ms": {
            "mean": round(total / count * 1000, 1) if count else None,
            "p50": _percentile(buckets, count, 0.50),
            "p95": _percentile(buckets, count, 0.95),
            "p99": _percentile(buckets, count, 0.99),
        },
    }


async def _model_server() -> Optional[Dict[str, Any]]:
    url = (os.getenv("ZNYX_INFERENCE_URL") or "").rstrip("/")
    if not url:
        return None
    import httpx

    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            resp = await client.get(f"{url}/v1/models")
        data = resp.json() if resp.status_code == 200 else {}
    except Exception:  # noqa: BLE001 - the dashboard reports it as unreachable
        return {"url": url, "reachable": False, "models": []}
    models = data.get("models") if isinstance(data, dict) else data
    out = []
    for m in models or []:
        if isinstance(m, dict):
            out.append({k: m.get(k) for k in ("task", "model_id", "revision", "runner", "available",
                                              "active", "detail") if k in m})
    return {"url": url, "reachable": True, "models": out}


@router.get("/v1/local/summary", dependencies=[Depends(verify_runtime_auth)])
async def local_summary() -> Dict[str, Any]:
    if not dashboard_enabled():
        raise HTTPException(status_code=404)
    from znyx_core.middleware.metrics import MetricsCollector
    from znyx_core.utils.env import is_airgapped
    from znyx_runtime import __version__
    from znyx_runtime import main as runtime
    from znyx_runtime.workers import instance_snapshot, worker_count, worker_records

    snap = await asyncio.to_thread(instance_snapshot, MetricsCollector(), runtime.recent_decisions,
                                   runtime.worker_reporter)
    bundle = runtime.bundle_manager.bundle_info if runtime.bundle_manager else {}
    return {
        "version": os.getenv("ZNYX_PLATFORM_VERSION") or __version__,
        "runtime_version": __version__,
        "mode": runtime.config.mode if runtime.config else None,
        "airgapped": is_airgapped(),
        "state_backend": (os.getenv("ZNYX_STATE_BACKEND") or "memory").lower(),
        "uptime_seconds": round(time.time() - _STARTED),
        "policy": _policy_summary(bundle),
        "workers": {"count": worker_count(), "reporting": snap["workers"],
                    "processes": [{k: w.get(k) for k in ("index", "pid", "uptime_seconds",
                                                         "requests", "age_seconds")}
                                  for w in worker_records()]},
        "model_server": await _model_server(),
        **summarise(snap["metrics"]),
        "recent": list(reversed(snap["recent"]))[:50],
        "gateway": _gateway_summary(snap),
    }


def _policy_summary(bundle: dict) -> dict:
    """What the dashboard shows about the policy: the bundle, or each environment of a
    runtime serving several (ids and hashes only, never policy content)."""
    summary = {k: bundle.get(k) for k in ("mode", "policy_path", "bundle_id", "policy_hash",
                                          "published_at", "environment") if k in bundle}
    if bundle.get("bundle_set"):
        summary["environments"] = [
            {"project": m.get("project_slug") or m.get("project_id"), "environment": m.get("environment"),
             "bundle_id": m.get("bundle_id"), "policy_hash": m.get("policy_hash")}
            for m in bundle.get("members") or []]
        summary["missing"] = len(bundle.get("missing") or [])
    return summary


def _gateway_summary(snap: Dict[str, Any]) -> Dict[str, Any]:
    from znyx_runtime import main as runtime
    gw = getattr(runtime.app.state, "gateway", None)
    if gw is None or not gw.config.enabled:
        return {"enabled": False}
    return {"enabled": True, "providers": sorted(gw.config.providers), "source": gw.config.source,
            "keys": len(gw.config.keys_by_digest), **summarise_gateway(snap["metrics"]),
            "recent": list(reversed(snap.get("gateway_recent") or []))[:50]}
