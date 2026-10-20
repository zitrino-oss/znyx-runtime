"""Multi-worker support in the runtime: worker identity, liveness and metrics files,
merged metrics, the local dashboard and its summary endpoint."""
import asyncio
import json
import os
import subprocess
import sys

import httpx
import pytest
from fastapi.testclient import TestClient

from conftest import MINIMAL_POLICY_YAML
from znyx_core.middleware.metrics import MetricsCollector, merge_snapshots
from znyx_runtime import workers
from znyx_runtime.api import dashboard_routes
from znyx_runtime.workers import RecentDecisions, WorkerReporter, instance_snapshot

ALLOW_INPUT = [["context", "input"], ["decision", "ALLOW"]]


def _counter_snapshot(name, series):
    """A hand-built metrics snapshot: ``series`` is [(labels dict, value), ...]."""
    return {"counters": {name: [[[list(p) for p in labels.items()], v] for labels, v in series]},
            "gauges": {}, "histograms": {}, "help": {}}


@pytest.fixture()
def collector():
    """The (singleton) collector, empty before and after the test."""
    c = MetricsCollector()
    c.reset()
    yield c
    c.reset()


def _evaluations(snapshot):
    return sum(v for _, v in snapshot["counters"].get("guardrails_evaluations_total", []))


# ---------------------------------------------------------------------------
# Worker identity
# ---------------------------------------------------------------------------

def test_defaults_are_worker_zero_of_one(monkeypatch):
    for name in ("ZNYX_WORKER_INDEX", "ZNYX_WORKER_COUNT", "ZNYX_RUN_DIR"):
        monkeypatch.delenv(name, raising=False)
    assert workers.worker_index() == 0
    assert workers.worker_count() == 1
    assert workers.is_leader() is True
    assert workers.run_dir() is None


def test_identity_comes_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("ZNYX_WORKER_INDEX", "2")
    monkeypatch.setenv("ZNYX_WORKER_COUNT", "4")
    monkeypatch.setenv("ZNYX_RUN_DIR", str(tmp_path))
    assert workers.worker_index() == 2
    assert workers.worker_count() == 4
    assert workers.is_leader() is False
    assert workers.run_dir() == tmp_path


@pytest.mark.parametrize("raw,index,count", [
    ("junk", 0, 1), ("", 0, 1), ("-3", 0, 1), ("0", 0, 1), ("2.5", 0, 1),
])
def test_bad_values_fall_back(monkeypatch, raw, index, count):
    monkeypatch.setenv("ZNYX_WORKER_INDEX", raw)
    monkeypatch.setenv("ZNYX_WORKER_COUNT", raw)
    assert workers.worker_index() == index
    assert workers.worker_count() == count
    assert workers.is_leader() is True


def test_only_index_zero_leads(monkeypatch):
    monkeypatch.setenv("ZNYX_WORKER_INDEX", "1")
    assert workers.is_leader() is False
    monkeypatch.setenv("ZNYX_WORKER_INDEX", "0")
    assert workers.is_leader() is True


# ---------------------------------------------------------------------------
# RecentDecisions
# ---------------------------------------------------------------------------

def test_recent_decisions_keep_metadata_only():
    recent = RecentDecisions()
    recent.record({"timestamp": "2026-01-01T00:00:00Z", "context": "input", "decision": "BLOCK",
                   "risk_score": 90, "rule_hit_ids": ["secrets.x"], "latency_ms": 3.2,
                   "app_id": "a", "policy_version": "v1",
                   "text": "my private words", "sanitized_text": "my [REDACTED] words",
                   "original_text": "further prose"})
    [item] = recent.items()
    assert item["decision"] == "BLOCK" and item["rules"] == ["secrets.x"]
    assert set(item) == {"time", "context", "decision", "risk_score", "rules", "latency_ms",
                         "app_id", "policy_version", "worker"}
    blob = json.dumps(item)
    assert "private" not in blob and "REDACTED" not in blob and "prose" not in blob


def test_recent_decisions_cap_at_maxlen_and_keep_the_newest():
    recent = RecentDecisions(maxlen=3)
    for i in range(5):
        recent.record({"timestamp": str(i), "decision": "ALLOW"})
    assert [d["time"] for d in recent.items()] == ["2", "3", "4"]


def test_recent_decisions_limit_rules_to_ten_and_tolerate_missing_fields():
    recent = RecentDecisions()
    recent.record({"rule_hit_ids": [f"r{i}" for i in range(25)]})
    recent.record({})
    first, second = recent.items()
    assert len(first["rules"]) == 10
    assert second["rules"] == [] and second["decision"] is None


def test_recent_decisions_items_is_a_copy():
    recent = RecentDecisions()
    recent.record({"decision": "ALLOW"})
    recent.items().clear()
    assert len(recent.items()) == 1


def test_recent_decisions_record_the_worker_index(monkeypatch):
    monkeypatch.setenv("ZNYX_WORKER_INDEX", "3")
    recent = RecentDecisions()
    recent.record({"decision": "ALLOW"})
    assert recent.items()[0]["worker"] == 3


# ---------------------------------------------------------------------------
# WorkerReporter
# ---------------------------------------------------------------------------

def test_write_now_writes_liveness_and_metrics_files(tmp_path, collector):
    collector.counter_inc("guardrails_evaluations_total", 5, labels={"context": "input", "decision": "ALLOW"})
    recent = RecentDecisions()
    recent.record({"timestamp": "t", "decision": "ALLOW"})
    reporter = WorkerReporter(collector, recent, tmp_path, index=1, version="1.2.3")
    reporter.write_now()

    live = json.loads((tmp_path / "workers" / "w1.json").read_text())
    assert live["index"] == 1
    assert live["pid"] == os.getpid()
    assert live["requests"] == 5
    assert live["version"] == "1.2.3"
    assert live["beat_at"] >= live["started_at"]

    data = json.loads((tmp_path / "metrics" / "w1.json").read_text())
    assert _evaluations(data["metrics"]) == 5
    assert [d["decision"] for d in data["recent"]] == ["ALLOW"]
    assert not list(tmp_path.rglob("*.tmp")), "atomic replace leaves no temp files"


def test_write_now_uses_the_environment_index_by_default(tmp_path, collector, monkeypatch):
    monkeypatch.setenv("ZNYX_WORKER_INDEX", "2")
    WorkerReporter(collector, RecentDecisions(), tmp_path).write_now()
    assert (tmp_path / "workers" / "w2.json").is_file()


def test_a_restarted_worker_continues_its_predecessors_counters(tmp_path, collector):
    labels = {"context": "input", "decision": "ALLOW"}
    collector.counter_inc("guardrails_evaluations_total", 4, labels=labels)
    collector.histogram_observe("guardrails_evaluation_duration_seconds", 0.02, labels={"context": "input"})
    collector.gauge_set("guardrails_active_connections", 7)
    WorkerReporter(collector, RecentDecisions(), tmp_path, index=0).write_now()

    collector.reset()                                   # the old process is gone
    successor = WorkerReporter(collector, RecentDecisions(), tmp_path, index=0)
    successor.restore()
    collector.counter_inc("guardrails_evaluations_total", 2, labels=labels)
    snap = collector.snapshot()
    assert _evaluations(snap) == 6
    assert snap["histograms"]["guardrails_evaluation_duration_seconds"][0][1]["count"] == 1
    assert not snap["gauges"].get("guardrails_active_connections"), "gauges describe the present"


def test_restore_ignores_a_missing_or_corrupt_snapshot(tmp_path, collector):
    reporter = WorkerReporter(collector, RecentDecisions(), tmp_path, index=0)
    reporter.restore()                                  # no file
    (tmp_path / "metrics").mkdir()
    (tmp_path / "metrics" / "w0.json").write_text("{broken")
    reporter.restore()
    (tmp_path / "metrics" / "w0.json").write_text(json.dumps({"metrics": {"counters": "oops"}}))
    reporter.restore()
    assert _evaluations(collector.snapshot()) == 0


def test_restore_only_reads_its_own_index(tmp_path, collector):
    collector.counter_inc("guardrails_evaluations_total", 9, labels={"context": "input", "decision": "ALLOW"})
    WorkerReporter(collector, RecentDecisions(), tmp_path, index=1).write_now()
    collector.reset()
    WorkerReporter(collector, RecentDecisions(), tmp_path, index=0).restore()
    assert _evaluations(collector.snapshot()) == 0


def test_reporter_start_restores_then_reports_and_stop_writes_a_last_record(tmp_path, collector):
    collector.counter_inc("guardrails_evaluations_total", 1, labels={"context": "input", "decision": "ALLOW"})

    async def scenario():
        reporter = WorkerReporter(collector, RecentDecisions(), tmp_path, index=0)
        await reporter.start()
        await asyncio.sleep(0.2)
        assert reporter.liveness_path.is_file()
        collector.counter_inc("guardrails_evaluations_total", 1,
                              labels={"context": "input", "decision": "ALLOW"})
        await reporter.stop()
        return json.loads(reporter.liveness_path.read_text())

    assert asyncio.run(scenario())["requests"] == 2


# ---------------------------------------------------------------------------
# instance_snapshot
# ---------------------------------------------------------------------------

def test_instance_snapshot_merges_the_other_workers_files(tmp_path, collector, monkeypatch):
    monkeypatch.setenv("ZNYX_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("ZNYX_WORKER_COUNT", "2")
    collector.counter_inc("guardrails_evaluations_total", 3, labels={"context": "input", "decision": "ALLOW"})
    recent = RecentDecisions()
    recent.record({"timestamp": "2026-01-01T00:00:02Z", "decision": "ALLOW"})
    reporter = WorkerReporter(collector, recent, tmp_path, index=0)

    other = _counter_snapshot("guardrails_evaluations_total",
                              [({"context": "input", "decision": "ALLOW"}, 4),
                               ({"context": "output", "decision": "BLOCK"}, 1)])
    (tmp_path / "metrics").mkdir()
    (tmp_path / "metrics" / "w1.json").write_text(json.dumps({
        "metrics": other,
        "recent": [{"time": "2026-01-01T00:00:01Z", "decision": "BLOCK"},
                   {"time": "2026-01-01T00:00:03Z", "decision": "ALLOW"}]}))

    snap = instance_snapshot(collector, recent, reporter)
    assert snap["workers"] == 2
    totals = {tuple(map(tuple, k)): v for k, v in snap["metrics"]["counters"]["guardrails_evaluations_total"]}
    assert totals[(("context", "input"), ("decision", "ALLOW"))] == 7
    assert totals[(("context", "output"), ("decision", "BLOCK"))] == 1
    assert [d["time"] for d in snap["recent"]] == [
        "2026-01-01T00:00:01Z", "2026-01-01T00:00:02Z", "2026-01-01T00:00:03Z"]


def test_instance_snapshot_uses_this_workers_current_numbers(tmp_path, collector, monkeypatch):
    """The answering worker's file may be two seconds old: it is rewritten first."""
    monkeypatch.setenv("ZNYX_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("ZNYX_WORKER_COUNT", "2")
    reporter = WorkerReporter(collector, RecentDecisions(), tmp_path, index=0)
    reporter.write_now()                                # a stale file with zero evaluations
    collector.counter_inc("guardrails_evaluations_total", 5, labels={"context": "input", "decision": "ALLOW"})
    assert _evaluations(instance_snapshot(collector, RecentDecisions(), reporter)["metrics"]) == 5


def test_instance_snapshot_skips_unreadable_files(tmp_path, collector, monkeypatch):
    monkeypatch.setenv("ZNYX_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("ZNYX_WORKER_COUNT", "3")
    reporter = WorkerReporter(collector, RecentDecisions(), tmp_path, index=0)
    reporter.write_now()
    (tmp_path / "metrics" / "w1.json").write_text("not json")
    (tmp_path / "metrics" / "w2.json").write_text(json.dumps({"metrics": "nope"}))
    assert instance_snapshot(collector, RecentDecisions(), reporter)["workers"] == 1


def test_instance_snapshot_of_a_single_worker_is_just_this_process(tmp_path, collector, monkeypatch):
    monkeypatch.setenv("ZNYX_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("ZNYX_WORKER_COUNT", "1")
    collector.counter_inc("guardrails_evaluations_total", 2, labels={"context": "input", "decision": "ALLOW"})
    snap = instance_snapshot(collector, RecentDecisions(), WorkerReporter(collector, RecentDecisions(), tmp_path))
    assert snap["workers"] == 1 and _evaluations(snap["metrics"]) == 2
    assert not (tmp_path / "metrics").exists(), "nothing is written for a single worker"


def test_instance_snapshot_without_a_run_dir_is_just_this_process(collector, monkeypatch):
    monkeypatch.delenv("ZNYX_RUN_DIR", raising=False)
    monkeypatch.setenv("ZNYX_WORKER_COUNT", "2")
    from znyx_runtime.gateway.usage import recent_calls
    assert instance_snapshot(collector, None, None) == {
        "metrics": collector.snapshot(), "recent": [], "gateway_recent": recent_calls(), "workers": 1}


def test_worker_records_lists_liveness_files(tmp_path, collector, monkeypatch):
    monkeypatch.setenv("ZNYX_RUN_DIR", str(tmp_path))
    WorkerReporter(collector, RecentDecisions(), tmp_path, index=0).write_now()
    WorkerReporter(collector, RecentDecisions(), tmp_path, index=1).write_now()
    records = workers.worker_records()
    assert [r["index"] for r in records] == [0, 1]
    assert all(r["age_seconds"] >= 0 for r in records)


def test_worker_records_without_a_run_dir_is_empty(monkeypatch):
    monkeypatch.delenv("ZNYX_RUN_DIR", raising=False)
    assert workers.worker_records() == []


# ---------------------------------------------------------------------------
# Metrics snapshots
# ---------------------------------------------------------------------------

def _fill(collector, n):
    collector.counter_inc("guardrails_evaluations_total", n, labels={"context": "input", "decision": "ALLOW"})
    collector.gauge_set("guardrails_active_connections", n)
    for _ in range(n):
        collector.histogram_observe("guardrails_evaluation_duration_seconds", 0.02,
                                    labels={"context": "input"})


def test_merge_snapshots_sums_counters_gauges_and_histograms(collector):
    _fill(collector, 2)
    first = collector.snapshot()
    collector.reset()
    _fill(collector, 3)
    second = collector.snapshot()

    merged = merge_snapshots([first, second])
    assert _evaluations(merged) == 5
    assert merged["gauges"]["guardrails_active_connections"][0][1] == 5
    [(_, hist)] = merged["histograms"]["guardrails_evaluation_duration_seconds"]
    assert hist["count"] == 5
    assert hist["sum"] == pytest.approx(0.1)
    assert dict((b, c) for b, c in hist["buckets"])[0.025] == 5
    assert "guardrails_evaluations_total" in merged["help"]


def test_merge_snapshots_keeps_distinct_label_sets_apart():
    a = _counter_snapshot("m", [({"decision": "ALLOW"}, 1)])
    b = _counter_snapshot("m", [({"decision": "BLOCK"}, 2)])
    merged = merge_snapshots([a, b])
    assert sorted(v for _, v in merged["counters"]["m"]) == [1, 2]


def test_merge_snapshots_of_nothing_is_empty():
    assert merge_snapshots([]) == {"counters": {}, "gauges": {}, "histograms": {}, "help": {}}


def test_format_prometheus_of_a_snapshot_equals_the_live_output(collector):
    _fill(collector, 4)
    collector.counter_inc("guardrails_detector_hits_total", 2,
                          labels={"detector_name": "pii.email", "action": "REDACT"})
    assert collector.format_prometheus(collector.snapshot()) == collector.format_prometheus()
    assert collector.format_prometheus(merge_snapshots([collector.snapshot()])).count("\n") > 5


def test_format_prometheus_of_a_merged_snapshot_reports_the_sum(collector):
    _fill(collector, 2)
    snap = collector.snapshot()
    text = collector.format_prometheus(merge_snapshots([snap, snap]))
    assert 'guardrails_evaluations_total{context="input",decision="ALLOW"} 4' in text
    assert 'guardrails_evaluation_duration_seconds_count{context="input"} 4' in text
    assert 'guardrails_evaluation_duration_seconds_bucket{context="input",le="+Inf"} 4' in text


def test_restore_adds_counters_and_histograms_but_not_gauges(collector):
    _fill(collector, 2)
    snap = collector.snapshot()
    collector.restore(snap)                              # onto its own totals: they add
    after = collector.snapshot()
    assert _evaluations(after) == 4
    [(_, hist)] = after["histograms"]["guardrails_evaluation_duration_seconds"]
    assert hist["count"] == 4 and hist["sum"] == pytest.approx(0.08)
    assert after["gauges"]["guardrails_active_connections"][0][1] == 2


def test_restore_into_an_empty_collector_recreates_series(collector):
    _fill(collector, 3)
    snap = json.loads(json.dumps(collector.snapshot()))  # as read back from a file
    collector.reset()
    collector.restore(snap)
    after = collector.snapshot()
    assert _evaluations(after) == 3
    assert not after["gauges"].get("guardrails_active_connections")


# ---------------------------------------------------------------------------
# summarise()
# ---------------------------------------------------------------------------

def _hist(buckets, total, count):
    return {"buckets": buckets, "sum": total, "count": count}


def test_summarise_sums_decisions_contexts_and_ranks_rules():
    snapshot = {
        "counters": {
            "guardrails_evaluations_total": [
                [[["context", "input"], ["decision", "ALLOW"]], 3],
                [[["context", "input"], ["decision", "BLOCK"]], 1],
                [[["context", "output"], ["decision", "ALLOW"]], 2],
            ],
            "guardrails_detector_hits_total": [
                [[["detector_name", "pii.email"], ["action", "REDACT"]], 2],
                [[["detector_name", "secrets.pat"], ["action", "BLOCK"]], 7],
                [[["detector_name", "pii.email"], ["action", "BLOCK"]], 3],
            ],
        },
        "histograms": {},
    }
    out = dashboard_routes.summarise(snapshot)
    assert out["evaluations"] == 6
    assert out["decisions"] == {"ALLOW": 5, "BLOCK": 1}
    assert out["contexts"] == {"input": 4, "output": 2}
    assert out["top_rules"] == [{"rule": "secrets.pat", "hits": 7}, {"rule": "pii.email", "hits": 5}]
    assert out["latency_ms"] == {"mean": None, "p50": None, "p95": None, "p99": None}


def test_summarise_percentiles_come_from_the_merged_buckets():
    snapshot = {"counters": {}, "histograms": {"guardrails_evaluation_duration_seconds": [
        [[["context", "input"]], _hist([[0.01, 30], [0.1, 20]], 1.0, 50)],
        [[["context", "output"]], _hist([[0.01, 20], [0.1, 25], [1.0, 5]], 3.0, 50)],
    ]}}
    lat = dashboard_routes.summarise(snapshot)["latency_ms"]
    # merged buckets: 0.01 -> 50, 0.1 -> 45, 1.0 -> 5 (100 observations)
    assert lat["p50"] == 10.0
    assert lat["p95"] == 100.0
    assert lat["p99"] == 1000.0
    assert lat["mean"] == 40.0


def test_summarise_percentile_beyond_the_largest_bucket_is_unknown():
    snapshot = {"counters": {}, "histograms": {"guardrails_evaluation_duration_seconds": [
        [[], _hist([[0.01, 1]], 99.0, 10)]]}}
    lat = dashboard_routes.summarise(snapshot)["latency_ms"]
    assert lat["p50"] is None and lat["p99"] is None
    assert lat["mean"] == 9900.0


def test_summarise_of_an_empty_snapshot():
    out = dashboard_routes.summarise({"counters": {}, "histograms": {}})
    assert out["evaluations"] == 0 and out["decisions"] == {} and out["top_rules"] == []


def test_summarise_limits_top_rules_to_fifteen():
    snapshot = {"counters": {"guardrails_detector_hits_total": [
        [[["detector_name", f"r{i}"]], i + 1] for i in range(30)]}, "histograms": {}}
    rules = dashboard_routes.summarise(snapshot)["top_rules"]
    assert len(rules) == 15 and rules[0]["rule"] == "r29"


# ---------------------------------------------------------------------------
# Dashboard routes on the real app
# ---------------------------------------------------------------------------

KEY = "dash-test-key-123"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """The runtime app under TestClient (as in test_runtime_api), in local mode."""
    tmp = tmp_path_factory.mktemp("dash")
    policy_path = tmp / "policies.yaml"
    policy_path.write_text(MINIMAL_POLICY_YAML)
    env = {
        "ZNYX_MODE": "local", "ZNYX_TELEMETRY": "false", "RUNTIME_REQUIRE_AUTH": "true",
        "RUNTIME_API_KEY": KEY, "ZNYX_POLICY_PATH": str(policy_path),
        "ZNYX_AUDIT_SPOOL_PATH": str(tmp / "egress-audit.spool"),
        "ZNYX_JUDGE_AUDIT_SPOOL_PATH": str(tmp / "judge-audit.spool"),
        "ZNYX_DASHBOARD": "true",
    }
    saved = {key: os.environ.get(key) for key in env}
    saved.update({k: os.environ.get(k) for k in ("ZNYX_RUN_DIR", "ZNYX_WORKER_COUNT", "ZNYX_WORKER_INDEX")})
    os.environ.update(env)
    for name in ("ZNYX_RUN_DIR", "ZNYX_WORKER_COUNT", "ZNYX_WORKER_INDEX"):
        os.environ.pop(name, None)

    import znyx_runtime.install_state as install_state
    saved_state = (install_state.STATE_DIR, install_state.STATE_FILE)
    install_state.STATE_DIR = tmp / ".znyx"
    install_state.STATE_FILE = install_state.STATE_DIR / "state.json"
    try:
        from znyx_runtime.main import app
        with TestClient(app) as test_client:
            yield test_client
    finally:
        install_state.STATE_DIR, install_state.STATE_FILE = saved_state
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


AUTH = {"X-API-Key": KEY}


def test_dashboard_page_is_html_with_a_strict_csp(client):
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    csp = resp.headers["content-security-policy"]
    assert "script-src 'self'" in csp
    assert "default-src 'none'" in csp
    assert "frame-ancestors 'none'" in csp
    assert "unsafe-inline" not in csp
    assert "<html" in resp.text.lower()


def test_dashboard_page_needs_no_key_and_loads_nothing_from_elsewhere(client):
    html = client.get("/dashboard").text
    assert "http://" not in html and "https://" not in html
    assert "app.js" in html and "app.css" in html


def test_dashboard_assets_are_served_with_their_types(client):
    js = client.get("/dashboard/app.js")
    css = client.get("/dashboard/app.css")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
    assert css.status_code == 200 and css.headers["content-type"].startswith("text/css")
    assert "/v1/local/summary" in js.text
    assert js.headers["content-security-policy"] == client.get("/dashboard").headers["content-security-policy"]


@pytest.mark.parametrize("name", ["other", "index.html", "..%2Fmain.py", "app.js.map"])
def test_unknown_dashboard_paths_are_404(client, name):
    assert client.get(f"/dashboard/{name}").status_code == 404


def test_dashboard_can_be_switched_off(client, monkeypatch):
    monkeypatch.setenv("ZNYX_DASHBOARD", "false")
    for path in ("/dashboard", "/dashboard/app.js", "/dashboard/app.css"):
        assert client.get(path).status_code == 404, path
    assert client.get("/v1/local/summary", headers=AUTH).status_code == 404


@pytest.mark.parametrize("value", ["0", "no", "off", "FALSE"])
def test_dashboard_off_spellings(client, monkeypatch, value):
    monkeypatch.setenv("ZNYX_DASHBOARD", value)
    assert client.get("/dashboard").status_code == 404


def test_summary_requires_the_api_key(client):
    assert client.get("/v1/local/summary").status_code == 401
    assert client.get("/v1/local/summary", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/v1/local/summary", headers=AUTH).status_code == 200
    assert client.get("/v1/local/summary",
                      headers={"Authorization": f"Bearer {KEY}"}).status_code == 200


def test_summary_after_an_evaluation_has_counts_latency_and_no_text(client):
    marker = "zzq-unique-marker-text"
    body = {"request_id": "dash-1", "tenant_id": "default", "app_id": "dash-app",
            "text": f"hello {marker}"}
    assert client.post("/v1/evaluate/input", json=body, headers=AUTH).status_code == 200

    resp = client.get("/v1/local/summary", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["evaluations"] >= 1
    assert isinstance(data["decisions"], dict) and data["decisions"]
    assert sum(data["decisions"].values()) == data["evaluations"]
    assert set(data["latency_ms"]) == {"mean", "p50", "p95", "p99"}
    assert data["mode"] == "local"
    assert data["airgapped"] is False
    assert data["workers"]["count"] == 1
    assert data["model_server"] is None
    mine = [r for r in data["recent"] if r["app_id"] == "dash-app"]
    assert mine and mine[0]["decision"] == "ALLOW" and mine[0]["context"] == "input"
    for entry in data["recent"]:
        assert not {"text", "sanitized_text", "original_text"} & set(entry)
    assert marker not in resp.text


def test_summary_lists_the_most_recent_decision_first(client):
    for i in range(2):
        client.post("/v1/evaluate/input", headers=AUTH, json={
            "request_id": f"o{i}", "tenant_id": "default", "app_id": f"order-{i}", "text": "hi"})
    recent = client.get("/v1/local/summary", headers=AUTH).json()["recent"]
    apps = [r["app_id"] for r in recent if str(r["app_id"]).startswith("order-")]
    assert apps[:2] == ["order-1", "order-0"]


def test_summary_reports_a_blocked_decision(client, fake_pat):
    client.post("/v1/evaluate/input", headers=AUTH, json={
        "request_id": "b1", "tenant_id": "default", "app_id": "default", "text": "tok " + fake_pat})
    data = client.get("/v1/local/summary", headers=AUTH).json()
    assert data["decisions"].get("BLOCK", 0) >= 1
    assert any(r["rule"].startswith("secrets.") for r in data["top_rules"])


def test_summary_reports_an_unreachable_model_server(client, monkeypatch):
    monkeypatch.setenv("ZNYX_INFERENCE_URL", "http://127.0.0.1:1")
    data = client.get("/v1/local/summary", headers=AUTH).json()
    assert data["model_server"] == {"url": "http://127.0.0.1:1", "reachable": False, "models": []}


def test_summary_reports_platform_version_and_airgap(client, monkeypatch):
    monkeypatch.setenv("ZNYX_PLATFORM_VERSION", "7.7.7")
    monkeypatch.setenv("ZNYX_AIRGAPPED", "true")
    data = client.get("/v1/local/summary", headers=AUTH).json()
    assert data["version"] == "7.7.7"
    assert data["airgapped"] is True
    assert client.get("/").json()["platform_version"] == "7.7.7"


def test_worker_header_appears_only_with_several_workers(client, monkeypatch):
    monkeypatch.delenv("ZNYX_WORKER_COUNT", raising=False)
    assert "x-znyx-worker" not in client.get("/healthz").headers
    monkeypatch.setenv("ZNYX_WORKER_COUNT", "2")
    monkeypatch.setenv("ZNYX_WORKER_INDEX", "1")
    assert client.get("/healthz").headers["x-znyx-worker"] == "1"


def test_metrics_endpoint_still_renders_for_one_worker(client, monkeypatch):
    monkeypatch.delenv("ZNYX_WORKER_COUNT", raising=False)
    client.post("/v1/evaluate/input", headers=AUTH, json={
        "request_id": "m1", "tenant_id": "default", "app_id": "default", "text": "hi"})
    text = client.get("/metrics").text
    assert "# TYPE guardrails_evaluations_total counter" in text
    assert 'guardrails_evaluations_total{context="input",decision="ALLOW"}' in text


# ---------------------------------------------------------------------------
# _on_evaluation
# ---------------------------------------------------------------------------

def test_on_evaluation_records_then_emits(monkeypatch):
    from znyx_runtime import main as runtime

    class Sink:
        def __init__(self):
            self.events = []

        def emit(self, event):
            self.events.append(event)

        def record(self, event):
            self.events.append(("recent", event["decision"]))

    sink = Sink()
    monkeypatch.setattr(runtime, "recent_decisions", sink)
    monkeypatch.setattr(runtime, "telemetry", sink)
    runtime._on_evaluation({"decision": "ALLOW"})
    assert sink.events == [("recent", "ALLOW"), {"decision": "ALLOW"}]


def test_a_failing_dashboard_recorder_never_blocks_telemetry(monkeypatch):
    from znyx_runtime import main as runtime

    class Boom:
        def record(self, event):
            raise RuntimeError("dashboard broke")

    emitted = []

    class Telemetry:
        def emit(self, event):
            emitted.append(event)

    monkeypatch.setattr(runtime, "recent_decisions", Boom())
    monkeypatch.setattr(runtime, "telemetry", Telemetry())
    runtime._on_evaluation({"decision": "BLOCK"})          # must not raise
    assert emitted == [{"decision": "BLOCK"}]


def test_on_evaluation_without_telemetry_only_records(monkeypatch):
    from znyx_runtime import main as runtime

    recent = RecentDecisions()
    monkeypatch.setattr(runtime, "recent_decisions", recent)
    monkeypatch.setattr(runtime, "telemetry", None)
    runtime._on_evaluation({"decision": "ALLOW"})
    assert len(recent.items()) == 1


# ---------------------------------------------------------------------------
# Telemetry: only the leader ships
# ---------------------------------------------------------------------------

def test_a_non_shipping_emitter_still_spools_events(tmp_path):
    from znyx_runtime.telemetry import TelemetryEmitter

    spool = tmp_path / "telemetry.spool"
    emitter = TelemetryEmitter("http://cp.invalid", "tok", enabled=True, spool_path=str(spool), ship=False)
    asyncio.run(emitter.start())
    assert emitter._shipper is None
    emitter.emit({"event_id": "e1"})
    emitter.emit({"event_id": "e2"})
    assert [json.loads(line)["event_id"] for line in spool.read_text().splitlines()] == ["e1", "e2"]
    asyncio.run(emitter.stop())                             # nothing to stop


def test_a_shipping_emitter_delivers_the_spool(tmp_path, monkeypatch):
    import znyx_runtime.telemetry as telemetry

    delivered = []

    async def fake_post(url, records):
        delivered.append((url, [r["event_id"] for r in records]))
        return True

    monkeypatch.setattr(telemetry, "http_post", lambda token: fake_post)
    spool = tmp_path / "telemetry.spool"
    emitter = telemetry.TelemetryEmitter("http://cp.invalid/", "tok", enabled=True,
                                         spool_path=str(spool), ship=True)

    async def scenario():
        await emitter.start()
        assert emitter._shipper is not None
        emitter.emit({"event_id": "e1"})
        await emitter.stop()                                # final attempt delivers

    asyncio.run(scenario())
    assert delivered == [("http://cp.invalid/v1/telemetry/events", ["e1"])]


def test_a_disabled_emitter_does_nothing(tmp_path):
    from znyx_runtime.telemetry import TelemetryEmitter

    spool = tmp_path / "telemetry.spool"
    emitter = TelemetryEmitter("http://cp.invalid", "tok", enabled=False, spool_path=str(spool))
    asyncio.run(emitter.start())
    emitter.emit({"event_id": "e1"})
    assert emitter._shipper is None and not spool.exists()


# ---------------------------------------------------------------------------
# Runtime report
# ---------------------------------------------------------------------------

@pytest.fixture()
def captured_posts(monkeypatch):
    """Route the reporter's httpx client through a MockTransport; returns the posts."""
    posts = []

    def handler(request: httpx.Request) -> httpx.Response:
        posts.append({"url": str(request.url), "json": json.loads(request.content),
                      "key": request.headers.get("x-api-key")})
        return httpx.Response(200, json={})

    real = httpx.AsyncClient

    class Patched(real):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", Patched)
    return posts


def test_runtime_report_carries_platform_version_and_worker_count(monkeypatch, captured_posts):
    from znyx_runtime.runtime_report import RuntimeReporter

    monkeypatch.setenv("ZNYX_PLATFORM_VERSION", "3.4.5")
    monkeypatch.setenv("ZNYX_WORKER_COUNT", "4")
    reporter = RuntimeReporter("https://cp.example.com/", "rt_token")
    assert asyncio.run(reporter.report({"bundle_id": "b1", "policy_hash": "h1"})) is True
    [post] = captured_posts
    assert post["url"] == "https://cp.example.com/v1/inference/heartbeat"
    assert post["key"] == "rt_token"
    assert post["json"]["platform_version"] == "3.4.5"
    assert post["json"]["worker_count"] == 4
    assert post["json"]["bundle_id"] == "b1" and post["json"]["reporter"] == "runtime"


def test_runtime_report_without_znyx_serve_has_no_platform_version(monkeypatch, captured_posts):
    from znyx_runtime.runtime_report import RuntimeReporter

    monkeypatch.delenv("ZNYX_PLATFORM_VERSION", raising=False)
    monkeypatch.delenv("ZNYX_WORKER_COUNT", raising=False)
    asyncio.run(RuntimeReporter("https://cp", "tok").report({}))
    [post] = captured_posts
    assert post["json"]["platform_version"] is None
    assert post["json"]["worker_count"] == 1


def test_runtime_report_is_silent_without_a_control_plane(captured_posts):
    from znyx_runtime.runtime_report import RuntimeReporter

    assert asyncio.run(RuntimeReporter("", "").report({})) is False
    assert captured_posts == []


# ---------------------------------------------------------------------------
# .env discovery
# ---------------------------------------------------------------------------

_PROBE = "import os, znyx_runtime.main; print(os.environ.get('ZNYX_TEST_DOTENV_PROBE', '<unset>'))"


def _import_main_in(cwd, **extra):
    env = {k: v for k, v in os.environ.items() if k != "ZNYX_TEST_DOTENV_PROBE"}
    env.pop("ZNYX_SKIP_DOTENV", None)
    env["HOME"] = str(cwd)
    env.update(extra)
    out = subprocess.run([sys.executable, "-c", _PROBE], cwd=cwd, env=env, capture_output=True,
                         text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip().splitlines()[-1]


def test_dotenv_is_read_from_the_working_directory(tmp_path):
    (tmp_path / ".env").write_text("ZNYX_TEST_DOTENV_PROBE=from-cwd\n")
    assert _import_main_in(tmp_path) == "from-cwd"


def test_dotenv_is_skipped_under_the_supervisor(tmp_path):
    (tmp_path / ".env").write_text("ZNYX_TEST_DOTENV_PROBE=from-cwd\n")
    assert _import_main_in(tmp_path, ZNYX_SKIP_DOTENV="1") == "<unset>"


def test_dotenv_does_not_override_the_environment(tmp_path):
    (tmp_path / ".env").write_text("ZNYX_TEST_DOTENV_PROBE=from-file\n")
    assert _import_main_in(tmp_path, ZNYX_TEST_DOTENV_PROBE="from-env") == "from-env"
