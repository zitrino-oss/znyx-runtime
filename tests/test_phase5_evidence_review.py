"""AI gateway Phase 5 part 4: the adversarial review's runtime findings (2026-10-23), kept as
regression tests. Each test failed before its fix."""
# ruff: noqa: F811
import json
from pathlib import Path

from znyx_core.core.models import Decision, DetectorResult, EvaluationRequest
from znyx_core.utils.spool import append_chained, verify_chain


def _lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


# -- 1. A lone surrogate in a caller-supplied field makes the chained egress audit write fail ---------

class _RecordingRegistry:
    def __init__(self):
        self.calls = []

    def detect(self, name, text, config):
        self.calls.append((name, text))
        return DetectorResult(decision=Decision.BLOCK, risk_score=95)


def test_a_lone_surrogate_in_trace_id_skips_a_remote_custom_detector(tmp_path):
    """Before part 4 the egress sink wrote ``json.dumps(...)`` (ASCII escapes), so a lone
    surrogate in a caller field was written fine. ``append_chained`` writes with
    ensure_ascii=False and a strict UTF-8 encode, so the write raises, the gate reports
    ``egress_audit_unavailable`` and the remote detector is skipped: the request is no longer
    checked by it. The evaluate API takes ``trace_id`` from the JSON body (stdlib json.loads
    keeps "\\ud800")."""
    from znyx_core.engine.detector_registry import DetectorRegistry
    from znyx_core.engine.orchestrator import DetectorOrchestrator
    from znyx_runtime.audit_sink import SpoolAuditSink, make_audit_egress_sink

    sink = make_audit_egress_sink(SpoolAuditSink(str(tmp_path / "egress.spool")))
    registry = _RecordingRegistry()
    orch = DetectorOrchestrator(DetectorRegistry(), plugin_registry=registry, egress_sink=sink)
    policy = {"custom_detectors": [{"name": "wh", "config": {
        "url": "https://api.vendor.com/x", "egress_allowlist": ["api.vendor.com"]}}]}

    clean = EvaluationRequest(request_id="r", tenant_id="org1", app_id="a", text="attack", trace_id="t1")
    orch.run_detectors("attack", policy, clean, context="input")
    assert len(registry.calls) == 1                       # control: the remote detector runs

    hostile = EvaluationRequest(request_id="r", tenant_id="org1", app_id="a", text="attack",
                                trace_id="t1\ud800")
    orch.run_detectors("attack", policy, hostile, context="input")
    assert len(registry.calls) == 2, "the remote detector was skipped for a trace_id holding a lone surrogate"


def test_a_lone_surrogate_in_trace_id_stops_escalation_to_a_remote_model(tmp_path):
    from znyx_core.engine.backends import build_strategy
    from znyx_core.engine.escalation import run_with_strategy
    from znyx_runtime.audit_sink import SpoolAuditSink, make_audit_egress_sink

    sink = make_audit_egress_sink(SpoolAuditSink(str(tmp_path / "egress.spool")))
    strategy = build_strategy({
        "strategy": {"order": ["local_deterministic", "remote_api"],
                     "escalate_when": {"deterministic_score_between": [0, 100]},
                     "fallback": "fallback_to_deterministic"},
        "backends": {"remote_api": {"endpoint_url": "https://api.vendor.com/mod"}},
        "egress_allowlist": ["api.vendor.com"]}, runtime_policy={})
    det = DetectorResult(decision=Decision.ALLOW, risk_score=10)
    called = []

    def caller(backend, text, timeout):
        called.append(text)
        return DetectorResult(decision=Decision.BLOCK, risk_score=95, confidence=0.9)

    request = type("R", (), {"tenant_id": "org1", "trace_id": "t1\ud800"})()
    out = run_with_strategy("jailbreak", det, strategy, "attack", backend_caller=caller, egress_sink=sink,
                            request=request)
    assert called and out.decision == Decision.BLOCK, (
        f"escalation fell back ({out.fallback_path}): the remote model never saw the request")


def test_a_lone_surrogate_in_a_governance_record_loses_the_record(tmp_path):
    """An agent names a tool "\\ud800..." (MCP ``tools/call`` accepts any str up to 200
    chars and ``saw_tool_call`` records it): the ``mcp.tool_call`` record cannot be written,
    the refused call leaves no governance record, and only the first failure is ever logged."""
    from znyx_runtime.governance_audit import GovernanceAudit
    audit = GovernanceAudit(str(tmp_path / "governance.spool"))
    audit.record("mcp.tool_call", tool="drop_tables\ud800", outcome="blocked", reason="unknown tool")
    assert (tmp_path / "governance.spool").exists() and _lines(tmp_path / "governance.spool"), \
        "the governance record of the refused call was dropped"


# -- 2. A crash between the spool write and the state write, then a rotation, forks the chain -------------

def test_a_crash_before_the_state_write_followed_by_a_rotation_does_not_fork(tmp_path):
    """The contract: 'after a crash the next writer continues from the spool's last record, never
    forking'. ``append_chained`` reconciles a stale ``.chain`` with the spool's tail only; once the
    shipper has delivered and rotated the spool (``_rotate_if_done`` empties it under the lock and
    leaves ``.chain`` alone), the tail is gone and the stale state wins: the next record reuses the
    seq of the one already delivered, with another hash. The control plane records a fork
    (status mismatch) for an honest runtime."""
    import asyncio
    from znyx_runtime.shipper import SpoolShipper

    spool = tmp_path / "egress.spool"
    append_chained(spool, [{"n": 1}, {"n": 2}], fsync=False)
    state = spool.with_name(spool.name + ".chain")
    stale = state.read_text()
    append_chained(spool, [{"n": 3}], fsync=False)
    state.write_text(stale)                               # killed after the record, before the state

    delivered = []

    async def post(url, records):
        delivered.extend(records)
        return True
    shipper = SpoolShipper(str(spool), "http://cp/x", post, rotate_bytes=1)
    assert asyncio.run(shipper.ship_once()) == 3          # delivered, then rotated (spool now empty)

    (fourth,) = append_chained(spool, [{"n": 4}], fsync=False)
    seqs = [r["chain"]["seq"] for r in delivered] + [fourth["chain"]["seq"]]
    assert seqs == [1, 2, 3, 4] and verify_chain(delivered + [fourth]) == [], f"seqs {seqs}: a fork"


# -- 1b. End to end: an agent's refused MCP tool call leaves no governance record ------------------------

from test_phase5_mcp import KEYS as MCP_KEYS, client as mcp_client, start as mcp_start  # noqa: E402,F401


def test_an_agents_refused_tool_call_with_a_lone_surrogate_leaves_a_governance_record(mcp_client):
    import time
    path = Path(mcp_client.gateway.governance.path)
    before = len(_lines(path)) if path.exists() else 0
    session = mcp_start(mcp_client)
    # control: the same refused call without the surrogate is recorded
    from test_phase5_mcp import rpc as mcp_rpc
    mcp_rpc(mcp_client, "crm", "tools/call", {"name": "drop_all", "arguments": {}}, session=session)
    deadline = time.time() + 3
    while time.time() < deadline and not any(
            r["type"] == "mcp.tool_call" and r.get("tool") == "drop_all" for r in _lines(path)[before:]):
        time.sleep(0.05)
    assert any(r.get("tool") == "drop_all" for r in _lines(path)[before:]), "control: no record"
    before = len(_lines(path))
    body = '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"drop_all\\ud800","arguments":{}}}'
    try:   # the refusal's JSON rendering also fails on the surrogate (a 500, older than part 4)
        mcp_client.post("/v1/mcp/crm", content=body.encode(), headers={
            "Authorization": f"Bearer {MCP_KEYS['plain']}", "content-type": "application/json",
            "accept": "application/json, text/event-stream", "mcp-session-id": session})
    except UnicodeEncodeError:
        pass
    record = mcp_client.records[-1]
    assert record.tools_called and record.tools_called[0].startswith("drop_all") and record.outcome != "ok"
    deadline = time.time() + 3
    while time.time() < deadline:
        after = _lines(path)[before:] if path.exists() else []
        if any(r["type"] == "mcp.tool_call" for r in after):
            break
        time.sleep(0.05)
    calls = [r for r in after if r["type"] == "mcp.tool_call"]
    assert calls, f"no governance record for the refused call (outcome {record.outcome}, reason {record.reason})"


# -- the chain state shortcut and the rotation hand-over ----------------------------------------------

def test_the_chain_state_is_trusted_only_while_the_spool_size_matches(tmp_path):
    spool = tmp_path / "audit.spool"
    append_chained(spool, [{"n": 1}], fsync=False)
    state_path = spool.with_name(spool.name + ".chain")
    state = json.loads(state_path.read_text())
    assert state["size"] == spool.stat().st_size and state["seq"] == 1
    with open(spool, "a") as fh:                                   # another writer's record, state not saved
        fh.write(json.dumps({"n": 2, "chain": {"id": state["id"], "seq": 2, "prev": state["hash"],
                                               "hash": __import__("znyx_core.utils.spool", fromlist=["x"]).chain_hash(
                                                   state["hash"], {"n": 2})}}) + "\n")
    append_chained(spool, [{"n": 3}], fsync=False)
    records = _lines(spool)
    assert [r["chain"]["seq"] for r in records] == [1, 2, 3] and verify_chain(records) == []


def test_any_text_can_be_recorded_and_still_verifies(tmp_path):
    spool = tmp_path / "audit.spool"
    append_chained(spool, [{"trace_id": "t1\ud800", "who": "Zoë"}], fsync=False)
    raw = spool.read_bytes()
    assert raw.isascii()                                           # escaped on disk
    assert verify_chain(_lines(spool)) == []


def test_governance_write_failures_are_counted(tmp_path, caplog):
    import logging
    from znyx_runtime.governance_audit import GovernanceAudit
    blocked = tmp_path / "file"
    blocked.write_text("x")
    audit = GovernanceAudit(str(blocked / "governance.spool"))
    with caplog.at_level(logging.ERROR):
        for _ in range(5):
            audit.record("token.issued", person="p")
    assert audit.failures == 5 and sum("cannot be written" in r.message for r in caplog.records) == 1
