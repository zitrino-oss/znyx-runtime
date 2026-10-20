"""Durable delivery of runtime spools to the control plane (znyx_runtime.shipper)."""
import asyncio
import json

from znyx_runtime.shipper import SpoolShipper, append_record


class FakeControlPlane:
    def __init__(self):
        self.received = []
        self.up = True

    async def post(self, url, records):
        if not self.up:
            return False
        self.received.extend(records)
        return True


def _shipper(tmp_path, cp, **kw):
    return SpoolShipper(str(tmp_path / "events.spool"), "http://cp/v1/x", cp.post, name="test", **kw)


def _run(coro):
    return asyncio.run(coro)


def test_delivers_new_records_once(tmp_path):
    cp = FakeControlPlane()
    sh = _shipper(tmp_path, cp)
    for i in range(3):
        append_record(sh.path, {"event_id": i})
    assert _run(sh.ship_once()) == 3
    assert _run(sh.ship_once()) == 0
    append_record(sh.path, {"event_id": 3})
    assert _run(sh.ship_once()) == 1
    assert [r["event_id"] for r in cp.received] == [0, 1, 2, 3]


def test_an_outage_delays_records_instead_of_losing_them(tmp_path):
    cp = FakeControlPlane()
    sh = _shipper(tmp_path, cp)
    cp.up = False
    for i in range(5):
        append_record(sh.path, {"event_id": i})
    assert _run(sh.ship_once()) == 0
    assert sh.failures == 1
    cp.up = True
    assert _run(sh.ship_once()) == 5
    assert [r["event_id"] for r in cp.received] == list(range(5))


def test_delivery_is_at_least_once_across_a_crash(tmp_path):
    """A crash after delivery but before the offset is written re-sends the batch;
    the control plane dedupes on event id."""
    cp = FakeControlPlane()
    sh = _shipper(tmp_path, cp)
    append_record(sh.path, {"event_id": "a"})
    _run(sh.ship_once())
    sh.offset_path.unlink()                       # the offset write never happened
    restarted = _shipper(tmp_path, cp)
    _run(restarted.ship_once())
    assert [r["event_id"] for r in cp.received] == ["a", "a"]


def test_a_half_written_line_waits_for_its_newline(tmp_path):
    cp = FakeControlPlane()
    sh = _shipper(tmp_path, cp)
    append_record(sh.path, {"event_id": 1})
    with open(sh.path, "a") as fh:
        fh.write('{"event_id": 2')                # a write still in progress
    assert _run(sh.ship_once()) == 1
    with open(sh.path, "a") as fh:
        fh.write("}\n")
    assert _run(sh.ship_once()) == 1
    assert [r["event_id"] for r in cp.received] == [1, 2]


def test_unparseable_lines_are_skipped(tmp_path):
    cp = FakeControlPlane()
    sh = _shipper(tmp_path, cp)
    append_record(sh.path, {"event_id": 1})
    with open(sh.path, "a") as fh:
        fh.write("not json\n")
    append_record(sh.path, {"event_id": 2})
    _run(sh.ship_once())
    assert [r["event_id"] for r in cp.received] == [1, 2]


def test_batches_respect_the_batch_size(tmp_path):
    batches = []

    async def post(url, records):
        batches.append(len(records))
        return True

    sh = SpoolShipper(str(tmp_path / "s.spool"), "http://cp", post, batch_size=4)
    for i in range(10):
        append_record(sh.path, {"event_id": i})
    _run(sh.ship_once())
    assert batches == [4, 4, 2]


def test_fully_delivered_spool_is_rotated(tmp_path):
    cp = FakeControlPlane()
    sh = _shipper(tmp_path, cp, rotate_bytes=200)
    for i in range(20):
        append_record(sh.path, {"event_id": i, "pad": "x" * 20})
    _run(sh.ship_once())
    assert not sh.path.exists() or sh.path.stat().st_size == 0
    assert sh._read_offset() == 0
    append_record(sh.path, {"event_id": 99})
    _run(sh.ship_once())
    assert cp.received[-1]["event_id"] == 99
    assert len(cp.received) == 21


def test_disk_limit_drops_only_the_oldest_and_counts_them(tmp_path):
    cp = FakeControlPlane()
    cp.up = False
    sh = _shipper(tmp_path, cp, max_bytes=300)
    for i in range(50):
        append_record(sh.path, {"event_id": i})
    _run(sh.ship_once())
    assert sh.dropped > 0
    cp.up = True
    _run(sh.ship_once())
    ids = [r["event_id"] for r in cp.received]
    assert ids == sorted(ids) and ids[-1] == 49
    assert len(ids) + sh.dropped == 50


def test_status_reports_pending_bytes(tmp_path):
    cp = FakeControlPlane()
    cp.up = False
    sh = _shipper(tmp_path, cp)
    append_record(sh.path, {"event_id": 1})
    assert sh.status()["pending_bytes"] > 0
    cp.up = True
    _run(sh.ship_once())
    assert sh.status()["pending_bytes"] == 0


def test_records_are_compact_json_lines(tmp_path):
    p = tmp_path / "x.spool"
    append_record(str(p), {"b": 1, "a": "z"})
    line = p.read_text().strip()
    assert json.loads(line) == {"b": 1, "a": "z"}


# -- the HTTP sender ---------------------------------------------------------

import httpx  # noqa: E402

from znyx_runtime.shipper import DROP, http_post  # noqa: E402


def _sender(status, seen=None):
    def handle(request):
        if seen is not None:
            seen.append(request)
        return httpx.Response(status, json={})
    return http_post("rt-token", transport=httpx.MockTransport(handle))


def test_sender_classifies_responses():
    assert _run(_sender(200)("http://cp/x", [{"a": 1}])) is True
    assert _run(_sender(503)("http://cp/x", [{"a": 1}])) is False     # retry later
    assert _run(_sender(429)("http://cp/x", [{"a": 1}])) is False
    assert _run(_sender(403)("http://cp/x", [{"a": 1}])) is False     # fixable config
    assert _run(_sender(422)("http://cp/x", [{"a": 1}])) == DROP      # malformed batch


def test_sender_uses_the_runtime_token_header_and_events_body():
    seen = []
    _run(_sender(200, seen)("http://cp/v1/runtime/audit/egress", [{"event_id": "e1"}]))
    assert seen[0].headers["X-API-Key"] == "rt-token"
    assert json.loads(seen[0].content) == {"events": [{"event_id": "e1"}]}


def test_a_rejected_batch_is_skipped_not_retried_forever(tmp_path):
    received = []

    async def post(url, records):
        if any(r.get("poison") for r in records):
            return DROP
        received.extend(records)
        return True

    sh = SpoolShipper(str(tmp_path / "s.spool"), "http://cp", post, batch_size=1)
    append_record(sh.path, {"event_id": 1})
    append_record(sh.path, {"event_id": 2, "poison": True})
    append_record(sh.path, {"event_id": 3})
    _run(sh.ship_once())
    assert [r["event_id"] for r in received] == [1, 3]
    assert sh.dropped == 1


# -- telemetry ---------------------------------------------------------------

def test_telemetry_is_not_lost_when_the_control_plane_is_down(tmp_path, monkeypatch):
    import znyx_runtime.telemetry as telemetry_mod
    cp = FakeControlPlane()
    cp.up = False
    monkeypatch.setattr(telemetry_mod, "http_post", lambda token: cp.post)

    async def scenario():
        emitter = telemetry_mod.TelemetryEmitter("http://cp", "rt", enabled=True,
                                                 spool_path=str(tmp_path / "telemetry.spool"))
        await emitter.start()
        for i in range(5):
            emitter.emit({"request_id": f"r{i}", "decision": "ALLOW"})
        await emitter._shipper.ship_once()          # control plane down: nothing lost
        cp.up = True
        await emitter.stop()                        # final attempt delivers the backlog

    _run(scenario())
    assert [e["request_id"] for e in cp.received] == [f"r{i}" for i in range(5)]


def test_disabled_telemetry_writes_nothing(tmp_path):
    import znyx_runtime.telemetry as telemetry_mod
    spool = tmp_path / "telemetry.spool"
    emitter = telemetry_mod.TelemetryEmitter("http://cp", "rt", enabled=False, spool_path=str(spool))
    emitter.emit({"request_id": "r"})
    assert not spool.exists()


def test_telemetry_carries_no_request_text(tmp_path):
    """Rule-hit messages can quote what matched (a URL, a snippet); only metadata
    leaves the customer's network."""
    import znyx_runtime.telemetry as telemetry_mod
    spool = tmp_path / "telemetry.spool"
    emitter = telemetry_mod.TelemetryEmitter("http://cp", "rt", enabled=True, spool_path=str(spool))
    emitter.emit({"request_id": "r1", "rule_hit_ids": ["malicious_url.data_uri"], "detector_results": [
        {"detector_name": "malicious_url", "rule_hits": [
            {"rule_id": "malicious_url.data_uri", "severity": "high",
             "message": "Data URI detected (potential XSS vector): data:text/html,<script>secret-user-text"}]}]})
    line = spool.read_text()
    assert "secret-user-text" not in line and "malicious_url.data_uri" in line
