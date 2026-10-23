"""AI gateway Phase 5, part 4 (evidence), runtime side: hash-chained audit spools, the
governance records, and prompt fingerprints on usage events."""
# The gateway fixtures are imported from the part 3 tests, so test parameters reuse their names.
# ruff: noqa: F811
import json
import logging
import threading
import time
from pathlib import Path

import pytest

from znyx_core import compliance, usage_events
from znyx_core.utils.spool import GENESIS, append_chained, chain_hash, verify_chain
from test_phase5_identity import KEYS, _fresh, chat, client  # noqa: F401  (fixtures)


def _lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def governance_mark(client):
    path = client.gateway.governance.path
    return len(_lines(path)) if Path(path).exists() else 0


def governance_records(client, since, kinds=(), wait=3.0):
    """Records written after ``since`` lines; waits for the kinds named (written in the background)."""
    path = client.gateway.governance.path
    deadline = time.time() + wait
    while True:
        records = _lines(path)[since:] if Path(path).exists() else []
        if all(any(r["type"] == k for r in records) for k in kinds) or time.time() > deadline:
            return records
        time.sleep(0.05)


# -- the chain ------------------------------------------------------------------------------

def test_a_chained_spool_links_every_record_to_the_one_before(tmp_path):
    spool = tmp_path / "audit.spool"
    first = append_chained(spool, [{"n": 1}, {"n": 2}])
    append_chained(spool, [{"n": 3}])
    records = _lines(spool)
    chains = [r["chain"] for r in records]
    assert [c["seq"] for c in chains] == [1, 2, 3] and len({c["id"] for c in chains}) == 1
    assert chains[0]["prev"] == GENESIS and chains[1]["prev"] == chains[0]["hash"]
    assert chains[2]["hash"] == chain_hash(chains[1]["hash"], {"n": 3})
    assert first[0]["chain"] == chains[0] and verify_chain(records) == []


def test_a_changed_or_removed_record_shows(tmp_path):
    spool = tmp_path / "audit.spool"
    append_chained(spool, [{"n": i, "who": "agent"} for i in range(5)])
    records = _lines(spool)
    records[1]["who"] = "someone else"
    assert verify_chain(records) == [{"chain": records[1]["chain"]["id"], "seq": 2, "problem": "mismatch"}]
    records = _lines(spool)
    del records[2]
    assert [f["problem"] for f in verify_chain(records)] == ["gap"]
    assert verify_chain([{"n": 1}]) == [{"seq": None, "problem": "unchained"}]


def test_the_chain_survives_a_lost_or_stale_state_file(tmp_path):
    spool = tmp_path / "audit.spool"
    append_chained(spool, [{"n": 1}, {"n": 2}])
    state = spool.with_name(spool.name + ".chain")
    stale = state.read_text()
    append_chained(spool, [{"n": 3}])
    state.write_text(stale)                                      # a crash after the write, before the state
    append_chained(spool, [{"n": 4}])
    state.unlink()                                               # the state lost altogether
    append_chained(spool, [{"n": 5}])
    records = _lines(spool)
    assert [r["chain"]["seq"] for r in records] == [1, 2, 3, 4, 5] and verify_chain(records) == []


def test_writers_in_parallel_extend_one_chain(tmp_path):
    spool = tmp_path / "audit.spool"

    def write(worker):
        for i in range(25):
            append_chained(spool, [{"worker": worker, "i": i}], fsync=False)
    threads = [threading.Thread(target=write, args=(w,)) for w in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    records = _lines(spool)
    assert [r["chain"]["seq"] for r in records] == list(range(1, 101)) and verify_chain(records) == []


def test_the_egress_audit_spool_is_chained(tmp_path):
    import asyncio
    from znyx_runtime.audit_sink import EgressAuditEvent, SpoolAuditSink
    sink = SpoolAuditSink(str(tmp_path / "egress.spool"))
    event = EgressAuditEvent(occurred_at="2026-10-23T00:00:00+00:00", mode="remote_llm", detector_key="gateway",
                             destination_host="api.openai.com", org_scope="acme", event_id="e1")
    sink.record_sync(event)
    asyncio.run(sink.record(event))
    records = _lines(tmp_path / "egress.spool")
    assert [r["chain"]["seq"] for r in records] == [1, 2] and verify_chain(records) == []


# -- governance records ---------------------------------------------------------------------

def test_a_governance_record_never_fails_the_call(tmp_path, caplog):
    from znyx_runtime.governance_audit import GovernanceAudit
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    audit = GovernanceAudit(str(blocked / "governance.spool"))
    with caplog.at_level(logging.ERROR):
        audit.record("token.issued", person="p", tools=[], reason="")
        audit.record("token.issued", person="p")
    assert sum("cannot be written" in r.message for r in caplog.records) == 1     # said once
    audit = GovernanceAudit(str(tmp_path / "governance.spool"))
    audit.record("token.revoked", person="p", tools=[], reason=None, token_ref="abc")
    (record,) = _lines(tmp_path / "governance.spool")
    assert record["type"] == "token.revoked" and "tools" not in record and "reason" not in record
    assert record["chain"]["seq"] == 1 and record["event_id"] and record["occurred_at"]


# -- prompt versions --------------------------------------------------------------------------

def test_a_prompt_fingerprint_ignores_formatting_but_not_wording():
    fp = usage_events.prompt_fingerprint
    a = fp(["You are a support agent.\n\nBe brief."])
    assert a == fp(["  You are a support   agent. Be brief.  "]) == fp(["You are a support agent.", "Be brief."])
    assert a != fp(["You are a support agent. Be very brief."]) and len(a) == 16
    assert fp([]) is None and fp(["   \n"]) is None
    keyed = fp(["You are a support agent. Be brief."], key=b"k" * 32, scope="org-1")
    assert keyed != a and keyed == fp(["You are a support agent.  Be brief."], key=b"k" * 32, scope="org-1")
    assert keyed != fp(["You are a support agent. Be brief."], key=b"k" * 32, scope="org-2")


@pytest.mark.parametrize("value,expected", [
    ("support-bot@v3", "support-bot@v3"), ("billing/refunds@2026-10-01", "billing/refunds@2026-10-01"),
    ("plain-name", "plain-name"), (" padded@1 ", "padded@1"),
    ("", None), ("@v1", None), ("bad name@v1", None), ("x" * 101, None), ("a@" + "v" * 41, None),
    ("<script>@1", None),
])
def test_a_prompt_name_must_look_like_one(value, expected):
    assert usage_events.prompt_ref(value) == expected


def test_usage_events_keep_only_well_formed_prompt_fields():
    base = usage_events.build(request_id="r", tenant_id="acme", app_id="a", key_id="k", api="openai",
                              outcome="ok", status=200)
    good = usage_events.validate({**base, "prompt_hash": "0123456789abcdef", "prompt_ref": "bot@v1"})
    assert good["prompt_hash"] == "0123456789abcdef" and good["prompt_ref"] == "bot@v1"
    bad = usage_events.validate({**base, "prompt_hash": "not-a-hash"})
    assert "prompt_hash" not in bad


def test_a_gateway_call_records_which_instructions_ran(client):
    system = "You are the assistant of ACME. Answer briefly."
    body = {"model": "gpt-4o-mini", "messages": [{"role": "system", "content": system},
                                                   {"role": "user", "content": "SAY:hi"}]}
    resp = client.post("/v1/chat/completions", json=body,
                       headers={"Authorization": f"Bearer {KEYS['assistant']}", "X-Znyx-Prompt": "assistant@v2"})
    assert resp.status_code == 200
    record = client.records[-1]
    from znyx_runtime.install_state import user_id_key
    assert record.prompt_hash == usage_events.prompt_fingerprint([system], key=user_id_key(), scope="")
    assert record.prompt_hash != usage_events.prompt_fingerprint([system])         # keyed: not guessable
    assert record.prompt_ref == "assistant@v2"
    event = usage_events.validate(json.loads(json.dumps(client.sink.event_for(record), default=str)))
    assert event["prompt_hash"] == record.prompt_hash and event["prompt_ref"] == "assistant@v2"
    assert system not in json.dumps(event, default=str)
    client.post("/v1/chat/completions", json=body,
                headers={"Authorization": f"Bearer {KEYS['assistant']}", "X-Znyx-Prompt": "not a name"})
    assert client.records[-1].prompt_ref is None
    assert chat(client, KEYS["assistant"]).status_code == 200 and client.records[-1].prompt_hash is None


# -- delegated tokens leave governance records -------------------------------------------------

def test_tokens_issued_refused_and_revoked_are_recorded(client):
    import hashlib
    from test_phase5_identity import exchange
    mark = governance_mark(client)
    token = exchange(client).json()["access_token"]
    assert exchange(client, key="loner").status_code == 403
    client.request("DELETE", "/v1/znyx/token", headers={"Authorization": f"Bearer {token}"})
    records = governance_records(client, mark, ("token.issued", "token.refused", "token.revoked"))
    by_type = {r["type"]: r for r in records}
    ref = hashlib.sha256(token.encode()).hexdigest()[:16]
    issued = by_type["token.issued"]
    assert issued["token_ref"] == ref and issued["agent_id"] == "agent-assistant" and issued["key_id"] == "k-assistant"
    assert issued["identity_provider"] == "corp" and issued["person"] and set(issued["tools"]) >= {"search"}
    assert by_type["token.refused"]["error"] == "unauthorized_client"
    assert by_type["token.revoked"]["token_ref"] == ref
    assert token not in Path(client.gateway.governance.path).read_text()      # never the token itself
    assert verify_chain(_lines(client.gateway.governance.path)) == []


def test_the_crosswalk_uses_only_known_checks():
    assert compliance.checks_used() == list(compliance.CHECKS)
    out = compliance.assess("nist_ai_rmf", {"kill_switch_ready": True, "approval_rules": False})
    status = {r["id"]: r["status"] for r in out["requirements"]}
    assert status["mg-2.4"] == "supported" and status["map-3.5"] == "not_yet" and status["gv-2.1"] == "organisational"
    assert status["mg-4.1"] == "partial" and sum(out["counts"].values()) == len(out["requirements"])
    assert "not a statement of compliance" in out["disclaimer"]
    with pytest.raises(KeyError):
        compliance.assess("soc2", {})
