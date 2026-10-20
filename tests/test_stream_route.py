"""POST /v1/evaluate/stream end to end: redaction reaches the wire, windows run on
the shared detector pool (admission control and deadline, fail-closed), and the
stream is recorded as one evaluation."""
import json
import os
import threading

import pytest
from fastapi.testclient import TestClient

from conftest import MINIMAL_POLICY_YAML

_RUNTIME_ENV = {
    "ZNYX_MODE": "local",
    "RUNTIME_REQUIRE_AUTH": "false",
    "ZNYX_TELEMETRY": "false",
}

_EMAIL = "jane.doe" + "@example.com"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("stream-runtime")
    policy_path = tmp / "policies.yaml"
    policy_path.write_text(MINIMAL_POLICY_YAML)

    env = dict(_RUNTIME_ENV)
    env["ZNYX_POLICY_PATH"] = str(policy_path)
    env["ZNYX_AUDIT_SPOOL_PATH"] = str(tmp / "egress-audit.spool")
    env["ZNYX_JUDGE_AUDIT_SPOOL_PATH"] = str(tmp / "judge-audit.spool")
    saved = {key: os.environ.get(key) for key in env}
    os.environ.update(env)

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


def _events(resp):
    out = []
    for block in resp.text.strip().split("\n\n"):
        lines = block.split("\n")
        name = lines[0].removeprefix("event: ")
        data = json.loads(lines[1].removeprefix("data: "))
        out.append({"event": name, "data": data})
    return out


def _stream(client, chunks, **extra):
    body = {"request_id": "stream-test", "context": "output", "chunks": chunks, **extra}
    return client.post("/v1/evaluate/stream", json=body)


def test_redaction_is_applied_on_the_wire(client):
    text = ("Thanks for reaching out. You can contact the account owner at " + _EMAIL
            + " during business hours. " + "We will follow up with the details shortly. " * 6)
    resp = _stream(client, [text[i:i + 25] for i in range(0, len(text), 25)])
    assert resp.status_code == 200
    # Nothing on the wire carries the raw address: not chunks, not previews.
    assert _EMAIL not in resp.text
    events = _events(resp)
    released = "".join(e["data"]["text"] for e in events if e["event"] == "chunk")
    assert "[EMAIL]" in released
    assert events[-1]["data"]["final_decision"] == "REDACT"


def test_saturated_pool_fails_closed(client, monkeypatch):
    import znyx_core.engine.evaluator as evaluator_mod
    monkeypatch.setattr(evaluator_mod, "_ADMISSION", threading.Semaphore(0))
    resp = _stream(client, ["hello there, this is a perfectly ordinary reply"])
    assert resp.status_code == 200
    events = _events(resp)
    assert [e["event"] for e in events] == ["block", "done"]
    assert events[-1]["data"]["final_decision"] == "BLOCK"
    assert events[-1]["data"]["released_text_length"] == 0
    assert "saturated" in events[-1]["data"]["reason"]


def test_missed_deadline_fails_closed(client, monkeypatch):
    import time as _time
    import znyx_core.engine.evaluator as evaluator_mod
    from znyx_core.engine.streaming import StreamingEvaluator

    monkeypatch.setattr(evaluator_mod, "_DEADLINE_S", 0.05)
    real_push = StreamingEvaluator.push

    def slow_push(self, chunk):
        _time.sleep(0.3)
        return real_push(self, chunk)

    monkeypatch.setattr(StreamingEvaluator, "push", slow_push)
    resp = _stream(client, ["a reply that takes too long to check"])
    events = _events(resp)
    assert events[0]["event"] == "block"
    assert events[-1]["data"]["final_decision"] == "BLOCK"
    assert "deadline" in events[-1]["data"]["reason"]
    assert not any(e["event"] == "chunk" for e in events)


def test_stream_is_recorded_as_one_evaluation(client, monkeypatch):
    from znyx_runtime.api.routes import get_evaluator
    seen = []
    monkeypatch.setattr(get_evaluator(), "on_evaluation", seen.append)
    text = "This is a long and entirely ordinary answer about planning. " * 20
    resp = _stream(client, [text[i:i + 30] for i in range(0, len(text), 30)])
    assert resp.status_code == 200
    windows = _events(resp)[-1]["data"]["windows_evaluated"]
    assert windows > 1
    assert len(seen) == 1
    assert seen[0]["request_id"] == "stream-test"
    assert seen[0]["context"] == "output"
    assert seen[0]["decision"] == "ALLOW"
