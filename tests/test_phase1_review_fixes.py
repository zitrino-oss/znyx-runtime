"""Regression tests for the adversarial review of the single install (2026-10-20).

One test (or a few) per confirmed finding: instances sharing a run directory, the
optional model server taking the instance down, credentials reaching the model
server, a shipper dying on one error, air-gap holes, model pins pushed once, .env
from a parent directory, the key in terminal output, torn liveness files and clock
steps, relative paths, torn spool lines, spool permissions, corrupt state rows.
"""
import asyncio
import json
import os
import socket
import threading
import time
from pathlib import Path

import pytest

from znyx import supervisor as sup_mod
from znyx.supervisor import RunDirError, Supervisor, _Child


def _sup(run_dir, **kw):
    kw.setdefault("workers", 1)
    return Supervisor(host="127.0.0.1", port=1, env={}, run_dir=run_dir, **kw)


# -- 1. one supervisor per run directory --------------------------------------------

def test_a_second_instance_cannot_take_a_run_directory_in_use(tmp_path):
    first = _sup(tmp_path / "run")
    first._prepare_run_dir()
    second = _sup(tmp_path / "run")
    with pytest.raises(RunDirError, match="another znyx instance"):
        second._prepare_run_dir()
    first._release_lock()
    second._prepare_run_dir()          # free again once the first one is gone
    second._release_lock()


def test_a_symlinked_run_directory_is_refused(tmp_path):
    real = tmp_path / "elsewhere"
    real.mkdir()
    link = tmp_path / "run"
    link.symlink_to(real)
    with pytest.raises(RunDirError, match="symbolic link"):
        _sup(link)._prepare_run_dir()


def test_the_default_run_directory_is_per_host_and_port(monkeypatch, tmp_path):
    from znyx.config import load_settings, run_dir
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(socket, "gethostname", lambda: "pod-a/b")
    settings = load_settings(env={"PORT": "8123"}, cwd=tmp_path)
    assert run_dir(settings).name == "pod-a_b-8123"


# -- 2. the optional model server never takes the instance down ---------------------

class _Proc:
    def __init__(self, alive=False, pid=4242):
        self.alive, self.pid, self.exitcode, self.killed = alive, pid, None if alive else 1, False

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        pass

    def kill(self):
        self.killed, self.alive, self.exitcode = True, False, -9


def test_a_crashing_model_server_is_retried_without_giving_up(tmp_path):
    sup = _sup(tmp_path)
    ml = _Child(name="ml", index=None, process=_Proc())
    now = time.monotonic()
    delays = []
    for i in range(12):
        ml.restart_due = None
        sup._check(ml, now + i)
        delays.append(ml.restart_due - (now + i))
    assert not sup._stop.is_set() and sup.exit_code == 0
    assert max(delays) == sup_mod.ML_MAX_BACKOFF_SECONDS


def test_a_model_server_port_in_use_fails_fast_and_releases_the_run_dir(tmp_path):
    taken = socket.socket()
    taken.bind(("127.0.0.1", 0))
    taken.listen(1)
    free = socket.socket()
    free.bind(("127.0.0.1", 0))
    main_port = free.getsockname()[1]
    free.close()
    try:
        sup = Supervisor(host="127.0.0.1", port=main_port, workers=1, env={},
                         run_dir=tmp_path / "run", ml_port=taken.getsockname()[1])
        assert sup.run() == 1
        assert sup._children == []                  # nothing was started
        _sup(tmp_path / "run")._prepare_run_dir()   # the lock was released
    finally:
        taken.close()


# -- 3. the model server holds no credential -----------------------------------------

def test_the_model_server_environment_is_an_allowlist():
    env = {"PATH": "/usr/bin", "LC_ALL": "C", "PYTHONPATH": "/x", "ZNYX_INFERENCE_TASKS": "{}",
           "HF_TOKEN": "hf", "REDIS_URL": "redis://:pw@h", "DATABASE_URL": "postgres://u:pw@h/db",
           "GITHUB_TOKEN": "g", "AZURE_CLIENT_SECRET": "a", "GUARDRAILS_REDIS_URL": "r",
           "RUNTIME_API_KEY": "k", "SOME_NEW_VENDOR_KEY": "v"}
    sup_mod.scrub_credentials(env)
    assert env == {"PATH": "/usr/bin", "LC_ALL": "C", "PYTHONPATH": "/x",
                   "ZNYX_INFERENCE_TASKS": "{}", "HF_TOKEN": "hf"}


def test_the_supervisor_environment_is_restored_after_starting_the_model_server(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://u:pw@h/db")
    before = dict(os.environ)
    with sup_mod._environment_for_ml({"PORT": "1"}):
        assert "DATABASE_URL" not in os.environ and os.environ["PORT"] == "1"
    assert dict(os.environ) == before


# -- 10/11. liveness: torn reads and clock steps --------------------------------------

def _beat(run_dir, index, pid, seq):
    (run_dir / "workers").mkdir(parents=True, exist_ok=True)
    (run_dir / "workers" / f"w{index}.json").write_text(json.dumps({"pid": pid, "seq": seq}))


def test_a_torn_liveness_read_does_not_kill_a_healthy_worker(tmp_path):
    sup = _sup(tmp_path, worker_timeout=10)
    proc = _Proc(alive=True)
    child = _Child(name="worker-0", index=0, process=proc, started_at=0.0)
    _beat(tmp_path, 0, proc.pid, 1)
    sup._check(child, 200.0)                     # first beat seen at t=200
    (tmp_path / "workers" / "w0.json").write_text('{"pid": 42')     # torn
    sup._check(child, 209.0)
    assert not proc.killed                       # within worker_timeout of the last good beat
    _beat(tmp_path, 0, proc.pid, 2)
    sup._check(child, 215.0)                     # a new beat resets the clock
    sup._check(child, 224.0)
    assert not proc.killed
    sup._check(child, 226.0)                     # no new beat for > 10 s: restart
    assert proc.killed


def test_liveness_ignores_the_wall_clock(tmp_path):
    """The record's own timestamps are irrelevant: only the counter changing, on the
    supervisor's monotonic clock, counts."""
    sup = _sup(tmp_path, worker_timeout=10)
    proc = _Proc(alive=True)
    child = _Child(name="worker-0", index=0, process=proc, started_at=0.0)
    path = tmp_path / "workers" / "w0.json"
    for seq, t in ((1, 200.0), (2, 205.0), (3, 210.0)):
        (tmp_path / "workers").mkdir(exist_ok=True)
        path.write_text(json.dumps({"pid": proc.pid, "seq": seq, "beat_at": 0}))
        os.utime(path, (1.0, 1.0))                 # mtime decades in the past
        sup._check(child, t)
    assert not proc.killed


def test_worker_reports_carry_an_increasing_counter(tmp_path):
    from znyx_core.middleware.metrics import MetricsCollector
    from znyx_runtime.workers import RecentDecisions, WorkerReporter
    reporter = WorkerReporter(MetricsCollector(), RecentDecisions(), tmp_path, index=0)
    seqs = []
    for _ in range(3):
        reporter.write_now()
        seqs.append(json.loads(reporter.liveness_path.read_text())["seq"])
    assert seqs == [1, 2, 3]


def test_concurrent_reports_never_leave_a_torn_file(tmp_path):
    from znyx_core.middleware.metrics import MetricsCollector
    from znyx_runtime.workers import RecentDecisions, WorkerReporter
    reporter = WorkerReporter(MetricsCollector(), RecentDecisions(), tmp_path, index=0)
    reporter.write_now()
    stop, errors, bad = threading.Event(), [], []

    def write():
        while not stop.is_set():
            try:
                reporter.write_now()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    def read():
        while not stop.is_set():
            for path in (reporter.liveness_path, reporter.metrics_path):
                try:
                    json.loads(path.read_text())
                except ValueError:
                    bad.append(path)

    threads = [threading.Thread(target=write) for _ in range(3)] + [threading.Thread(target=read)]
    for t in threads:
        t.start()
    time.sleep(1.0)
    stop.set()
    for t in threads:
        t.join()
    assert errors == [] and bad == []
    assert not list(tmp_path.glob("**/*.tmp"))


# -- 4. shippers -----------------------------------------------------------------

def test_a_delivery_loop_survives_an_unexpected_error(tmp_path):
    from znyx_runtime.shipper import SpoolShipper, append_record
    received = []

    async def post(url, records):
        received.extend(records)
        return True

    shipper = SpoolShipper(str(tmp_path / "s.spool"), "http://cp", post, interval=0.01)
    calls = {"n": 0}
    real = shipper.ship_once

    async def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise IsADirectoryError("transient")
        return await real()

    shipper.ship_once = flaky

    async def scenario():
        append_record(shipper.path, {"event_id": 1})
        shipper.start()
        for _ in range(200):
            if received:
                break
            await asyncio.sleep(0.01)
        assert not shipper._task.done()
        await shipper.stop(final_attempt=False)

    asyncio.run(scenario())
    assert [r["event_id"] for r in received] == [1]
    assert shipper.failures >= 1


def test_only_one_shipper_owns_a_spool(tmp_path):
    from znyx_runtime.shipper import SpoolShipper

    async def post(url, records):
        return True

    a = SpoolShipper(str(tmp_path / "s.spool"), "http://cp", post)
    b = SpoolShipper(str(tmp_path / "s.spool"), "http://cp", post)
    assert a._acquire_ownership() is True
    assert b._acquire_ownership() is False
    a._release_ownership()
    assert b._acquire_ownership() is True
    b._release_ownership()


def test_offset_writes_use_unique_temporary_files(tmp_path):
    from znyx_runtime.shipper import SpoolShipper

    async def post(url, records):
        return True

    a = SpoolShipper(str(tmp_path / "s.spool"), "http://cp", post)
    b = SpoolShipper(str(tmp_path / "s.spool"), "http://cp", post)
    errors = []

    def hammer(shipper, n):
        for i in range(300):
            try:
                shipper._write_offset(n + i)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    threads = [threading.Thread(target=hammer, args=(s, k * 1000)) for k, s in enumerate((a, b))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert int(a.offset_path.read_text()) >= 0


# -- 13/14. spool lines and permissions ------------------------------------------------

def test_a_record_after_a_torn_line_is_still_delivered(tmp_path):
    from znyx_runtime.shipper import SpoolShipper, append_record
    spool = tmp_path / "s.spool"
    append_record(str(spool), {"event_id": "ev-1"})
    with open(spool, "a") as fh:
        fh.write('{"event_id": "ev-2", "half')     # a writer killed mid-append
    append_record(str(spool), {"event_id": "ev-3"})
    received = []

    async def post(url, records):
        received.extend(records)
        return True

    asyncio.run(SpoolShipper(str(spool), "http://cp", post).ship_once())
    assert [r["event_id"] for r in received] == ["ev-1", "ev-3"]


def test_spools_are_private(tmp_path):
    from znyx_core.utils.spool import append_lines
    spool = tmp_path / "new-dir" / "telemetry.spool"
    append_lines(spool, ["{}"])
    assert spool.stat().st_mode & 0o777 == 0o600
    assert spool.parent.stat().st_mode & 0o777 == 0o700


# -- 5/6. air-gapped ----------------------------------------------------------------

def test_the_first_run_ping_is_never_sent_when_air_gapped(monkeypatch):
    import znyx_runtime.heartbeat as hb_mod
    monkeypatch.setenv("ZNYX_AIRGAPPED", "true")
    monkeypatch.setattr(hb_mod, "heartbeat_endpoint", lambda: "https://telemetry.example/v1")
    sent = []

    class Boom:
        def __init__(self, *a, **k):
            sent.append(1)

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", Boom)
    hb = hb_mod.Heartbeat(enabled=True)
    asyncio.run(hb.send_first_run_ping(run_count=1))
    asyncio.run(hb._send_ping())
    assert sent == []


@pytest.mark.parametrize("host,expected", [
    ("127.0.0.1", True), ("::1", True), ("10.1.2.3", True), ("192.168.0.4", True),
    ("172.16.9.9", True), ("fd00::1", True), ("169.254.1.1", True), ("localhost", True),
    ("inference", True), ("ml.internal", True), ("svc.ns.svc.cluster.local", True),
    ("printer.local", True), ("8.8.8.8", False), ("api.groq.com", False),
    ("inference.example.com", False), ("", False), (None, False),
])
def test_in_network(host, expected):
    from znyx_core.engine.egress import in_network
    assert in_network(host) is expected


def test_operator_listed_hosts_count_as_in_network(monkeypatch):
    from znyx_core.engine.egress import in_network
    monkeypatch.setenv("ZNYX_AIRGAPPED_ALLOW", "ml.corp.example.com, .models.example.net")
    assert in_network("ml.corp.example.com")
    assert in_network("a.models.example.net") and in_network("models.example.net")
    assert not in_network("evil.example.com")


def _decide(mode, url, in_boundary=None):
    from znyx_core.engine.egress import evaluate_egress
    return evaluate_egress(mode, endpoint_url=url, region=None, in_boundary=in_boundary,
                           no_external_calls=False, egress_allowlist=None, allowed_regions=None,
                           redact_pii=False, redact_secrets=False)


@pytest.mark.parametrize("mode,url", [
    ("local_llm", "https://api.groq.com/openai/v1/chat/completions"),
    ("local_ml", "https://inference.example.com/v1/infer/toxicity"),
    ("remote_llm", "https://api.openai.com/v1/chat/completions"),
    ("remote_api", None),
])
def test_air_gapped_refuses_any_call_leaving_the_network(monkeypatch, mode, url):
    monkeypatch.setenv("ZNYX_AIRGAPPED", "true")
    decision = _decide(mode, url)
    assert decision.allowed is False and decision.reason == "no_external_calls"


def test_air_gapped_allows_an_in_network_model_server(monkeypatch):
    monkeypatch.setenv("ZNYX_AIRGAPPED", "true")
    assert _decide("local_ml", "http://ml.internal:9000/v1/infer/toxicity", in_boundary=False).allowed
    assert _decide("local_ml", "http://127.0.0.1:9000/v1/infer/toxicity", in_boundary=True).allowed
    assert _decide("local_deterministic", None).allowed


# -- 7. model pins are re-pushed in local mode -------------------------------------------

def test_local_mode_repeats_the_listener_cycle(tmp_path):
    from znyx_runtime.bundle_manager import BundleManager
    from znyx_runtime.config import RuntimeConfig
    policy = Path(__file__).resolve().parent.parent / "config" / "policies.yaml"
    manager = BundleManager(RuntimeConfig(mode="local", policy_path=str(policy),
                                          bundle_poll_interval=0.05))
    calls = []

    async def listener(_policy):
        calls.append(1)

    async def scenario():
        manager.add_cycle_listener(listener)
        await manager.start()
        await asyncio.sleep(0.4)
        await manager.stop()

    asyncio.run(scenario())
    assert len(calls) >= 3


# -- 8/9/12. config, .env and the start banner -------------------------------------------

def test_relative_paths_in_a_config_file_are_relative_to_that_file(tmp_path):
    from znyx.config import load_settings
    cfg_dir = tmp_path / "etc"
    cfg_dir.mkdir()
    (cfg_dir / "znyx.yaml").write_text("policy: ./policies.yaml\nauth: {api_key_file: keys/k}\n"
                                       "ml: {artifacts_dir: ~/models}\n")
    settings = load_settings(str(cfg_dir / "znyx.yaml"), env={}, cwd=tmp_path / "elsewhere")
    assert settings["policy"] == str(cfg_dir / "policies.yaml")
    assert settings["auth.api_key_file"] == str(cfg_dir / "keys" / "k")
    assert settings["ml.artifacts_dir"] == "~/models"


def test_an_empty_variable_means_not_set(tmp_path):
    from znyx.config import load_settings
    settings = load_settings(env={"ZNYX_WORKERS": "", "ZNYX_ML": " ", "PORT": ""}, cwd=tmp_path)
    assert settings["server.workers"] == "auto" and settings["ml.enabled"] is False
    assert settings["server.port"] == 8080


def test_only_the_current_directorys_env_file_is_read(tmp_path, monkeypatch):
    from znyx import cli, config
    (tmp_path / ".env").write_text("RUNTIME_REQUIRE_AUTH=false\nZNYX_FAIL_MODE=open\n")
    project = tmp_path / "a" / "b"
    project.mkdir(parents=True)
    monkeypatch.chdir(project)
    for name in ("RUNTIME_REQUIRE_AUTH", "ZNYX_FAIL_MODE"):
        monkeypatch.delenv(name, raising=False)
    assert cli.load_dotenv_once() is None
    assert "RUNTIME_REQUIRE_AUTH" not in os.environ
    (project / ".env").write_text("ZNYX_FAIL_MODE=open\n")
    monkeypatch.setattr(config, "DOTENV_NAMES", set())
    assert cli.load_dotenv_once() == project / ".env"
    settings = config.load_settings(cwd=project)
    assert settings.sources["fail_mode"] == ".env ZNYX_FAIL_MODE"


def test_the_start_banner_never_prints_the_key(capsys, tmp_path):
    from znyx import cli
    from znyx.config import load_settings
    settings = load_settings(env={}, cwd=tmp_path)
    cli._print_start(settings, 2, "zx_super_secret_value", ("new", tmp_path / "api-key"))
    err = capsys.readouterr().err
    assert "zx_super_secret_value" not in err and "#key=" not in err
    assert "znyx key" in err


# -- 16. corrupt state rows -------------------------------------------------------------

def test_a_corrupt_state_row_is_treated_as_absent(tmp_path):
    from znyx_core.state_store import SqliteStateStore
    store = SqliteStateStore(str(tmp_path / "s.sqlite3"))
    conn = store._conn()
    for key in ("bucket", "counter", "list"):
        conn.execute("INSERT INTO kv (key, value, expires) VALUES (?, ?, NULL)", ("znyx:" + key, "not json"))
    assert store.consume_token("bucket", rate_per_second=0.001, capacity=2) is True
    assert store.incr("counter", {"n": 1}, ttl_seconds=60) == {"n": 1.0}
    assert store.push_capped("list", "a", max_len=3, ttl_seconds=60) == ["a"]
    assert store.get("counter") == {"n": 1.0}


# -- 17. worker 0 reports for every worker ----------------------------------------------

def test_sdk_callers_are_reported_for_the_whole_instance(tmp_path, monkeypatch):
    from znyx_runtime import workers
    monkeypatch.setenv("ZNYX_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("ZNYX_WORKER_COUNT", "2")
    (tmp_path / "metrics").mkdir()
    (tmp_path / "metrics" / "w1.json").write_text(json.dumps({"metrics": {}, "sdk_clients": [
        {"source": "znyx-sdk-python", "version": "1.2.3", "last_seen_at": "2026-10-20T10:00:00"},
        {"source": "znyx-sdk-ts", "version": "1.0.0", "last_seen_at": "2026-10-20T09:00:00"}]}))
    local = [{"source": "znyx-sdk-python", "version": "1.2.0", "last_seen_at": "2026-10-20T08:00:00"}]
    merged = workers.instance_sdk_clients(local)
    assert [(c["source"], c["version"]) for c in merged] == [("znyx-sdk-python", "1.2.3"),
                                                            ("znyx-sdk-ts", "1.0.0")]


# -- the model server's Host check ------------------------------------------------------

def test_the_model_server_refuses_foreign_host_headers_when_asked(monkeypatch):
    import importlib

    from fastapi.testclient import TestClient

    import znyx_inference.main as ml_main
    monkeypatch.setenv("ZNYX_INFERENCE_TRUSTED_HOSTS", "127.0.0.1,localhost")
    try:
        app = importlib.reload(ml_main).app
        client = TestClient(app, base_url="http://127.0.0.1")
        assert client.get("/healthz").status_code == 200
        assert client.get("/healthz", headers={"Host": "rebind.attacker.example"}).status_code == 400
    finally:
        monkeypatch.delenv("ZNYX_INFERENCE_TRUSTED_HOSTS")
        importlib.reload(ml_main)
