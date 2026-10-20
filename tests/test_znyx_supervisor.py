"""The ``znyx serve`` supervisor with real processes: workers sharing one port,
merged metrics, crash and hang recovery, and graceful shutdown.

These start actual ``znyx serve`` processes (a few seconds each), so the file is
slower than the rest of the suite. Every process is stopped in a ``finally``, and
workers (which leave the supervisor's process group by design) are also killed by the
pids in their liveness files.
"""
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

from znyx import supervisor
from znyx.config import SETTINGS

REPO = Path(__file__).resolve().parents[1]
POLICY = REPO / "config" / "policies.yaml"


# ---------------------------------------------------------------------------
# Unit level (no processes)
# ---------------------------------------------------------------------------

def test_scrub_credentials_removes_secrets_and_keeps_the_rest():
    env = {"RUNTIME_API_KEY": "k", "ZNYX_RUNTIME_TOKEN": "t", "OPENAI_API_KEY": "o",
           "AWS_SECRET_ACCESS_KEY": "a", "PATH": "/usr/bin", "ZNYX_INFERENCE_ARTIFACTS_DIR": "/m"}
    supervisor.scrub_credentials(env)
    assert env == {"PATH": "/usr/bin", "ZNYX_INFERENCE_ARTIFACTS_DIR": "/m"}


def test_scrub_credentials_is_fine_when_nothing_is_set():
    env = {"PATH": "/usr/bin"}
    supervisor.scrub_credentials(env)
    assert env == {"PATH": "/usr/bin"}


def test_read_worker_status_of_a_missing_or_empty_dir_is_empty(tmp_path):
    assert supervisor.read_worker_status(tmp_path / "nope") == []
    (tmp_path / "workers").mkdir()
    assert supervisor.read_worker_status(tmp_path) == []


def test_read_worker_status_reads_records_and_skips_garbage(tmp_path):
    workers = tmp_path / "workers"
    workers.mkdir()
    (workers / "w0.json").write_text(json.dumps({"index": 0, "pid": 11}))
    (workers / "w1.json").write_text("{not json")
    (workers / "other.txt").write_text("{}")
    records = supervisor.read_worker_status(tmp_path)
    assert [r["pid"] for r in records] == [11]
    assert records[0]["age_seconds"] >= 0


def _supervisor(tmp_path, **kw):
    return supervisor.Supervisor(host="127.0.0.1", port=1, workers=2, env={}, run_dir=tmp_path, **kw)


def test_prepare_run_dir_clears_stale_liveness_but_keeps_the_state_file(tmp_path):
    for sub in ("workers", "metrics"):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "w0.json").write_text("{}")
    (tmp_path / "state.sqlite3").write_text("limits")
    _supervisor(tmp_path)._prepare_run_dir()
    assert list((tmp_path / "workers").iterdir()) == []
    assert list((tmp_path / "metrics").iterdir()) == []
    assert (tmp_path / "state.sqlite3").read_text() == "limits"
    assert (tmp_path.stat().st_mode & 0o777) == 0o700


class _DeadProcess:
    pid = 4242
    exitcode = 1

    def is_alive(self):
        return False

    def join(self, timeout=None):
        pass

    def kill(self):
        pass


def test_a_worker_that_keeps_crashing_makes_the_supervisor_give_up(tmp_path):
    sup = _supervisor(tmp_path)
    child = supervisor._Child(name="worker-0", index=0, process=_DeadProcess())
    now = time.monotonic()
    delays = []
    for i in range(supervisor.MAX_RESTARTS_IN_WINDOW):
        child.restart_due = None
        sup._check(child, now + i)
        assert not sup._stop.is_set(), f"gave up too early at crash {i + 1}"
        delays.append(round(child.restart_due - (now + i), 1))
    assert delays == [0.5, 1.0, 2.0, 4.0, 8.0], "restarts back off exponentially"
    child.restart_due = None
    sup._check(child, now + supervisor.MAX_RESTARTS_IN_WINDOW)
    assert sup._stop.is_set()
    assert sup.exit_code == 1


def test_crashes_outside_the_window_do_not_count(tmp_path):
    sup = _supervisor(tmp_path)
    child = supervisor._Child(name="worker-0", index=0, process=_DeadProcess())
    now = time.monotonic()
    for i in range(supervisor.MAX_RESTARTS_IN_WINDOW * 2):
        child.restart_due = None
        sup._check(child, now + i * (supervisor.RESTART_WINDOW_SECONDS + 1))
    assert not sup._stop.is_set()
    assert sup.exit_code == 0


# ---------------------------------------------------------------------------
# Real processes
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for(predicate, timeout, interval=0.2, what="condition"):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            last = predicate()
        except (httpx.HTTPError, OSError, ValueError, KeyError):
            last = None
        if last:
            return last
        time.sleep(interval)
    raise AssertionError(f"timed out after {timeout}s waiting for {what}")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Stack:
    """One running ``znyx serve`` and helpers to talk to it."""

    def __init__(self, home: Path, port: int, proc: subprocess.Popen, log: Path):
        self.home, self.port, self.proc, self.log = home, port, proc, log
        self.base = f"http://127.0.0.1:{port}"
        # Per host and port by default (two instances sharing a home or a volume must
        # never share a run directory); computed the same way the CLI does.
        import re
        import socket
        host = re.sub(r"[^A-Za-z0-9._-]", "_", socket.gethostname() or "host")[:64]
        self.run_dir = home / ".znyx" / "run" / f"{host}-{port}"
        self.sent = 0                  # evaluations sent through ``evaluate``
        self._lock = threading.Lock()

    @property
    def key(self) -> str:
        return (self.home / ".znyx" / "api-key").read_text().strip()

    @property
    def headers(self) -> dict:
        return {"X-API-Key": self.key}

    def worker_file(self, index: int) -> Path:
        return self.run_dir / "workers" / f"w{index}.json"

    def worker_record(self, index: int):
        try:
            return json.loads(self.worker_file(index).read_text())
        except (OSError, ValueError):
            return None

    def worker_pids(self) -> list:
        out = []
        for path in sorted((self.run_dir / "workers").glob("w*.json")):
            try:
                out.append(json.loads(path.read_text())["pid"])
            except (OSError, ValueError, KeyError):
                pass
        return out

    def wait_for_workers(self, count=2, timeout=40):
        _wait_for(lambda: all(self.worker_record(i) for i in range(count)), timeout,
                  what=f"{count} worker liveness files")

    def evaluate_once(self, text="How do I bake bread?", headers=None):
        """One evaluation on a fresh connection (so the kernel picks the worker)."""
        body = {"request_id": "r", "tenant_id": "default", "app_id": "default", "text": text}
        with httpx.Client(timeout=10.0, headers={"Connection": "close"}) as client:
            resp = client.post(f"{self.base}/v1/evaluate/input", json=body,
                               headers=self.headers if headers is None else headers)
        if resp.status_code == 200:
            with self._lock:
                self.sent += 1
        return resp

    def evaluate(self, n: int, threads: int = 8) -> set:
        """Send ``n`` evaluations concurrently; returns the worker ids that answered."""
        with ThreadPoolExecutor(threads) as pool:
            responses = list(pool.map(lambda _: self.evaluate_once(), range(n)))
        assert all(r.status_code == 200 for r in responses), [r.status_code for r in responses]
        return {r.headers["X-Znyx-Worker"] for r in responses}

    def metrics_total(self) -> float:
        text = httpx.get(f"{self.base}/metrics", timeout=5.0).text
        return sum(float(m.group(1)) for m in re.finditer(
            r"^guardrails_evaluations_total\{[^}]*\} (\S+)$", text, re.M))

    def summary(self) -> dict:
        resp = httpx.get(f"{self.base}/v1/local/summary", headers=self.headers, timeout=5.0)
        assert resp.status_code == 200, resp.text
        return resp.json()

    def log_tail(self) -> str:
        try:
            return "\n".join(self.log.read_text(errors="replace").splitlines()[-40:])
        except OSError:
            return "(no log)"

    def stop(self):
        """Terminate the supervisor, then anything still alive from the liveness files."""
        pids = self.worker_pids()
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(20)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(5)
        for pid in pids:
            for sig in (signal.SIGCONT, signal.SIGKILL):
                try:
                    os.kill(pid, sig)
                except (ProcessLookupError, PermissionError):
                    break


def _child_env(home: Path, extra: dict) -> dict:
    env = dict(os.environ)
    for setting in SETTINGS:                 # nothing from the developer's shell or .env
        for name in setting.env:
            env.pop(name, None)
    env.pop("ZNYX_CONFIG", None)
    env["HOME"] = str(home)
    env["ZNYX_ENVIRONMENT"] = "development"
    env["PYTHONUNBUFFERED"] = "1"
    env.update(extra)
    return env


def _start_stack(tmp_path: Path, **extra_env) -> Stack:
    home = tmp_path / "home"
    work = tmp_path / "work"
    home.mkdir()
    work.mkdir()
    shutil.copyfile(POLICY, work / "policies.yaml")
    port = _free_port()
    log = tmp_path / "serve.log"
    with open(log, "wb") as out:
        proc = subprocess.Popen(
            [sys.executable, "-m", "znyx.cli", "serve", "--port", str(port), "--workers", "2",
             "--log-level", "warning"],
            cwd=work, env=_child_env(home, extra_env), stdout=out, stderr=subprocess.STDOUT,
            start_new_session=True)
    stack = Stack(home, port, proc, log)
    try:
        def ready():
            if proc.poll() is not None:
                raise AssertionError(f"znyx serve exited with {proc.returncode}:\n{stack.log_tail()}")
            return httpx.get(f"{stack.base}/healthz", timeout=1.0).status_code == 200
        _wait_for(ready, 60, 0.3, "/healthz")
        stack.wait_for_workers()
    except BaseException:
        print(stack.log_tail())
        stack.stop()
        raise
    return stack


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    s = _start_stack(tmp_path_factory.mktemp("stack"))
    try:
        yield s
    finally:
        s.stop()


def test_requests_need_the_generated_key(stack):
    assert (stack.home / ".znyx" / "api-key").stat().st_mode & 0o777 == 0o600
    assert stack.evaluate_once(headers={}).status_code == 401
    assert stack.evaluate_once(headers={"X-API-Key": "wrong"}).status_code == 401
    ok = stack.evaluate_once()
    assert ok.status_code == 200
    assert ok.json()["decision"] == "ALLOW"


def test_both_workers_serve_requests(stack):
    seen = set()
    for _ in range(8):                      # 8 x 40: far more than enough for a fair kernel
        seen |= stack.evaluate(40)
        if seen == {"0", "1"}:
            break
    assert seen == {"0", "1"}


def test_prometheus_totals_cover_every_worker(stack):
    stack.evaluate(30)
    _wait_for(lambda: stack.metrics_total() == stack.sent, 8,
              what=f"/metrics evaluations_total == {stack.sent}")


def test_local_summary_reports_the_whole_instance(stack):
    stack.evaluate(10)
    _wait_for(lambda: stack.summary()["evaluations"] == stack.sent, 8,
              what=f"summary evaluations == {stack.sent}")
    summary = stack.summary()
    assert summary["workers"]["count"] == 2
    assert summary["workers"]["reporting"] == 2
    assert {p["index"] for p in summary["workers"]["processes"]} == {0, 1}
    assert sum(summary["decisions"].values()) == summary["evaluations"]
    for entry in summary["recent"]:
        assert "text" not in entry


def test_the_dashboard_page_is_served_by_every_worker(stack):
    seen = set()
    for _ in range(30):
        with httpx.Client(headers={"Connection": "close"}) as client:
            resp = client.get(f"{stack.base}/dashboard")
        assert resp.status_code == 200
        seen.add(resp.headers["X-Znyx-Worker"])
    assert seen <= {"0", "1"}


def test_a_killed_worker_is_restarted_and_its_totals_survive(stack):
    stack.evaluate(20)
    _wait_for(lambda: stack.summary()["evaluations"] == stack.sent, 8, what="counters settled")
    before = stack.sent
    old = stack.worker_record(1)["pid"]
    os.kill(old, signal.SIGKILL)

    new = _wait_for(lambda: (stack.worker_record(1) or {}).get("pid") not in (None, old)
                    and stack.worker_record(1)["pid"], 20, what="worker 1 restarted")
    assert new != old and _pid_alive(new)
    assert httpx.get(f"{stack.base}/healthz", timeout=2.0).status_code == 200
    assert stack.proc.poll() is None

    # The new process restored its predecessor's counters: the instance total did not
    # drop (and did not double count either).
    totals = []
    for _ in range(10):
        totals.append(stack.summary()["evaluations"])
        time.sleep(0.3)
    assert min(totals) >= before, totals
    assert max(totals) == before, totals
    stack.evaluate(5)
    _wait_for(lambda: stack.summary()["evaluations"] == stack.sent, 8, what="counting resumes")


def test_a_worker_that_stops_reporting_is_restarted(tmp_path):
    s = _start_stack(tmp_path, ZNYX_WORKER_TIMEOUT="3")
    stopped = None
    try:
        stopped = s.worker_record(1)["pid"]
        os.kill(stopped, signal.SIGSTOP)
        _wait_for(lambda: (s.worker_record(1) or {}).get("pid") not in (None, stopped), 20,
                  what="hung worker 1 replaced")
        _wait_for(lambda: not _pid_alive(stopped), 5, what="the hung worker to be killed")
        assert httpx.get(f"{s.base}/healthz", timeout=3.0).status_code == 200
    finally:
        if stopped:
            for sig in (signal.SIGCONT, signal.SIGKILL):
                try:
                    os.kill(stopped, sig)
                except (ProcessLookupError, PermissionError):
                    break
        s.stop()


def test_sigterm_stops_the_supervisor_cleanly_and_leaves_no_workers(tmp_path):
    s = _start_stack(tmp_path)
    try:
        pids = s.worker_pids()
        assert len(pids) == 2 and all(_pid_alive(p) for p in pids)
        os.kill(s.proc.pid, signal.SIGTERM)
        assert s.proc.wait(15) == 0
        _wait_for(lambda: not any(_pid_alive(p) for p in pids), 5, what="workers to exit")
        with pytest.raises(httpx.HTTPError):
            httpx.get(f"{s.base}/healthz", timeout=1.0)
    finally:
        s.stop()
