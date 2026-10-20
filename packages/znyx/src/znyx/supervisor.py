"""The ``znyx serve`` supervisor.

Python threads do not run detectors in parallel, so throughput comes from
processes. The supervisor binds the one port, starts N worker processes that share
it (each serves whole requests: the kernel spreads connections across them), and
optionally the ML model server on loopback. It restarts a worker that crashes or
stops reporting, gives up (exit code 1) if one keeps crashing, and on SIGTERM or
Ctrl+C lets in-flight requests finish before stopping.

Children run in their own process group, so a terminal's Ctrl+C reaches only the
supervisor, which then stops them once and gracefully. Each child also watches for
the supervisor disappearing (killed, not stopped) and shuts itself down, so no
orphan keeps serving the port.

One supervisor owns a run directory at a time (an exclusive lock): two instances
sharing one, for example replicas mounting the same volume, would otherwise delete
and misread each other's liveness files and restart each other's healthy workers.

Worker-local duties (the singleton jobs only worker 0 runs, liveness files, shared
metrics) are implemented by the runtime itself; it reads ``ZNYX_WORKER_INDEX``,
``ZNYX_WORKER_COUNT`` and ``ZNYX_RUN_DIR``, which the supervisor sets.
"""
from __future__ import annotations

import json
import logging
import multiprocessing
import os
import signal
import stat
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("znyx")

RUNTIME_APP = "znyx_runtime.main:app"
ML_APP = "znyx_inference.main:app"

RESTART_WINDOW_SECONDS = 60.0
MAX_RESTARTS_IN_WINDOW = 5
BOOT_GRACE_SECONDS = 120.0
ML_READY_TIMEOUT_SECONDS = 120.0
ML_MAX_BACKOFF_SECONDS = 30.0
TICK_SECONDS = 0.5


class RunDirError(RuntimeError):
    """The run directory cannot be used safely; the message says why."""


# ---------------------------------------------------------------------------
# Child entry points (module level: they are pickled by the spawn start method)
# ---------------------------------------------------------------------------

def _detach_and_watch_parent(graceful_timeout: float) -> None:
    """Leave the terminal's process group and stop when the supervisor is gone.

    Waits on the parent's process sentinel, which works on Windows too (there
    ``os.getppid()`` keeps returning the dead parent's id); polling the parent id is
    the fallback when no sentinel is available."""
    if hasattr(os, "setpgrp"):
        try:
            os.setpgrp()
        except OSError:
            pass
    parent = multiprocessing.parent_process()
    parent_pid = os.getppid()

    def stop_self() -> None:
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(graceful_timeout + 5)
        os._exit(1)

    def watch() -> None:
        if parent is not None and getattr(parent, "sentinel", None) is not None:
            from multiprocessing.connection import wait
            wait([parent.sentinel])
            stop_self()
        while True:
            time.sleep(1.0)
            if os.getppid() != parent_pid:
                stop_self()

    threading.Thread(target=watch, name="znyx-parent-watch", daemon=True).start()


def _worker_entry(sockets: list, env: Dict[str, str], uvicorn_kwargs: Dict[str, Any]) -> None:
    _detach_and_watch_parent(float(uvicorn_kwargs.get("timeout_graceful_shutdown") or 30))
    os.environ.update(env)
    import uvicorn

    config = uvicorn.Config(RUNTIME_APP, **uvicorn_kwargs)
    uvicorn.Server(config).run(sockets=sockets)


def _ml_entry(sockets: list, env: Dict[str, str], uvicorn_kwargs: Dict[str, Any]) -> None:
    _detach_and_watch_parent(float(uvicorn_kwargs.get("timeout_graceful_shutdown") or 30))
    scrub_credentials(os.environ)
    os.environ.update(env)
    import uvicorn

    uvicorn.Server(uvicorn.Config(ML_APP, **uvicorn_kwargs)).run(sockets=sockets)


# ---------------------------------------------------------------------------
# The model server's environment: an allowlist, not a denylist
# ---------------------------------------------------------------------------

# The model server has no authentication of its own and must hold no credential: no
# runtime or control-plane token, no database or Redis URL, no cloud or provider key,
# whatever the shell or .env that started the supervisor contained. It gets only what a
# Python process serving local models needs. HF_TOKEN is the one secret it may need (it
# fetches weights, and gated models require one).
ML_ENV_NAMES = frozenset({
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LANGUAGE", "TZ", "TERM",
    "TMPDIR", "TMP", "TEMP", "VIRTUAL_ENV",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
    "HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
    "TRANSFORMERS_OFFLINE", "TRANSFORMERS_CACHE", "TORCH_HOME", "XDG_CACHE_HOME",
    "ZNYX_AIRGAPPED", "ZNYX_PLATFORM_VERSION", "ZNYX_ENVIRONMENT", "ZNYX_ENV", "ENVIRONMENT",
    "GUARDRAILS_ENVIRONMENT", "PORT",
    # Windows needs these to start a Python process at all.
    "SYSTEMROOT", "SystemRoot", "WINDIR", "COMSPEC", "PATHEXT", "USERPROFILE", "APPDATA",
    "LOCALAPPDATA", "PROGRAMDATA", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE",
})
ML_ENV_PREFIXES = ("LC_", "PYTHON", "ZNYX_INFERENCE_", "OMP_", "MKL_", "OPENBLAS_", "ORT_",
                   "CUDA_", "NVIDIA_")


def ml_env_allowed(name: str) -> bool:
    return name in ML_ENV_NAMES or name.startswith(ML_ENV_PREFIXES)


def scrub_credentials(environ: Dict[str, str]) -> None:
    """Remove from ``environ`` everything the model server is not allowed to see."""
    for name in [n for n in environ if not ml_env_allowed(n)]:
        environ.pop(name, None)


class _environment_for_ml:
    """This process's environment, reduced to the allowlist, for the duration of the
    model server's start. A spawned child starts with the parent's environment, so
    removing variables inside the child would be too late; the supervisor is
    single-threaded while it starts children."""

    def __init__(self, extra: Dict[str, str]):
        self.extra = extra

    def __enter__(self) -> None:
        self._saved = dict(os.environ)
        keep = {k: v for k, v in self._saved.items() if ml_env_allowed(k)}
        os.environ.clear()
        os.environ.update(keep)
        os.environ.update(self.extra)

    def __exit__(self, *exc: Any) -> None:
        os.environ.clear()
        os.environ.update(self._saved)


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

@dataclass
class _Child:
    name: str
    index: Optional[int]                 # worker index; None for the ML process
    process: Any = None
    started_at: float = 0.0
    restart_times: List[float] = field(default_factory=list)
    restart_due: Optional[float] = None
    last_seq: Optional[int] = None       # liveness counter last seen from this process
    last_seen: float = 0.0               # supervisor's monotonic time when it changed
    ready: bool = False                  # ML only: has answered /healthz
    warned: bool = False                 # ML only: "not ready yet" logged for this start


class Supervisor:
    def __init__(self, *, host: str, port: int, workers: int, env: Dict[str, str],
                 run_dir: Path, ml_port: Optional[int] = None, log_level: str = "info",
                 graceful_timeout: int = 30, worker_timeout: int = 60,
                 ml_env: Optional[Dict[str, str]] = None) -> None:
        self.host = host
        self.port = port
        self.workers = workers
        self.env = dict(env)
        self.ml_env = dict(ml_env if ml_env is not None else {})
        self.run_dir = run_dir
        self.ml_port = ml_port
        self.log_level = log_level
        self.graceful_timeout = graceful_timeout
        self.worker_timeout = worker_timeout
        self.exit_code = 0
        self._ctx = multiprocessing.get_context("spawn")
        self._stop = threading.Event()
        self._sockets: list = []
        self._ml_sockets: list = []
        self._children: List[_Child] = []
        self._lock_fd: Optional[int] = None
        self._ml_checked_at = 0.0

    # -- layout -------------------------------------------------------------

    @property
    def workers_dir(self) -> Path:
        return self.run_dir / "workers"

    def _secure_run_dir(self) -> None:
        """The run directory holds the shared limits, liveness records and metrics: it
        must be this user's own, private directory, never a link to somewhere else."""
        self.run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        st = os.lstat(self.run_dir)
        if stat.S_ISLNK(st.st_mode):
            raise RunDirError(f"{self.run_dir} is a symbolic link; use a real directory")
        if hasattr(os, "getuid") and st.st_uid != os.getuid():
            raise RunDirError(f"{self.run_dir} belongs to another user (uid {st.st_uid}); "
                              "set server.run_dir to a directory you own")
        try:
            os.chmod(self.run_dir, 0o700)
        except OSError as exc:
            raise RunDirError(f"cannot make {self.run_dir} private: {exc}") from exc

    def _lock_run_dir(self) -> None:
        """Exclusive ownership of the run directory for this supervisor's lifetime."""
        path = self.run_dir / "supervisor.lock"
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            try:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except ImportError:  # pragma: no cover - Windows
                import msvcrt
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            try:
                holder = os.pread(fd, 32, 0).decode(errors="replace").strip() if hasattr(os, "pread") else ""
            except OSError:
                holder = ""
            os.close(fd)
            raise RunDirError(
                f"another znyx instance{f' (pid {holder})' if holder else ''} is using "
                f"{self.run_dir}; give each instance its own server.run_dir") from None
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        self._lock_fd = fd

    def _prepare_run_dir(self) -> None:
        """A private, exclusively owned layout for this instance: stale liveness and
        metrics files from a previous run must not be mistaken for live workers. The
        state file is kept, so limits and budgets survive a restart."""
        self._secure_run_dir()
        if self._lock_fd is None:
            self._lock_run_dir()
        for sub in ("workers", "metrics"):
            path = self.run_dir / sub
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            for stale in path.iterdir():
                if stale.is_file() or stale.is_symlink():
                    stale.unlink(missing_ok=True)

    def _bind(self, host: str, port: int) -> list:
        import uvicorn

        sock = uvicorn.Config(RUNTIME_APP, host=host, port=port).bind_socket()
        return [sock]

    def _uvicorn_kwargs(self, **extra: Any) -> Dict[str, Any]:
        kwargs = {"log_level": self.log_level, "timeout_graceful_shutdown": self.graceful_timeout,
                  "proxy_headers": False}
        kwargs.update(extra)
        return kwargs

    # -- children -----------------------------------------------------------

    def _start(self, child: _Child) -> None:
        if child.index is None:
            process = self._ctx.Process(target=_ml_entry, name="znyx-ml",
                                        args=(self._ml_sockets, self.ml_env, self._uvicorn_kwargs()),
                                        daemon=False)
            with _environment_for_ml(self.ml_env):
                process.start()
        else:
            env = {**self.env, "ZNYX_WORKER_INDEX": str(child.index),
                   "ZNYX_SUPERVISOR_PID": str(os.getpid())}
            process = self._ctx.Process(target=_worker_entry, name=child.name,
                                        args=(self._sockets, env, self._uvicorn_kwargs()),
                                        daemon=False)
            process.start()
        child.process = process
        child.started_at = time.monotonic()
        child.restart_due = None
        child.last_seq = None
        child.ready = False
        child.warned = False
        logger.info("znyx: started %s (pid %s)", child.name, process.pid)

    def _read_beat(self, child: _Child) -> Optional[int]:
        """The liveness counter in the worker's record, if the record is this process's
        and readable. A torn or missing read is "no news", never "dead"."""
        try:
            data = json.loads((self.workers_dir / f"w{child.index}.json").read_text())
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or data.get("pid") != child.process.pid:
            return None
        seq = data.get("seq")
        return seq if isinstance(seq, int) else None

    def _check_ml_ready(self, child: _Child, now: float) -> None:
        if child.ready or now - self._ml_checked_at < 2.0:
            return
        self._ml_checked_at = now
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.ml_port}/healthz",  # noqa: S310
                                        timeout=0.3) as resp:
                if resp.status == 200:
                    child.ready = True
                    logger.info("znyx: model server ready on 127.0.0.1:%s", self.ml_port)
        except OSError:
            if now - child.started_at > ML_READY_TIMEOUT_SECONDS and not child.warned:
                logger.warning("znyx: model server not ready after %.0fs; checks use rules "
                               "until it is", now - child.started_at)
                child.warned = True

    def _check(self, child: _Child, now: float) -> None:
        process = child.process
        if child.restart_due is not None:
            if now >= child.restart_due:
                self._start(child)
            return
        if process.is_alive():
            if child.index is None:
                self._check_ml_ready(child, now)
                return
            if self.worker_timeout <= 0:
                return
            seq = self._read_beat(child)
            if seq is not None and seq != child.last_seq:
                child.last_seq, child.last_seen = seq, now
            if child.last_seq is None:
                silent_since, limit = child.started_at, max(BOOT_GRACE_SECONDS, self.worker_timeout)
            else:
                silent_since, limit = child.last_seen, self.worker_timeout
            if now - silent_since > limit:
                logger.error("znyx: %s (pid %s) stopped reporting for %.0fs; restarting it",
                             child.name, process.pid, now - silent_since)
                process.kill()
                process.join(5)
            else:
                return
        # The child has exited without being asked to.
        code = process.exitcode
        if child.index is None:
            # The model server is optional: the instance keeps serving with rules and keeps
            # trying to bring it back, but never gives up on the whole instance because of it.
            child.restart_times = [t for t in child.restart_times if now - t < RESTART_WINDOW_SECONDS]
            child.restart_times.append(now)
            delay = min(ML_MAX_BACKOFF_SECONDS, 0.5 * 2 ** (len(child.restart_times) - 1))
            logger.warning("znyx: model server exited (code %s); checks use rules; restarting "
                           "in %.1fs", code, delay)
            child.restart_due = now + delay
            return
        child.restart_times = [t for t in child.restart_times if now - t < RESTART_WINDOW_SECONDS]
        child.restart_times.append(now)
        if len(child.restart_times) > MAX_RESTARTS_IN_WINDOW:
            logger.error("znyx: %s exited %d times within %.0fs (last exit code %s); giving up",
                         child.name, len(child.restart_times), RESTART_WINDOW_SECONDS, code)
            self.exit_code = 1
            self._stop.set()
            return
        delay = min(10.0, 0.5 * 2 ** (len(child.restart_times) - 1))
        logger.warning("znyx: %s exited (code %s); restarting in %.1fs", child.name, code, delay)
        child.restart_due = now + delay

    # -- lifecycle ----------------------------------------------------------

    def request_stop(self, *_args: Any) -> None:
        self._stop.set()

    def _install_signals(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self.request_stop)

    def run(self) -> int:
        try:
            self._prepare_run_dir()
        except (RunDirError, OSError) as exc:
            logger.error("znyx: %s", exc)
            return 1
        try:
            self._sockets = self._bind(self.host, self.port)
        except (OSError, SystemExit) as exc:
            logger.error("znyx: cannot listen on %s:%s (%s)", self.host, self.port, exc)
            self._release_lock()
            return 1
        if self.ml_port:
            # Bound here, before anything starts, for the same reason as the main port: a
            # port already in use fails now with a clear message, and no other process
            # can be listening where the workers send text for the model checks.
            try:
                self._ml_sockets = self._bind("127.0.0.1", self.ml_port)
            except (OSError, SystemExit) as exc:
                logger.error("znyx: cannot start the model server on 127.0.0.1:%s (%s); "
                             "set ml.port to a free port", self.ml_port, exc)
                self._close_sockets(self._sockets)
                self._release_lock()
                return 1
        self._install_signals()
        if self.ml_port:
            ml = _Child(name="ml", index=None)
            self._children.append(ml)
            self._start(ml)
        for index in range(self.workers):
            if self._stop.is_set():
                break
            child = _Child(name=f"worker-{index}", index=index)
            self._children.append(child)
            self._start(child)
        logger.info("znyx: serving on http://%s:%s with %d worker%s%s", self.host, self.port,
                    self.workers, "" if self.workers == 1 else "s",
                    " and the model server" if self.ml_port else "")
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                for child in self._children:
                    if self._stop.is_set():
                        break
                    self._check(child, now)
                self._stop.wait(TICK_SECONDS)
        finally:
            self._shutdown()
        return self.exit_code

    def _close_sockets(self, sockets: list) -> None:
        for sock in sockets:
            try:
                sock.close()
            except OSError:
                pass

    def _shutdown(self) -> None:
        workers = [c for c in self._children if c.index is not None]
        others = [c for c in self._children if c.index is None]
        for group, sockets in ((workers, self._sockets), (others, self._ml_sockets)):
            live = [c for c in group if c.process is not None and c.process.is_alive()]
            for child in live:
                child.process.terminate()
            # The workers close their copies of the listening socket as they start to
            # drain; closing ours too means new connections are refused at once instead
            # of waiting in the backlog until everything has stopped.
            self._close_sockets(sockets)
            deadline = time.monotonic() + self.graceful_timeout + 5
            for child in live:
                child.process.join(max(0.1, deadline - time.monotonic()))
                if child.process.is_alive():
                    logger.warning("znyx: %s did not stop in time; killing it", child.name)
                    child.process.kill()
                    child.process.join(5)
        self._release_lock()
        logger.info("znyx: stopped")

    def _release_lock(self) -> None:
        if self._lock_fd is not None:
            try:
                os.close(self._lock_fd)
            except OSError:
                pass
            self._lock_fd = None


def read_worker_status(run_dir: Path) -> List[Dict[str, Any]]:
    """Liveness records the workers write (``znyx status`` and the dashboard)."""
    out = []
    folder = run_dir / "workers"
    if not folder.is_dir():
        return out
    for path in sorted(folder.glob("w*.json")):
        try:
            data = json.loads(path.read_text())
            data["age_seconds"] = round(time.time() - path.stat().st_mtime, 1)
            out.append(data)
        except (OSError, ValueError):
            continue
    return out
