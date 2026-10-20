"""The ``znyx.yaml`` config file.

One file configures the whole install. Precedence, highest first: command-line
flags, then environment variables, then the file, then the defaults of the image or
package it runs in (``ZNYX_DEFAULTS``, a file in the same format: the container image
uses it to listen on 0.0.0.0, for example), then built-in defaults. The
services read environment variables, so the supervisor turns the effective
settings into those variables for its child processes. An install configured only
with environment variables (every existing deployment) therefore behaves exactly as
before, and a value set in the environment always beats the file.

String values may reference the environment as ``${NAME}`` or ``${NAME:-default}``,
so secrets need not be written into the file. Unknown settings are an error, not a
silent no-op: a misspelt ``fail_mode`` must not quietly leave the default in place.
"""
from __future__ import annotations

import difflib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

CONFIG_FILENAME = "znyx.yaml"
CONFIG_VERSION = 1

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


class ConfigError(ValueError):
    """The configuration cannot be used as given. The message says what to fix."""


@dataclass(frozen=True)
class Setting:
    key: str                     # dotted path in znyx.yaml
    env: Tuple[str, ...]         # canonical environment variable first, then legacy names
    kind: str                    # str | int | port | bool | workers | choice | json
    default: Any = None
    choices: Tuple[str, ...] = ()
    secret: bool = False
    help: str = ""


SETTINGS: Tuple[Setting, ...] = (
    Setting("server.host", ("HOST",), "str", "127.0.0.1",
            help="Address to listen on. The container image sets 0.0.0.0."),
    Setting("server.port", ("PORT",), "port", 8080, help="The one port for every API."),
    Setting("server.workers", ("ZNYX_WORKERS",), "workers", "auto",
            help="Worker processes sharing the port: a number, or auto (one per CPU, up to 4)."),
    Setting("server.graceful_timeout", ("ZNYX_GRACEFUL_TIMEOUT",), "int", 30,
            help="Seconds to let in-flight requests finish on shutdown."),
    Setting("server.worker_timeout", ("ZNYX_WORKER_TIMEOUT",), "int", 60,
            help="Restart a worker that has not reported for this many seconds."),
    Setting("server.run_dir", ("ZNYX_RUN_DIR",), "str", "",
            help="Where workers share state, metrics and liveness files."),
    Setting("server.allowed_origins", ("ALLOWED_ORIGINS",), "str", "",
            help="Comma-separated CORS origins."),
    Setting("mode", ("ZNYX_MODE", "GUARDRAILS_MODE"), "choice", "local", ("local", "managed"),
            help="local: policies from a file. managed: signed bundles from the control plane."),
    Setting("policy", ("ZNYX_POLICY_PATH", "GUARDRAILS_POLICY_PATH", "POLICY_PATH"), "str", "",
            help="Policy file (local mode). Empty: ./policies.yaml, else the built-in starter policy."),
    Setting("fail_mode", ("ZNYX_FAIL_MODE", "GUARDRAILS_FAIL_MODE"), "choice", "closed",
            ("open", "closed"), help="What happens when no policy is available."),
    Setting("control_plane.url", ("ZNYX_CONTROL_PLANE_URL", "GUARDRAILS_CONTROL_PLANE_URL"), "str", ""),
    Setting("control_plane.token", ("ZNYX_RUNTIME_TOKEN", "GUARDRAILS_RUNTIME_TOKEN"), "str", "",
            secret=True),
    Setting("control_plane.bundle_public_key",
            ("ZNYX_BUNDLE_PUBLIC_KEY", "GUARDRAILS_BUNDLE_PUBLIC_KEY"), "str", ""),
    Setting("control_plane.require_signed_bundles",
            ("ZNYX_REQUIRE_SIGNED_BUNDLES", "GUARDRAILS_REQUIRE_SIGNED_BUNDLES"), "bool", None),
    Setting("control_plane.poll_interval",
            ("ZNYX_BUNDLE_POLL_INTERVAL", "GUARDRAILS_BUNDLE_POLL_INTERVAL"), "int", None),
    Setting("auth.require", ("RUNTIME_REQUIRE_AUTH",), "bool", None,
            help="Require an API key on every evaluate call."),
    Setting("auth.api_key", ("RUNTIME_API_KEY",), "str", "", secret=True,
            help="Empty: znyx serve creates one on first run and keeps it in auth.api_key_file."),
    Setting("auth.api_key_file", ("ZNYX_API_KEY_FILE",), "str", "",
            help="Where the generated key is kept (default ~/.znyx/api-key)."),
    Setting("ml.enabled", ("ZNYX_ML",), "bool", False,
            help="Start the ML model server alongside the workers (loopback only)."),
    Setting("ml.port", ("ZNYX_ML_PORT",), "port", 9000, help="Loopback port of the model server."),
    Setting("ml.artifacts_dir", ("ZNYX_INFERENCE_ARTIFACTS_DIR",), "str", "",
            help="Where model weights live."),
    Setting("ml.url", ("ZNYX_INFERENCE_URL",), "str", "",
            help="Use a model server you run elsewhere instead of starting one."),
    Setting("state.backend", ("ZNYX_STATE_BACKEND",), "choice", "auto",
            ("auto", "memory", "sqlite", "redis"),
            help="Where limits and budgets live. auto: shared on-disk store when there is more "
                 "than one worker, memory otherwise. Use redis across replicas."),
    Setting("state.redis_url", ("ZNYX_REDIS_URL",), "str", "", secret=True),
    Setting("state.fail_mode", ("ZNYX_STATE_FAIL_MODE",), "choice", "local", ("local", "closed")),
    Setting("telemetry.enabled", ("ZNYX_TELEMETRY_ENABLED", "GUARDRAILS_TELEMETRY_ENABLED"), "bool", None,
            help="Per-evaluation telemetry to your control plane (managed mode default: on)."),
    Setting("telemetry.heartbeat", ("ZNYX_TELEMETRY", "GUARDRAILS_TELEMETRY"), "bool", False,
            help="Anonymous daily install heartbeat to Zitrino. Off unless switched on."),
    Setting("dashboard.enabled", ("ZNYX_DASHBOARD",), "bool", True,
            help="The local dashboard at /dashboard."),
    Setting("gateway.config", ("ZNYX_GATEWAY_CONFIG",), "str", "",
            help="The AI gateway's providers and keys (local mode). Empty: ./gateway.yaml if "
                 "present, else providers from OPENAI_API_KEY / ANTHROPIC_API_KEY."),
    Setting("airgapped", ("ZNYX_AIRGAPPED",), "bool", False,
            help="No outbound calls: no external model or judge calls, no heartbeat, "
                 "no model downloads."),
    Setting("airgapped_allow", ("ZNYX_AIRGAPPED_ALLOW",), "str", "",
            help="Air-gapped: hosts inside your network that model and judge calls may still "
                 "reach (comma-separated), beyond loopback, private addresses and "
                 "single-label or cluster-internal names."),
    Setting("log_level", ("ZNYX_LOG_LEVEL",), "choice", "info",
            ("critical", "error", "warning", "info", "debug")),
)

_BY_KEY: Dict[str, Setting] = {s.key: s for s in SETTINGS}

# File paths: relative ones written in a config file mean "next to that file", not
# "wherever the command happens to run".
PATH_KEYS = frozenset({"policy", "auth.api_key_file", "server.run_dir", "ml.artifacts_dir",
                       "gateway.config"})


def _relative_to(value: Any, key: str, base: Path) -> Any:
    if key in PATH_KEYS and isinstance(value, str) and value and not value.startswith("~") \
            and not Path(value).is_absolute():
        return os.path.normpath(str(base / value))
    return value

# Environment variables that came from a .env file rather than the real environment
# (filled in by the CLI), so the effective settings can say so.
DOTENV_NAMES: set = set()
_SECTIONS = {s.key.split(".")[0] for s in SETTINGS if "." in s.key}


@dataclass
class Settings:
    values: Dict[str, Any]
    sources: Dict[str, str]
    config_file: Optional[Path] = None
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse(setting: Setting, raw: Any, origin: str) -> Any:
    where = f"{setting.key} ({origin})"
    kind = setting.kind
    if raw is None:
        return None
    if kind == "str":
        if isinstance(raw, (dict, list)):
            raise ConfigError(f"{where}: expected text, got {type(raw).__name__}")
        return str(raw)
    if kind == "bool":
        if isinstance(raw, bool):
            return raw
        text = str(raw).strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        raise ConfigError(f"{where}: expected true or false, got {raw!r}")
    if kind in ("int", "port"):
        if isinstance(raw, bool):
            raise ConfigError(f"{where}: expected a number, got {raw!r}")
        try:
            value = int(str(raw).strip())
        except ValueError:
            raise ConfigError(f"{where}: expected a number, got {raw!r}") from None
        if kind == "port" and not 1 <= value <= 65535:
            raise ConfigError(f"{where}: port must be 1 to 65535, got {value}")
        if kind == "int" and value < 0:
            raise ConfigError(f"{where}: must not be negative, got {value}")
        return value
    if kind == "workers":
        text = str(raw).strip().lower()
        if text == "auto":
            return "auto"
        try:
            value = int(text)
        except ValueError:
            raise ConfigError(f"{where}: expected a number or auto, got {raw!r}") from None
        if value < 1:
            raise ConfigError(f"{where}: need at least 1 worker, got {value}")
        return value
    if kind == "choice":
        text = str(raw).strip().lower()
        if text not in setting.choices:
            raise ConfigError(f"{where}: expected one of {', '.join(setting.choices)}, got {raw!r}")
        return text
    if kind == "json":
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except ValueError as exc:
                raise ConfigError(f"{where}: not valid JSON ({exc})") from None
        return raw
    raise AssertionError(f"unknown setting kind {kind}")


_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _interpolate(value: Any, env: Mapping[str, str], path: str) -> Any:
    if isinstance(value, str):
        def sub(match: "re.Match[str]") -> str:
            name, default = match.group(1), match.group(2)
            if name in env:
                return env[name]
            if default is not None:
                return default
            raise ConfigError(f"{path}: environment variable {name} is not set "
                              f"(write ${{{name}:-default}} to give a fallback)")
        return _REF.sub(sub, value)
    if isinstance(value, dict):
        return {k: _interpolate(v, env, f"{path}.{k}" if path else str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v, env, path) for v in value]
    return value


def _flatten(data: Mapping[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in data.items():
        key = str(key)
        if key == "version":
            continue
        if key in _SECTIONS:
            if value is None:
                continue
            if not isinstance(value, dict):
                raise ConfigError(f"{key}: expected a section of settings, got {value!r}")
            for sub, sub_value in value.items():
                flat[f"{key}.{sub}"] = sub_value
        else:
            flat[key] = value
    return flat


def _unknown(key: str) -> ConfigError:
    near = difflib.get_close_matches(key, list(_BY_KEY), n=1, cutoff=0.6)
    hint = f" (did you mean {near[0]}?)" if near else ""
    return ConfigError(f"unknown setting {key}{hint}")


def read_file(path: Path, env: Mapping[str, str]) -> Dict[str, Any]:
    """The file's settings as ``{dotted key: raw value}``, references resolved."""
    import yaml

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping of settings at the top level")
    version = data.get("version", CONFIG_VERSION)
    if version != CONFIG_VERSION:
        raise ConfigError(f"{path}: version {version!r} is not supported (this release reads "
                          f"version {CONFIG_VERSION})")
    flat = _flatten(_interpolate(data, env, ""))
    for key in flat:
        if key not in _BY_KEY:
            raise _unknown(key)
    return flat


def find_config_file(explicit: Optional[str], env: Mapping[str, str],
                     cwd: Optional[Path] = None) -> Optional[Path]:
    """``--config``, else ``$ZNYX_CONFIG`` (both must exist), else ./znyx.yaml if present."""
    for origin, value in (("--config", explicit), ("ZNYX_CONFIG", env.get("ZNYX_CONFIG"))):
        if value:
            path = Path(value).expanduser()
            if not path.is_file():
                raise ConfigError(f"config file not found: {path} (from {origin})")
            return path.resolve()
    candidate = (cwd or Path.cwd()) / CONFIG_FILENAME
    return candidate.resolve() if candidate.is_file() else None


def load_settings(config_path: Optional[str] = None,
                  overrides: Optional[Mapping[str, Any]] = None,
                  env: Optional[Mapping[str, str]] = None,
                  cwd: Optional[Path] = None) -> Settings:
    """Effective settings from defaults, the file, the environment and ``overrides``
    (dotted keys, from command-line flags), in rising order of precedence."""
    env = os.environ if env is None else env
    values: Dict[str, Any] = {s.key: s.default for s in SETTINGS}
    sources: Dict[str, str] = {s.key: "default" for s in SETTINGS}

    # Defaults of the environment Znyx runs in (the container image), below the
    # user's own file: an image must never override what the operator configured.
    defaults_file = env.get("ZNYX_DEFAULTS")
    if defaults_file:
        defaults_path = Path(defaults_file).expanduser()
        if not defaults_path.is_file():
            raise ConfigError(f"ZNYX_DEFAULTS points at a missing file: {defaults_path}")
        for key, raw in read_file(defaults_path, env).items():
            values[key] = _relative_to(_parse(_BY_KEY[key], raw, str(defaults_path.name)), key,
                                       defaults_path.resolve().parent)
            sources[key] = "image defaults"

    path = find_config_file(config_path, env, cwd)
    if path is not None:
        for key, raw in read_file(path, env).items():
            values[key] = _relative_to(_parse(_BY_KEY[key], raw, str(path.name)), key, path.parent)
            sources[key] = "file"

    for setting in SETTINGS:
        for name in setting.env:
            if name in env:
                raw = env[name]
                # An empty variable (VAR= from a template) means "not set" for anything
                # but text, rather than a parse error that stops the install.
                if setting.kind != "str" and not str(raw).strip():
                    break
                values[setting.key] = _parse(setting, raw, name)
                sources[setting.key] = f".env {name}" if name in DOTENV_NAMES else f"env {name}"
                break

    for key, raw in (overrides or {}).items():
        if raw is None:
            continue
        if key not in _BY_KEY:
            raise _unknown(key)
        values[key] = _parse(_BY_KEY[key], raw, "command line")
        sources[key] = "command line"

    settings = Settings(values=values, sources=sources, config_file=path)
    _validate(settings)
    return settings


def _validate(settings: Settings) -> None:
    v = settings.values
    if v["mode"] == "managed":
        if not v["control_plane.url"]:
            raise ConfigError("managed mode needs control_plane.url (ZNYX_CONTROL_PLANE_URL)")
        if not v["control_plane.token"]:
            raise ConfigError("managed mode needs control_plane.token (ZNYX_RUNTIME_TOKEN)")
    if v["state.backend"] == "redis" and not v["state.redis_url"]:
        raise ConfigError("state.backend redis needs state.redis_url (ZNYX_REDIS_URL)")
    if v["ml.enabled"] and v["ml.url"]:
        raise ConfigError("set either ml.enabled (start the model server here) or ml.url "
                          "(use one that runs elsewhere), not both")
    if v["ml.enabled"] and v["ml.port"] == v["server.port"]:
        raise ConfigError(f"ml.port and server.port are both {v['server.port']}")
    if v["airgapped"] and v["telemetry.heartbeat"]:
        raise ConfigError("airgapped and telemetry.heartbeat contradict each other: an air-gapped "
                          "install makes no outbound calls")


# ---------------------------------------------------------------------------
# Derived values
# ---------------------------------------------------------------------------

MAX_AUTO_WORKERS = 4


def available_cpus() -> int:
    """CPUs this process may use, honouring a container's CPU quota (cgroup v2 or
    v1), which ``os.cpu_count()`` does not."""
    count = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        try:
            count = len(os.sched_getaffinity(0)) or count
        except OSError:
            pass
    quota = cgroup_cpu_quota()
    if quota is not None:
        count = max(1, min(count, quota))
    return count


def cgroup_cpu_quota(root: Path = Path("/sys/fs/cgroup")) -> Optional[int]:
    """CPUs allowed by the container's CPU quota (rounded up), or None for no quota."""
    import math
    try:
        text = (root / "cpu.max").read_text().split()
        if text and text[0] != "max":
            return max(1, math.ceil(int(text[0]) / int(text[1])))
        return None
    except (OSError, ValueError, IndexError, ZeroDivisionError):
        pass
    try:
        quota = int((root / "cpu" / "cpu.cfs_quota_us").read_text())
        period = int((root / "cpu" / "cpu.cfs_period_us").read_text())
        if quota > 0 and period > 0:
            return max(1, math.ceil(quota / period))
    except (OSError, ValueError):
        pass
    return None


def worker_count(settings: Settings) -> int:
    workers = settings["server.workers"]
    if workers == "auto":
        return max(1, min(available_cpus(), MAX_AUTO_WORKERS))
    return int(workers)


def state_backend(settings: Settings, workers: int) -> str:
    backend = settings["state.backend"]
    if backend == "auto":
        return "sqlite" if workers > 1 else "memory"
    if backend == "memory" and workers > 1:
        settings.notes.append(
            f"state.backend memory with {workers} workers: each worker keeps its own limits, "
            "so rate limits and budgets are that many times looser. Use auto or sqlite.")
    return backend


def run_dir(settings: Settings) -> Path:
    """Where this instance's workers share state. Per host and port by default, so two
    instances sharing a home directory or a volume (replicas mounting one /data) never
    share one; the supervisor also refuses to start on a directory another instance holds."""
    configured = settings["server.run_dir"]
    if configured:
        return Path(configured).expanduser().resolve()
    import re
    import socket
    host = re.sub(r"[^A-Za-z0-9._-]", "_", socket.gethostname() or "host")[:64]
    return (Path.home() / ".znyx" / "run" / f"{host}-{settings['server.port']}").resolve()


def starter_policy_path() -> Path:
    from importlib.resources import files
    return Path(str(files("znyx") / "templates" / "policies.yaml"))


def resolve_policy(settings: Settings, cwd: Optional[Path] = None) -> Optional[str]:
    """The policy file a local-mode install uses. An explicit path is used as given
    (the runtime refuses to start if it is missing; a typo must not silently swap in
    another policy). With none configured: ./policies.yaml, ./config/policies.yaml,
    else the built-in starter policy."""
    if settings["mode"] != "local":
        return settings["policy"] or None
    if settings["policy"]:
        return str(Path(settings["policy"]).expanduser())
    base = cwd or Path.cwd()
    for candidate in (base / "policies.yaml", base / "config" / "policies.yaml"):
        if candidate.is_file():
            return str(candidate.resolve())
    settings.notes.append("No policy file configured or found: using the built-in starter "
                          "policy. Run `znyx init` to write one you can edit.")
    return str(starter_policy_path())


def service_env(settings: Settings, workers: int, *, policy_path: Optional[str],
                run_directory: Path, platform_version: str) -> Dict[str, str]:
    """Environment variables for the child processes: every configured value under
    its canonical name, plus the values derived from the supervisor's own layout."""
    env: Dict[str, str] = {}
    supervisor_only = {"server.workers", "server.graceful_timeout", "server.worker_timeout",
                       "server.run_dir", "ml.enabled", "ml.port", "ml.url", "state.backend",
                       "policy", "log_level", "auth.api_key_file"}
    for setting in SETTINGS:
        if setting.key in supervisor_only:
            continue
        value = settings[setting.key]
        if value is None or value == "":
            continue
        env[setting.env[0]] = _to_env(value)
    if policy_path:
        env["ZNYX_POLICY_PATH"] = policy_path
    backend = state_backend(settings, workers)
    env["ZNYX_STATE_BACKEND"] = backend
    if backend == "sqlite":
        env["ZNYX_STATE_SQLITE_PATH"] = str(run_directory / "state.sqlite3")
    if settings["ml.enabled"]:
        env["ZNYX_INFERENCE_URL"] = f"http://127.0.0.1:{settings['ml.port']}"
    elif settings["ml.url"]:
        env["ZNYX_INFERENCE_URL"] = settings["ml.url"]
    if settings["airgapped"]:
        # Defence in depth for libraries that would otherwise reach the internet.
        env["ZNYX_TELEMETRY"] = "false"
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    env["ZNYX_RUN_DIR"] = str(run_directory)
    env["ZNYX_SKIP_DOTENV"] = "1"            # the supervisor already read .env
    env["ZNYX_WORKER_COUNT"] = str(workers)
    env["ZNYX_PLATFORM_VERSION"] = platform_version
    return env


def _to_env(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def describe(settings: Settings) -> List[Tuple[str, str, str]]:
    """``(key, shown value, source)`` rows for ``znyx config``, secrets masked."""
    rows = []
    for setting in SETTINGS:
        value = settings[setting.key]
        if setting.secret and value:
            shown = "***" if len(str(value)) < 8 else str(value)[:4] + "***"
        elif value is None or value == "":
            shown = "(not set)"
        else:
            shown = _to_env(value)
        rows.append((setting.key, shown, settings.sources[setting.key]))
    return rows


def iter_settings() -> Iterable[Setting]:
    return iter(SETTINGS)


# ---------------------------------------------------------------------------
# The API key
# ---------------------------------------------------------------------------

def api_key_path(settings: Settings) -> Path:
    configured = settings["auth.api_key_file"]
    return Path(configured).expanduser() if configured else Path.home() / ".znyx" / "api-key"


def needs_api_key(settings: Settings, env: Optional[Mapping[str, str]] = None) -> bool:
    """The runtime requires a key unless auth is switched off outside production
    (production always requires one; see znyx_runtime.api.routes)."""
    if settings["auth.require"] is not False:
        return True
    env = os.environ if env is None else env
    from znyx_core.utils.env import ENV_VARS, _NONPROD
    return not any((env.get(v) or "").strip().lower() in _NONPROD for v in ENV_VARS)


def ensure_api_key(settings: Settings, rotate: bool = False) -> Tuple[str, Path, bool]:
    """The configured key, else the one kept in the key file, else a new one written
    there (owner-only). Returns ``(key, file, created)``."""
    import secrets

    path = api_key_path(settings)
    if settings["auth.api_key"]:
        # A configured key is the operator's; it is never replaced by a generated one,
        # even on rotate (rotate it where it is configured).
        return settings["auth.api_key"], path, False
    if path.is_file() and not rotate:
        key = path.read_text(encoding="utf-8").strip()
        if key:
            return key, path, False
    key = "zx_" + secrets.token_urlsafe(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(key + "\n")
    os.chmod(path, 0o600)
    return key, path, True

