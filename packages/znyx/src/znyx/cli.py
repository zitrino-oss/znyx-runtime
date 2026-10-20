"""The ``znyx`` command.

    znyx serve                 # evaluate requests on http://127.0.0.1:8080
    znyx serve --ml            # also start the ML model server (loopback only)
    znyx init                  # write znyx.yaml and policies.yaml to edit
    znyx config                # the effective settings and where each came from
    znyx status                # the running workers
    znyx key                   # this install's API key (--rotate for a new one)
    znyx gateway-key --id APP  # a gateway key for an app or agent (printed once)
    znyx version               # one platform version (and its components)
    znyx models list | fetch   # vetted model weights, fetched explicitly and sha256-pinned
    znyx benchmark ...         # certify a model-backed detector (see `znyx benchmark -h`)

``znyx start`` is accepted as another name for ``serve`` (the npm installer's README
has always used it).
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from znyx import __version__, component_versions
from znyx.config import (
    CONFIG_FILENAME,
    ConfigError,
    describe,
    ensure_api_key,
    load_settings,
    needs_api_key,
    resolve_policy,
    run_dir,
    service_env,
    starter_policy_path,
    worker_count,
)

logger = logging.getLogger("znyx")


def _add_config_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", metavar="PATH",
                        help=f"Config file (default: ./{CONFIG_FILENAME} if present, or $ZNYX_CONFIG).")


def _add_serve_flags(parser: argparse.ArgumentParser) -> None:
    _add_config_flag(parser)
    parser.add_argument("--host", help="Address to listen on (default 127.0.0.1).")
    parser.add_argument("--port", type=int, help="Port for every API (default 8080).")
    parser.add_argument("--workers", help="Worker processes: a number, or auto.")
    parser.add_argument("--policy", metavar="PATH", help="Policy file (local mode).")
    parser.add_argument("--mode", choices=("local", "managed"))
    ml = parser.add_mutually_exclusive_group()
    ml.add_argument("--ml", dest="ml", action="store_true", default=None,
                    help="Start the ML model server too.")
    ml.add_argument("--no-ml", dest="ml", action="store_false")
    parser.add_argument("--ml-port", type=int, help="Loopback port of the model server (default 9000).")
    parser.add_argument("--airgapped", action="store_true", default=None,
                        help="Make no outbound calls (no external models or judges, no heartbeat).")
    parser.add_argument("--log-level", choices=("critical", "error", "warning", "info", "debug"))
    parser.add_argument("--reload", action="store_true",
                        help="Development: one process that reloads on code changes.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="znyx", description="Znyx: guardrails for every AI call.")
    parser.add_argument("-V", "--version", action="store_true", help="Print the version and exit.")
    sub = parser.add_subparsers(dest="command", metavar="command")

    for name in ("serve", "start"):
        _add_serve_flags(sub.add_parser(
            name, help="Run Znyx." if name == "serve" else "Another name for serve."))

    init = sub.add_parser("init", help=f"Write a {CONFIG_FILENAME} and a policies.yaml to edit.")
    init.add_argument("--output", "-o", default=".", metavar="DIR", help="Directory (default: current).")
    init.add_argument("--force", action="store_true", help="Overwrite existing files.")

    cfg = sub.add_parser("config", help="Show the effective settings and where each came from.")
    _add_config_flag(cfg)
    cfg.add_argument("--json", action="store_true")

    status = sub.add_parser("status", help="Show the running workers.")
    _add_config_flag(status)
    status.add_argument("--json", action="store_true")

    keycmd = sub.add_parser("key", help="Print this install's API key (or make a new one).")
    _add_config_flag(keycmd)
    keycmd.add_argument("--rotate", action="store_true",
                        help="Replace the generated key (restart znyx serve to use it).")

    gkey = sub.add_parser("gateway-key", help="Create a gateway key for an app or agent (local mode).")
    gkey.add_argument("--id", required=True, help="A name for the key, e.g. support-bot.")
    gkey.add_argument("--app", help="App id (default: the key id).")
    gkey.add_argument("--agent", help="Agent id, for an agent key.")
    gkey.add_argument("--tenant", default="default")
    gkey.add_argument("--providers", help="Comma-separated provider names (default: all).")
    gkey.add_argument("--models", help="Comma-separated approved models or patterns (default: any).")

    ver = sub.add_parser("version", help="Print the platform version and its components.")
    ver.add_argument("--json", action="store_true")

    models = sub.add_parser("models", help="List or fetch vetted model weights.")
    msub = models.add_subparsers(dest="models_command", metavar="action")
    mlist = msub.add_parser("list", help="List vetted models.")
    mlist.add_argument("--task")
    mlist.add_argument("--open-only", action="store_true", help="Only OSI-open licences.")
    mfetch = msub.add_parser("fetch", help="Download, export and sha256-pin one model.")
    _add_config_flag(mfetch)
    mfetch.add_argument("--task", required=True)
    mfetch.add_argument("--model-id")
    mfetch.add_argument("--revision")
    mfetch.add_argument("--allow-unvetted", action="store_true")

    from znyx_runtime.benchmark import add_arguments as _benchmark_args
    _benchmark_args(sub.add_parser(
        "benchmark", help="Benchmark a model-backed detector and stamp its scorecard."))
    return parser


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _serve_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "server.host": args.host, "server.port": args.port, "server.workers": args.workers,
        "policy": args.policy, "mode": args.mode, "ml.enabled": args.ml, "ml.port": args.ml_port,
        "airgapped": args.airgapped, "log_level": args.log_level,
    }


def load_dotenv_once() -> Optional[Path]:
    """Read ``./.env`` (the current directory only, never a parent's: a .env two levels
    up belongs to some other project, and could switch off auth) into the environment,
    without overriding variables already set. Each variable it supplies is recorded, so
    `znyx config` labels it ".env" rather than plain environment. The workers are told
    not to read it again, so every process sees the same settings."""
    import os

    from znyx import config as _config

    path = Path.cwd() / ".env"
    if not path.is_file():
        return None
    try:
        from dotenv import dotenv_values
    except ImportError:  # pragma: no cover - python-dotenv ships with znyx-runtime
        return None
    for name, value in dotenv_values(path).items():
        if value is not None and name not in os.environ:
            os.environ[name] = value
            _config.DOTENV_NAMES.add(name)
    return path


def cmd_serve(args: argparse.Namespace) -> int:
    import os

    dotenv = load_dotenv_once()
    settings = load_settings(args.config, _serve_overrides(args))
    level = settings["log_level"]
    logging.basicConfig(level=getattr(logging, level.upper()), format="%(message)s", stream=sys.stderr)
    if dotenv is not None:
        from_dotenv = sorted(k for k, src in settings.sources.items() if src.startswith(".env"))
        logger.warning("znyx: read %s%s", dotenv,
                       f"; it sets {', '.join(from_dotenv)}" if from_dotenv else "")
    workers = 1 if args.reload else worker_count(settings)
    directory = run_dir(settings)
    policy = resolve_policy(settings)
    env = service_env(settings, workers, policy_path=policy, run_directory=directory,
                      platform_version=__version__)
    key_source = None
    if needs_api_key(settings):
        key, key_file, created = ensure_api_key(settings)
        env["RUNTIME_API_KEY"] = key
        key_source = (settings.sources["auth.api_key"] if settings["auth.api_key"]
                      else ("new", key_file) if created else ("file", key_file))
    if settings.config_file:
        logger.info("znyx %s: config %s", __version__, settings.config_file)
    else:
        logger.info("znyx %s: no %s, using defaults and environment variables", __version__,
                    CONFIG_FILENAME)
    if policy and settings["mode"] == "local":
        logger.info("znyx: policy %s", policy)
    for note in settings.notes:
        logger.warning("znyx: %s", note)
    _print_start(settings, workers, env.get("RUNTIME_API_KEY"), key_source)

    if args.reload:
        if settings["ml.enabled"]:
            logger.warning("znyx: --reload runs the runtime only; the model server is not started")
        os.environ.update(env)
        os.environ["ZNYX_WORKER_INDEX"] = "0"
        import uvicorn
        uvicorn.run("znyx_runtime.main:app", host=settings["server.host"],
                    port=settings["server.port"], reload=True, log_level=level)
        return 0

    from znyx.supervisor import Supervisor
    # The model server reads only its own settings; the supervisor strips every
    # credential from its environment (it must hold none).
    ml_env = {"PORT": str(settings["ml.port"]),
              "ZNYX_INFERENCE_ARTIFACTS_DIR": (settings["ml.artifacts_dir"]
                                               or os.environ.get("ZNYX_INFERENCE_ARTIFACTS_DIR")
                                               or str(Path.home() / ".znyx" / "models"))}
    ml_env.update({k: env[k] for k in ("ZNYX_AIRGAPPED", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                                       "ZNYX_PLATFORM_VERSION") if k in env})
    # Only the workers on this machine may call it: refuse any other Host header, which
    # is what a DNS-rebinding page in a local browser would send.
    ml_env["ZNYX_INFERENCE_TRUSTED_HOSTS"] = "127.0.0.1,localhost"
    supervisor = Supervisor(
        host=settings["server.host"], port=settings["server.port"], workers=workers, env=env,
        run_dir=directory, ml_port=settings["ml.port"] if settings["ml.enabled"] else None,
        log_level=level, graceful_timeout=settings["server.graceful_timeout"],
        worker_timeout=settings["server.worker_timeout"], ml_env=ml_env,
    )
    return supervisor.run()


_ZNYX_YAML = """\
# Znyx configuration. Every setting is optional; `znyx config` shows the effective
# value of each and where it came from. Environment variables override this file,
# and command-line flags override both. Reference secrets as ${NAME} rather than
# writing them here.
version: 1

server:
  host: 127.0.0.1        # 0.0.0.0 to accept connections from other machines
  port: 8080             # the one port for every API
  workers: auto          # worker processes sharing the port (auto: one per CPU, up to 4)

mode: local              # local: policies.yaml below. managed: signed bundles from the console.
policy: ./policies.yaml
fail_mode: closed        # with no policy available, block rather than allow

# control_plane:         # managed mode
#   url: https://console.example.com
#   token: ${ZNYX_RUNTIME_TOKEN}

# auth:                  # require an API key on every evaluate call
#   require: true
#   api_key: ${RUNTIME_API_KEY}

ml:
  enabled: false         # true: start the ML model server alongside (needs `pip install "znyx[ml]"`)
  port: 9000             # loopback only, never exposed

state:
  backend: auto          # auto, memory, sqlite, or redis (shared across replicas)
  # redis_url: ${ZNYX_REDIS_URL}

telemetry:
  heartbeat: false       # anonymous daily install ping to Zitrino; off unless you switch it on

dashboard:
  enabled: true          # http://127.0.0.1:8080/dashboard

airgapped: false         # true: no outbound calls at all
"""


def _print_start(settings, workers: int, key: Optional[str], key_source) -> None:
    """Where to point a browser and a first request. The key itself is never printed: a
    terminal is often a container's log stream too (`docker run -t`, `tty: true`).
    `znyx key` shows it on request."""
    host = settings["server.host"]
    shown = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    base = f"http://{shown}:{settings['server.port']}"
    lines = [f"znyx {__version__} starting on {base} ({workers} worker{'s' if workers != 1 else ''}"
             f"{', model server' if settings['ml.enabled'] else ''})"]
    if isinstance(key_source, tuple):
        kind, path = key_source
        lines.append(f"  API key    {'created and saved in' if kind == 'new' else 'from'} {path}"
                     "  (show it with: znyx key)")
        header = f'-H "X-API-Key: $(cat {path})"'
    elif key_source:
        lines.append(f"  API key    from {key_source}")
        header = '-H "X-API-Key: $RUNTIME_API_KEY"'
    else:
        lines.append("  API key    not required (auth is off outside production)")
        header = ""
    if settings["dashboard.enabled"]:
        lines.append(f"  Dashboard  {base}/dashboard  (sign in with the API key)")
    body = ('{"request_id":"1","tenant_id":"local","app_id":"demo",'
            '"text":"My card is 4111 1111 1111 1111"}')
    curl = f"curl -s {base}/v1/evaluate/input"
    if header:
        curl += f" {header}"
    curl += f" -H 'Content-Type: application/json' -d '{body}'"
    lines.append(f"  Try it     {curl}")
    print("\n".join(lines), file=sys.stderr, flush=True)


def cmd_key(args: argparse.Namespace) -> int:
    load_dotenv_once()
    settings = load_settings(args.config)
    if settings["auth.api_key"]:
        print(f"the API key comes from {settings.sources['auth.api_key']}; "
              f"{'change it there' if args.rotate else 'znyx does not store it'}", file=sys.stderr)
        return 1
    key, path, created = ensure_api_key(settings, rotate=args.rotate)
    print(key)
    print(f"({'new key saved in' if created else 'from'} {path})", file=sys.stderr)
    return 0


def cmd_gateway_key(args: argparse.Namespace) -> int:
    """A new gateway key: printed once, with the gateway.yaml entry that holds only
    its digest (the key itself never needs to be written anywhere)."""
    import hashlib
    import secrets

    import yaml

    key = "zxg_" + secrets.token_urlsafe(32)
    entry: Dict[str, Any] = {"id": args.id, "key_sha256": hashlib.sha256(key.encode()).hexdigest(),
                             "tenant_id": args.tenant, "app_id": args.app or args.id}
    if args.agent:
        entry["agent_id"] = args.agent
    if args.providers:
        entry["providers"] = [p.strip() for p in args.providers.split(",") if p.strip()]
    if args.models:
        entry["allowed_models"] = [m.strip() for m in args.models.split(",") if m.strip()]
    print(key)
    print("\nGive this key to the app (it is shown only now). Add this under `keys:` in "
          "gateway.yaml:\n", file=sys.stderr)
    print(yaml.safe_dump([entry], sort_keys=False).rstrip(), file=sys.stderr)
    return 0


_GATEWAY_YAML = """\
# The AI gateway: point an app's SDK at this instance instead of the provider.
#   OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="<gateway key>")
#   Anthropic(base_url="http://127.0.0.1:8080", api_key="<gateway key>")
# Every call is checked by the policy, governed by its key and recorded.
# Provider keys are references (env:NAME or file:/path), never the key itself.
version: 1

providers:
  openai:
    format: openai
    base_url: https://api.openai.com/v1
    api_key: env:OPENAI_API_KEY
    region: us
  anthropic:
    format: anthropic
    base_url: https://api.anthropic.com
    api_key: env:ANTHROPIC_API_KEY
    region: us

# One key per app or agent: `znyx gateway-key --id support-bot` prints a key and the
# entry to paste here. Without keys, the instance's own API key works for everything.
keys: []
#  - id: support-bot
#    key_sha256: <from znyx gateway-key>
#    app_id: support-bot
#    providers: [openai]
#    allowed_models: [gpt-4o-mini]
#    limits: {requests_per_minute: 600, tokens_per_day: 2000000, spend_usd_per_day: 50}

# prices:                 # override or add list prices (US dollars per million tokens)
#   my-finetune: {input_per_mtok: 0.30, output_per_mtok: 1.20}
"""


def cmd_init(args: argparse.Namespace) -> int:
    out = Path(args.output).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    targets = {out / CONFIG_FILENAME: _ZNYX_YAML, out / "policies.yaml": None,
               out / "gateway.yaml": _GATEWAY_YAML}
    existing = [p for p in targets if p.exists()]
    if existing and not args.force:
        for path in existing:
            print(f"{path} already exists (use --force to overwrite)", file=sys.stderr)
        return 1
    for path, text in targets.items():
        if text is None:
            shutil.copyfile(starter_policy_path(), path)
        else:
            path.write_text(text, encoding="utf-8")
        print(f"wrote {path}")
    print(f"\nNext: cd {out} && znyx serve")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    load_dotenv_once()
    settings = load_settings(args.config)
    rows = describe(settings)
    if args.json:
        print(json.dumps({"config_file": str(settings.config_file) if settings.config_file else None,
                          "settings": {k: {"value": v, "source": s} for k, v, s in rows}}, indent=2))
        return 0
    print(f"config file: {settings.config_file or '(none)'}")
    width = max(len(k) for k, _, _ in rows)
    for key, value, source in rows:
        print(f"  {key:<{width}}  {value:<28}  {source}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    load_dotenv_once()
    from znyx.supervisor import read_worker_status

    settings = load_settings(args.config)
    workers = read_worker_status(run_dir(settings))
    if args.json:
        print(json.dumps(workers, indent=2))
        return 0 if workers else 1
    if not workers:
        print(f"no running workers found for port {settings['server.port']}")
        return 1
    for w in workers:
        print(f"worker {w.get('index')}: pid {w.get('pid')}, up {w.get('uptime_seconds', '?')}s, "
              f"last report {w.get('age_seconds')}s ago, {w.get('requests', 0)} requests")
    return 0


def cmd_version(args: argparse.Namespace) -> int:
    components = component_versions()
    if args.json:
        print(json.dumps({"version": __version__, "components": components}, indent=2))
        return 0
    print(f"znyx {__version__}")
    for dist, ver in components.items():
        print(f"  {dist} {ver or '(not installed)'}")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    if args.models_command is None:
        print("usage: znyx models {list,fetch} ...", file=sys.stderr)
        return 2
    try:
        from znyx_inference.runners._fetch import fetch_model, list_candidates, resolve_fetch_target
    except ImportError as exc:
        print(f"model tooling is not installed ({exc}); pip install \"znyx[export]\"", file=sys.stderr)
        return 2
    if args.models_command == "list":
        rows = list_candidates(task=args.task, open_only=args.open_only)
        for r in rows:
            print(f"{r.get('task', '?'):<16} {r.get('model_id', '?'):<55} {r.get('license', '?')}")
        return 0 if rows else 1
    settings = load_settings(args.config)
    if settings["airgapped"]:
        print("this install is air-gapped: fetch weights on a connected machine and mount them "
              "into ml.artifacts_dir", file=sys.stderr)
        return 2
    try:
        target = resolve_fetch_target(args.task, model_id=args.model_id, revision=args.revision,
                                      allow_unvetted=args.allow_unvetted)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"fetching {target['model_id']}@{target['revision']} for {target['task']} "
          f"-> {target['dest_dir']}", file=sys.stderr)
    digest = fetch_model(target["model_id"], target["revision"], target["dest_dir"],
                         runner=target["runner"])
    print(f"model_id: {target['model_id']}\nrevision: {target['revision']}\nsha256:   {digest}\n"
          f"dir:      {target['dest_dir']}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from znyx_runtime.benchmark import run_benchmark
    return run_benchmark(args)


COMMANDS = {"serve": cmd_serve, "start": cmd_serve, "init": cmd_init, "config": cmd_config,
            "status": cmd_status, "version": cmd_version, "models": cmd_models,
            "benchmark": cmd_benchmark, "key": cmd_key, "gateway-key": cmd_gateway_key}


def main(argv: Optional[List[str]] = None) -> None:
    import multiprocessing

    multiprocessing.freeze_support()     # no-op unless running as a frozen binary
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        print(f"znyx {__version__}")
        sys.exit(0)
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    try:
        sys.exit(COMMANDS[args.command](args))
    except ConfigError as exc:
        print(f"znyx: configuration error: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
