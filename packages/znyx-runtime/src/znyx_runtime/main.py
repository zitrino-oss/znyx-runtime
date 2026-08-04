"""
ZNYX Runtime - Standalone OSS evaluation server.

Stateless, zero DB dependency. Loads policy from local YAML/bundle
or fetches signed bundles from the control plane.

Usage:
    # Local YAML mode (default):
    ZNYX_MODE=local ZNYX_POLICY_PATH=./config/policies.yaml \
        uvicorn znyx_runtime.main:app --port 8080

    # Managed mode (fetch from control plane):
    ZNYX_MODE=managed ZNYX_CONTROL_PLANE_URL=https://cp.example.com \
        ZNYX_RUNTIME_TOKEN=rt_xxx uvicorn znyx_runtime.main:app --port 8080
"""
import os
import logging
from contextlib import asynccontextmanager
from typing import Callable, Optional

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file if present

from fastapi import FastAPI, Header, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from znyx_runtime.config import RuntimeConfig
from znyx_runtime.bundle_manager import BundleManager
from znyx_runtime.telemetry import TelemetryEmitter
from znyx_runtime.heartbeat import Heartbeat
from znyx_runtime.install_state import record_run
from znyx_runtime.api.routes import router
from znyx_runtime.api.stream_routes import router as stream_router
from znyx_core.engine.evaluator import GuardrailsEvaluator
from znyx_core.detectors.plugin import PluginRegistry, init_plugins

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

VERSION = "1.0.0"
CONSOLE_URL = os.getenv("ZNYX_CONSOLE_URL", "")

def _build_welcome_banner(version: str, console_url: str, inner_width: int = 54) -> str:
    """Build a fixed-width box banner. Each text row is `║ <padded content> ║`."""
    top = "╔" + "═" * inner_width + "╗"
    bot = "╚" + "═" * inner_width + "╝"
    blank = "║" + " " * inner_width + "║"

    def line(text: str) -> str:
        return "║ " + text.ljust(inner_width - 2) + " ║"

    return "\n".join([
        top,
        line(f"Welcome to ZNYX AI Runtime v{version}"),
        blank,
        line("Running in LOCAL mode - policies from YAML file."),
        line("Your data never leaves this machine."),
        blank,
        line("Want a policy editor, traces, and analytics?"),
        line(f"-> {console_url}"),
        blank,
        line("No telemetry by default - never phones home."),
        line("Opt in to anonymous install heartbeats:"),
        line("  export ZNYX_TELEMETRY=true"),
        line("  (ZNYX_HEARTBEAT_URL overrides the destination)"),
        bot,
    ])


_WELCOME_BANNER = _build_welcome_banner(VERSION, CONSOLE_URL)

# Module-level state (set during lifespan)
config: RuntimeConfig = None
bundle_manager: BundleManager = None
telemetry: TelemetryEmitter = None
heartbeat: Heartbeat = None
evaluator: GuardrailsEvaluator = None
runtime_judge = None  # RuntimeJudgeAudit — durable judge audit + cached budget


@asynccontextmanager
async def lifespan(app: FastAPI):
    global config, bundle_manager, telemetry, heartbeat, evaluator, runtime_judge

    logger.info("ZNYX Runtime starting...")
    config = RuntimeConfig.from_env()

    # Initialize bundle manager
    bundle_manager = BundleManager(config)

    # Desired-state fan-out. The runtime is the only component that talks to the control
    # plane, so on every bundle cycle it (a) hands the sidecar the model pins the bundle
    # declares and (b) reports the active bundle plus what the sidecar loaded. Registered
    # BEFORE start() so the boot cycle already carries them.
    from znyx_runtime.inference_sync import InferenceSync
    from znyx_runtime.runtime_report import RuntimeReporter

    inference_sync = InferenceSync()
    runtime_reporter = RuntimeReporter(
        control_plane_url=config.control_plane_url,
        runtime_token=config.runtime_token,
        mode=config.mode,
    )

    async def _on_bundle_cycle(policy) -> None:
        # Order matters: push first so the report carries the sidecar's state as of this
        # cycle rather than the previous one.
        await inference_sync.push(policy)
        await runtime_reporter.report(
            bundle_manager.bundle_info,
            sidecar_models=inference_sync.reported_models,
            sidecar_version=inference_sync.sidecar_version,
        )

    bundle_manager.add_cycle_listener(_on_bundle_cycle)
    app.state.inference_sync = inference_sync
    app.state.runtime_reporter = runtime_reporter

    await bundle_manager.start()
    logger.info(f"Runtime mode: {config.mode}")

    # Initialize telemetry
    telemetry = TelemetryEmitter(
        control_plane_url=config.control_plane_url,
        runtime_token=config.runtime_token,
        enabled=config.telemetry_enabled,
    )
    await telemetry.start()

    # Initialize anonymous heartbeat (opt-in: set ZNYX_TELEMETRY=true - off by default)
    heartbeat = Heartbeat(enabled=config.heartbeat_enabled, mode=config.mode)
    await heartbeat.start()

    # Initialize plugin registry for custom detectors
    plugin_registry = PluginRegistry()
    init_plugins()

    # Egress audit: durable, fail-closed spool sink the escalation gate writes to
    # BEFORE any boundary-crossing call (no DB in the runtime → spool; the control
    # plane drains it / telemetry ships it). A failed audit write denies the egress.
    from znyx_runtime.audit_sink import make_audit_egress_sink, make_audit_sink
    audit_sink = make_audit_sink(
        mode=config.audit_sink_mode,
        fail_mode=config.audit_fail_mode,
        spool_path=config.audit_spool_path or None,
    )
    egress_sink = make_audit_egress_sink(audit_sink)

    # Initialize evaluator (no policy resolver - policy passed per request)
    evaluator = GuardrailsEvaluator(
        policy_resolver=None,
        log_redacted_text=config.log_redacted_text,
        on_evaluation=telemetry.emit,
        plugin_registry=plugin_registry,
        egress_sink=egress_sink,
    )
    app.state.egress_audit_sink = audit_sink

    #: runtime-local judge audit (durable spool the CP drains) + cached
    # deny-of-wallet (bundle-delivered caps vs the runtime's own spend tally). Lets a
    # co-located judge run on the stateless path while still feeding the central audit
    # trail + budgets. Judges run unaudited/unbudgeted only if explicitly disabled.
    from znyx_runtime.judge_audit_sink import RuntimeJudgeAudit
    runtime_judge = RuntimeJudgeAudit(
        spool_path=config.judge_audit_spool_path or None,
        enabled=config.judge_audit_enabled,
    )
    app.state.runtime_judge = runtime_judge

    # Record this startup in shared install state (increments run_count).
    state = record_run(mode=config.mode)
    run_count = state["run_count"]

    # First-run welcome banner: shown exactly once per install (local mode).
    # Also fires a one-shot first_run telemetry ping (best-effort).
    if config.mode == "local" and run_count == 1:
        print(_WELCOME_BANNER)
        if config.heartbeat_enabled:
            try:
                await heartbeat.send_first_run_ping(run_count=run_count)
            except Exception:
                pass  # never block startup on telemetry

    # Upgrade nudge every 5th run in local mode
    if config.mode == "local" and run_count > 1 and run_count % 5 == 0:
        logger.info(
            f"[znyx] TIP: Run {run_count} - connect to the console for policy management "
            f"-> {CONSOLE_URL}"
        )

    logger.info("ZNYX Runtime ready")
    yield

    # Shutdown
    await heartbeat.stop()
    await telemetry.stop()
    await bundle_manager.stop()
    logger.info("ZNYX Runtime shut down")


# Hide the API docs + OpenAPI schema in prod (parity with the control plane).
# Uses the canonical is_production() helper (secure-by-default; reads all env
# families) so docs gating can't desync from the other prod checks.
from znyx_core.utils.env import is_production
_RT_PROD = is_production()

app = FastAPI(
    title="ZNYX Runtime",
    description="Lightweight, stateless guardrails evaluation engine for LLM applications",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None if _RT_PROD else "/docs",
    redoc_url=None if _RT_PROD else "/redoc",
    openapi_url=None if _RT_PROD else "/openapi.json",
)

# CORS — strict parsing. Wildcard origins rejected under allow_credentials
# (browsers already reject them, but we fail fast at startup).
def _parse_runtime_cors() -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "")
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    is_prod = is_production()
    bad: list[str] = []
    for o in origins:
        if o == "*":
            bad.append("wildcard '*' not allowed with credentials")
        elif is_prod and o.startswith("http://") and not o.startswith("http://localhost"):
            bad.append(f"{o} — production requires https:// origins")
    if bad:
        raise RuntimeError(
            "Invalid ALLOWED_ORIGINS for runtime CORS: " + "; ".join(bad)
        )
    return origins


ALLOWED_ORIGINS = _parse_runtime_cors()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
    max_age=600,
)


class RuntimeSecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Baseline security headers for the runtime. No CSP by default because
    the runtime serves JSON to SDKs — customers shouldn't be rendering it
    in a browser. Still sets HSTS + frame-deny + nosniff."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=()"
        )
        if is_production():
            response.headers["Strict-Transport-Security"] = (
                "max-age=63072000; includeSubDomains"
            )
        if not request.url.path.startswith(("/static", "/_next", "/docs", "/redoc")):
            response.headers["Cache-Control"] = "no-store"
        return response


app.add_middleware(RuntimeSecurityHeadersMiddleware)


class LocalModeNudgeMiddleware(BaseHTTPMiddleware):
    """Attach X-Znyx-Console and X-Znyx-Mode headers when running in local mode."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        if config and config.mode == "local":
            if CONSOLE_URL:
                response.headers["X-Znyx-Console"] = CONSOLE_URL
            response.headers["X-Znyx-Mode"] = "local"
        return response


app.add_middleware(LocalModeNudgeMiddleware)

# Prometheus metrics middleware + /metrics endpoint. Same shared collector
# the CP uses, so the same business counters (evaluations_total, decisions,
# detector_hits) work without rewiring.
from znyx_core.middleware.metrics import MetricsCollector, MetricsMiddleware
app.add_middleware(MetricsMiddleware, collector=MetricsCollector())


@app.get("/metrics", include_in_schema=False)
async def runtime_metrics(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None),
):
    """Prometheus exposition.

    Unauthenticated by default — same as CP's pre-auth posture for /metrics
    in dev. Production deploys should put a network policy in front of this
    or set ZNYX_METRICS_REQUIRE_KEY=true to require the runtime API key.
    """
    from starlette.responses import Response as _Response

    if os.getenv("ZNYX_METRICS_REQUIRE_KEY", "false").lower() in ("1", "true", "yes"):
        # Actually validate the key against RUNTIME_API_KEY (timing-safe), the
        # same credential the evaluation routes use. Accept X-API-Key or a
        # Bearer token. Kept inline to avoid the full auth dependency stack.
        import hmac as _hmac

        expected = os.getenv("RUNTIME_API_KEY", "").strip()
        token = x_api_key
        if not token and authorization and authorization.startswith("Bearer "):
            token = authorization[7:]
        if not expected or not token or not _hmac.compare_digest(token, expected):
            return _Response(
                content="metrics endpoint requires a valid X-API-Key in this deployment\n",
                status_code=401,
                media_type="text/plain",
            )
    body = MetricsCollector().format_prometheus()
    return _Response(content=body, media_type="text/plain; version=0.0.4")


app.include_router(router)
app.include_router(stream_router)


@app.get("/")
async def root():
    return {
        "service": "ZNYX Runtime",
        "version": "1.0.0",
        "mode": config.mode if config else "initializing",
        "endpoints": {
            "evaluation": "/v1/evaluate/{input|output|tool}",
            "streaming": "/v1/evaluate/stream",
            "health": "/healthz",
            "readiness": "/readyz",
            "bundle_status": "/v1/bundle/status",
            "docs": "/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
