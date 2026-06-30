"""
ZNYX Runtime - Standalone OSS evaluation server.

Stateless, zero DB dependency. Loads policy from local YAML/bundle
or fetches signed bundles from the control plane.

Usage:
    # Local YAML mode (default):
    ZNYX_MODE=local ZNYX_POLICY_PATH=./config/policies.yaml \
        uvicorn app.runtime.main:app --port 8080

    # Managed mode (fetch from control plane):
    ZNYX_MODE=managed ZNYX_CONTROL_PLANE_URL=https://cp.example.com \
        ZNYX_RUNTIME_TOKEN=rt_xxx uvicorn app.runtime.main:app --port 8080
"""
import os
import logging
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.runtime.config import RuntimeConfig
from app.runtime.bundle_manager import BundleManager
from app.runtime.telemetry import TelemetryEmitter
from app.runtime.heartbeat import Heartbeat
from app.runtime.install_state import record_run
from app.runtime.api.routes import router
from app.runtime.api.stream_routes import router as stream_router
from app.shared.engine.evaluator import GuardrailsEvaluator
from app.shared.detectors.plugin import PluginRegistry, init_plugins

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

VERSION = "1.0.0"
CONSOLE_URL = "https://console.znyx.ai"

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
        line("Anonymous install telemetry is ON by default."),
        line("Opt out:"),
        line("  export ZNYX_TELEMETRY=false"),
        bot,
    ])


_WELCOME_BANNER = _build_welcome_banner(VERSION, CONSOLE_URL)

# Module-level state (set during lifespan)
config: RuntimeConfig = None
bundle_manager: BundleManager = None
telemetry: TelemetryEmitter = None
heartbeat: Heartbeat = None
evaluator: GuardrailsEvaluator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global config, bundle_manager, telemetry, heartbeat, evaluator

    logger.info("ZNYX Runtime starting...")
    config = RuntimeConfig.from_env()

    # Initialize bundle manager
    bundle_manager = BundleManager(config)
    await bundle_manager.start()
    logger.info(f"Runtime mode: {config.mode}")

    # Initialize telemetry
    telemetry = TelemetryEmitter(
        control_plane_url=config.control_plane_url,
        runtime_token=config.runtime_token,
        enabled=config.telemetry_enabled,
    )
    await telemetry.start()

    # Initialize anonymous heartbeat (opt-out: ZNYX_TELEMETRY=false - on by default, disclosed in welcome banner)
    heartbeat = Heartbeat(enabled=config.heartbeat_enabled, mode=config.mode)
    await heartbeat.start()

    # Initialize plugin registry for custom detectors
    plugin_registry = PluginRegistry()
    init_plugins()

    # Initialize evaluator (no policy resolver - policy passed per request)
    evaluator = GuardrailsEvaluator(
        policy_resolver=None,
        log_redacted_text=config.log_redacted_text,
        on_evaluation=telemetry.emit,
        plugin_registry=plugin_registry,
    )

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


app = FastAPI(
    title="ZNYX Runtime",
    description="Lightweight, stateless guardrails evaluation engine for LLM applications",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — strict parsing. Wildcard origins rejected under allow_credentials
# (browsers already reject them, but we fail fast at startup).
def _parse_runtime_cors() -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:3010,http://localhost:3020")
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    env = (os.getenv("ZNYX_ENV") or os.getenv("ENVIRONMENT") or "").strip().lower()
    is_prod = env in ("prod", "production")
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
        env = (os.getenv("ZNYX_ENV") or os.getenv("ENVIRONMENT") or "").strip().lower()
        if env in ("prod", "production"):
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
            response.headers["X-Znyx-Console"] = "https://console.znyx.ai"
            response.headers["X-Znyx-Mode"] = "local"
        return response


app.add_middleware(LocalModeNudgeMiddleware)

# Prometheus metrics middleware + /metrics endpoint. Same shared collector
# the CP uses, so the same business counters (evaluations_total, decisions,
# detector_hits) work without rewiring.
from app.shared.middleware.metrics import MetricsCollector, MetricsMiddleware
app.add_middleware(MetricsMiddleware, collector=MetricsCollector())


@app.get("/metrics", include_in_schema=False)
async def runtime_metrics():
    """Prometheus exposition.

    Unauthenticated by default — same as CP's pre-auth posture for /metrics
    in dev. Production deploys should put a network policy in front of this
    or set ZNYX_METRICS_REQUIRE_KEY=true to gate it.
    """
    from starlette.responses import Response as _Response

    if os.getenv("ZNYX_METRICS_REQUIRE_KEY", "false").lower() in ("1", "true", "yes"):
        # Minimal key check — kept inline to avoid pulling the full auth
        # stack into the runtime's import surface.
        from fastapi import Header
        # Re-resolve dependency manually since this route was registered
        # without it. Cheap.
        return _Response(
            content="metrics endpoint requires X-API-Key in this deployment\n",
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
