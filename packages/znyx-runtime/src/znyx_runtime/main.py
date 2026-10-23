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
import asyncio
import json
import os
import logging
from contextlib import asynccontextmanager
from typing import Callable, Optional

from dotenv import load_dotenv
# A .env in the directory the service is started from, and only there. Searching from
# this module's own location, as before, read whatever .env sat above the installed
# package (an application's own .env when the runtime is installed in that
# application's virtual environment), and a parent directory's .env belongs to some
# other project; either could switch auth off. Under `znyx serve` the supervisor has
# already read it, so its workers do not read it again.
if not os.getenv("ZNYX_SKIP_DOTENV") and os.path.isfile(".env"):
    load_dotenv(".env")

from fastapi import FastAPI, Header, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from znyx_runtime import __version__
from znyx_runtime.console import SafeStreamHandler, safe_print
from znyx_runtime.sdk_registry import get_sdk_registry
from znyx_runtime.console import SafeStreamHandler, safe_print
from znyx_runtime.config import RuntimeConfig
from znyx_runtime.bundle_manager import BundleManager
from znyx_runtime.telemetry import TelemetryEmitter
from znyx_runtime.heartbeat import Heartbeat
from znyx_runtime.install_state import record_run
from znyx_runtime.api.routes import router
from znyx_runtime.api.stream_routes import router as stream_router
from znyx_runtime.api.dashboard_routes import router as dashboard_router
from znyx_runtime.gateway.routes import router as gateway_router
from znyx_runtime.workers import (
    RecentDecisions, WorkerReporter, instance_snapshot, is_leader, run_dir as worker_run_dir,
    worker_count, worker_index,
)
from znyx_core.engine.evaluator import GuardrailsEvaluator
from znyx_core.detectors.plugin import plugin_registry, init_plugins
from znyx_core.utils.env import is_airgapped

logging.basicConfig(level=logging.INFO, format="%(message)s",
                    handlers=[SafeStreamHandler()])
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

    rows = [
        top,
        line(f"Welcome to ZNYX AI Runtime v{version}"),
        blank,
        line("Running in LOCAL mode - policies from YAML file."),
        line("Your data never leaves this machine."),
    ]
    if console_url:
        rows += [
            blank,
            line("Policy editor, traces, and analytics:"),
            line(f"-> {console_url}"),
        ]
    rows += [
        blank,
        line("No telemetry by default - never phones home."),
        line("Opt in to anonymous install heartbeats:"),
        line("  export ZNYX_TELEMETRY=true"),
        line("  (ZNYX_HEARTBEAT_URL overrides the destination)"),
        bot,
    ]
    return "\n".join(rows)


_WELCOME_BANNER = _build_welcome_banner(VERSION, CONSOLE_URL)

# Module-level state (set during lifespan)
config: RuntimeConfig = None
bundle_manager: BundleManager = None
telemetry: TelemetryEmitter = None
heartbeat: Heartbeat = None
evaluator: GuardrailsEvaluator = None
runtime_judge = None  # RuntimeJudgeAudit — durable judge audit + cached budget
# The last decisions this worker made (metadata only), for the local dashboard.
recent_decisions = RecentDecisions()
worker_reporter: Optional[WorkerReporter] = None
otlp_exporter = None          # znyx_runtime.otel.OtlpExporter, when OTLP is configured


def _on_evaluation(event: dict) -> None:
    """Every evaluation's metadata: kept for the local dashboard, then queued for the
    control plane (when telemetry is on)."""
    try:
        recent_decisions.record(event)
    except Exception:  # noqa: BLE001 - the dashboard must never affect a decision
        pass
    if telemetry is not None:
        telemetry.emit(event)
    if otlp_exporter is not None:
        otlp_exporter.export_evaluation(event)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global config, bundle_manager, telemetry, heartbeat, evaluator, runtime_judge, worker_reporter
    global otlp_exporter

    # With several worker processes (znyx serve), only worker 0 runs the jobs that
    # must happen once per instance: delivering spools, the heartbeat, the benchmark
    # queue, model-pin pushes and control-plane reports. See workers.py.
    leader = is_leader()
    if worker_count() > 1:
        logger.info("ZNYX Runtime worker %d of %d starting...", worker_index(), worker_count())
    else:
        logger.info("ZNYX Runtime starting...")
    config = RuntimeConfig.from_env()
    # Managed mode: secret references arrive in bundles and may only name what this
    # side allows (znyx_core.secret_refs); local configuration is the operator's own.
    from znyx_core import secret_refs
    secret_refs.restrict_references(config.mode == "managed")

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
        # cycle rather than the previous one. One report per environment served.
        await inference_sync.push(policy)
        for info in bundle_manager.report_infos():
            await runtime_reporter.report(
                info,
                sidecar_models=inference_sync.reported_models,
                sidecar_version=inference_sync.sidecar_version,
            )

    if leader:
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
        spool_path=config.telemetry_spool_path or None,
        ship=leader,
    )
    await telemetry.start()

    # Initialize anonymous heartbeat (opt-in: set ZNYX_TELEMETRY=true - off by default).
    # Never from an air-gapped install, and once per instance.
    heartbeat = Heartbeat(enabled=config.heartbeat_enabled and leader and not is_airgapped(),
                          mode=config.mode)
    await heartbeat.start()

    # Initialize plugin registry for custom detectors. `plugin_registry` here is the
    # module-global instance init_plugins() populates (imported above, not a fresh
    # PluginRegistry()) -- there must be exactly one registry instance by the time the
    # evaluator is built below, or GUARDRAILS_PLUGIN_DIR custom detectors loaded by
    # init_plugins() would land in an object request-serving code never sees.
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
        on_evaluation=_on_evaluation,
        plugin_registry=plugin_registry,
        egress_sink=egress_sink,
    )
    app.state.egress_audit_sink = audit_sink

    # The AI gateway (OpenAI- and Anthropic-compatible endpoints). Configured from
    # gateway.yaml, or from OPENAI_API_KEY / ANTHROPIC_API_KEY with nothing else; an
    # explicit but broken configuration stops the start rather than serving without it.
    from znyx_runtime.gateway import config as gateway_config
    from znyx_runtime.gateway.pipeline import Gateway
    if config.mode == "managed":
        # The console's providers and keys arrive in the signed bundles (one per
        # environment served) and are re-read on every bundle cycle, in every worker.
        # Each environment's section stands alone: a broken one leaves only that
        # environment without a gateway.
        initial = gateway_config.from_bundle_members(bundle_manager.gateway_documents(),
                                                     shared=bundle_manager.serves_several)
    else:
        initial = gateway_config.load()
    # Usage and traces (Phase 3): each gateway call's usage event goes to a spool the
    # leader ships to the control plane (managed mode), and every call and check to the
    # customer's OTLP collector when one is configured. Metadata only.
    from znyx_runtime.otel import OtlpExporter
    from znyx_runtime.usage_sink import DEFAULT_USAGE_SPOOL, UsageSink
    otlp_exporter = OtlpExporter.from_env()
    if otlp_exporter is not None:
        otlp_exporter.start()
        logger.info("Traces exported over OTLP to %s", otlp_exporter.status()["endpoint_host"])
    app.state.otlp_exporter = otlp_exporter
    usage_spool = None
    if config.mode == "managed" and config.control_plane_url and config.runtime_token:
        usage_spool = os.getenv("ZNYX_USAGE_SPOOL_PATH") or str(DEFAULT_USAGE_SPOOL)
    app.state.usage_spool = usage_spool
    app.state.gateway = Gateway(initial, evaluator=evaluator, bundle_manager=bundle_manager,
                                egress_sink=egress_sink,
                                on_call=UsageSink(bundle_manager, usage_spool, otlp_exporter))
    if config.mode == "managed":
        gateway_state = {"doc": json.dumps([bundle_manager.serves_several, bundle_manager.gateway_documents()],
                                           sort_keys=True, default=str)}

        def _refresh_gateway() -> None:
            # Run by the bundle manager the moment its bundles change: a key disabled or
            # removed in the console stops working with the next fetch, in every worker.
            documents = bundle_manager.gateway_documents()
            doc = json.dumps([bundle_manager.serves_several, documents], sort_keys=True, default=str)
            if doc == gateway_state["doc"]:
                return
            app.state.gateway.update_config(gateway_config.from_bundle_members(
                documents, shared=bundle_manager.serves_several))
            gateway_state["doc"] = doc
            logger.info("Gateway configuration updated from the bundles")

        bundle_manager.add_change_listener(_refresh_gateway)
    if app.state.gateway.config.enabled:
        logger.info("Gateway: %s (%s)", ", ".join(sorted(app.state.gateway.config.providers)),
                    app.state.gateway.config.source)

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

    # Durable delivery of the two audit spools to the control plane (managed mode
    # only). They used to be read only by a control plane sharing this disk, so a
    # runtime in a customer's network never delivered them.
    audit_shippers = []
    if leader and config.mode == "managed" and config.control_plane_url and config.runtime_token:
        from znyx_runtime.shipper import SpoolShipper, http_post
        cp = config.control_plane_url.rstrip("/")
        sender = http_post(config.runtime_token)
        egress_path = getattr(audit_sink, "path", None)
        if egress_path is not None:
            audit_shippers.append(SpoolShipper(str(egress_path), f"{cp}/v1/runtime/audit/egress",
                                               sender, name="egress-audit"))
        judge_spool = getattr(runtime_judge, "spool", None)
        if judge_spool is not None and getattr(judge_spool, "path", None) is not None:
            audit_shippers.append(SpoolShipper(str(judge_spool.path), f"{cp}/v1/runtime/audit/judge",
                                               sender, name="judge-audit"))
        if app.state.usage_spool:
            audit_shippers.append(SpoolShipper(app.state.usage_spool, f"{cp}/v1/runtime/llm-calls",
                                               sender, name="llm-calls"))
        # Governance records (Phase 5 part 4): MCP tool calls, approvals, delegated tokens.
        audit_shippers.append(SpoolShipper(str(app.state.gateway.governance.path),
                                           f"{cp}/v1/runtime/audit/governance", sender, name="governance-audit"))
        for shipper in audit_shippers:
            shipper.start()
    app.state.audit_shippers = audit_shippers

    # Control-plane-queued benchmarks. Only this runtime can reach its own inference
    # sidecar, so a model-backed benchmark is evaluated here rather than in the control
    # plane (which must never dial a tenant-supplied URL and has no route into a customer
    # network). Given its own task deliberately, NOT a bundle-cycle listener: a
    # multi-minute run parked there would starve model-pin pushes and heartbeats, because
    # _spawn_cycle skips a tick while the previous one is still in flight.
    from znyx_runtime.benchmark_worker import BenchmarkWorker
    benchmark_worker = None
    if leader:
        benchmark_worker = BenchmarkWorker(
            control_plane_url=config.control_plane_url,
            runtime_token=config.runtime_token,
            evaluator=evaluator,
        )
        await benchmark_worker.start()
    app.state.benchmark_worker = benchmark_worker

    # Low-urgency gateway calls (technique 8): the leader submits them through the
    # providers' batch APIs and checks the answers as they return.
    batch_runner = None
    if leader:
        from znyx_runtime.gateway.batch import BatchRunner
        batch_runner = BatchRunner(app.state.gateway)
        batch_runner.start()
    app.state.batch_runner = batch_runner

    # Liveness and metrics for the supervisor and the dashboard (znyx serve only).
    directory = worker_run_dir()
    if directory is not None:
        from znyx_core.middleware.metrics import MetricsCollector as _Collector
        worker_reporter = WorkerReporter(_Collector(), recent_decisions, directory,
                                         version=__version__)
        await worker_reporter.start()

    # Record this startup in shared install state (increments run_count). Once per
    # instance start, not once per worker.
    run_count = record_run(mode=config.mode)["run_count"] if leader else 0

    # First-run welcome banner: shown exactly once per install (local mode).
    # Also fires a one-shot first_run telemetry ping (best-effort).
    if config.mode == "local" and run_count == 1:
        safe_print(_WELCOME_BANNER)
        if heartbeat.enabled:
            try:
                await heartbeat.send_first_run_ping(run_count=run_count)
            except Exception:
                pass  # never block startup on telemetry

    # Periodic reminder, only when the operator has configured a console URL.
    if config.mode == "local" and CONSOLE_URL and run_count > 1 and run_count % 5 == 0:
        logger.info(
            f"[znyx] TIP: Run {run_count} - connect to the console for policy management "
            f"-> {CONSOLE_URL}"
        )

    logger.info("ZNYX Runtime ready")
    yield

    # Shutdown
    gateway = getattr(app.state, "gateway", None)
    if gateway is not None:
        await gateway.close()
    if worker_reporter is not None:
        await worker_reporter.stop()
    if benchmark_worker is not None:
        await benchmark_worker.stop()
    await heartbeat.stop()
    await telemetry.stop()
    if otlp_exporter is not None:
        await otlp_exporter.close()
    if getattr(app.state, "batch_runner", None) is not None:
        await app.state.batch_runner.stop()
    for shipper in getattr(app.state, "audit_shippers", []):
        await shipper.stop(final_attempt=True)
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
    version=__version__,
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
    """Attach X-Znyx-Console and X-Znyx-Mode headers when running in local mode, and
    X-Znyx-Worker (which worker process answered) when there are several."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        if config and config.mode == "local":
            if CONSOLE_URL:
                response.headers["X-Znyx-Console"] = CONSOLE_URL
            response.headers["X-Znyx-Mode"] = "local"
        if worker_count() > 1:
            response.headers["X-Znyx-Worker"] = str(worker_index())
        return response


app.add_middleware(LocalModeNudgeMiddleware)


class SdkIdentificationMiddleware(BaseHTTPMiddleware):
    """Record the calling SDK's identity from X-Znyx-Sdk / X-Znyx-Sdk-Version.

    Middleware rather than an endpoint dependency because every evaluate route
    takes its pydantic body as `request` and holds no FastAPI Request; adding one
    to each would churn seven signatures and still miss the streaming routes.

    Headers are untrusted caller input - sanitised in the registry, used only as
    display metadata, never for policy. Wrapped so identification can never fail
    an evaluation.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            source = request.headers.get("x-znyx-sdk")
            if source:
                get_sdk_registry().record(source, request.headers.get("x-znyx-sdk-version"))
        except Exception:  # noqa: BLE001 - identification must never break a request
            pass
        return await call_next(request)


app.add_middleware(SdkIdentificationMiddleware)

# Prometheus metrics middleware + /metrics endpoint. Same shared collector
# the CP uses, so the same business counters (evaluations_total, decisions,
# detector_hits) work without rewiring.
from znyx_core.middleware.metrics import MetricsCollector, MetricsMiddleware
app.add_middleware(MetricsMiddleware, collector=MetricsCollector())

# Bodies are read before a route's authentication runs: capped in size and time.
from znyx_runtime.request_limits import BodyLimitMiddleware
app.add_middleware(BodyLimitMiddleware)

# The caller's trace context (traceparent), for the spans of evaluate-API checks.
from znyx_runtime.otel import TraceContextMiddleware
app.add_middleware(TraceContextMiddleware)


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
    collector = MetricsCollector()
    if worker_count() > 1:
        # Any worker may answer the scrape: report the whole instance, not one worker.
        snapshot = await asyncio.to_thread(instance_snapshot, collector, recent_decisions,
                                           worker_reporter)
        body = collector.format_prometheus(snapshot["metrics"])
    else:
        body = collector.format_prometheus()
    return _Response(content=body, media_type="text/plain; version=0.0.4")


app.include_router(router)
app.include_router(stream_router)
app.include_router(dashboard_router)
app.include_router(gateway_router)


@app.get("/")
async def root():
    return {
        "service": "ZNYX Runtime",
        "version": __version__,
        "platform_version": os.getenv("ZNYX_PLATFORM_VERSION") or __version__,
        "mode": config.mode if config else "initializing",
        "endpoints": {
            "evaluation": "/v1/evaluate/{input|output|tool}",
            "streaming": "/v1/evaluate/stream",
            "health": "/healthz",
            "readiness": "/readyz",
            "bundle_status": "/v1/bundle/status",
            "dashboard": "/dashboard",
            "docs": "/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
