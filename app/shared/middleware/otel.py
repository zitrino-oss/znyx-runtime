"""OpenTelemetry correlation middleware.

Reads W3C ``traceparent`` header, creates spans per evaluation and per detector.
All dependencies are optional - if ``opentelemetry`` is not installed, the
middleware degrades to a lightweight pass-through that propagates trace context
via request state.

Environment variables:
    OTEL_ENABLED              - "true" to enable (default "false")
    OTEL_SERVICE_NAME         - service name (default "guardrails")
    OTEL_EXPORTER_OTLP_ENDPOINT - OTLP endpoint URL
"""
import os
import re
import logging
from typing import Optional, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger(__name__)

# W3C traceparent format: version-trace_id-parent_id-trace_flags
_TRACEPARENT_RE = re.compile(
    r"^([0-9a-f]{2})-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$"
)

# Optional OpenTelemetry imports
_otel_available = False
_tracer = None

try:
    if os.getenv("OTEL_ENABLED", "false").lower() == "true":
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({
            "service.name": os.getenv("OTEL_SERVICE_NAME", "guardrails"),
        })
        provider = TracerProvider(resource=resource)

        # Try OTLP exporter if endpoint is configured
        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if otlp_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                provider.add_span_processor(BatchSpanProcessor(exporter))
                logger.info(f"OTel OTLP exporter configured: {otlp_endpoint}")
            except ImportError:
                logger.warning("opentelemetry-exporter-otlp not installed; spans won't be exported")

        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("guardrails")
        _otel_available = True
        logger.info("OpenTelemetry tracing enabled")
except ImportError:
    logger.debug("opentelemetry SDK not installed - OTel middleware disabled")
except Exception as e:
    logger.warning(f"Failed to initialize OpenTelemetry: {e}")


def parse_traceparent(header: str) -> Optional[Tuple[str, str, str, str]]:
    """Parse W3C traceparent header into (version, trace_id, parent_id, flags)."""
    match = _TRACEPARENT_RE.match(header.strip())
    if match:
        return match.group(1), match.group(2), match.group(3), match.group(4)
    return None


def get_tracer():
    """Return the OTel tracer or None if not available."""
    return _tracer


def is_otel_enabled() -> bool:
    return _otel_available


class OTelMiddleware(BaseHTTPMiddleware):
    """Extracts W3C traceparent and propagates trace context.

    If OpenTelemetry SDK is available, creates a server span per request.
    Otherwise, just parses the header and stores trace info in request.state.
    """

    async def dispatch(self, request: Request, call_next):
        traceparent = request.headers.get("traceparent", "")
        parsed = parse_traceparent(traceparent) if traceparent else None

        # Store trace context in request.state for downstream use
        if parsed:
            request.state.otel_trace_id = parsed[1]
            request.state.otel_parent_id = parsed[2]
            request.state.otel_trace_flags = parsed[3]
        else:
            request.state.otel_trace_id = None
            request.state.otel_parent_id = None
            request.state.otel_trace_flags = None

        if _otel_available and _tracer:
            return await self._dispatch_with_span(request, call_next, parsed)

        response = await call_next(request)

        # Echo traceparent back if we received one
        if parsed:
            response.headers["traceparent"] = traceparent

        return response

    async def _dispatch_with_span(self, request, call_next, parsed):
        """Create a server span wrapping the request."""
        from opentelemetry import trace, context
        from opentelemetry.trace import SpanKind, StatusCode

        span_name = f"{request.method} {request.url.path}"
        with _tracer.start_as_current_span(span_name, kind=SpanKind.SERVER) as span:
            if parsed:
                span.set_attribute("trace.parent_id", parsed[2])
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.url", str(request.url))

            response = await call_next(request)

            span.set_attribute("http.status_code", response.status_code)
            if response.status_code >= 400:
                span.set_status(StatusCode.ERROR)

            # Propagate trace context in response
            current_span = trace.get_current_span()
            ctx = current_span.get_span_context()
            if ctx and ctx.trace_id:
                trace_id_hex = format(ctx.trace_id, "032x")
                span_id_hex = format(ctx.span_id, "016x")
                response.headers["traceparent"] = f"00-{trace_id_hex}-{span_id_hex}-01"

            return response


def create_detector_span(detector_name: str):
    """Create a child span for a detector execution.

    Returns a context manager span, or a no-op if OTel is unavailable.

    Usage:
        with create_detector_span("pii") as span:
            result = detector.detect(text)
            if span:
                span.set_attribute("detector.decision", result.decision)
    """
    if _otel_available and _tracer:
        return _tracer.start_as_current_span(
            f"detector.{detector_name}",
            attributes={"detector.name": detector_name},
        )
    return _NoOpSpanContext()


class _NoOpSpanContext:
    """No-op context manager when OTel is not available."""
    def __enter__(self):
        return None
    def __exit__(self, *args):
        pass
