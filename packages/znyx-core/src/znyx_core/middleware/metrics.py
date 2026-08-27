"""
Prometheus Metrics Middleware - lightweight, zero-dependency implementation.

Provides a custom MetricsCollector (singleton) that tracks counters, gauges,
and histograms, plus a Starlette middleware for HTTP request metrics.

Usage:
    from znyx_core.middleware.metrics import MetricsCollector, MetricsMiddleware

    collector = MetricsCollector()
    collector.counter_inc("guardrails_evaluations_total", labels={"context": "input", "decision": "ALLOW"})
"""
import time
import threading
import logging
from collections import defaultdict
from typing import Dict, Optional, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

# Default histogram buckets (seconds) - tuned for HTTP / evaluation latency
DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

# Label value that absorbs series once a metric hits its label-set cap
OVERFLOW_LABEL_VALUE = "other"


def _labels_key(labels: Optional[Dict[str, str]]) -> Tuple[Tuple[str, str], ...]:
    """Convert a labels dict to a hashable, sorted tuple of pairs."""
    if not labels:
        return ()
    return tuple(sorted(labels.items()))


class MetricsCollector:
    """Thread-safe singleton metrics collector.

    Supports three metric types:
      - counter  - monotonically increasing value
      - gauge    - value that can go up or down
      - histogram - observations bucketed by value
    """

    _instance: Optional["MetricsCollector"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "MetricsCollector":
        with cls._lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._initialized = False
                cls._instance = inst
            return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        self._mu = threading.Lock()

        # Cardinality guard: cap distinct label-sets per metric so unbounded
        # label values (raw URL paths, client-supplied ids) cannot grow
        # memory forever. Overflow collapses into a single "other" series.
        from znyx_core.config.tunables import METRICS_MAX_LABEL_SETS
        self._max_label_sets = METRICS_MAX_LABEL_SETS

        # counters: {name: {labels_key: float}}
        self._counters: Dict[str, Dict[Tuple, float]] = defaultdict(lambda: defaultdict(float))
        # gauges: {name: {labels_key: float}}
        self._gauges: Dict[str, Dict[Tuple, float]] = defaultdict(lambda: defaultdict(float))
        # histograms: {name: {labels_key: {"buckets": {bound: count}, "sum": float, "count": int}}}
        self._histograms: Dict[str, Dict[Tuple, dict]] = defaultdict(dict)
        # Track bucket definitions per histogram name
        self._histogram_buckets: Dict[str, Tuple[float, ...]] = {}
        # Metric help text
        self._help: Dict[str, str] = {}
        # Metric type
        self._type: Dict[str, str] = {}

        self._register_defaults()

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------

    def _register_defaults(self) -> None:
        """Pre-register all Guardrails metrics with help text."""
        self.register("guardrails_evaluations_total", "counter",
                      "Total number of guardrail evaluations")
        self.register("guardrails_evaluation_duration_seconds", "histogram",
                      "Duration of guardrail evaluations in seconds")
        self.register("guardrails_detector_hits_total", "counter",
                      "Total detector rule hits")
        self.register("guardrails_http_requests_total", "counter",
                      "Total HTTP requests")
        self.register("guardrails_http_request_duration_seconds", "histogram",
                      "HTTP request duration in seconds")
        self.register("guardrails_policy_cache_hits_total", "counter",
                      "Policy cache hits")
        self.register("guardrails_policy_cache_misses_total", "counter",
                      "Policy cache misses")
        self.register("guardrails_active_connections", "gauge",
                      "Number of active HTTP connections")

    def register(self, name: str, metric_type: str, help_text: str = "",
                 buckets: Optional[Tuple[float, ...]] = None) -> None:
        self._help[name] = help_text
        self._type[name] = metric_type
        if metric_type == "histogram":
            self._histogram_buckets[name] = buckets or DEFAULT_BUCKETS

    def _bounded_key(self, series: Dict[Tuple, object],
                     key: Tuple[Tuple[str, str], ...]) -> Tuple[Tuple[str, str], ...]:
        """Return ``key``, or an overflow key once the metric is at its cap.

        Existing series keep updating; only NEW label-sets beyond the cap are
        collapsed (every label value becomes ``other``). Must be called with
        ``self._mu`` held.
        """
        if key in series or len(series) < self._max_label_sets:
            return key
        return tuple((k, OVERFLOW_LABEL_VALUE) for k, _ in key)

    # ------------------------------------------------------------------
    # Counter
    # ------------------------------------------------------------------

    def counter_inc(self, name: str, value: float = 1.0,
                    labels: Optional[Dict[str, str]] = None) -> None:
        key = _labels_key(labels)
        with self._mu:
            series = self._counters[name]
            series[self._bounded_key(series, key)] += value

    # ------------------------------------------------------------------
    # Gauge
    # ------------------------------------------------------------------

    def gauge_set(self, name: str, value: float,
                  labels: Optional[Dict[str, str]] = None) -> None:
        key = _labels_key(labels)
        with self._mu:
            series = self._gauges[name]
            series[self._bounded_key(series, key)] = value

    def gauge_inc(self, name: str, value: float = 1.0,
                  labels: Optional[Dict[str, str]] = None) -> None:
        key = _labels_key(labels)
        with self._mu:
            series = self._gauges[name]
            series[self._bounded_key(series, key)] += value

    def gauge_dec(self, name: str, value: float = 1.0,
                  labels: Optional[Dict[str, str]] = None) -> None:
        key = _labels_key(labels)
        with self._mu:
            series = self._gauges[name]
            series[self._bounded_key(series, key)] -= value

    # ------------------------------------------------------------------
    # Histogram
    # ------------------------------------------------------------------

    def histogram_observe(self, name: str, value: float,
                          labels: Optional[Dict[str, str]] = None) -> None:
        key = _labels_key(labels)
        buckets = self._histogram_buckets.get(name, DEFAULT_BUCKETS)
        with self._mu:
            key = self._bounded_key(self._histograms[name], key)
            if key not in self._histograms[name]:
                self._histograms[name][key] = {
                    "buckets": {b: 0 for b in buckets},
                    "sum": 0.0,
                    "count": 0,
                }
            entry = self._histograms[name][key]
            entry["sum"] += value
            entry["count"] += 1
            # Increment only the individual bucket; cumulation happens at format time
            for b in sorted(buckets):
                if value <= b:
                    entry["buckets"][b] += 1
                    break

    # ------------------------------------------------------------------
    # Prometheus text exposition format
    # ------------------------------------------------------------------

    def format_prometheus(self) -> str:
        """Return all metrics in Prometheus text exposition format (0.0.4)."""
        lines: list = []

        with self._mu:
            # Counters
            for name, series in sorted(self._counters.items()):
                lines.append(f"# HELP {name} {self._help.get(name, '')}")
                lines.append(f"# TYPE {name} counter")
                for lk, val in sorted(series.items()):
                    lines.append(f"{name}{self._format_labels(lk)} {self._fmt_value(val)}")

            # Gauges
            for name, series in sorted(self._gauges.items()):
                lines.append(f"# HELP {name} {self._help.get(name, '')}")
                lines.append(f"# TYPE {name} gauge")
                for lk, val in sorted(series.items()):
                    lines.append(f"{name}{self._format_labels(lk)} {self._fmt_value(val)}")

            # Histograms
            for name, series in sorted(self._histograms.items()):
                lines.append(f"# HELP {name} {self._help.get(name, '')}")
                lines.append(f"# TYPE {name} histogram")
                for lk, entry in sorted(series.items()):
                    cumulative = 0
                    for bound in sorted(entry["buckets"].keys()):
                        cumulative += entry["buckets"][bound]
                        le_labels = dict(lk) if lk else {}
                        le_labels["le"] = self._fmt_value(bound)
                        le_key = _labels_key(le_labels)
                        lines.append(f"{name}_bucket{self._format_labels(le_key)} {cumulative}")
                    # +Inf bucket
                    inf_labels = dict(lk) if lk else {}
                    inf_labels["le"] = "+Inf"
                    inf_key = _labels_key(inf_labels)
                    lines.append(f"{name}_bucket{self._format_labels(inf_key)} {entry['count']}")
                    lines.append(f"{name}_sum{self._format_labels(lk)} {self._fmt_value(entry['sum'])}")
                    lines.append(f"{name}_count{self._format_labels(lk)} {entry['count']}")

        lines.append("")  # trailing newline
        return "\n".join(lines)

    @staticmethod
    def _format_labels(labels_key: Tuple[Tuple[str, str], ...]) -> str:
        if not labels_key:
            return ""
        inner = ",".join(f'{k}="{v}"' for k, v in labels_key)
        return "{" + inner + "}"

    @staticmethod
    def _fmt_value(v: float) -> str:
        if v == int(v):
            return str(int(v))
        return f"{v:.6g}"

    # ------------------------------------------------------------------
    # Reset (for testing)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all collected metrics. Intended for testing."""
        with self._mu:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


# ------------------------------------------------------------------
# HTTP Metrics Middleware
# ------------------------------------------------------------------

class MetricsMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that records HTTP request metrics.

    Tracked:
      - guardrails_http_requests_total   (counter)
      - guardrails_http_request_duration_seconds (histogram)
      - guardrails_active_connections    (gauge)
    """

    def __init__(self, app, collector: Optional[MetricsCollector] = None) -> None:
        super().__init__(app)
        self.collector = collector or MetricsCollector()

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip metrics endpoint itself to avoid self-referential noise
        if request.url.path == "/metrics":
            return await call_next(request)

        method = request.method

        self.collector.gauge_inc("guardrails_active_connections")
        start = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            self.collector.gauge_dec("guardrails_active_connections")
            raise

        duration = time.perf_counter() - start
        status_code = str(response.status_code)

        # Label by the matched route template ("/items/{item_id}") rather than
        # the raw path, so path parameters do not create a series per value.
        # Routing has run by now, so the scope carries the matched route (set
        # by FastAPI). Unmatched requests fall back to the raw path, which the
        # collector's label-set cap keeps bounded.
        route = request.scope.get("route")
        path = (getattr(route, "path_format", None)
                or getattr(route, "path", None)
                or request.url.path)

        self.collector.gauge_dec("guardrails_active_connections")
        self.collector.counter_inc(
            "guardrails_http_requests_total",
            labels={"method": method, "path": path, "status_code": status_code},
        )
        self.collector.histogram_observe(
            "guardrails_http_request_duration_seconds",
            duration,
            labels={"method": method, "path": path},
        )

        return response
