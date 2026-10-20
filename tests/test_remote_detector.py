"""Remote detector: fail-open/fail-closed resolution, malformed responses,
retries, circuit breaker, and the SSRF guard."""
import types

import httpx
import pytest

import znyx_core.detectors.remote as remote_mod
from znyx_core.core.models import Decision
from znyx_core.detectors.remote import CircuitBreaker, RemoteDetector

ENDPOINT = "http://127.0.0.1:9099/detect"


@pytest.fixture()
def transport(monkeypatch):
    """Route the detector's outbound HTTP through an in-process handler.

    Returns a mutable holder: set ``holder.handler`` per test; ``holder.calls``
    collects every request the detector sent.
    """
    holder = types.SimpleNamespace(handler=None, calls=[])
    real_client = httpx.AsyncClient

    def handle(request):
        holder.calls.append(request)
        return holder.handler(request)

    def factory(**kwargs):
        return real_client(transport=httpx.MockTransport(handle),
                           timeout=kwargs.get("timeout"))

    monkeypatch.setattr(remote_mod, "httpx",
                        types.SimpleNamespace(AsyncClient=factory))
    # The detector pools its client per event loop and the loop is process-wide, so a client
    # built from a previous test's factory would still be cached here — serving this test's
    # requests to the previous test's handler. Drop the pool on both sides of the test.
    remote_mod.reset_transport_pool()
    yield holder
    remote_mod.reset_transport_pool()


def _detector(**overrides):
    config = {"endpoint_url": ENDPOINT, "max_retries": 0, "retry_backoff": 0.001}
    config.update(overrides)
    return RemoteDetector(config)


class TestContract:
    def test_valid_block_response(self, transport):
        transport.handler = lambda req: httpx.Response(
            200, json={"decision": "block", "risk_score": 87, "message": "bad"})
        result = _detector().detect("hello")
        assert result.decision == Decision.BLOCK
        assert result.risk_score == 87
        assert len(result.rule_hits) == 1

    def test_valid_allow_response(self, transport):
        transport.handler = lambda req: httpx.Response(
            200, json={"decision": "ALLOW", "risk_score": 0})
        result = _detector().detect("hello")
        assert result.decision == Decision.ALLOW
        assert result.rule_hits == []

    def test_scores_are_clamped_not_fatal(self, transport):
        transport.handler = lambda req: httpx.Response(
            200, json={"decision": "BLOCK", "risk_score": 250, "confidence": 1.7})
        result = _detector().detect("hello")
        assert result.decision == Decision.BLOCK
        assert result.risk_score == 100
        assert result.confidence == 1.0

    def test_payload_and_auth_headers(self, transport):
        transport.handler = lambda req: httpx.Response(
            200, json={"decision": "ALLOW", "risk_score": 0})
        detector = _detector(input_field="content", task="toxicity",
                             auth_type="api_key", auth_header="X-Custom-Key",
                             auth_value="k-123")
        detector.detect("scan me")
        request = transport.calls[0]
        import json
        body = json.loads(request.content)
        assert body["content"] == "scan me"
        assert body["task"] == "toxicity"
        assert request.headers["X-Custom-Key"] == "k-123"

    def test_bearer_auth_header(self, transport):
        transport.handler = lambda req: httpx.Response(
            200, json={"decision": "ALLOW", "risk_score": 0})
        _detector(auth_type="bearer", auth_value="tok").detect("x")
        assert transport.calls[0].headers["Authorization"] == "Bearer tok"


class TestFailurePolicy:
    def _connect_error(self, req):
        raise httpx.ConnectError("connection refused")

    def test_fail_open_allows_on_transport_error(self, transport):
        transport.handler = self._connect_error
        result = _detector(fail_open=True).detect("hello")
        assert result.decision == Decision.ALLOW
        assert result.risk_score == 0
        assert "fail-open" in result.developer_message

    def test_fail_closed_blocks_on_transport_error(self, transport):
        transport.handler = self._connect_error
        result = _detector(fail_open=False).detect("hello")
        assert result.decision == Decision.BLOCK
        assert result.risk_score == 100
        assert any(h.rule_id.endswith(".unavailable") for h in result.rule_hits)

    def test_http_error_status_uses_failure_policy(self, transport):
        transport.handler = lambda req: httpx.Response(500, text="boom")
        assert _detector(fail_open=False).detect("x").decision == Decision.BLOCK

    def test_retries_then_fails(self, transport):
        transport.handler = self._connect_error
        result = _detector(max_retries=2, fail_open=True).detect("hello")
        assert result.decision == Decision.ALLOW
        assert len(transport.calls) == 3

    def test_no_endpoint_is_a_noop_allow(self):
        result = RemoteDetector({}).detect("hello")
        assert result.decision == Decision.ALLOW


class TestMalformedResponses:
    # A malformed payload must resolve through the failure policy, never
    # silently as ALLOW: a broken endpoint would bypass fail-closed otherwise.
    @pytest.mark.parametrize("payload", [
        [1, 2, 3],                                   # not an object
        {"risk_score": 50},                          # decision missing
        {"decision": "MAYBE", "risk_score": 50},     # unknown decision value
    ], ids=["not-a-dict", "missing-decision", "invalid-decision"])
    def test_malformed_blocks_when_fail_closed(self, transport, payload):
        transport.handler = lambda req: httpx.Response(200, json=payload)
        result = _detector(fail_open=False).detect("hello")
        assert result.decision == Decision.BLOCK
        assert result.risk_score == 100

    def test_malformed_allows_when_fail_open(self, transport):
        transport.handler = lambda req: httpx.Response(200, json=[1])
        result = _detector(fail_open=True).detect("hello")
        assert result.decision == Decision.ALLOW
        assert "fail-open" in result.developer_message

    def test_malformed_counts_as_circuit_failure(self, transport):
        transport.handler = lambda req: httpx.Response(200, json={"nope": 1})
        detector = _detector(fail_open=False, circuit_failure_threshold=1)
        detector.detect("hello")
        assert detector.circuit_state == "open"


class TestCircuitBreaker:
    def test_opens_after_threshold_and_recovers(self):
        breaker = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.0)
        assert breaker.allow_request()
        breaker.record_failure()
        assert breaker.state.value == "closed"
        breaker.record_failure()
        assert breaker.state.value == "open"
        # cooldown of 0 means the next request probes half-open immediately
        assert breaker.allow_request()
        assert breaker.state.value == "half_open"
        breaker.record_success()
        assert breaker.state.value == "closed"

    def test_open_circuit_blocks_requests_within_cooldown(self):
        breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=60.0)
        breaker.record_failure()
        assert not breaker.allow_request()

    def test_open_circuit_resolves_via_failure_policy(self, transport):
        def boom(req):
            raise httpx.ConnectError("down")
        transport.handler = boom
        detector = _detector(fail_open=False, circuit_failure_threshold=1,
                             circuit_cooldown=60.0)
        detector.detect("first")           # opens the circuit
        result = detector.detect("second")  # short-circuits
        assert result.decision == Decision.BLOCK
        assert "Circuit breaker open" in result.developer_message
        assert len(transport.calls) == 1


class TestSSRFGuard:
    def test_metadata_endpoint_is_blocked_without_any_call(self, transport):
        transport.handler = lambda req: httpx.Response(
            200, json={"decision": "ALLOW", "risk_score": 0})
        detector = _detector(endpoint_url="http://169.254.169.254/latest/meta-data",
                             fail_open=False)
        result = detector.detect("hello")
        assert result.decision == Decision.BLOCK
        assert transport.calls == []


class TestSharedCircuitBreaker:
    """Escalation and the remote plugin build a new RemoteDetector for every call.
    The breaker must belong to the endpoint, not the instance, or it never trips."""

    def test_breaker_survives_across_detector_instances(self, transport):
        def boom(req):
            raise httpx.ConnectError("down")
        transport.handler = boom
        _detector(fail_open=False, circuit_failure_threshold=2, circuit_cooldown=60.0).detect("a")
        _detector(fail_open=False, circuit_failure_threshold=2, circuit_cooldown=60.0).detect("b")
        # A third, fresh instance for the same endpoint short-circuits without a call.
        result = _detector(fail_open=False, circuit_failure_threshold=2,
                           circuit_cooldown=60.0).detect("c")
        assert "Circuit breaker open" in result.developer_message
        assert len(transport.calls) == 2

    def test_endpoints_have_independent_breakers(self, transport):
        def boom(req):
            raise httpx.ConnectError("down")
        transport.handler = boom
        for _ in range(2):
            _detector(fail_open=False, circuit_failure_threshold=1, circuit_cooldown=60.0).detect("x")
        other = _detector(endpoint_url="http://10.0.0.6:9000/classify", fail_open=False,
                          circuit_failure_threshold=1, circuit_cooldown=60.0)
        assert other.circuit_state == "closed"

    def test_half_open_admits_one_probe_at_a_time(self):
        from znyx_core.detectors.remote import CircuitBreaker
        breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.0)
        breaker.record_failure()
        assert breaker.allow_request()        # the probe
        assert breaker.state.value == "half_open"
        assert not breaker.allow_request()    # a second caller waits for the probe
        breaker.record_failure()              # the probe failed: open again
        assert breaker.state.value == "open"

    def test_abandoned_probe_is_replaced_after_cooldown(self, monkeypatch):
        import znyx_core.detectors.remote as remote
        now = [1000.0]
        monkeypatch.setattr(remote.time, "time", lambda: now[0])
        breaker = remote.CircuitBreaker(failure_threshold=1, cooldown_seconds=5.0)
        breaker.record_failure()
        now[0] += 5.0
        assert breaker.allow_request()        # probe starts, never reports back
        assert not breaker.allow_request()
        now[0] += 5.0
        assert breaker.allow_request()        # abandoned probe replaced

    def test_breaker_is_thread_safe_under_concurrent_failures(self):
        import threading
        from znyx_core.detectors.remote import CircuitBreaker
        breaker = CircuitBreaker(failure_threshold=1000, cooldown_seconds=60.0)
        threads = [threading.Thread(target=lambda: [breaker.record_failure() for _ in range(250)])
                   for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert breaker.failure_count == 2000
        assert breaker.state.value == "open"

    def test_escalation_path_trips_the_breaker(self, transport):
        """The real path that rebuilt the detector per call: _call_backend."""
        from znyx_core.engine.backends import DetectorBackend
        from znyx_core.engine.escalation import BackendUnavailable, _call_backend
        def boom(req):
            raise httpx.ConnectError("down")
        transport.handler = boom
        backend = DetectorBackend(mode="local_ml", endpoint_url=ENDPOINT, task="toxicity")
        for _ in range(4):
            with pytest.raises(BackendUnavailable):
                _call_backend(backend, "text", timeout_ms=1000)
        # Default threshold 5 with 3 attempts per call: the circuit opens during the
        # second call, and the third and fourth calls never reach the endpoint.
        assert len(transport.calls) == 5
