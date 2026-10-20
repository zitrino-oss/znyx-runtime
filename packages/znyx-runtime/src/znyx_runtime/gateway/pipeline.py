"""One LLM call through the gateway: identify, permit, guard input, forward, guard
output, record (plan section 4; the optimisation step arrives in Phase 4).

Pure pass-through: the call goes to the provider and model in the request. A call
that is not allowed (unknown key, model not approved for the app, provider outside
the allowed regions, budget spent, a guardrail block) is answered with a clear
reason in the provider's own error shape. It is never quietly sent somewhere else.

Text checks reuse the evaluator exactly as ``/v1/evaluate/*`` does (same pool,
admission control, deadline, egress audit, telemetry). Every role's text is read,
including documents, tool results and the history's tool calls; the text parts of
one message are read together, so text split across parts is seen whole. Text
already checked clean under the same policy is skipped (techniques 34 and 35): a
conversation replays its whole history on every call. The user text a call adds is
always checked, and only it is charged to stateful checks (flood detection, session
budgets), so an agent's tool loop is not counted as a flood of user messages.

A text the policy wants changed is sent changed or not at all: a redaction that
cannot be applied stops the request (or withholds the answer), it never lets the
original through.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from znyx_runtime.gateway import formats as fmts
from znyx_runtime.gateway.config import (
    GatewayConfig, GatewayKey, Provider, instance_key, key_digest,
)
from znyx_runtime.gateway.pricing import PriceTable
from znyx_runtime.gateway.upstream import Upstream, UpstreamError

logger = logging.getLogger(__name__)

_DECISION_RANK = {"ALLOW": 0, "WARN": 1, "TRANSFORM": 2, "REDACT": 3, "BLOCK": 4}
# Invisible characters removed from untrusted text before it is sent (technique 25):
# soft hyphen, combining grapheme joiner, zero-width space, word joiner and the
# invisible operators, bidirectional embedding, override and isolate controls, BOM,
# the Unicode tag block (hidden ASCII instructions) and other default-ignorable
# controls. Characters that render blank in place of a space become a space. Kept,
# because scripts and emoji need them: the joiners (ZWJ, ZWNJ), variation selectors
# and direction marks. Checks also read a copy without those (``detection_copy``).
_INVISIBLE = re.compile("[­͏឴឵​‪-‮⁠-⁯﻿"
                        "￰-￸\U0001bca0-\U0001bca3\U0001d173-\U0001d17a\U000e0000-\U000e007f]")
_BLANK = re.compile("[ᅟᅠㅤﾠ]")
_DEPLOYMENT = re.compile(r"^(?!\.+$)[A-Za-z0-9._-]{1,128}$")
_API_VERSION = re.compile(r"^(\d{4}-\d{2}-\d{2}(-preview)?|preview|latest|v1)$")
_ALL_ROLES = ("system", "developer", "user", "assistant", "tool", "function")
VERDICT_TTL_SECONDS = 3600
# Usage is estimated when the provider's count never arrives (a stream cut short, a
# client that left): about four characters per token, and a flat amount per image or
# file. Budgets are then charged the estimate rather than nothing.
CHARS_PER_TOKEN = 4
OPAQUE_PART_TOKENS = 85


def quiet_policy(policy: Dict[str, Any]) -> Dict[str, Any]:
    """The policy for every check of a call other than the one charged: the same
    detectors, with per-request accounting (abuse rate limit, session budget,
    conversation turns) off, so one call is counted once, not once per message and
    once more for the answer."""
    from znyx_core.engine.streaming import _without_abuse
    return _without_abuse(policy)


def strip_invisible(text: str) -> Tuple[str, int]:
    cleaned, removed = _INVISIBLE.subn("", text)
    cleaned, blanks = _BLANK.subn(" ", cleaned)
    return cleaned, removed + blanks


def detection_copy(text: str) -> str:
    """The text without any default-ignorable character (joiners and variation
    selectors too) and with blank fillers read as spaces: what a check reads when the
    text contains such characters, so they cannot hide a word from it."""
    from znyx_core.core.text_normalize import strip_zero_width
    return strip_zero_width(text)


def estimate_tokens(chars: int) -> int:
    return math.ceil(chars / CHARS_PER_TOKEN) if chars > 0 else 0


def _worst(a: str, b: str) -> str:
    return a if _DECISION_RANK.get(a, 0) >= _DECISION_RANK.get(b, 0) else b


@dataclass
class CallRecord:
    """Metadata of one gateway call (never text)."""
    request_id: str
    key_id: str
    tenant_id: str
    app_id: str
    agent_id: str
    api: str
    provider: str = ""
    model: Optional[str] = None
    response_model: Optional[str] = None
    stream: bool = False
    status: int = 0
    outcome: str = "ok"                    # ok | blocked_input | blocked_output | denied | error
    input_decision: str = "ALLOW"
    output_decision: str = "ALLOW"
    rule_ids: List[str] = field(default_factory=list)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    latency_ms: int = 0
    upstream_ms: Optional[int] = None
    ttft_ms: Optional[int] = None
    guard_ms: int = 0
    techniques: Dict[str, int] = field(default_factory=dict)
    reason: Optional[str] = None
    started: float = field(default_factory=time.perf_counter)
    estimated_input_tokens: int = 0        # used only when the provider's count is missing
    member: str = ""                       # the environment (bundle set member) of the key
    # Phase 3: what the usage event and the trace span carry.
    occurred_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_ns: int = field(default_factory=time.time_ns)
    trace_id: str = ""
    span_id: str = ""
    parent_span_id: Optional[str] = None
    user: Optional[str] = None             # the end user, pseudonymised (settings.user_ids)
    provider_format: Optional[str] = None
    cache_write_tokens: Optional[int] = None
    usage_estimated: bool = False
    price_known: bool = False

    def note(self, technique: str, count: int = 1) -> None:
        self.techniques[technique] = self.techniques.get(technique, 0) + count

    def to_event(self) -> Dict[str, Any]:
        data = {k: v for k, v in self.__dict__.items()
                if k not in ("started", "estimated_input_tokens", "member", "started_ns")}
        data["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return data


class Denied(Exception):
    """A call the gateway answers itself; carries the status, message and code."""

    def __init__(self, status: int, message: str, code: str, outcome: str = "denied",
                 error_type: Optional[str] = None, retry_after: Optional[int] = None):
        super().__init__(message)
        self.status, self.message, self.code, self.outcome = status, message, code, outcome
        self.error_type, self.retry_after = error_type, retry_after


@dataclass
class _Verdict:
    """What a check decided, independent of which evaluator call made it."""
    decision: str
    rule_ids: List[str]
    sanitized_text: Optional[str] = None
    sanitized_tool_args: Optional[Dict[str, Any]] = None
    user_message: Optional[str] = None
    hits: int = 0

    @classmethod
    def of(cls, result) -> "_Verdict":
        return cls(result.decision.value, [h.rule_id for h in result.rule_hits], result.sanitized_text,
                   result.sanitized_tool_args, result.user_message, len(result.rule_hits))

    def merge(self, other: "_Verdict") -> "_Verdict":
        """This verdict (on the text as sent) with one on its detection copy: the worse
        decision; the copy's rewrite when it has one (it is the text without the
        characters that hid something)."""
        return _Verdict(
            _worst(self.decision, other.decision), self.rule_ids + other.rule_ids,
            other.sanitized_text if other.sanitized_text is not None else self.sanitized_text,
            other.sanitized_tool_args if other.sanitized_tool_args is not None else self.sanitized_tool_args,
            self.user_message or other.user_message, self.hits + other.hits)

    @property
    def wants_change(self) -> bool:
        return self.decision in ("REDACT", "TRANSFORM")


@dataclass
class _Unit:
    """Parts read as one text: the text parts of one message, or of one tool result."""
    parts: List[fmts.Part]

    @property
    def first(self) -> fmts.Part:
        return self.parts[0]

    @property
    def text(self) -> str:
        return "".join(p.text for p in self.parts)


def _units(parts: List[fmts.Part]) -> List[_Unit]:
    groups: "OrderedDict[Tuple[Any, ...], List[fmts.Part]]" = OrderedDict()
    for part in parts:
        groups.setdefault((part.kind, part.container), []).append(part)
    return [_Unit(group) for group in groups.values()]


class _VerdictCache:
    """Content hashes already checked clean, per scope and policy version (technique
    29: never shared across tenants or policy versions). In this process only: in the
    shared store they would take room from budgets and rate limits (one entry per
    distinct text), and another worker checking a text once more costs one check."""

    def __init__(self, size: int = 50_000):
        self._local: "OrderedDict[str, float]" = OrderedDict()
        self._size = size

    @staticmethod
    def key(scope: Tuple[str, ...], stage: str, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()
        return hashlib.sha256("\x1f".join((*scope, stage, digest)).encode("utf-8", "surrogatepass")).hexdigest()

    def seen(self, key: str) -> bool:
        hit = self._local.get(key)
        if hit is not None and time.monotonic() - hit < VERDICT_TTL_SECONDS:
            self._local.move_to_end(key)
            return True
        return False

    def add(self, key: str) -> None:
        self._local[key] = time.monotonic()
        self._local.move_to_end(key)
        while len(self._local) > self._size:
            self._local.popitem(last=False)


class Gateway:
    def __init__(self, config: GatewayConfig, *, evaluator, bundle_manager, egress_sink=None,
                 on_call=None):
        self.config = config
        self.evaluator = evaluator
        self.bundle_manager = bundle_manager
        self.egress_sink = egress_sink
        self.on_call = on_call
        self.prices = PriceTable(config.prices)
        self._member_prices = {m: PriceTable(p) for m, p in config.member_prices.items()}
        self.upstream = Upstream(max_connections=config.settings.max_concurrent)
        self.verdicts = _VerdictCache()
        self._instance_key = instance_key(config.settings)
        self._slots = asyncio.Semaphore(config.settings.max_concurrent)
        self._slots_size = config.settings.max_concurrent
        self._known_apps = self._app_labels(config)

    @staticmethod
    def _app_labels(config: GatewayConfig) -> set:
        """The apps a metric may name: those of configured keys (bounded), so a label's
        values come from the configuration, never from callers."""
        return set(sorted({k.app_id for k in config.keys_by_digest.values()})[:200])

    def update_config(self, config: GatewayConfig) -> None:
        """Swap in a new configuration (a new bundle). Calls in flight finish on the
        connection pool and the concurrency slots they started with; the old pool is
        closed a minute later."""
        old_upstream = self.upstream
        self.config = config
        self.prices = PriceTable(config.prices)
        self._member_prices = {m: PriceTable(p) for m, p in config.member_prices.items()}
        self.upstream = Upstream(max_connections=config.settings.max_concurrent)
        self._instance_key = instance_key(config.settings)
        self._known_apps = self._app_labels(config)
        if config.settings.max_concurrent != self._slots_size:
            self._slots = asyncio.Semaphore(config.settings.max_concurrent)
            self._slots_size = config.settings.max_concurrent

        async def retire() -> None:
            await asyncio.sleep(60)
            await old_upstream.close()

        try:
            asyncio.get_running_loop().create_task(retire())
        except RuntimeError:
            pass

    async def close(self) -> None:
        await self.upstream.close()

    def prices_for(self, member: str) -> PriceTable:
        """The price table of the key's own environment (its bundle's)."""
        return self._member_prices.get(member) or self.prices

    @staticmethod
    def _state_id(key: GatewayKey) -> str:
        """Who limits and budgets are counted for. Key ids are unique within one
        project's environment only, so on a runtime serving several the environment is
        part of it."""
        return f"{key.member}/{key.id}" if key.member else key.id

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def handle(self, request: Request, api: str, deployment: Optional[str] = None) -> Response:
        from znyx_core.engine.evaluator import EvaluatorOverloadedError

        fmt = fmts.FORMATS[api]
        record = CallRecord(request_id=request.headers.get("x-request-id") or f"gw-{uuid.uuid4().hex}",
                            key_id="", tenant_id="", app_id="", agent_id="", api=api)
        self._trace(record, request)
        # Who is calling is settled from the headers before anything else: a caller
        # without a valid key gets no concurrency slot and no body read.
        try:
            key, client_credential = self._identify(request)
        except Denied as denied:
            return self._finish(record, fmt, denied)
        record.key_id, record.tenant_id, record.app_id, record.agent_id = (
            key.id, key.tenant_id, key.app_id, key.agent_id)
        record.member = key.member
        slots = self._slots
        if slots.locked():
            return self._finish(record, fmt, Denied(429, "Too many concurrent requests to the gateway; "
                                                         "retry shortly", "rate_limit_exceeded",
                                                    error_type="rate_limit_error", retry_after=1))
        await slots.acquire()
        released = False

        def release() -> None:
            nonlocal released
            if not released:
                released = True
                slots.release()

        handed_off = False
        try:
            try:
                response = await self._handle(request, api, fmt, record, deployment, key,
                                              client_credential, release)
            except Denied as denied:
                response = self._finish(record, fmt, denied)
            except UpstreamError as exc:
                response = self._finish(record, fmt, Denied(exc.status, str(exc), "upstream_error",
                                                            outcome="error", error_type="api_error"))
            except EvaluatorOverloadedError:
                response = self._finish(record, fmt, Denied(
                    503, "The gateway is busy checking other requests; retry shortly", "overloaded",
                    outcome="error", retry_after=1))
            except Exception:  # noqa: BLE001 - answered in the provider's shape, never a bare 500
                logger.exception("gateway: unexpected error handling %s", record.request_id)
                response = self._finish(record, fmt, Denied(500, "The gateway could not handle this request",
                                                            "internal_error", outcome="error",
                                                            error_type="api_error"))
            # A guarded stream keeps its slot until it ends (it releases it itself).
            handed_off = bool(getattr(response, "owns_slot", False))
            return response
        finally:
            if not handed_off:
                release()

    @staticmethod
    def _trace(record: CallRecord, request: Request) -> None:
        """The call's place in a trace (W3C trace context): the caller's ``traceparent``
        makes this call a child of the caller's span; ``X-Znyx-Trace-Id`` joins a trace by
        id alone; otherwise the call starts a trace of its own."""
        from znyx_core import usage_events
        parent = usage_events.parse_traceparent(request.headers.get("traceparent"))
        if parent:
            record.trace_id, record.parent_span_id = parent
        else:
            given = (request.headers.get("x-znyx-trace-id") or "").strip().lower().replace("-", "")
            record.trace_id = given if re.fullmatch(r"[0-9a-f]{32}", given or "") else usage_events.new_trace_id()
        record.span_id = usage_events.new_span_id()

    @staticmethod
    def trace_headers(record: CallRecord) -> Dict[str, str]:
        from znyx_core import usage_events
        return {"x-znyx-request-id": record.request_id, "x-znyx-trace-id": record.trace_id,
                "traceparent": usage_events.traceparent(record.trace_id, record.span_id)}

    def _user_of(self, request: Request, body: Dict[str, Any]) -> Optional[str]:
        """The end user the app called for: ``X-Znyx-User``, else the API's own field
        (OpenAI ``user``, Anthropic ``metadata.user_id``). Pseudonymised by default
        (``settings.user_ids``: hash | plain | off): a keyed hash whose key stays on the
        runtime side, so the console can group spend by user without learning, or guessing,
        who the users are."""
        mode = self.config.settings.user_ids
        if mode == "off":
            return None
        raw = request.headers.get("x-znyx-user")
        if not raw:
            raw = body.get("user") if isinstance(body.get("user"), str) else None
        if not raw and isinstance(body.get("metadata"), dict):
            value = body["metadata"].get("user_id")
            raw = value if isinstance(value, str) else None
        if not raw:
            return None
        raw = raw.strip()[:200]
        if mode == "plain":
            return raw
        import hmac
        from znyx_runtime.install_state import user_id_key
        scope = self.bundle_manager.members[0].bundle.org_id if getattr(self.bundle_manager, "members", None) else ""
        digest = hmac.new(user_id_key(), f"{scope}\x1f{raw}".encode("utf-8", "surrogatepass"), hashlib.sha256)
        return "u_" + digest.hexdigest()[:16]

    def _finish(self, record: CallRecord, fmt, denied: Denied) -> Response:
        record.status = denied.status
        record.outcome = denied.outcome
        record.reason = denied.code
        status, body = fmt.error(denied.status, denied.message, code=denied.code,
                                 error_type=denied.error_type,
                                 extra={"request_id": record.request_id, "reason": denied.code})
        headers = self.trace_headers(record)
        if denied.retry_after:
            headers["retry-after"] = str(denied.retry_after)
        self._record(record)
        return JSONResponse(body, status_code=status, headers=headers)

    # ------------------------------------------------------------------
    # The steps
    # ------------------------------------------------------------------

    async def _read_body(self, request: Request) -> Dict[str, Any]:
        settings = self.config.settings
        limit = settings.max_request_bytes
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > limit:
            raise Denied(413, f"Request body larger than {limit} bytes", "request_too_large")

        async def read() -> bytes:
            chunks, size = [], 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > limit:
                    raise Denied(413, f"Request body larger than {limit} bytes", "request_too_large")
                chunks.append(chunk)
            return b"".join(chunks)

        try:
            raw = await asyncio.wait_for(read(), timeout=settings.body_timeout_seconds)
        except asyncio.TimeoutError:
            raise Denied(408, "The request body did not arrive in time", "request_timeout") from None
        try:
            body = json.loads(raw or b"null")
        except (ValueError, RecursionError):
            raise Denied(400, "Request body is not valid JSON", "invalid_json") from None
        if not isinstance(body, dict):
            raise Denied(400, "Request body must be a JSON object", "invalid_body")
        if "stream" in body and not isinstance(body["stream"], bool):
            # A provider that reads "true" as true would stream while the gateway
            # treats the call as a single answer.
            raise Denied(400, "stream must be true or false", "invalid_stream")
        return body

    def _identify(self, request: Request) -> Tuple[GatewayKey, Optional[str]]:
        """The caller's gateway key, and (for client-key providers) the provider
        credential the caller sent. X-Znyx-Key always identifies; otherwise the SDK's
        own credential header does."""
        from znyx_runtime.api.routes import _runtime_auth_required

        headers = request.headers
        explicit = headers.get("x-znyx-key")
        standard = None
        auth = headers.get("authorization") or ""
        if auth.lower().startswith("bearer "):
            standard = auth[7:].strip()
        standard = standard or headers.get("x-api-key") or headers.get("api-key")
        presented = explicit or standard
        if not presented:
            if not _runtime_auth_required():
                return GatewayKey(id="anonymous", key_sha256="", instance_key=True), None
            raise Denied(401, "A gateway key is required (send it as the API key of your SDK)",
                         "invalid_api_key", error_type="authentication_error")
        digest = key_digest(presented)
        key = self.config.keys_by_digest.get(digest)
        if key is None and self._instance_key is not None and digest == self._instance_key.key_sha256:
            key = self._instance_key
        if key is None or key.disabled:
            raise Denied(401, "Invalid gateway key", "invalid_api_key", error_type="authentication_error")
        client_credential = standard if explicit and standard else None
        return key, client_credential

    def _scope(self, key: GatewayKey, request: Request, member=None) -> Dict[str, str]:
        """Who the call is for. The environment is the key's own: on a runtime serving
        several environments, the one the key belongs to."""
        env = self.bundle_manager.environment_for(member) or key.environment or "prod"
        return {"tenant_id": key.tenant_id, "app_id": key.app_id, "agent_id": key.agent_id,
                "env": env, "session_id": request.headers.get("x-znyx-session-id") or ""}

    def _check_limits(self, key: GatewayKey, model: Optional[str]) -> None:
        from znyx_core.state_store import StateUnavailable, get_state_store

        limits = key.limits
        if limits.spend_usd_per_day and not key.instance_key and self.prices_for(key.member).lookup(model) is None:
            # Without a price the call would cost nothing against the limit.
            raise Denied(403, f"Model {model} has no price on this gateway, so this key's daily spend "
                              "limit cannot be enforced; add a price for it", "price_unknown",
                         error_type="permission_error")
        store = get_state_store()
        try:
            if limits.requests_per_minute:
                rpm = float(limits.requests_per_minute)
                if not store.consume_token(f"pin:gw:rpm:{self._state_id(key)}", rate_per_second=rpm / 60.0,
                                           capacity=rpm):
                    raise Denied(429, "Rate limit for this key reached; retry shortly",
                                 "rate_limit_exceeded", error_type="rate_limit_error", retry_after=1)
            if limits.tokens_per_day or limits.spend_usd_per_day:
                used = store.get(self._day_key(key))
                if limits.tokens_per_day and used.get("tokens", 0) >= limits.tokens_per_day:
                    raise Denied(429, "Daily token budget for this key is spent", "insufficient_quota",
                                 error_type="rate_limit_error")
                if limits.spend_usd_per_day and used.get("cost", 0) >= limits.spend_usd_per_day:
                    raise Denied(429, "Daily spend budget for this key is spent", "insufficient_quota",
                                 error_type="rate_limit_error")
        except StateUnavailable:
            raise Denied(503, "Limits cannot be checked right now (fail-closed)", "limits_unavailable",
                         outcome="error", error_type="api_error", retry_after=1) from None

    @classmethod
    def _day_key(cls, key: GatewayKey) -> str:
        # "pin:": never evicted from the in-memory store by other traffic.
        return f"pin:gw:day:{cls._state_id(key)}:{time.strftime('%Y%m%d', time.gmtime())}"

    def _charge(self, key: GatewayKey, record: CallRecord) -> None:
        tokens = (record.input_tokens or 0) + (record.output_tokens or 0)
        if not (key.limits.tokens_per_day or key.limits.spend_usd_per_day) or key.instance_key:
            return
        try:
            from znyx_core.state_store import get_state_store
            get_state_store().incr(self._day_key(key), {"tokens": tokens, "cost": record.cost_usd or 0.0,
                                                        "requests": 1}, ttl_seconds=2 * 86400)
        except Exception:  # noqa: BLE001 - the call already happened; never fail its response
            logger.warning("gateway: could not record usage for key %s", key.id)

    def estimate_missing_usage(self, record: CallRecord, output_chars: int) -> None:
        """Fill in usage the provider never reported (a stream cut short, a client that
        left, an unreadable answer) from what was sent and generated, so budgets are
        charged for it."""
        estimated = False
        if record.input_tokens is None and record.estimated_input_tokens:
            record.input_tokens = record.estimated_input_tokens
            estimated = True
        if record.output_tokens is None and output_chars:
            record.output_tokens = estimate_tokens(output_chars)
            estimated = True
        if estimated:
            record.note("usage_estimated")
            record.usage_estimated = True
            record.price_known = self.prices_for(record.member).lookup(record.response_model or record.model) is not None
            record.cost_usd = self.prices_for(record.member).cost(
                record.response_model or record.model,
                fmts.Usage(record.input_tokens, record.output_tokens, record.cached_input_tokens,
                           record.cache_write_tokens))

    async def _permit_egress(self, provider: Provider, policy: Dict[str, Any], body_bytes: int,
                             scope: Dict[str, str], record: CallRecord) -> None:
        """Data residency and the egress allowlist through the same gate model calls
        use. A boundary-crossing call is audited before it is made, and a call that
        cannot be audited is not made (fail-closed), exactly as for model backends."""
        from znyx_core.engine.egress import EgressEvent, evaluate_egress

        rp = policy.get("runtime_policy") or {}
        mode = "local_llm" if provider.in_boundary else "remote_llm"
        decision = evaluate_egress(
            mode, endpoint_url=provider.base_url, region=provider.region,
            in_boundary=provider.in_boundary, no_external_calls=bool(rp.get("no_external_calls")),
            egress_allowlist=rp.get("egress_allowlist"), allowed_regions=rp.get("allowed_regions"),
            redact_pii=False, redact_secrets=False)
        reason = None if decision.allowed else (decision.reason or "egress_denied")
        if reason is None and decision.is_egress:
            if self.egress_sink is None:
                reason = "egress_audit_unconfigured"
            else:
                try:
                    # The audit write is durable (fsync): off the event loop.
                    await asyncio.to_thread(self.egress_sink, EgressEvent(
                        event_id=str(uuid.uuid4()), occurred_at=datetime.now(timezone.utc).isoformat(),
                        detector_key="gateway", mode=mode, destination_host=decision.host,
                        destination_region=decision.region, org_scope=scope["tenant_id"],
                        trace_id=record.request_id, bytes_out=body_bytes, redacted=False,
                        allowlisted=decision.allowlisted, model_version=record.model))
                except Exception:  # noqa: BLE001 - an unauditable crossing is not made
                    reason = "egress_audit_unavailable"
        if reason is not None:
            messages = {
                "no_external_calls": "This policy does not allow calls outside your network",
                "residency_denied": "The provider is outside the regions this policy allows",
                "egress_not_allowlisted": "The provider is not on this policy's egress allowlist",
                "egress_audit_unavailable": "The call could not be audited (fail-closed)",
                "egress_audit_unconfigured": "The call could not be audited (fail-closed)",
            }
            raise Denied(403, messages.get(reason, "The call is not allowed by policy"), reason,
                         error_type="permission_error")

    @staticmethod
    def _filter_tools(api: str, body: Dict[str, Any], policy: Dict[str, Any],
                      record: CallRecord) -> None:
        """Least-privilege tools (technique 26): the model only sees tools the policy
        allows, when the policy names them. Legacy OpenAI ``functions`` too, and a
        forced choice of a removed tool is dropped."""
        tools_policy = policy.get("tools") or {}
        if not isinstance(tools_policy, dict) or not tools_policy.get("enabled", True):
            return
        allowed = set(tools_policy.get("allowed") or tools_policy.get("allowed_tools") or [])
        denied = set(tools_policy.get("denied") or tools_policy.get("blocked_tools") or [])
        if not (allowed or denied):
            return

        def permitted(name: str) -> bool:
            return (not allowed or name in allowed) and name not in denied

        def name_of(tool: Any) -> str:
            if not isinstance(tool, dict):
                return ""
            if api == "anthropic":
                return str(tool.get("name") or "")
            fn = tool.get("function")
            return str(fn.get("name") or "") if isinstance(fn, dict) else ""

        removed: set = set()
        tools = body.get("tools")
        if isinstance(tools, list):
            kept = [t for t in tools if permitted(name_of(t))]
            removed |= {name_of(t) for t in tools if not permitted(name_of(t))}
            if len(kept) != len(tools):
                if kept:
                    body["tools"] = kept
                else:
                    body.pop("tools", None)
                    body.pop("tool_choice", None)
        functions = body.get("functions") if api != "anthropic" else None
        if isinstance(functions, list):
            def fname(f: Any) -> str:
                return str(f.get("name") or "") if isinstance(f, dict) else ""
            kept_functions = [f for f in functions if permitted(fname(f))]
            removed |= {fname(f) for f in functions if not permitted(fname(f))}
            if len(kept_functions) != len(functions):
                if kept_functions:
                    body["functions"] = kept_functions
                else:
                    body.pop("functions", None)
                    body.pop("function_call", None)
        if not removed:
            return
        record.note("tools_filtered", len(removed))
        choice = body.get("tool_choice")
        if isinstance(choice, dict):
            forced = choice.get("name") if api == "anthropic" else (choice.get("function") or {}).get("name")
            if forced in removed:
                body.pop("tool_choice", None)
        call = body.get("function_call")
        if isinstance(call, dict) and call.get("name") in removed:
            body.pop("function_call", None)

    @staticmethod
    def _in_scope(entry: Dict[str, Any], key: GatewayKey, project: str, environment: str) -> bool:
        scope = entry.get("scope")
        return (scope == "org"
                or (scope == "project" and entry.get("project_id") == project)
                or (scope == "environment" and entry.get("environment_id") == environment)
                or (scope == "app" and entry.get("app_id") == key.app_id
                    and entry.get("project_id") in (None, "", project))
                or (scope == "key" and entry.get("key_id") == key.id
                    and entry.get("project_id") in (None, "", project)
                    and entry.get("environment_id") in (None, "", environment)))

    def _check_spend_controls(self, key: GatewayKey, member, model: Optional[str] = None) -> None:
        """Budgets the console set to block (signed in the bundle set). A spent one refuses
        calls in its scope until its period ends or it is raised; while any is active, a
        model without a price is refused in its scope, since such calls would cost nothing
        against it."""
        controls = getattr(self.bundle_manager, "controls", None) or {}
        if not isinstance(controls, dict) or not controls:
            return
        project = member.project_id if member is not None else ""
        environment = member.environment_id if member is not None else ""
        now = datetime.now(timezone.utc)
        for entry in controls.get("spend_blocked") or []:
            if not isinstance(entry, dict):
                continue
            try:
                until = datetime.fromisoformat(str(entry.get("until")).replace("Z", "+00:00"))
                if until.tzinfo is None:
                    until = until.replace(tzinfo=timezone.utc)
                if until <= now:
                    continue                     # the budget's period is over
            except ValueError:
                pass                             # no end given: blocked until it is lifted
            if self._in_scope(entry, key, project, environment):
                name = entry.get("name") or "a budget"
                raise Denied(429, f"Spend budget {name!r} is spent; calls are refused until it resets "
                                  "or is raised", "budget_exhausted", error_type="rate_limit_error")
        if model is not None and self.prices_for(key.member).lookup(model) is None:
            for entry in controls.get("priced_only") or []:
                if isinstance(entry, dict) and self._in_scope(entry, key, project, environment):
                    name = entry.get("name") or "a budget"
                    raise Denied(403, f"Model {model} has no price on this gateway, and spend here is "
                                      f"limited by budget {name!r}; add a price for it", "price_unknown",
                                 error_type="permission_error")

    def _check_provider_side_calls(self, api: str, body: Dict[str, Any]) -> None:
        if api == "anthropic" and body.get("mcp_servers") and not self.config.settings.allow_provider_mcp:
            raise Denied(400, "MCP servers called by the provider bypass the gateway's tool policy and "
                              "are not allowed (gateway setting allow_provider_mcp)",
                         "provider_mcp_not_allowed")

    # -- guardrails --------------------------------------------------------------

    def _request(self, scope: Dict[str, str], record: CallRecord, suffix: str, text: str,
                 role: str, context: Optional[Dict[str, Any]] = None):
        from znyx_core.core.models import EvaluationRequest
        metadata = {"source": "gateway", "api": record.api, "provider": record.provider,
                    "model": record.model or "", "role": role}
        if context:
            # The request's own system prompt and tool schemas, for the hidden-context
            # leak check on the answer (technique 27). Used in memory, never recorded.
            metadata.update(context)
        return EvaluationRequest(
            request_id=f"{record.request_id}-{suffix}", tenant_id=scope["tenant_id"],
            app_id=scope["app_id"], agent_id=scope["agent_id"], env=scope["env"], text=text,
            # span_id: the span this check belongs to (the call's), so a check is a child
            # span of the call in the trace.
            session_id=scope["session_id"] or None, trace_id=record.trace_id or record.request_id,
            span_id=record.span_id or None, metadata=metadata)

    async def _check_one(self, text: str, kind: str, role: str, tool_name: Optional[str], stage: str,
                         scope: Dict[str, str], record: CallRecord, policy: Dict[str, Any], label: str,
                         context: Optional[Dict[str, Any]] = None) -> _Verdict:
        from znyx_core.core.models import ToolEvaluationRequest
        from znyx_runtime.api.routes import runtime_judge_ctx

        if kind in ("tool_result", "tool_call"):
            args: Dict[str, Any] = {}
            if kind == "tool_call":
                try:
                    parsed = json.loads(text) if text else {}
                    args = parsed if isinstance(parsed, dict) else {"value": parsed}
                except ValueError:
                    args = {"raw": text}
            tool_req = ToolEvaluationRequest(
                request_id=f"{record.request_id}-{label}", tenant_id=scope["tenant_id"],
                app_id=scope["app_id"], agent_id=scope["agent_id"], env=scope["env"],
                tool_name=tool_name or "tool", tool_args=args,
                tool_result=text if kind == "tool_result" else None,
                session_id=scope["session_id"] or None, trace_id=record.trace_id or record.request_id,
                span_id=record.span_id or None,
                metadata={"source": "gateway", "api": record.api, "stage": stage})
            result = await self.evaluator.evaluate_tool(tool_req, policy=policy,
                                                        judge_ctx=runtime_judge_ctx(policy, tool_req))
            return _Verdict.of(result)
        req = self._request(scope, record, label, text, role, context)
        result = await self.evaluator.evaluate(req, context=stage, policy=policy,
                                               judge_ctx=runtime_judge_ctx(policy, req))
        return _Verdict.of(result)

    async def _check_text(self, text: str, kind: str, role: str, tool_name: Optional[str], stage: str,
                          scope: Dict[str, str], record: CallRecord, policy: Dict[str, Any],
                          quiet: Dict[str, Any], label: str,
                          context: Optional[Dict[str, Any]] = None) -> _Verdict:
        """A check of ``text``, and of its detection copy when invisible characters
        could be hiding something from it."""
        verdict = await self._check_one(text, kind, role, tool_name, stage, scope, record, policy,
                                        label, context)
        probe = detection_copy(text)
        if probe == text:
            return verdict
        record.note("checked_without_invisible")
        return verdict.merge(await self._check_one(probe, kind, role, tool_name, stage, scope, record,
                                                   quiet, label + "n", context))

    async def _guard_input(self, api: str, fmt, body: Dict[str, Any], scope: Dict[str, str],
                           policy: Dict[str, Any], record: CallRecord) -> None:
        settings = self.config.settings
        parts = fmt.input_parts(body, settings.check_roles)
        opaque = [p for p in parts if p.kind == "opaque"]
        if opaque:
            record.note("uncheckable_parts", len(opaque))
            if settings.uncheckable_content == "block":
                kinds = ", ".join(sorted({p.text for p in opaque}))
                raise Denied(400, f"This request carries content the gateway cannot check ({kinds}), "
                                  "and this gateway refuses such content", "uncheckable_content",
                             outcome="blocked_input", error_type="invalid_request_error")
        parts = [p for p in parts if p.kind != "opaque"]
        if not parts:
            return
        if settings.normalise_before_send:
            for part in parts:
                if part.kind == "history_tool_call" and fmt is fmts.AnthropicMessages:
                    continue                       # an object, not a string; its copy is checked
                cleaned, removed = strip_invisible(part.text)
                if removed:
                    record.note("invisible_removed", removed)
                    fmt.set_text(body, part, cleaned)
                    part.text = cleaned
        units = _units(parts)
        # The user text this call adds: user messages after the last assistant turn.
        # Its last unit is always checked, with the full policy (stateful checks count
        # it once per call). An agent loop's later steps add no user text, so they are
        # not counted as new user messages.
        messages = body.get("messages") if isinstance(body.get("messages"), list) else []
        last_assistant = max((i for i, m in enumerate(messages)
                              if isinstance(m, dict) and m.get("role") == "assistant"), default=-1)
        fresh = [u for u in units if u.first.role == "user" and u.first.kind == "text"
                 and u.first.message is not None and u.first.message > last_assistant]
        charged = fresh[-1] if fresh else None
        version = str(policy.get("policy_version") or "")
        scope_key = (scope["tenant_id"], scope["app_id"], scope["agent_id"], version)
        quiet = quiet_policy(policy)
        gate = asyncio.Semaphore(settings.max_parallel_checks)

        async def check(n: int, unit: _Unit) -> Optional[_Verdict]:
            text = unit.text
            cache_key = self.verdicts.key(scope_key, "input:" + unit.first.kind, text)
            if unit is not charged and self.verdicts.seen(cache_key):
                record.note("checks_skipped_seen")
                return None
            async with gate:
                verdict = await self._check_text(
                    text, unit.first.kind, unit.first.role, unit.first.tool_name, "input", scope, record,
                    policy if unit is charged else quiet, quiet, f"input{n}")
            if verdict.decision == "ALLOW" and not verdict.hits:
                self.verdicts.add(cache_key)       # remembered as it completes, not after all
            return verdict

        t0 = time.perf_counter()
        verdicts = await asyncio.gather(*(check(n, unit) for n, unit in enumerate(units)))
        record.guard_ms += int((time.perf_counter() - t0) * 1000)
        block_message = None
        removals: List[fmts.Part] = []
        for unit, verdict in zip(units, verdicts):
            if verdict is None:
                continue
            record.input_decision = _worst(record.input_decision, verdict.decision)
            record.rule_ids.extend(verdict.rule_ids)
            if verdict.decision == "BLOCK":
                block_message = block_message or verdict.user_message
                continue
            rewrite = verdict.sanitized_text
            if rewrite is not None and rewrite != unit.text:
                try:
                    # Several parts read as one: the rewritten text replaces the first,
                    # the others are removed (a redaction may span them).
                    fmt.set_text(body, unit.first, rewrite)
                except (ValueError, TypeError):
                    record.input_decision = "BLOCK"
                    continue
                removals.extend(unit.parts[1:])
                record.note("input_rewritten")
            elif verdict.wants_change:
                # The policy wants this text changed and no changed text came back:
                # it is not sent as it is.
                record.input_decision = "BLOCK"
        if block_message is not None or record.input_decision == "BLOCK":
            raise Denied(400, block_message or settings.block_message, "content_filter",
                         outcome="blocked_input", error_type="invalid_request_error")
        if removals:
            fmts.remove_blocks(body, removals)

    async def _guard_output(self, fmt, response: Dict[str, Any], scope: Dict[str, str],
                            policy: Dict[str, Any], record: CallRecord,
                            context: Optional[Dict[str, Any]] = None) -> None:
        parts = fmt.output_parts(response)
        if not parts:
            return
        quiet = quiet_policy(policy)               # the call was charged on the way in
        gate = asyncio.Semaphore(self.config.settings.max_parallel_checks)

        async def check(i: int, part: fmts.Part) -> _Verdict:
            kind = "tool_call" if part.kind == "tool_call" else "text"
            # The hidden-context leak check reads what the user is shown, not the model's
            # reasoning (which often restates its instructions) or quoted documents.
            ctx = context if part.kind in ("text", "audio") else None
            async with gate:
                return await self._check_text(part.text, kind, "assistant", part.tool_name, "output",
                                              scope, record, quiet, quiet, f"output{i}", ctx)

        t0 = time.perf_counter()
        verdicts = await asyncio.gather(*(check(i, p) for i, p in enumerate(parts)))
        record.guard_ms += int((time.perf_counter() - t0) * 1000)
        withheld, rewritten = set(), set()
        for part, verdict in zip(parts, verdicts):
            record.output_decision = _worst(record.output_decision, verdict.decision)
            record.rule_ids.extend(verdict.rule_ids)
            if verdict.decision == "BLOCK":
                withheld.add(part.choice)
            elif part.kind == "tool_call":
                if verdict.sanitized_tool_args is not None:
                    value = verdict.sanitized_tool_args
                    fmt.set_text(response, part, value if fmt is fmts.AnthropicMessages else json.dumps(value))
                    rewritten.add(part.choice)
                    record.note("output_rewritten")
                elif verdict.wants_change:
                    withheld.add(part.choice)
            elif verdict.sanitized_text is not None and verdict.sanitized_text != part.text:
                if part.kind in ("audio", "thinking"):
                    # Audio cannot be edited to match its transcript, nor a signed
                    # thinking block: either is passed on clean or not at all.
                    withheld.add(part.choice)
                else:
                    fmt.set_text(response, part, verdict.sanitized_text)
                    rewritten.add(part.choice)
                    record.note("output_rewritten")
            elif verdict.wants_change:
                withheld.add(part.choice)
        for choice in rewritten - withheld:
            fmt.rewritten_choice(response, choice)
        for choice in withheld:
            fmt.withhold_choice(response, choice, self.config.settings.output_block_message)
        if withheld:
            record.outcome = "blocked_output"

    # -- the call ----------------------------------------------------------------

    async def _handle(self, request: Request, api: str, fmt, record: CallRecord,
                      deployment: Optional[str], key: GatewayKey, client_credential: Optional[str],
                      release) -> Response:
        body = await self._read_body(request)
        family = "azure" if api == "azure" else api
        requested_provider = request.headers.get("x-znyx-provider")
        try:
            provider = self.config.provider_for(key, family, requested_provider)
        except LookupError as exc:
            raise Denied(400 if requested_provider else 403, str(exc), "provider_not_available",
                         error_type="permission_error") from None
        record.provider = provider.name
        record.provider_format = provider.format
        record.user = self._user_of(request, body)
        if provider.blocked:
            raise Denied(403, provider.blocked, "provider_not_allowed_here", error_type="permission_error")
        if provider.uses_client_key and not client_credential:
            raise Denied(401, f"Provider {provider.name} uses your own provider key: send the gateway "
                         "key as X-Znyx-Key and the provider key as your SDK's API key",
                         "provider_key_required", error_type="authentication_error")
        api_version = None
        if api == "azure":
            if not deployment or not _DEPLOYMENT.match(deployment):
                raise Denied(400, "Invalid deployment name", "invalid_deployment")
            api_version = request.query_params.get("api-version")
            if api_version is not None and not _API_VERSION.match(api_version):
                raise Denied(400, "Invalid api-version", "invalid_api_version")
            record.model = body.get("model") if isinstance(body.get("model"), str) else deployment
        else:
            record.model = fmt.model(body)
            if not record.model:
                raise Denied(400, "The request names no model", "model_required")
        if not key.model_allowed(record.model if api != "azure" else deployment):
            raise Denied(403, f"Model {record.model} is not approved for this app", "model_not_allowed",
                         error_type="permission_error")
        self._check_limits(key, record.model)

        from znyx_runtime.bundle_manager import ScopeNotServed
        try:
            member = self.bundle_manager.member_by_key(key.member or None)
        except ScopeNotServed as exc:
            raise Denied(403, str(exc), "project_not_served", error_type="permission_error") from None
        self._check_spend_controls(key, member, record.model)
        scope = self._scope(key, request, member)
        try:
            policy = self.bundle_manager.get_policy(tenant_id=scope["tenant_id"], app_id=scope["app_id"],
                                                    agent_id=scope["agent_id"], env=scope["env"],
                                                    member=member)
        except RuntimeError:
            raise Denied(503, "Policy unavailable (fail-closed)", "policy_unavailable",
                         outcome="error", error_type="api_error") from None
        self._check_provider_side_calls(api, body)
        self._filter_tools(api, body, policy, record)
        await self._guard_input(api, fmt, body, scope, policy, record)
        if self.config.settings.canary_marker:
            fmt.add_system_text(body, self.canary())
            record.note("canary")
        context = {"hidden_context": fmt.hidden_context(body)}
        if isinstance(body.get("tools"), list):
            context["tools"] = body["tools"]
        sent = fmt.input_parts(body, _ALL_ROLES)
        record.estimated_input_tokens = (
            estimate_tokens(sum(len(p.text) for p in sent if p.kind != "opaque"))
            + OPAQUE_PART_TOKENS * sum(1 for p in sent if p.kind == "opaque"))
        await self._permit_egress(provider, policy, len(json.dumps(body)), scope, record)

        credential = client_credential if provider.uses_client_key else None
        if not provider.uses_client_key:
            try:
                credential = provider.credential()
            except ValueError as exc:
                logger.error("gateway: %s", exc)
                raise Denied(503, f"Provider {provider.name} has no usable credential", "provider_unconfigured",
                             outcome="error", error_type="api_error") from None
        url = self.upstream.url(provider, deployment=deployment, api_version=api_version)
        headers = self.upstream.headers(provider, request.headers, credential)
        record.stream = fmt.is_stream(body)
        if record.stream:
            from znyx_runtime.gateway.streaming import stream_call
            return await stream_call(self, request, api, fmt, provider, url, headers, body,
                                     key, scope, policy, record, context, release)
        return await self._forward(request, fmt, provider, url, headers, body, key, scope, policy,
                                   record, context)

    async def _forward(self, request: Request, fmt, provider: Provider, url: str,
                       headers: Dict[str, str], body: Dict[str, Any], key: GatewayKey,
                       scope: Dict[str, str], policy: Dict[str, Any], record: CallRecord,
                       context: Optional[Dict[str, Any]] = None) -> Response:
        t_up = time.perf_counter()
        upstream_task = asyncio.ensure_future(self.upstream.send(provider, url, headers, body))
        # Cancel the provider call if the client goes away (it would be paid for nothing).
        while not upstream_task.done():
            done, _ = await asyncio.wait({upstream_task}, timeout=0.5)
            if not done and await request.is_disconnected():
                upstream_task.cancel()
                record.status, record.outcome, record.reason = 499, "error", "client_disconnected"
                # The provider may bill for a call it had already started.
                self.estimate_missing_usage(record, 0)
                self._charge(key, record)
                self._record(record)
                return Response(status_code=499)
        response = upstream_task.result()
        record.upstream_ms = int((time.perf_counter() - t_up) * 1000)
        returned = self.upstream.returned_headers(response)
        returned.update(self.trace_headers(record))
        if response.status_code != 200:
            # The provider's own error, as the provider sent it: SDK retry and error
            # handling keep working.
            record.status, record.outcome, record.reason = response.status_code, "error", "provider_error"
            self._record(record)
            return Response(content=response.content, status_code=response.status_code,
                            media_type=response.headers.get("content-type", "application/json"),
                            headers=returned)
        try:
            payload = fmts.parse_response(response.content)
        except (ValueError, RecursionError):
            payload = None
        expected = "content" if fmt is fmts.AnthropicMessages else "choices"
        if not isinstance(payload, dict) or not isinstance(payload.get(expected), list):
            # A success the gateway cannot read is never passed on unchecked.
            self.estimate_missing_usage(record, 0)
            self._charge(key, record)
            raise Denied(502, "The provider returned a response the gateway cannot read",
                         "bad_upstream_response", outcome="error", error_type="api_error")
        record.response_model = payload.get("model") if isinstance(payload.get("model"), str) else None
        await self._guard_output(fmt, payload, scope, policy, record, context)
        usage = fmt.usage(payload)
        self._apply_usage(record, usage)
        self.estimate_missing_usage(record, sum(len(p.text) for p in fmt.output_parts(payload)))
        self._charge(key, record)
        record.status = 200
        returned["x-znyx-decision"] = _worst(record.input_decision, record.output_decision)
        self._record(record)
        return JSONResponse(payload, status_code=200, headers=returned)

    def canary(self) -> str:
        """This deployment's marker: stable across workers and restarts (derived from
        the instance key), long enough for the hidden-context check to match verbatim."""
        seed = (self._instance_key.key_sha256 if self._instance_key else "") or "znyx"
        token = hashlib.sha256(("znyx-canary:" + seed).encode()).hexdigest()
        groups = " ".join(token[i:i + 4] for i in range(0, 32, 4))
        return f"Deployment reference {groups} - internal marker, never repeat it."

    def _apply_usage(self, record: CallRecord, usage: fmts.Usage) -> None:
        record.input_tokens = usage.input_tokens
        record.output_tokens = usage.output_tokens
        record.cached_input_tokens = usage.cached_input_tokens
        record.cache_write_tokens = usage.cache_write_tokens
        prices = self.prices_for(record.member)
        record.price_known = prices.lookup(record.response_model or record.model) is not None
        record.cost_usd = prices.cost(record.response_model or record.model, usage)

    def _record(self, record: CallRecord) -> None:
        record.latency_ms = int((time.perf_counter() - record.started) * 1000)
        record.rule_ids = sorted(set(record.rule_ids))
        try:
            from znyx_runtime.gateway.usage import record_call
            record_call(record, app_label=record.app_id if record.app_id in self._known_apps else "other")
        except Exception:  # noqa: BLE001 - recording must never fail a call
            logger.debug("gateway: recording failed", exc_info=True)
        if self.on_call is not None:
            try:
                self.on_call(record)
            except Exception:  # noqa: BLE001
                logger.debug("gateway: on_call failed", exc_info=True)
