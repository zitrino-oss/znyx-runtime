"""One LLM call through the gateway: identify, permit, optimise, guard input, forward,
guard output, record (plan section 4). The optimisation layer (``gateway.optimise``)
changes the request before the input guard reads it and adds cache markers after;
it cannot skip or weaken a check.

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
from znyx_core import identity, usage_events

from znyx_runtime.gateway import formats as fmts
from znyx_runtime.gateway.config import (
    GatewayConfig, GatewayKey, Provider, instance_key, key_digest,
)
from znyx_runtime.gateway.optimise import Optimiser
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
BATCH_PRICE_SHARE = 0.5              # batch calls are billed at half the standard price
# Usage is estimated when the provider's count never arrives (a stream cut short, a
# client that left): about four characters per token, and a flat amount per image or
# file. Budgets are then charged the estimate rather than nothing.
CHARS_PER_TOKEN = 4
OPAQUE_PART_TOKENS = 85


def _clean_spans(spans: Any, length: int) -> List[Tuple[int, int]]:
    """Spans a model server returned, made safe: whole numbers within the text, in order,
    not overlapping; anything malformed is dropped."""
    out: List[Tuple[int, int]] = []
    if not isinstance(spans, list):
        return out
    for item in spans[:10_000]:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        s, e = item[0], item[1]
        if isinstance(s, bool) or isinstance(e, bool) or not isinstance(s, int) or not isinstance(e, int):
            continue
        if 0 <= s < e <= length:
            out.append((s, e))
    out.sort()
    merged: List[Tuple[int, int]] = []
    for s, e in out:
        if merged and s < merged[-1][1]:
            continue                          # overlapping: the earlier span wins
        merged.append((s, e))
    return merged


def stage_c_parts(fmt, body: Dict[str, Any]) -> List[fmts.Part]:
    from znyx_runtime.gateway.optimise import stage_c
    return stage_c.retrieved_parts(fmt, body)


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
    # Phase 4: the optimisation ledger (``optimise.Plan.settle`` fills these).
    profile: Optional[str] = None
    optimisations: List[Dict[str, Any]] = field(default_factory=list)
    saved_usd: Optional[float] = None
    baseline_cost_usd: Optional[float] = None
    # Phase 5: the tools offered to the model and those it called (names only), for the
    # inventory.
    tools_offered: List[str] = field(default_factory=list)
    tools_called: List[str] = field(default_factory=list)
    mcp_tools_changed: List[str] = field(default_factory=list)   # "tool@sha256" held by the MCP gateway
    mcp_tools_seen: List[str] = field(default_factory=list)      # "tool@sha256" every allowed tool listed
    # Phase 5 part 3: the call ran for a person a delegated token names (``user`` is that person).
    delegated: bool = False
    identity_provider: Optional[str] = None
    # Phase 5 part 4: which instructions ran (usage_events.prompt_fingerprint) and the caller's
    # name for them (X-Znyx-Prompt).
    prompt_hash: Optional[str] = None
    prompt_ref: Optional[str] = None
    plan: Any = None

    def note(self, technique: str, count: int = 1) -> None:
        self.techniques[technique] = self.techniques.get(technique, 0) + count

    def saw_tool_call(self, name: Optional[str]) -> None:
        if name and name not in self.tools_called and len(self.tools_called) < 50:
            self.tools_called.append(name[:200])

    def to_event(self) -> Dict[str, Any]:
        data = {k: v for k, v in self.__dict__.items()
                if not k.startswith("_") and k not in ("started", "estimated_input_tokens", "member", "started_ns", "plan", "_recorded",
                             "_cache_user", "_stops", "_no_hold", "_delegated_tools", "_person")}
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
        self.optimiser = Optimiser()
        self._instance_key = instance_key(config.settings)
        self._slots = asyncio.Semaphore(config.settings.max_concurrent)
        self._slots_size = config.settings.max_concurrent
        self._known_apps = self._app_labels(config)
        self._stops_memo: Optional[tuple] = None
        self._live_memo: Optional[tuple] = None
        from znyx_runtime.gateway.approval import ApprovalBroker
        from znyx_runtime.gateway.mcp import McpGateway
        from znyx_runtime.gateway.identity import DelegationService
        from znyx_runtime.governance_audit import GovernanceAudit
        self.governance = GovernanceAudit()
        from znyx_runtime.gateway.defence import DefenceOutbox
        self.defence = DefenceOutbox(self.governance.path.with_name("shared-defence.spool"))
        self.approvals = ApprovalBroker(self)
        self.mcp = McpGateway(self)
        self.delegations = DelegationService(self)

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
        await self.drain()
        await self.governance.drain()
        await self.approvals.close()
        await self.upstream.close()

    def prices_for(self, member: str) -> PriceTable:
        """The price table of the key's own environment (its bundle's)."""
        return self._member_prices.get(member) or self.prices

    @staticmethod
    def _caller_id(key: GatewayKey) -> str:
        """Whose calls optimisation may share or remember: the key itself (its digest),
        so two keys never meet even when their ids are the same."""
        return f"{key.member}/{key.key_sha256 or key.id}"

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
        record._reauthorize = lambda: self.current_identity(request)
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
                if getattr(record, "_reservation", None) is not None:
                    # Includes cancellation and errors before _forward can settle.
                    if getattr(record, "_upstream_started", False):
                        self.estimate_missing_usage(record, 0)
                    self._charge(key, record)

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

    def _user_of(self, request: Request, body: Dict[str, Any], mode: Optional[str] = None) -> Optional[str]:
        """The end user the app called for: ``X-Znyx-User``, else the API's own field
        (OpenAI ``user``, Anthropic ``metadata.user_id``). Pseudonymised by default
        (``settings.user_ids``: hash | plain | off): a keyed hash whose key stays on the
        runtime side, so the console can group spend by user without learning, or guessing,
        who the users are."""
        mode = mode or self.config.settings.user_ids
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
        if getattr(record, "_reservation", None) is not None and record.plan is not None:
            if getattr(record, "_upstream_started", False):
                self.estimate_missing_usage(record, 0)
            self._charge(record.plan.key, record)
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
        from znyx_runtime.gateway.identity import TOKEN_PREFIX
        request.state.znyx_delegation = None
        if presented.startswith(TOKEN_PREFIX):
            return self._identify_delegated(request, presented), None
        digest = key_digest(presented)
        key = self.config.keys_by_digest.get(digest)
        if key is None and self._instance_key is not None and digest == self._instance_key.key_sha256:
            key = self._instance_key
        if key is None or key.disabled or not self.key_usable(key):
            raise Denied(401, "Invalid gateway key", "invalid_api_key", error_type="authentication_error")
        client_credential = standard if explicit and standard else None
        return key, client_credential

    def _identify_delegated(self, request: Request, token: str) -> GatewayKey:
        """A delegated token (part 3): the agent's key, which must still be usable, and what the
        token may do now (``request.state.znyx_delegation``)."""
        from znyx_runtime.bundle_manager import ScopeNotServed
        from znyx_runtime.gateway.identity import TokenRefused
        refused = Denied(401, "Invalid or expired delegated token", "invalid_api_key",
                         error_type="authentication_error")
        entry = self.delegations.resolve(token)
        if entry is None:
            raise refused
        key = self.config.keys_by_digest.get(str(entry.get("key") or ""))
        if key is None or key.disabled or not self.key_usable(key):
            raise refused
        try:
            member = self.bundle_manager.member_by_key(key.member or None)
            request.state.znyx_delegation = self.delegations.delegation_now(entry, key, member)
        except (ScopeNotServed, TokenRefused):
            raise refused from None
        return key

    def delegation_context(self, request: Request, key: GatewayKey, member):
        """The call's delegation, or None; refused when the agent's rules say it acts only for a
        signed-in person and the call came with the key alone."""
        delegation = getattr(request.state, "znyx_delegation", None)
        if delegation is None and not key.instance_key:
            rules = self.delegations.agent_rules(key, member)
            if rules is not None and rules.require_delegation:
                raise Denied(403, "This agent acts only for a signed-in person: exchange the person's token at "
                                  "/v1/znyx/token and call with the token it gives", "delegation_required",
                             error_type="permission_error")
        return delegation

    def current_identity(self, request: Request):
        """Authenticate again after a long wait, including revocation, token expiry,
        changed delegation rules and an agent newly requiring delegation."""
        key, _ = self._identify(request)
        member = self.bundle_manager.member_by_key(key.member or None)
        return key, member, self.delegation_context(request, key, member)

    def current_caller(self, record: CallRecord):
        refresh = getattr(record, "_reauthorize", None)
        if refresh is not None:
            return refresh()
        # Internal callers without an HTTP request may use a static key, but may
        # never manufacture or extend a delegated identity.
        key = next((k for k in self.config.keys_by_digest.values()
                    if k.id == record.key_id and k.member == record.member), None)
        if key is None or key.disabled or not self.key_usable(key) or record.delegated:
            raise Denied(401, "The caller is no longer authorized", "invalid_api_key")
        return key, self.bundle_manager.member_by_key(key.member or None), None

    def apply_delegation(self, record: CallRecord, delegation) -> None:
        if delegation is None:
            return
        record.user = self._person_id(delegation.person)
        record._cache_user = record.user or self._person_id(delegation.person, mode="hash")
        record.delegated = True
        record.identity_provider = delegation.provider
        record._delegated_tools = delegation.tools
        record._person = delegation.person                 # the raw id, for stops only; never recorded
        record.note("delegated")

    def _person_id(self, raw: str, mode: Optional[str] = None) -> Optional[str]:
        """A verified person's id as usage records keep it (``settings.user_ids``)."""
        mode = mode or self.config.settings.user_ids
        if mode == "off" or not raw:
            return None
        if mode == "plain":
            return raw[:200]
        import hmac
        from znyx_runtime.install_state import user_id_key
        scope = self.bundle_manager.members[0].bundle.org_id if getattr(self.bundle_manager, "members", None) else ""
        digest = hmac.new(user_id_key(), f"{scope}\x1f{raw[:200]}".encode("utf-8", "surrogatepass"), hashlib.sha256)
        return "u_" + digest.hexdigest()[:16]

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

    @staticmethod
    def _session_of(request: Request, record: CallRecord) -> Optional[str]:
        """What a runaway-loop allowance is counted for: the session the caller names, else a
        trace it names (a call that starts its own trace is a session of one)."""
        session = (request.headers.get("x-znyx-session-id") or "").strip()
        if session:
            return "s:" + session[:128]
        if request.headers.get("traceparent") or request.headers.get("x-znyx-trace-id"):
            return "t:" + record.trace_id
        return None

    def _check_loop(self, key: GatewayKey, plan) -> None:
        """Technique 6: a session that has used its allowance of calls or tokens within the
        window is stopped before anything is checked or sent."""
        limits = plan.loop_limits()
        if not limits:
            return
        from znyx_core.state_store import StateUnavailable, get_state_store
        from znyx_runtime.gateway.optimise.stage_b import loop_key
        max_calls, max_tokens, window = limits
        try:
            used = get_state_store().get(loop_key(self._caller_id(key), plan.session))
        except StateUnavailable:
            raise Denied(503, "Limits cannot be checked right now (fail-closed)", "limits_unavailable",
                         outcome="error", error_type="api_error", retry_after=1) from None
        over = ((max_calls and used.get("calls", 0) >= max_calls)
                or (max_tokens and used.get("tokens", 0) >= max_tokens))
        if over:
            plan.effect("output_caps")
            what = f"{max_calls} calls" if max_calls and used.get("calls", 0) >= max_calls else f"{max_tokens} tokens"
            raise Denied(429, f"This session reached its allowance of {what} within {window} seconds and was "
                              "stopped; start a new session or wait", "agent_loop_stopped",
                         error_type="rate_limit_error", retry_after=min(window, 60))

    def _count_loop(self, key: GatewayKey, record: CallRecord, tokens: int) -> None:
        plan = record.plan
        limits = plan.loop_limits() if plan is not None else None
        if not limits:
            return
        try:
            from znyx_core.state_store import get_state_store
            from znyx_runtime.gateway.optimise.stage_b import loop_key
            get_state_store().incr(loop_key(self._caller_id(key), plan.session), {"calls": 1, "tokens": tokens},
                                   ttl_seconds=limits[2])
        except Exception:  # noqa: BLE001 - the call already happened; never fail its response
            logger.warning("gateway: could not count the session's usage for key %s", key.id)

    def _count_risk(self, record: CallRecord) -> None:
        plan = record.plan
        if not plan.on("risk_adaptive") or not plan.session:
            return
        from znyx_runtime.gateway.optimise import stage_b
        points = stage_b.risk_points(record)
        if not points:
            return
        try:
            from znyx_core.state_store import get_state_store
            get_state_store().incr(stage_b.risk_key(plan.caller, plan.session), {"score": points},
                                   ttl_seconds=stage_b.RISK_TTL_SECONDS)
        except Exception:  # noqa: BLE001 - recording must never fail a call
            logger.warning("gateway: could not record the session's risk for key %s", record.key_id)

    def _near_budget(self, key: GatewayKey, member, plan) -> bool:
        """Technique 7: whether one of the key's budgets is nearly used: its daily limits on
        this runtime, or a console budget in its scope (``controls.spend_near``)."""
        from znyx_runtime.gateway.optimise.stage_b import DEFAULT_DEGRADE_AT_PERCENT
        share = (plan.opts.setting("degrade_at_percent") or DEFAULT_DEGRADE_AT_PERCENT) / 100.0
        limits = key.limits
        if (limits.tokens_per_day or limits.spend_usd_per_day) and not key.instance_key:
            try:
                from znyx_core.state_store import get_state_store
                used = get_state_store().get(self._day_key(key))
            except Exception:  # noqa: BLE001 - the limits themselves were checked on the way in
                used = {}
            if limits.tokens_per_day and used.get("tokens", 0) >= limits.tokens_per_day * share:
                return True
            if limits.spend_usd_per_day and used.get("cost", 0) >= limits.spend_usd_per_day * share:
                return True
        controls = getattr(self.bundle_manager, "controls", None) or {}
        project = member.project_id if member is not None else ""
        environment = member.environment_id if member is not None else ""
        now = datetime.now(timezone.utc)
        wanted = int(share * 100) // 10 * 10           # the key's own percentage, in the bands sent
        for entry in (controls.get("spend_near") or []) if isinstance(controls, dict) else []:
            if not isinstance(entry, dict) or not self._in_scope(entry, key, project, environment):
                continue
            band = entry.get("band")
            if isinstance(band, (int, float)) and not isinstance(band, bool) and band < wanted:
                continue                                 # used, but not yet as far as this key wants
            try:
                until = datetime.fromisoformat(str(entry.get("until")).replace("Z", "+00:00"))
                if (until if until.tzinfo else until.replace(tzinfo=timezone.utc)) <= now:
                    continue
            except ValueError:
                pass
            return True
        return False

    @classmethod
    def _day_key(cls, key: GatewayKey) -> str:
        # "pin:": never evicted from the in-memory store by other traffic.
        return f"pin:gw:day:{cls._state_id(key)}:{time.strftime('%Y%m%d', time.gmtime())}"

    def _charge(self, key: GatewayKey, record: CallRecord) -> None:
        """The call's own usage to the key's day and its session (extra calls, such as
        retries and votes, were charged when they were made)."""
        plan = record.plan
        added_tokens = plan.extra_tokens_added if plan is not None else 0
        added_cost = plan.extra_cost_added if plan is not None else 0.0
        tokens = max(0, (record.input_tokens or 0) + (record.output_tokens or 0) - added_tokens)
        self._count_loop(key, record, tokens + added_tokens)
        cost = max(0.0, (record.cost_usd or 0.0) - added_cost)
        reserved = getattr(record, "_reservation", None)
        if reserved is not None:
            record._reservation = None
            from znyx_core.state_store import get_state_store
            try:
                # Settle on the day reserved, even when a stream crosses midnight.
                get_state_store().incr(reserved[0], {"tokens": tokens - reserved[1],
                                                     "cost": cost - reserved[2], "requests": 1}, 2 * 86400)
            except Exception:
                # Keep the reservation when settlement cannot reach the store.
                logger.warning("gateway: could not settle usage for key %s", key.id)
        else:
            self._charge_usage(key, tokens, cost, requests=1)

    def _reserve_call(self, key: GatewayKey, record: CallRecord, body: Dict[str, Any], provider,
                      *, factor: int = 1, discount: float = 1.0, price_override=None) -> None:
        """Reserve the request and its maximum answer atomically before sending.
        When no answer cap was supplied, bound it to the remaining allowance. Token
        counts are estimates unless a tokenizer is installed; provider usage settles them."""
        limits = key.limits
        if key.instance_key or not (limits.tokens_per_day or limits.spend_usd_per_day):
            return
        from znyx_core.state_store import StateUnavailable, get_state_store
        store = get_state_store()
        day = self._day_key(key)
        old = getattr(record, "_reservation", None)
        if old is not None:
            # Re-preparing after a discarded parallel call: its usage was charged
            # by _bill_early, and this is a different provider request.
            store.incr(old[0], {"tokens": -old[1], "cost": -old[2]}, 2 * 86400)
            record._reservation = None
        model = body.get("model") or record.model
        price = price_override or self.prices_for(record.member).lookup(model)
        if limits.spend_usd_per_day and price is None:
            raise Denied(403, "The model has no price for reserving its spend", "price_unknown")
        # Include the message envelope and a margin, not just the visible text.
        from znyx_runtime.gateway.optimise import tokens as token_counts
        input_tokens = max(record.estimated_input_tokens,
                           token_counts.count(json.dumps(body, ensure_ascii=False), provider.format, model) + 32)
        record.estimated_input_tokens = input_tokens
        asked = body.get("max_completion_tokens") or body.get("max_tokens")
        explicit = isinstance(asked, int) and not isinstance(asked, bool) and asked > 0
        output_tokens = asked if explicit else 1000
        try:
            used = store.get(day)
            if not explicit:
                if limits.tokens_per_day:
                    output_tokens = min(output_tokens, math.floor(
                        (limits.tokens_per_day - used.get("tokens", 0)) / factor - input_tokens))
                if limits.spend_usd_per_day and price and price["output_per_mtok"] > 0:
                    available = (limits.spend_usd_per_day - used.get("cost", 0)) / (factor * discount)
                    output_tokens = min(output_tokens, math.floor(
                        (available * 1_000_000 - input_tokens * price["input_per_mtok"]) / price["output_per_mtok"]))
            if output_tokens < 1:
                raise Denied(429, "The remaining daily budget cannot cover this request", "insufficient_quota",
                             error_type="rate_limit_error")
            reserved_tokens = factor * (input_tokens + output_tokens)
            reserved_cost = factor * discount * (
                input_tokens * price["input_per_mtok"] + output_tokens * price["output_per_mtok"]
            ) / 1_000_000 if price else 0.0
            bounds = {}
            if limits.tokens_per_day:
                bounds["tokens"] = limits.tokens_per_day
            if limits.spend_usd_per_day:
                bounds["cost"] = limits.spend_usd_per_day
            if not store.reserve(day, {"tokens": reserved_tokens, "cost": reserved_cost}, bounds, 2 * 86400):
                raise Denied(429, "Daily budget is spent or reserved by other calls; retry after they finish",
                             "insufficient_quota", error_type="rate_limit_error")
        except StateUnavailable:
            raise Denied(503, "Budgets cannot be reserved right now (fail-closed)", "limits_unavailable",
                         outcome="error", retry_after=1) from None
        record._reservation = (day, reserved_tokens, reserved_cost)
        if not explicit:
            from znyx_runtime.gateway.optimise.stage_b import apply_output_cap
            apply_output_cap(record.api, provider, body, output_tokens)

    def _charge_usage(self, key: GatewayKey, tokens: int, cost: float, requests: int = 0) -> None:
        """Add usage to the key's day (its daily limits)."""
        if not (key.limits.tokens_per_day or key.limits.spend_usd_per_day) or key.instance_key:
            return
        try:
            from znyx_core.state_store import get_state_store
            get_state_store().incr(self._day_key(key), {"tokens": tokens, "cost": cost, "requests": requests},
                                   ttl_seconds=2 * 86400)
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
    def _has_tool_blocks(body: Dict[str, Any]) -> bool:
        for message in body.get("messages") or []:
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, list) and any(isinstance(b, dict) and str(b.get("type", "")).endswith(
                    ("tool_use", "tool_result")) for b in content):
                return True
        return False

    def _note_prompt(self, fmt, body: Dict[str, Any], request: Request, record: CallRecord) -> None:
        """The prompt version (part 4), from the caller's instructions before the gateway adds
        anything of its own; keyed like user pseudonyms (same key and scope)."""
        try:
            from znyx_runtime.install_state import user_id_key
            scope = self.bundle_manager.members[0].bundle.org_id if getattr(self.bundle_manager, "members", None) else ""
            record.prompt_hash = usage_events.prompt_fingerprint(fmt.hidden_context(body), key=user_id_key(),
                                                                 scope=scope or "")
        except Exception:  # noqa: BLE001 - a fingerprint never fails a call
            record.prompt_hash = None
        record.prompt_ref = usage_events.prompt_ref(request.headers.get("x-znyx-prompt"))

    @staticmethod
    def _note_offered(body: Dict[str, Any], record: CallRecord, name_of, fname) -> None:
        tools = body.get("tools")
        names = [name_of(t) for t in tools] if isinstance(tools, list) else []
        if isinstance(body.get("functions"), list):
            names += [fname(f) for f in body["functions"]]
        record.tools_offered = list(dict.fromkeys(n[:200] for n in names if n))[:50]

    @staticmethod
    def _filter_tools(api: str, body: Dict[str, Any], policy: Dict[str, Any],
                      record: CallRecord) -> None:
        """Least-privilege tools (technique 26): the model only sees tools the policy
        allows, when the policy names them, and none a stop concerns (a stopped tool; every
        tool for a quarantined caller). Legacy OpenAI ``functions`` too, and a forced choice
        of a removed tool is dropped. The names offered are kept for the inventory."""
        tools_policy = policy.get("tools") or {}
        if not isinstance(tools_policy, dict) or not tools_policy.get("enabled", True):
            tools_policy = {}
        allowed = set(tools_policy.get("allowed") or tools_policy.get("allowed_tools") or [])
        denied = set(tools_policy.get("denied") or tools_policy.get("blocked_tools") or [])
        stops = getattr(record, "_stops", None)
        delegated = getattr(record, "_delegated_tools", None)
        restricted = (stops is not None and stops.restricts_tools) or delegated is not None

        def permitted(name: str) -> bool:
            # Under a stop or a delegated token, a tool whose name cannot be read is removed too.
            if restricted and not name:
                return False
            if stops is not None and (stops.quarantined is not None or stops.tool_stop(name) is not None):
                return False
            if delegated is not None and not identity.tool_granted(delegated, name):
                return False
            return (not allowed or name in allowed) and name not in denied

        def name_of(tool: Any) -> str:
            if not isinstance(tool, dict):
                return ""
            if api == "anthropic":
                return str(tool.get("name") or "")
            for kind in ("function", "custom"):
                inner = tool.get(kind)
                if isinstance(inner, dict) and inner.get("name"):
                    return str(inner["name"])
            return ""

        def fname(f: Any) -> str:
            return str(f.get("name") or "") if isinstance(f, dict) else ""

        if not (allowed or denied or restricted):
            Gateway._note_offered(body, record, name_of, fname)
            return
        if restricted and api == "anthropic" and body.get("mcp_servers"):
            # MCP servers the provider calls itself: their tools cannot be filtered one by one.
            body.pop("mcp_servers", None)
            record.note("mcp_servers_removed")
        offered_before = list(body.get("tools") or []) if isinstance(body.get("tools"), list) else []

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
            kept_functions = [f for f in functions if permitted(fname(f))]
            removed |= {fname(f) for f in functions if not permitted(fname(f))}
            if len(kept_functions) != len(functions):
                if kept_functions:
                    body["functions"] = kept_functions
                else:
                    body.pop("functions", None)
                    body.pop("function_call", None)
        Gateway._note_offered(body, record, name_of, fname)
        if not removed:
            return
        record.note("tools_filtered", len(removed))
        if restricted and stops.quarantined is not None:
            record.note("quarantined")
        if not body.get("tools") and not body.get("functions"):
            # Settings that are only valid with tools would make the provider refuse the request.
            for name in ("parallel_tool_calls", "function_call"):
                body.pop(name, None)
            if api == "anthropic" and offered_before and Gateway._has_tool_blocks(body):
                # Anthropic refuses a conversation with tool blocks and no tools: offer one inert
                # placeholder the model may not call.
                body["tools"] = [{"name": "znyx_no_tools", "description": "No tools are available.",
                                  "input_schema": {"type": "object", "properties": {}}}]
                body["tool_choice"] = {"type": "none"}
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

    # -- the kill switch (Phase 5) ---------------------------------------------------

    def _controls(self) -> Dict[str, Any]:
        controls = getattr(self.bundle_manager, "controls", None) or {}
        return controls if isinstance(controls, dict) else {}

    def key_usable(self, key: GatewayKey) -> bool:
        """False for a key the console no longer lists as usable (disabled, deleted, rotated or
        expired since the last publish): the signed controls carry the usable keys' digests,
        so such a key stops working with the next poll. Only when the controls carry the list
        (managed mode, where every key comes from the console); never the instance key."""
        if key.instance_key:
            return True
        live = self._controls().get("live_keys")
        if not isinstance(live, list):
            return True
        memo = self._live_memo
        if memo is None or memo[0] is not live:
            memo = self._live_memo = (live, frozenset(d for d in live if isinstance(d, str)))
        return key.key_sha256 in memo[1]

    def stop_set(self):
        """The active stops: the console's (signed controls) and the gateway file's."""
        from znyx_core.governance import StopSet, clean_stops
        raw = self._controls().get("stops")
        memo = self._stops_memo
        if memo is not None and memo[0] is raw and memo[1] is self.config:
            return memo[2]
        stops = StopSet(clean_stops(raw, "bundle set controls") + tuple(self.config.stops))
        self._stops_memo = (raw, self.config, stops)
        return stops

    def stops_for(self, key: GatewayKey, member, app_id: Optional[str] = None, agent_id: Optional[str] = None,
                  person: Optional[str] = None):
        from znyx_core.governance import NO_STOPS
        stops = self.stop_set()
        if not stops:
            return NO_STOPS
        return stops.for_caller(project_id=member.project_id if member is not None else "",
                                environment_id=member.environment_id if member is not None else "",
                                app_id=app_id or key.app_id, agent_id=agent_id or key.agent_id,
                                key_id=None if key.instance_key else key.id, user=person)

    def _check_controls(self, key: GatewayKey, member, model: Optional[str] = None, provider=None,
                        deployment: Optional[str] = None, person: Optional[str] = None):
        """The console's controls for a call: stops (the caller, the model, the provider),
        then budgets. Returns the caller's stops (quarantine, stopped tools). On Azure the
        deployment is what runs, so a model stop is checked against it as well."""
        from znyx_core.governance import NO_STOPS
        call = NO_STOPS
        stops = self.stop_set()
        if stops:
            project = member.project_id if member is not None else ""
            environment = member.environment_id if member is not None else ""
            call = self.stops_for(key, member, person=person)
            hit = (call.stopped or stops.model(model, project, environment)
                   or (stops.model(deployment, project, environment) if deployment else None)
                   or (stops.provider(provider.name, project, environment) if provider is not None else None))
            if hit is not None:
                raise Denied(403, hit.message(), "stopped", error_type="permission_error")
        self._check_spend_controls(key, member, model)
        return call

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

        if kind == "tool_call":
            record.saw_tool_call(tool_name)
            delegated = getattr(record, "_delegated_tools", None)
            if delegated is not None and not identity.tool_granted(delegated, tool_name):
                record.note("tool_call_not_delegated")
                return _Verdict("BLOCK", [identity.RULE_NOT_DELEGATED],
                                user_message="This tool is not among those granted for the person this call is for",
                                hits=1)
            stops = getattr(record, "_stops", None)
            hit = (stops.quarantined or stops.tool_stop(tool_name)) if stops is not None and stops.restricts_tools \
                else None
            if hit is not None:
                from znyx_core.governance import RULE_QUARANTINED, RULE_TOOL_STOPPED
                record.note("tool_call_stopped")
                return _Verdict("BLOCK", [RULE_QUARANTINED if hit is stops.quarantined else RULE_TOOL_STOPPED],
                                user_message=hit.message(), hits=1)
        if kind in ("tool_result", "tool_call"):
            args: Dict[str, Any] = {}
            if kind == "tool_call":
                try:
                    parsed = json.loads(text) if text else {}
                    args = parsed if isinstance(parsed, dict) else {"value": parsed}
                except ValueError:
                    args = {"raw": text}
            content_block = None
            if kind == "tool_call" and text:
                # What the call would send (a secret, say) under the policy's content checks; only
                # a block counts: a tool needs the values it is given, so nothing is redacted.
                content_req = self._request(scope, record, label + "c", text, role, None)
                content = await self.evaluator.evaluate(content_req, context="output", policy=policy,
                                                        judge_ctx=runtime_judge_ctx(policy, content_req))
                if content.decision.value == "BLOCK":
                    content_block = _Verdict.of(content)
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
            verdict = _Verdict.of(result)
            return verdict.merge(content_block) if content_block is not None else verdict
        req = self._request(scope, record, label, text, role, context)
        result = await self.evaluator.evaluate(req, context=stage, policy=policy,
                                               judge_ctx=runtime_judge_ctx(policy, req))
        return _Verdict.of(result)

    async def _check_text(self, text: str, kind: str, role: str, tool_name: Optional[str], stage: str,
                          scope: Dict[str, str], record: CallRecord, policy: Dict[str, Any],
                          quiet: Dict[str, Any], label: str,
                          context: Optional[Dict[str, Any]] = None, approve: bool = True) -> _Verdict:
        """A check of ``text``, and of its detection copy when invisible characters
        could be hiding something from it. A tool call in a model's answer that the policy
        marks for approval then waits for a person (``approve``: False where the caller asks
        itself, as the MCP gateway does)."""
        from znyx_core.defence import match
        feed = self._controls().get("shared_defence") or {}
        if match(text, feed):
            return _Verdict("BLOCK", ["shared_defence.attack_fingerprint"],
                            user_message="This content matches a reviewed attack fingerprint", hits=1)
        verdict = await self._check_one(text, kind, role, tool_name, stage, scope, record, policy,
                                        label, context)
        probe = detection_copy(text)
        if probe != text:
            record.note("checked_without_invisible")
            verdict = verdict.merge(await self._check_one(probe, kind, role, tool_name, stage, scope, record,
                                                          quiet, label + "n", context))
        if kind == "tool_call" and approve and verdict.decision != "BLOCK":
            verdict = await self._approved(text, tool_name, policy, record, verdict)
        if verdict.decision == "BLOCK":
            try:
                await asyncio.to_thread(self.defence.record, text, verdict.rule_ids, feed)
            except Exception:
                logger.warning("Shared defence contribution could not be recorded")
        return verdict

    async def _approved(self, text: str, tool_name: Optional[str], policy: Dict[str, Any], record: CallRecord,
                        verdict: _Verdict) -> _Verdict:
        """Technique 33 on a tool call in a model's answer: the answer waits while a person
        decides; not approved (or in a batch, where nobody can wait), the call is withheld."""
        from znyx_core import approvals
        rules = approvals.rules_of(policy)
        reason = rules.reason(tool_name) if rules is not None else None
        if reason is None:
            return verdict
        if getattr(record, "_no_hold", False):
            record.note("approval_not_possible")
            return _Verdict("BLOCK", verdict.rule_ids + [approvals.RULE_BATCH],
                            user_message="This tool call needs a person's approval, which a batch call cannot wait for",
                            hits=verdict.hits + 1)
        member = None
        try:
            member = self.bundle_manager.member_by_key(record.member or None)
        except Exception:  # noqa: BLE001 - the request names no environment then
            member = None
        arguments = verdict.sanitized_tool_args if verdict.sanitized_tool_args is not None else text
        try:
            approvals.parse_arguments(arguments)
        except approvals.AmbiguousArguments as exc:
            return _Verdict("BLOCK", verdict.rule_ids + [approvals.RULE_AMBIGUOUS],
                            user_message=f"This tool call's arguments are ambiguous ({exc})", hits=verdict.hits + 1)
        approved, why = await self.approvals.ask(rules=rules, reason=reason, tool=tool_name or "tool",
                                                 arguments=arguments, server=None, record=record, member=member)
        if approved:
            try:
                key, member, delegation = self.current_caller(record)
                self.apply_delegation(record, delegation)
                stops = self.stops_for(key, member, person=delegation.person if delegation else None)
                record._stops = stops
                current_scope = {"tenant_id": key.tenant_id, "app_id": key.app_id, "agent_id": key.agent_id,
                                 "env": self.bundle_manager.environment_for(member) or key.environment or "prod",
                                 "session_id": ""}
                current_policy = self.bundle_manager.get_policy(
                    tenant_id=key.tenant_id, app_id=key.app_id, agent_id=key.agent_id,
                    env=current_scope["env"], member=member)
            except Exception:  # authentication, shared state or policy unavailable: refuse
                return _Verdict("BLOCK", verdict.rule_ids + [approvals.RULE_NOT_APPROVED],
                                user_message="Approved, but the caller is no longer authorized", hits=verdict.hits + 1)
            hit = stops.stopped or stops.quarantined or stops.tool_stop(tool_name)
            if hit is not None:
                return _Verdict("BLOCK", verdict.rule_ids + [approvals.RULE_NOT_APPROVED],
                                user_message=f"Approved, but stopped meanwhile: {hit.message()}", hits=verdict.hits + 1)
            # Permissions and argument rules may have changed during the wait too.
            quiet = quiet_policy(current_policy)
            checked = await self._check_text(text, "tool_call", "assistant", tool_name, "output",
                                              current_scope, record, quiet, quiet, "approved", approve=False)
            if checked.sanitized_tool_args is not None and checked.sanitized_tool_args != approvals.parse_arguments(arguments):
                return _Verdict("BLOCK", verdict.rule_ids + [approvals.RULE_NOT_APPROVED],
                                user_message="The current policy changed the approved arguments; request approval again",
                                hits=verdict.hits + 1)
            return verdict.merge(checked)
        return _Verdict("BLOCK", verdict.rule_ids + [approvals.RULE_NOT_APPROVED],
                        user_message=f"This tool call was not approved: {why}", hits=verdict.hits + 1)

    def _normalise(self, fmt, body: Dict[str, Any], record: Optional[CallRecord],
                   parts: Optional[List[fmts.Part]] = None) -> None:
        """Remove invisible characters from the texts the input guard reads (technique 25),
        before they are checked and sent."""
        if not self.config.settings.normalise_before_send:
            return
        if parts is None:
            parts = [p for p in fmt.input_parts(body, self.config.settings.check_roles) if p.kind != "opaque"]
        for part in parts:
            if part.kind == "history_tool_call" and fmt is fmts.AnthropicMessages:
                continue                       # an object, not a string; its copy is checked
            cleaned, removed = strip_invisible(part.text)
            if removed:
                if record is not None:
                    record.note("invisible_removed", removed)
                fmt.set_text(body, part, cleaned)
                part.text = cleaned

    async def _guard_input(self, api: str, fmt, body: Dict[str, Any], scope: Dict[str, str],
                           policy: Dict[str, Any], record: CallRecord) -> None:
        settings = self.config.settings
        parts = fmt.input_parts(body, settings.check_roles)
        opaque = [p for p in parts if p.kind == "opaque"]
        if opaque and settings.media_inspection:
            from znyx_runtime.gateway.media import inspect_body
            if await inspect_body(self, body, scope, policy, record) < len(opaque):
                raise Denied(400, "Some media could not be inspected", "uncheckable_content", outcome="blocked_input")
            opaque = []
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
        self._normalise(fmt, body, record, parts)
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
        feed = self._controls().get("shared_defence") or {}
        from znyx_runtime.gateway.defence import feed_state
        feed_version = feed_state(feed)
        scope_key = (scope["tenant_id"], scope["app_id"], scope["agent_id"], version, feed_version)
        quiet = quiet_policy(policy)
        gate = asyncio.Semaphore(settings.max_parallel_checks)
        # A session checked more strictly (technique 32) has its whole history read again.
        recheck = bool(getattr(record.plan, "elevated", False))

        async def check(n: int, unit: _Unit) -> Optional[_Verdict]:
            text = unit.text
            cache_key = self.verdicts.key(scope_key, "input:" + unit.first.kind, text)
            if unit is not charged and not recheck and self.verdicts.seen(cache_key):
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
            await self._guard_media_output(fmt, response, scope, policy, record)
            return
        # Content blocks of one answer are one text, just as multipart input is. A
        # secret or PII value may span blocks; checking each independently loses it.
        units: List[_Unit] = []
        text_units: Dict[int, _Unit] = {}
        for part in parts:
            if part.kind == "text":
                unit = text_units.get(part.choice)
                if unit is None:
                    unit = text_units[part.choice] = _Unit([])
                    units.append(unit)
                unit.parts.append(part)
            else:
                units.append(_Unit([part]))
        quiet = quiet_policy(policy)               # the call was charged on the way in
        gate = asyncio.Semaphore(self.config.settings.max_parallel_checks)

        async def check(i: int, unit: _Unit) -> _Verdict:
            part = unit.first
            kind = "tool_call" if part.kind == "tool_call" else "text"
            # The hidden-context leak check reads what the user is shown, not the model's
            # reasoning (which often restates its instructions) or quoted documents.
            ctx = context if part.kind in ("text", "audio") else None
            async with gate:
                return await self._check_text(unit.text, kind, "assistant", part.tool_name, "output",
                                              scope, record, quiet, quiet, f"output{i}", ctx)

        t0 = time.perf_counter()
        verdicts = await asyncio.gather(*(check(i, u) for i, u in enumerate(units)))
        record.guard_ms += int((time.perf_counter() - t0) * 1000)
        withheld, rewritten = set(), set()
        for unit, verdict in zip(units, verdicts):
            part = unit.first
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
            elif verdict.sanitized_text is not None and verdict.sanitized_text != unit.text:
                if part.kind in ("audio", "thinking"):
                    # Audio cannot be edited to match its transcript, nor a signed
                    # thinking block: either is passed on clean or not at all.
                    withheld.add(part.choice)
                else:
                    fmt.set_text(response, part, verdict.sanitized_text)
                    for rest in unit.parts[1:]:
                        fmt.set_text(response, rest, "")
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
        await self._guard_media_output(fmt, response, scope, policy, record)

    async def _guard_media_output(self, fmt, response, scope, policy, record):
        if not self.config.settings.media_inspection:
            return
        from znyx_runtime.gateway import media
        try:
            await media.inspect_body(self, response, scope, policy, record, "output")
            if fmt is fmts.OpenAIChat:
                for choice in response.get("choices") or []:
                    audio = (choice.get("message") or {}).get("audio")
                    if isinstance(audio, dict) and audio.get("data"):
                        data, _ = media.decode(audio["data"])
                        await media.check(self, data, "audio/wav", scope, policy, record, "output")
        except Denied:
            indices = [choice.get("index", i) for i, choice in enumerate(response.get("choices") or [])]
            for index in indices or [0]:
                fmt.withhold_choice(response, index, self.config.settings.output_block_message)
            record.output_decision = "BLOCK"
            record.outcome = "blocked_output"
            record.note("media_withheld")

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
        # The semantic cache keeps end users apart even where user ids are not recorded.
        record._cache_user = record.user or self._user_of(request, body, mode="hash")
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
        delegation = self.delegation_context(request, key, member)
        self.apply_delegation(record, delegation)
        record._stops = self._check_controls(key, member, record.model, provider,
                                             deployment if api == "azure" else None,
                                             person=delegation.person if delegation is not None else None)
        scope = self._scope(key, request, member)
        try:
            policy = self.bundle_manager.get_policy(tenant_id=scope["tenant_id"], app_id=scope["app_id"],
                                                    agent_id=scope["agent_id"], env=scope["env"],
                                                    member=member)
        except RuntimeError:
            raise Denied(503, "Policy unavailable (fail-closed)", "policy_unavailable",
                         outcome="error", error_type="api_error") from None
        self._check_provider_side_calls(api, body)
        self._note_prompt(fmt, body, request, record)
        self._filter_tools(api, body, policy, record)
        if fmt.is_stream(body):
            from znyx_runtime.gateway.optimise.verify import contract_of
            contract = contract_of(policy)
            if contract is not None and str(contract.on_fail or "").lower() == "block":
                # A complete JSON contract cannot be checked before streaming text is
                # released. Refuse this combination rather than weaken the policy.
                raise Denied(400, "This policy requires a complete output contract; send stream=false",
                             "streaming_contract_not_supported", error_type="invalid_request_error")
        plan = record.plan = self.optimiser.plan(key, api, provider, body, request.headers,
                                                 self._caller_id(key), record.model)
        plan.session = self._session_of(request, record)
        plan.key = key
        self._check_loop(key, plan)
        if plan.wants_contract():
            from znyx_runtime.gateway.optimise.verify import contract_of
            plan.contract = contract_of(policy)
        if isinstance(policy.get("hallucination"), dict):
            plan.grounding_config = policy["hallucination"]
        from znyx_runtime.gateway.optimise import stage_b
        policy = self._adapt_to_risk(key, plan, policy, record)
        # Request-side techniques first: the input guard reads the request as it is sent.
        plan.before_guard(fmt, body, record, self.config.settings.check_roles)
        if plan.on("compress") and not fmt.is_stream(body):
            await self._compress(fmt, body, plan, record, live=True)   # the guard reads the shortened text
        input_policy = self._tokenising(fmt, body, plan, policy, record)
        send = dict(fmt=fmt, provider=provider, key=key, member=member, scope=scope, policy=policy, record=record,
                    request=request, client_credential=client_credential, deployment=deployment,
                    api_version=api_version)
        if (plan.on("parallel_input") and not fmt.is_stream(body)
                and not any(p.kind == "opaque" for p in fmt.input_parts(body, self.config.settings.check_roles))
                and not stage_b.input_may_be_rewritten(policy)):
            # Technique 40: send while the input checks run. The policy cannot rewrite the
            # input, so what is sent is what the checks read; it is used only if they pass and
            # left the text exactly as sent, and cancelled otherwise. Nothing is released
            # before the output guard.
            plan.parallel = True             # nothing decided from the checks' findings: they have not run
            early_body = fmts.deep_copy(body)
            self._normalise(fmt, early_body, None)
            expected = self._sent_texts(fmt, early_body)
            context, url, headers = await self._prepare_send(body=early_body, **send)
            early = asyncio.ensure_future(self.upstream.send(provider, url, headers, early_body))
            record._upstream_started = True
            try:
                await self._guard_input(api, fmt, body, scope, input_policy, record)
            except BaseException:
                early.cancel()
                self._bill_early(fmt, key, plan, record, early, apart=False)    # blocked: still paid
                raise
            if self._sent_texts(fmt, body) == expected:
                record.note("parallel_input")
                plan.effect("parallel_input").latency_ms = -record.guard_ms
                return await self._forward(request, fmt, provider, url, headers, early_body, key, scope, policy,
                                           record, context, early=early)
            early.cancel()
            record.note("parallel_input_discarded")
            self._bill_early(fmt, key, plan, record, early, apart=True)
        else:
            await self._guard_input(api, fmt, body, scope, input_policy, record)
        context, url, headers = await self._prepare_send(body=body, **send)
        if self._batchable(request, fmt, provider, body, plan):
            return await self._enqueue_batch(fmt, provider, url, body, key, scope, record, plan)
        if record.stream:
            from znyx_runtime.gateway.streaming import stream_call
            return await stream_call(self, request, api, fmt, provider, url, headers, body,
                                     key, scope, policy, record, context, release)
        return await self._forward(request, fmt, provider, url, headers, body, key, scope, policy,
                                   record, context)

    async def _prepare_send(self, *, fmt, provider: Provider, body: Dict[str, Any], key: GatewayKey, member,
                            scope: Dict[str, str], policy: Dict[str, Any], record: CallRecord, request: Request,
                            client_credential: Optional[str], deployment: Optional[str],
                            api_version: Optional[str]) -> Tuple[Dict[str, Any], str, Dict[str, str]]:
        """Everything after the input guard that makes the request ready to send: the
        canary, the context the output checks read, caps and cache markers, the egress gate
        (audited) and the provider's credential and URL."""
        plan = record.plan
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
        if plan.on("summarise_history") and not fmt.is_stream(body):
            await self._summarise(fmt, provider, url, headers, body, key, scope, policy, record, plan, live=True)
        if self.config.settings.canary_marker:
            fmt.add_system_text(body, self.canary())
            record.note("canary")
        # The request's documents, search results and tool results: what the answer may rest
        # on, for the policy's groundedness and citation checks (in memory, never recorded).
        context = self._output_context(fmt, body)
        plan.sources = context.get("grounding_sources") or []
        sent = fmt.input_parts(body, _ALL_ROLES)
        record.estimated_input_tokens = (
            estimate_tokens(sum(len(p.text) for p in sent if p.kind != "opaque"))
            + OPAQUE_PART_TOKENS * sum(1 for p in sent if p.kind == "opaque"))
        if plan.on("budget_degrade"):
            plan.near_budget = self._near_budget(key, member, plan)
        self._right_size(fmt, body, key, policy, record, plan)
        plan.before_send(fmt, provider, body, record, self.optimiser.seen, self._caller_id(key))
        factor = max(plan.vote_count(body, request.headers),
                     body.get("n") if isinstance(body.get("n"), int) and not isinstance(body.get("n"), bool) else 1, 1)
        self._reserve_call(key, record, body, provider, factor=factor,
                           discount=BATCH_PRICE_SHARE if self._batchable(request, fmt, provider, body, plan) else 1.0)
        self._plan_shadow(fmt, body, key, policy, record, plan, url)
        await self._permit_egress(provider, policy, len(json.dumps(body)), scope, record)
        record.stream = fmt.is_stream(body)
        return context, url, headers

    def _bill_early(self, fmt, key: GatewayKey, plan, record: CallRecord, early, apart: bool) -> None:
        """The early call of the parallel input check, not used (blocked, or the checks
        changed the text): the provider may have billed it, so it is charged, from its own
        usage when it had answered, else as an estimate. ``apart``: as an extra call next to
        the one sent instead; otherwise as this (blocked) call's own usage."""
        usage, estimated = None, False
        if early.done() and not early.cancelled() and early.exception() is None:
            response = early.result()
            if response.status_code != 200:
                return                                   # a provider error is not billed
            try:
                usage = fmt.usage(fmts.parse_response(response.content))
            except (ValueError, RecursionError):
                usage = None
        if usage is None or usage.input_tokens is None:
            usage, estimated = fmts.Usage(record.estimated_input_tokens, 0), True
        if apart:
            self._extra_call(plan, record, "parallel_input", None, usage, estimated=estimated)
            return
        self._apply_usage(record, usage)
        if estimated:
            record.usage_estimated = True
            record.note("usage_estimated")
        self._charge(key, record)

    def _tokenising(self, fmt, body: Dict[str, Any], plan, policy: Dict[str, Any],
                    record: CallRecord) -> Dict[str, Any]:
        """Technique 30: the policy the input guard uses. The guard still reads the original
        text with every check; only the PII check's redaction writes placeholders whose values
        this call keeps, to put back in the answer."""
        from znyx_runtime.gateway.optimise import stage_b
        if not plan.on("pii_tokenise") or fmt.is_stream(body) or not stage_b.pii_applies_to_input(policy.get("pii")):
            return policy
        self._normalise(fmt, body, record)       # the values as the guard will read them
        token_key = self._pii_token_key()
        plan.vault = stage_b.build_vault(fmt, body, self.config.settings.check_roles, policy["pii"], token_key)
        if not len(plan.vault):
            return policy
        record.note("pii_tokenise", len(plan.vault))
        plan.effect("pii_tokenise")
        return stage_b.tokenising_policy(policy, token_key)

    def _pii_token_key(self) -> str:
        """The key placeholders are derived with: the same on every worker of an install
        (from the end-user pseudonym key), so the same value gets the same placeholder."""
        key = getattr(self, "_token_key_cache", None)
        if key is None:
            import hmac
            from znyx_runtime.install_state import user_id_key
            key = self._token_key_cache = hmac.new(user_id_key(), b"znyx-pii-token", hashlib.sha256).hexdigest()
        return key

    # -- batch submission (technique 8) ----------------------------------------------

    def batch_store(self):
        store = getattr(self, "_batch_store", None)
        if store is None:
            from znyx_runtime.gateway.batch import BatchStore
            store = self._batch_store = BatchStore()
        return store

    def _batchable(self, request: Request, fmt, provider: Provider, body: Dict[str, Any], plan) -> bool:
        """A call the app marks as not urgent, on a provider whose batch API the gateway
        speaks, and nothing in this call that only lives in memory (PII placeholders)."""
        from znyx_runtime.gateway.batch import supports_batch
        if str(request.headers.get("x-znyx-urgency") or "").strip().lower() != "low":
            return False
        if getattr(request.state, "znyx_delegation", None) is not None:
            # A call for a person runs live: its tools and the person's stops are checked when
            # it runs, which a queued job could not keep up with.
            return False
        if not plan.on("batch") or fmt.is_stream(body) or len(plan.vault) or plan.shadow is not None:
            return False
        return supports_batch(provider)

    async def _enqueue_batch(self, fmt, provider: Provider, url: str, body: Dict[str, Any], key: GatewayKey,
                             scope: Dict[str, str], record: CallRecord, plan) -> Response:
        """Keep the checked call for the batch API and answer 202. What it will likely use is
        charged to the key's day and session now (a reservation), so queued calls cannot get
        around the key's limits; the difference is settled when the answer arrives."""
        import dataclasses
        asked = body.get("max_completion_tokens") or body.get("max_tokens")
        asked = asked if isinstance(asked, int) and not isinstance(asked, bool) and asked > 0 else 1000
        reserve_tokens = record.estimated_input_tokens + asked
        standard = self.prices_for(record.member).cost(record.model, fmts.Usage(record.estimated_input_tokens, asked))
        reserve_cost = round((standard or 0.0) * BATCH_PRICE_SHARE, 8)
        reserved = getattr(record, "_reservation", None)
        if reserved is not None:
            reserve_tokens, reserve_cost = reserved[1:]
        else:
            self._charge_usage(key, reserve_tokens, reserve_cost, requests=1)
        self._count_loop(key, record, reserve_tokens)
        fields = {f.name: getattr(record, f.name) for f in dataclasses.fields(record) if f.name != "plan"}
        fields.update(profile=plan.profile, scope=scope, reserved={"tokens": reserve_tokens, "cost": reserve_cost,
                                                                 "day": reserved[0] if reserved else self._day_key(key),
                                                                 "held": reserved is not None})
        provider_key = f"{provider.member}/{provider.name}" if provider.member else provider.name
        job_id = await asyncio.to_thread(
            self.batch_store().add, key_sha256=key.key_sha256 or key.id, member=key.member, api=record.api,
            provider=provider_key, url=url, body=body, record=fields)
        record._reservation = None                  # ownership passes to the durable job
        headers = self.trace_headers(record)
        return JSONResponse({"id": job_id, "object": "znyx.batch_job", "status": "queued",
                             "poll": f"/v1/znyx/batches/{job_id}"}, status_code=202, headers=headers)

    def _batch_key(self, job: Dict[str, Any], usable: bool = True) -> Optional[GatewayKey]:
        return next((k for k in self.config.keys_by_digest.values()
                     if (k.key_sha256 or k.id) == job["key_sha256"]
                     and (not usable or (not k.disabled and self.key_usable(k)))), None)

    async def batch_job_allowed(self, job: Dict[str, Any]) -> Optional[str]:
        """Why a queued call may no longer be sent (None when it may): the key, its budgets
        and the egress gate are checked again at the real crossing (audited then)."""
        key = self._batch_key(job)
        if key is None:
            return "the gateway key is no longer valid"
        saved = json.loads(job["record"] or "{}")
        scope = saved.get("scope") or {}
        try:
            member = self.bundle_manager.member_by_key(key.member or None)
            body = json.loads(job["body"] or "{}")
            stops = self._check_controls(key, member, body.get("model"), self.config.providers.get(job["provider"]))
            if stops.restricts_tools:
                # Queued before a quarantine or a tool stop: sending it now would offer what is stopped.
                offered = CallRecord(request_id="", key_id="", tenant_id="", app_id="", agent_id="", api=job["api"])
                offered._stops = stops
                probe = json.loads(job["body"] or "{}")
                self._filter_tools("anthropic" if job["api"] == "anthropic" else "openai", probe, {}, offered)
                if offered.techniques.get("tools_filtered"):
                    hit = stops.quarantined or next(iter(stops.tools))
                    return hit.message()
            policy = self.bundle_manager.get_policy(tenant_id=scope.get("tenant_id", key.tenant_id),
                                                    app_id=scope.get("app_id", key.app_id),
                                                    agent_id=scope.get("agent_id", key.agent_id),
                                                    env=scope.get("env", "prod"), member=member)
            provider = self.config.providers.get(job["provider"])
            if provider is None:
                return "the provider is no longer available"
            record = CallRecord(request_id=saved.get("request_id") or job["id"], key_id=key.id,
                                tenant_id=key.tenant_id, app_id=key.app_id, agent_id=key.agent_id, api=job["api"],
                                model=body.get("model"))
            await self._permit_egress(provider, policy, len(job["body"] or ""), scope, record)
        except Denied as denied:
            return denied.message
        except Exception as exc:  # noqa: BLE001 - when in doubt the call is not sent
            return f"the call cannot be sent now ({type(exc).__name__})"
        return None

    def _output_context(self, fmt, body: Dict[str, Any]) -> Dict[str, Any]:
        """What the output checks read besides the answer: the request's own system prompt and
        tools (hidden-context leak check) and its documents and tool results (groundedness)."""
        from znyx_runtime.gateway.optimise.verify import grounding_sources
        context: Dict[str, Any] = {"hidden_context": fmt.hidden_context(body)}
        if isinstance(body.get("tools"), list):
            context["tools"] = body["tools"]
        sources = grounding_sources(fmt, body)
        if sources:
            context["grounding_sources"] = sources
            context["source_context"] = "\n\n".join(s["text"] for s in sources)
        return context

    async def finish_batch_job(self, job: Dict[str, Any], payload: Optional[Dict[str, Any]],
                               error: Optional[str]) -> None:
        """A batch call's answer arrived (or failed): check it like a live answer under the
        policy in force now, settle what it cost against the reservation, record the call,
        and keep the checked answer for the app."""
        import dataclasses
        from znyx_runtime.gateway.optimise import Plan
        store = self.batch_store()
        fmt = fmts.FORMATS[job["api"]]
        saved = json.loads(job["record"] or "{}")
        scope = saved.pop("scope", None) or {}
        profile = saved.pop("profile", None)
        reserved = saved.pop("reserved", None) or {"tokens": 0, "cost": 0.0}
        names = {f.name for f in dataclasses.fields(CallRecord)}
        record = CallRecord(**{k: v for k, v in saved.items() if k in names and k not in ("plan", "started")})
        # Latency counted from when the app queued the call.
        record.started = time.perf_counter() - max(0.0, (time.time_ns() - record.started_ns) / 1e9)
        key = self._batch_key(job)
        owner = key or self._batch_key(job, usable=False)          # charged even if since disabled
        provider = self.config.providers.get(job["provider"])
        record.plan = Plan(profile, ["batch"], provider.format if provider else "openai", record.model)
        expected = "content" if fmt is fmts.AnthropicMessages else "choices"
        readable = isinstance(payload, dict) and isinstance(payload.get(expected), list)
        if isinstance(payload, dict):
            # Billed whether or not it can be released: usage and cost are recorded.
            record.response_model = payload.get("model") if isinstance(payload.get("model"), str) else None
            self._apply_usage(record, fmt.usage(payload))
            standard = record.cost_usd
            if standard is not None:
                record.cost_usd = round(standard * BATCH_PRICE_SHARE, 8)
                effect = record.plan.effect("batch")
                effect.basis, effect.saved_usd = "fixed", round(standard - record.cost_usd, 8)
        if owner is not None and reserved.get("held", bool(owner.limits.tokens_per_day or owner.limits.spend_usd_per_day)):
            actual_tokens = (record.input_tokens or 0) + (record.output_tokens or 0)
            from znyx_core.state_store import get_state_store
            get_state_store().incr(reserved.get("day") or self._day_key(owner), {
                "tokens": actual_tokens - int(reserved.get("tokens", 0)),
                "cost": (record.cost_usd or 0.0) - float(reserved.get("cost", 0.0))}, 2 * 86400)
        try:
            if key is None:
                raise Denied(403, "the gateway key is no longer valid", "batch_failed", outcome="error",
                             error_type="api_error")
            if error or not readable:
                raise Denied(502, error or "The provider returned an answer the gateway cannot read", "batch_failed",
                             outcome="error", error_type="api_error")
            member = self.bundle_manager.member_by_key(key.member or None)
            record._stops = self.stops_for(key, member)            # tool calls a stop concerns are withheld
            record._no_hold = True                                 # nobody can wait for an approval here
            if record._stops.stopped is not None:
                raise Denied(403, record._stops.stopped.message(), "stopped", error_type="permission_error")
            policy = self.bundle_manager.get_policy(tenant_id=scope.get("tenant_id", key.tenant_id),
                                                    app_id=scope.get("app_id", key.app_id),
                                                    agent_id=scope.get("agent_id", key.agent_id),
                                                    env=scope.get("env", "prod"), member=member)
            body = json.loads(job["body"] or "{}")
            await self._enforce_contract(fmt, payload, policy, record)
            await self._guard_output(fmt, payload, scope, policy, record, self._output_context(fmt, body))
            record.status = 200
            await asyncio.to_thread(store.update, job["id"], status="done", result=json.dumps(payload), body=None,
                                    error=None)
        except Denied as denied:
            record.status, record.outcome, record.reason = denied.status, denied.outcome, denied.code
            await asyncio.to_thread(store.update, job["id"], status="failed", error=denied.message, body=None)
        except Exception:  # noqa: BLE001 - a job that cannot be finished is failed, never released unchecked
            logger.exception("gateway: batch job %s could not be finished", job["id"])
            record.status, record.outcome, record.reason = 500, "error", "batch_failed"
            await asyncio.to_thread(store.update, job["id"], status="failed",
                                    error="the answer could not be checked", body=None)
        self._record(record)

    async def batch_status(self, request: Request, job_id: str) -> Response:
        """``GET /v1/znyx/batches/{id}``: the job's state, and its checked answer once done.
        Only the key that queued it may read it."""
        try:
            key, _ = self._identify(request)
            from znyx_runtime.bundle_manager import ScopeNotServed
            try:
                member = self.bundle_manager.member_by_key(key.member or None)
            except ScopeNotServed:
                member = None
            delegation = self.delegation_context(request, key, member)
            caller_stops = self.stops_for(key, member, person=delegation.person if delegation is not None else None)
            if caller_stops.stopped is not None:
                raise Denied(403, caller_stops.stopped.message(), "stopped", error_type="permission_error")
        except Denied as denied:
            return JSONResponse({"error": {"message": denied.message, "code": denied.code}}, status_code=denied.status)
        job = await asyncio.to_thread(self.batch_store().get, job_id) if job_id.startswith("zbatch_") else None
        if job is None or job["key_sha256"] != (key.key_sha256 or key.id):
            return JSONResponse({"error": {"message": "No such batch job", "code": "not_found"}}, status_code=404)
        out: Dict[str, Any] = {"id": job["id"], "object": "znyx.batch_job", "status": job["status"],
                               "created": int(job["created"])}
        if job["status"] == "done":
            out["result"] = json.loads(job["result"])
            if caller_stops.restricts_tools:
                # Finished before a quarantine or a tool stop: its tool calls are not released now.
                fmt = fmts.FORMATS[job["api"]]
                withheld = {p.choice for p in fmt.output_parts(out["result"]) if p.kind == "tool_call"
                            and (caller_stops.quarantined or caller_stops.tool_stop(p.tool_name))}
                for choice in withheld:
                    fmt.withhold_choice(out["result"], choice, self.config.settings.output_block_message)
        if job["status"] == "failed":
            out["error"] = job["error"]
        return JSONResponse(out)

    # -- right-sizing and shadow tests (part 3) ----------------------------------------

    def _right_size_target(self, fmt, body: Dict[str, Any], key: GatewayKey, policy: Dict[str, Any],
                           record: CallRecord, plan) -> Optional[str]:
        """The smaller model this call may go to, or None: never for streams, agent steps,
        calls whose input checks found something, or an app the policy marks critical."""
        from znyx_runtime.gateway.optimise import rightsize
        target = plan.opts.setting("right_size_to")
        if not target or record.api == "azure" or fmt.is_stream(body) or target == record.model:
            return None
        if plan.parallel or plan.batch_requested:
            return None                       # decided before the checks ran, or never verified
        marks = policy.get("optimisation") if isinstance(policy.get("optimisation"), dict) else {}
        if marks.get("critical") or record.input_decision != "ALLOW" or record.rule_ids:
            return None
        if rightsize.is_agent_step(body):
            return None
        if not self._model_gates_pass(key, target, record, "right_size_refused"):
            return None
        return target

    def _right_size(self, fmt, body: Dict[str, Any], key: GatewayKey, policy: Dict[str, Any],
                    record: CallRecord, plan) -> None:
        """Technique 14, once approved: a request the rules or the difficulty score find easy
        goes to the smaller model (verified after, retried on the requested model if it
        fails). Never hidden: the response and the trace say which model answered."""
        if not plan.on("right_size"):
            return
        target = self._right_size_target(fmt, body, key, policy, record, plan)
        if target is None:
            return
        from znyx_runtime.gateway.optimise import rightsize
        route, _why, _p, _values = rightsize.decide(body, plan.opts.tuned())
        if route != "small":
            return
        plan.right_sized_from = record.model
        body["model"] = target
        record.note("right_sized")
        plan.effect("right_size").basis = "fixed"

    def _plan_shadow(self, fmt, body: Dict[str, Any], key: GatewayKey, policy: Dict[str, Any],
                     record: CallRecord, plan, url: str = "") -> None:
        """Pick a shadow test for this call (sampled, within the key's daily allowance): the
        variant request it would send, or a routing verdict that needs no call."""
        if not plan.shadow_candidates or fmt.is_stream(body) or plan.parallel or plan.optimise_off:
            return
        from znyx_runtime.gateway.optimise import rightsize, shadow as shadows
        # Sample first: building variants costs time and is wasted on calls not tested.
        if not shadows.sampled(plan.opts.setting("shadow_percent")):
            return
        candidates = []
        for technique in plan.shadow_candidates:
            if technique == "right_size":
                target = self._right_size_target(fmt, body, key, policy, record, plan)
                if target:
                    candidates.append((technique, target))
            else:
                variant = self._stage_c_variant(technique, fmt, body, plan, record)
                if isinstance(variant, tuple) and variant[0] == "cache":
                    # In shadow the cache looks further (down to 60% of words the same), so the
                    # results show where the cut-off can safely sit (technique 23 tunes it).
                    hit = self._semantic_lookup(body, key, url, policy, plan, record, similarity=60)
                    variant = ("answer", hit[0], hit[3]) if hit is not None else None
                if variant is not None:
                    candidates.append((technique, variant))
        if not candidates:
            return
        technique, detail = candidates[0] if len(candidates) == 1 else shadows.pick(candidates)
        test: Dict[str, Any] = {"technique": technique, "variant": None}
        if technique == "right_size":
            route, _why, _p, values = rightsize.decide(body, plan.opts.tuned())
            test["routed"], test["features"] = route, rightsize.vector(values)
            if route == "small":
                test["variant"] = {**fmts.deep_copy(body), "model": detail}
        elif isinstance(detail, tuple) and detail[0] == "answer":
            test["answer"] = detail[1]                  # a stored answer: nothing to send
            test["features"] = [round(detail[2], 4)]    # how similar the questions were
        elif isinstance(detail, tuple) and detail[0] == "build":
            test["build"], test["body"] = detail[1], fmts.deep_copy(body)
        else:
            test["variant"] = detail
        if test["variant"] is not None or test.get("build"):
            from znyx_core.state_store import get_state_store
            if not shadows.within_daily_allowance(get_state_store(), plan.caller,
                                                  plan.opts.setting("shadow_calls_per_day")):
                record.note("shadow_allowance_used")
                return
        plan.shadow = test

    def _stage_c_variant(self, technique: str, fmt, body: Dict[str, Any], plan, record: CallRecord):
        """The request as a stage C technique would send it (a request, or for the model-backed
        ones a coroutine that builds it), or None when it would not change it."""
        from znyx_runtime.gateway.optimise import stage_c
        copy = fmts.deep_copy(body)
        if technique == "tool_pruning":
            changed = stage_c.prune_tools(record.api, copy, plan.opts.setting("tool_keep"))
        elif technique == "relevance_filter":
            changed = stage_c.filter_irrelevant(fmt, copy)
        elif technique == "context_order":
            changed = stage_c.order_context(copy) if fmt is fmts.AnthropicMessages else 0
        elif technique == "mark_untrusted":
            changed = stage_c.mark_untrusted(fmt, copy)
        elif technique == "summarise_history":
            if stage_c.summary_cut(copy, plan.opts.setting("summarise_after_turns")) is None:
                return None
            return ("build", technique)
        elif technique == "compress":
            if not self._compress_available() or not self._compress_parts(fmt, copy):
                return None
            return ("build", technique)
        elif technique == "semantic_cache":
            return ("cache", technique)
        else:
            return None
        return copy if changed else None

    async def _run_shadow(self, fmt, provider: Provider, url: str, headers: Dict[str, str], plan,
                          record: CallRecord, key: GatewayKey, scope: Dict[str, str], policy: Dict[str, Any],
                          served_text: str, served_usage: fmts.Usage) -> None:
        """After the client has its answer: send the variant, compare, keep only the verdict.
        The call is recorded when the test is done (its event carries the verdict)."""
        from znyx_runtime.gateway.optimise import shadow as shadows, verify
        test = plan.shadow
        technique = test["technique"]
        effect = plan.effect(technique)
        effect.shadow, effect.basis, effect.saved_usd = True, "fixed", 0.0
        from znyx_core.optimisation import shadow_config_hash
        effect.config_hash = shadow_config_hash(
            technique, choices={"enable": key.optimisation.enable, "disable": key.optimisation.disable,
                                "approved": key.optimisation.approved, "settings": dict(key.optimisation.settings)},
            profile=key.profile if key.profile_chosen else None, providers=key.providers,
            default_provider=key.default_provider, allowed_models=key.allowed_models,
            app_id=key.app_id, agent_id=key.agent_id)
        effect.routed, effect.features = test.get("routed"), test.get("features")
        try:
            if test.get("answer") is not None:
                # The semantic cache: the stored answer it would have served, against the fresh one.
                effect.features = test.get("features")
                answer = fmts.parse_response(test["answer"])
                text = verify.answer_text(fmt, answer) if isinstance(answer, dict) else None
                served_cost = self.prices_for(record.member).cost(record.response_model or record.model, served_usage)
                effect.projected_saved_usd = served_cost
                if text is None:
                    record.note("shadow_failed")
                    return
                nli = shadows.nli_from_policy(policy, self.egress_sink,
                                              self._request(scope, record, "shadow", served_text, "assistant"))
                agreed, score, method = await asyncio.to_thread(shadows.compare, served_text, text, nli)
                effect.verified, effect.agreement, effect.method = ("pass" if agreed else "fail"), score, method
                record.note("shadow_tested")
                return
            variant = test["variant"]
            try:
                member = self.bundle_manager.member_by_key(key.member or None)
                self._check_controls(key, member, (variant or {}).get("model") or record.model)
            except Exception:  # noqa: BLE001 - a spent budget or a member no longer served: no test
                record.note("shadow_refused")
                return
            if test.get("build"):
                variant = await self._build_variant(test["build"], fmt, provider, url, headers, test["body"], key,
                                                    scope, policy, record, plan)
                if variant is None:
                    record.note("shadow_failed")
                    return
            await self._permit_egress(provider, policy, len(json.dumps(variant)), scope, record)
            response = await self._send_extra(provider, url, headers, variant, key, record)
            if response.status_code != 200:
                record.note("shadow_failed")
                return
            answer = fmts.parse_response(response.content)
            text = verify.answer_text(fmt, answer) if isinstance(answer, dict) else None
            usage = fmt.usage(answer) if isinstance(answer, dict) else fmts.Usage()
            model = (answer.get("model") if isinstance(answer, dict) and isinstance(answer.get("model"), str)
                     else variant.get("model") or record.model)
            prices = self.prices_for(record.member)
            if usage.input_tokens is not None:
                cost = prices.cost(model, usage)
                self._extra_call(plan, record, technique, model, usage, charged=True)
                served_cost = prices.cost(record.response_model or record.model, served_usage)
                if served_cost is not None and cost is not None:
                    effect.projected_saved_usd = round(served_cost - cost, 8)
            if text is None:
                record.note("shadow_failed")
                return
            nli = shadows.nli_from_policy(policy, self.egress_sink,
                                          self._request(scope, record, "shadow", served_text, "assistant"))
            agreed, score, method = await asyncio.to_thread(shadows.compare, served_text, text, nli)
            effect.verified, effect.agreement, effect.method = ("pass" if agreed else "fail"), score, method
            record.note("shadow_tested")
        except Exception:  # noqa: BLE001 - a shadow test never affects the call; it is noted
            record.note("shadow_failed")
            logger.debug("gateway: shadow test failed for %s", record.request_id, exc_info=True)
        finally:
            self._record(record)

    async def _build_variant(self, technique: str, fmt, provider: Provider, url: str, headers: Dict[str, str],
                             body: Dict[str, Any], key: GatewayKey, scope: Dict[str, str], policy: Dict[str, Any],
                             record: CallRecord, plan) -> Optional[Dict[str, Any]]:
        if technique == "summarise_history":
            done = await self._summarise(fmt, provider, url, headers, body, key, scope, policy, record, plan, live=False)
            return body if done else None
        if technique == "compress":
            before = {id(p): p.text for p in self._compress_parts(fmt, body)}
            if not await self._compress(fmt, body, plan, record, live=False):
                return None
            # Shortened text can read differently from the checked original: checked before it leaves.
            quiet = quiet_policy(policy)
            for part in self._compress_parts(fmt, body) + stage_c_parts(fmt, body):
                if part.text in before.values():
                    continue
                verdict = await self._check_text(part.text, part.kind, part.role, part.tool_name, "input", scope,
                                                 record, quiet, quiet, "shadow-compress")
                if verdict.decision == "BLOCK" or verdict.wants_change:
                    record.note("shadow_refused")
                    return None
            return body
        return None

    async def _summarise(self, fmt, provider: Provider, url: str, headers: Dict[str, str], body: Dict[str, Any],
                         key: GatewayKey, scope: Dict[str, str], policy: Dict[str, Any], record: CallRecord, plan,
                         live: bool) -> bool:
        """Technique 12: replace older turns by a summary from the key's summary model (same
        provider, the same gates as any model). The summary is checked like input before it is
        sent, and kept per conversation prefix so the next turn reuses it. Returns True when
        the request changed."""
        from znyx_runtime.gateway.optimise import stage_c, verify
        model = plan.opts.setting("summary_model")
        if not model or record.api == "azure":
            return False
        cut = stage_c.summary_cut(body, plan.opts.setting("summarise_after_turns"))
        if cut is None or not self._model_gates_pass(key, model, record, "summary_refused"):
            return False
        store = getattr(self, "_summaries", None)
        if store is None:
            store = self._summaries = stage_c.SummaryStore()
        prefix = stage_c.prefix_key(f"{plan.caller}|{policy.get('policy_version') or ''}|{scope.get('app_id')}",
                                    body, cut)
        summary = store.get(prefix)
        owner = "summarise_history"
        if summary is None:
            ask = stage_c.summary_request(model, record.api, stage_c.transcript(body, cut))
            try:
                await self._permit_egress(provider, policy, len(json.dumps(ask)), scope, record)
                response = await self._send_extra(provider, url, headers, ask, key, record)
                answer = fmts.parse_response(response.content) if response.status_code == 200 else None
            except (Denied, UpstreamError, ValueError, RecursionError):
                answer = None
            summary = verify.answer_text(fmt, answer) if isinstance(answer, dict) else None
            if isinstance(answer, dict):
                usage = fmt.usage(answer)
                if usage.input_tokens is not None:
                    self._extra_call(plan, record, owner, model, usage, charged=True)
            if not summary:
                record.note("summary_failed")
                return False
            store.put(prefix, summary)
        # The summary is the model's text about checked text: checked before every use, under
        # the policy in force (a stored one too; a summary checked clean is skipped cheaply).
        quiet = quiet_policy(policy)
        verdict = await self._check_text(summary, "text", "system", None, "input", scope, record, quiet,
                                         quiet, "summary")
        if verdict.decision == "BLOCK" or verdict.wants_change:
            record.note("summary_refused")
            return False
        removed = sum(len(json.dumps(m)) for m in body["messages"][:cut]) - len(summary)
        stage_c.apply_summary(fmt, body, cut, summary)
        if live:
            record.note("summarised")
            effect = plan.effect(owner)
            effect.basis, effect.estimated = "fixed", True
            effect.saved_input_tokens += max(0, removed // 4)
            price = self.prices_for(record.member).lookup(record.model) or {}
            effect.saved_usd = round((effect.saved_usd or 0.0) + max(0, removed // 4) *
                                     price.get("input_per_mtok", 0.0) / 1_000_000, 8)
        return True

    @staticmethod
    def _compress_available() -> bool:
        import os
        return bool(os.environ.get("ZNYX_INFERENCE_URL"))

    @staticmethod
    def _compress_parts(fmt, body: Dict[str, Any]) -> List[fmts.Part]:
        from znyx_runtime.gateway.optimise import stage_c
        return [p for p in stage_c.retrieved_parts(fmt, body)
                if len(p.text) >= 2000 and not stage_c._cited(stage_c._block(body, p.path))]

    async def _compress(self, fmt, body: Dict[str, Any], plan, record: CallRecord, live: bool) -> bool:
        """Technique 13: long tool results and documents shortened by the model server's
        compression model (task ``compress``: the spans of text worth keeping). In the
        perimeter only: a model server outside the network is not called. Before the input
        guard, which reads the shortened text. Returns True when the request changed."""
        import os
        from urllib.parse import urlsplit
        import ipaddress
        parts = self._compress_parts(fmt, body)
        if not parts or not self._compress_available():
            return False
        base = os.environ["ZNYX_INFERENCE_URL"].rstrip("/")
        host = (urlsplit(base).hostname or "").lower()
        try:
            private = host == "localhost" or ipaddress.ip_address(host).is_private or ipaddress.ip_address(host).is_loopback
        except ValueError:
            private = host.endswith((".internal", ".local", ".svc", ".cluster.local")) or "." not in host
        if not private:
            record.note("compress_refused")
            return False
        results = await self._compress_call(base, [p.text for p in parts])
        if not isinstance(results, list) or len(results) != len(parts):
            record.note("compress_unavailable")
            return False
        removed = 0
        for part, result in zip(parts, results):
            spans = _clean_spans(result.get("entity_spans") if isinstance(result, dict) else None, len(part.text))
            if not spans:
                continue
            kept = " ".join(part.text[s:e] for s, e in spans)
            if kept and len(kept) < len(part.text):
                try:
                    fmt.set_text(body, part, kept)
                except (ValueError, TypeError):
                    continue
                removed += len(part.text) - len(kept)
        if removed and live:
            record.note("compressed")
            plan.effect("compress", saved_input_tokens=removed // 4, estimated=True)
        return removed > 0

    @staticmethod
    async def _compress_call(base: str, texts: List[str]) -> Optional[List[Any]]:
        """The model server's ``compress`` task for these texts (its results), or None."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(f"{base}/v1/infer/compress", json={"texts": texts, "params": {"rate": 0.5}})
            return response.json().get("results") if response.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            return None

    def _semantic_lookup(self, body: Dict[str, Any], key: GatewayKey, url: str, policy: Dict[str, Any], plan,
                         record: CallRecord, similarity: Optional[int] = None):
        """A stored answer to a near-identical question in the same context, or None. Never
        for a call whose input checks found something."""
        from znyx_runtime.gateway.optimise import rightsize, stage_c
        if record.input_decision != "ALLOW" or record.rule_ids or rightsize.is_agent_step(body):
            return None                       # never for findings, never inside an agent loop
        cache = getattr(self, "_semantic", None)
        if cache is None:
            return None
        context = stage_c.context_key(plan.caller, url, str(policy.get("policy_version") or ""), body,
                                      getattr(record, "_cache_user", record.user))
        if similarity is None:
            tuned = plan.opts.tuned().get("semantic_cache") or {}
            similarity = (tuned.get("similarity") if isinstance(tuned.get("similarity"), (int, float)) else None) \
                or plan.opts.setting("semantic_similarity") or stage_c.DEFAULT_SIMILARITY
        return cache.find(context, stage_c.question_of(body), similarity)

    def _semantic_store(self, body: Dict[str, Any], url: str, policy: Dict[str, Any], plan, record: CallRecord,
                        content: bytes, headers: Dict[str, str], upstream_ms: Optional[int]) -> None:
        from znyx_runtime.gateway.optimise import rightsize, stage_c
        if record.input_decision != "ALLOW" or record.rule_ids or rightsize.is_agent_step(body):
            return
        cache = getattr(self, "_semantic", None)
        if cache is None:
            cache = self._semantic = stage_c.SemanticCache()
        context = stage_c.context_key(plan.caller, url, str(policy.get("policy_version") or ""), body,
                                      getattr(record, "_cache_user", record.user))
        cache.put(context, stage_c.question_of(body), content, headers, upstream_ms)

    def _schedule(self, coroutine, record: Optional[CallRecord] = None) -> None:
        """Run a background job (a shadow test). Its call is recorded even if the job is
        cancelled before it ran (a shutdown)."""
        tasks = getattr(self, "_background", None)
        if tasks is None:
            tasks = self._background = set()
        task = asyncio.ensure_future(coroutine)
        tasks.add(task)

        def done(t, record=record) -> None:
            tasks.discard(t)
            if record is not None and t.cancelled() and not getattr(record, "_recorded", False):
                record.note("shadow_cancelled")
                self._record(record)

        task.add_done_callback(done)

    async def drain(self, timeout: float = 5.0) -> None:
        """Let background jobs (shadow tests) finish before shutting down; cancel the rest."""
        tasks = list(getattr(self, "_background", None) or [])
        if not tasks:
            return
        _done, pending = await asyncio.wait(tasks, timeout=timeout)
        for task in pending:
            task.cancel()

    async def _check_right_sized(self, fmt, provider: Provider, url: str, headers: Dict[str, str],
                                 body: Dict[str, Any], payload: Dict[str, Any], plan, record: CallRecord,
                                 key: GatewayKey, scope: Dict[str, str], policy: Dict[str, Any]) -> Dict[str, Any]:
        """Verify the smaller model's answer (contract, rules, computed claims, groundedness);
        if it fails, ask the requested model and pay for it. The ledger gets the net saving."""
        from znyx_runtime.gateway.optimise import verify
        prices = self.prices_for(record.member)
        small = record.response_model or body.get("model")
        small_usage = fmts.Usage(record.input_tokens, record.output_tokens, record.cached_input_tokens,
                                 record.cache_write_tokens)
        effect = plan.effect("right_size")
        effect.basis, effect.estimated = "fixed", True
        text = verify.answer_text(fmt, payload)
        faults = await self._faults(fmt, text, plan, scope, record, policy, force=True) if text is not None else None
        if faults:
            big = {**fmts.deep_copy(body), "model": plan.right_sized_from}
            try:
                await self._permit_egress(provider, policy, len(json.dumps(big)), scope, record)
                response = await self._send_extra(provider, url, headers, big, key, record)
                answer = fmts.parse_response(response.content) if response.status_code == 200 else None
            except (Denied, UpstreamError, ValueError, RecursionError):
                answer = None
            expected = "content" if fmt is fmts.AnthropicMessages else "choices"
            if isinstance(answer, dict) and isinstance(answer.get(expected), list):
                served = answer.get("model") if isinstance(answer.get("model"), str) else plan.right_sized_from
                self._extra_call(plan, record, "right_size", served, fmt.usage(answer), charged=True)
                record.response_model = served
                record.note("right_size_retried")
                effect.verified = "fail"
                payload = answer
                plan.answer_changed = True
            else:
                # The smaller model's answer failed its checks and the requested model did not
                # answer: an unverified answer is not passed on.
                effect.verified = "fail"
                raise Denied(502, "The answer could not be verified and the requested model did not answer",
                             "right_size_unverified", outcome="error", error_type="api_error")
        else:
            effect.verified = "pass"
        counterfactual = prices.cost(plan.right_sized_from, small_usage)
        paid = prices.cost(small, small_usage)
        if counterfactual is not None and paid is not None:
            shape = plan.cache_shape
            if shape is not None and shape.prefix_chars and plan.prefix_seen:
                # The requested model would likely have read this prefix from its cache.
                price = prices.lookup(plan.right_sized_from) or {}
                cached = min(small_usage.input_tokens or 0, shape.prefix_chars // 4)
                counterfactual -= cached * (price.get("input_per_mtok", 0) - price.get(
                    "cached_input_per_mtok", price.get("input_per_mtok", 0))) / 1_000_000
            effect.saved_usd = round((effect.saved_usd or 0.0) + counterfactual - paid, 8)
        return payload

    @staticmethod
    def _sent_texts(fmt, body: Dict[str, Any]) -> List[Tuple[Any, str]]:
        return [(p.path, p.text) for p in fmt.input_parts(body, _ALL_ROLES)]

    def _adapt_to_risk(self, key: GatewayKey, plan, policy: Dict[str, Any], record: CallRecord) -> Dict[str, Any]:
        """Technique 32: a session whose risk climbed past the key's threshold is checked
        more strictly (never less: ``stage_b.stricter`` only tightens)."""
        if not plan.on("risk_adaptive") or not plan.session:
            return policy
        from znyx_runtime.gateway.optimise import stage_b
        try:
            from znyx_core.state_store import get_state_store
            score = get_state_store().get(stage_b.risk_key(self._caller_id(key), plan.session)).get("score", 0)
        except Exception:  # noqa: BLE001 - unknown risk: the checks stay as the policy sets them
            return policy
        threshold = plan.opts.setting("risk_threshold") or stage_b.DEFAULT_RISK_THRESHOLD
        if score < threshold:
            return policy
        plan.elevated = True
        record.note("risk_elevated")
        plan.effect("risk_adaptive")
        return stage_b.stricter(policy)

    async def _forward(self, request: Request, fmt, provider: Provider, url: str,
                       headers: Dict[str, str], body: Dict[str, Any], key: GatewayKey,
                       scope: Dict[str, str], policy: Dict[str, Any], record: CallRecord,
                       context: Optional[Dict[str, Any]] = None, early=None) -> Response:
        t_up = time.perf_counter()
        plan = record.plan
        upstream = self.upstream
        shared_key = cache_key = None
        votes = plan.vote_count(body, request.headers) if plan is not None and early is None else 0
        if (not votes and early is None and plan is not None
                and (plan.coalescing_allowed(body) or plan.caching_allowed(body))):
            # Hashing a large body takes a while: off the event loop.
            base = await asyncio.to_thread(self.optimiser.coalescer.key, self._caller_id(key), url, headers, body)
            shared_key = base if plan.coalescing_allowed(body) else None
            if plan.caching_allowed(body):
                # Technique 29: a stored answer is never served under another policy version.
                cache_key = hashlib.sha256(f"{base}|{policy.get('policy_version') or ''}".encode()).hexdigest()
                fresh = "no-cache" in (request.headers.get("cache-control") or "").lower()
                hit = None if fresh else self.optimiser.cache.get(cache_key, plan.cache_ttl())
                if hit is not None:
                    return await self._answer_from_cache(fmt, key, scope, policy, record, context, hit)
        if (not votes and early is None and plan is not None and plan.on("semantic_cache")
                and "no-cache" not in (request.headers.get("cache-control") or "").lower()):
            found = self._semantic_lookup(body, key, url, policy, plan, record)
            if found is not None:
                from znyx_runtime.gateway.optimise.cache import CachedAnswer
                content, stored_headers, upstream_ms, _score = found
                return await self._answer_from_cache(fmt, key, scope, policy, record, context,
                                                     CachedAnswer(content, stored_headers, upstream_ms, 0.0),
                                                     technique="semantic_cache")
        # Technique 2: an identical deterministic call of this key already in flight is
        # joined instead of sent again. The provider bills it once: one caller pays.
        if early is not None:
            def start():
                return early                       # already on its way (parallel input check)
        elif votes:
            def start():
                return self._vote(fmt, upstream, provider, url, headers, body, votes, plan, record,
                                  policy=policy, scope=scope)
        else:
            def start():
                return upstream.send(provider, url, headers, body)
        flight, leader = self.optimiser.coalescer.join(shared_key, start)
        record._upstream_started = True
        # Stop waiting if the client goes away; the provider call is cancelled when no
        # one else waits for it (it would be paid for nothing).
        try:
            while not flight.task.done():
                done, _ = await asyncio.wait({flight.task}, timeout=0.5)
                if not done and await request.is_disconnected():
                    last = flight.leave()
                    record.status, record.outcome, record.reason = 499, "error", "client_disconnected"
                    if last and flight.pay():
                        # The provider may bill for a call it had already started. While
                        # others still wait, the one that receives the answer pays.
                        self.estimate_missing_usage(record, 0)
                        for _ in range(max(0, votes - 1)):          # the other answers of a vote
                            self._extra_call(plan, record, "voting", None,
                                             fmts.Usage(record.estimated_input_tokens, 0), estimated=True)
                        self._add_extra_calls(record, plan)
                    self._charge(key, record)
                    self._record(record)
                    return Response(status_code=499)
        except asyncio.CancelledError:
            flight.leave()
            raise
        if flight.task.cancelled():
            raise UpstreamError(502, f"provider {provider.name} call was cancelled")
        response = flight.task.result()
        if response.status_code != 200 and plan is not None and plan.right_sized_from:
            # The smaller model refused the request: the requested model answers it instead.
            big = {**fmts.deep_copy(body), "model": plan.right_sized_from}
            try:
                await self._permit_egress(provider, policy, len(json.dumps(big)), scope, record)
                # The smaller model refused without usage. Replace its reservation
                # with one priced for the requested model before the fallback.
                self._reserve_call(key, record, big, provider)
                retried = await self.upstream.send(provider, url, headers, big)
            except (Denied, UpstreamError):
                retried = None
            if retried is not None:
                response, body = retried, big
                plan.right_sized_from = None
                plan.effects.pop("right_size", None)
                record.note("right_size_fell_back")
        record.upstream_ms = int((time.perf_counter() - t_up) * 1000)
        returned = self.upstream.returned_headers(response)
        returned.update(self.trace_headers(record))
        if not leader:
            record.note("coalesced")
            returned["x-znyx-coalesced"] = "true"
        if response.status_code != 200:
            # The provider's own error, as the provider sent it: SDK retry and error
            # handling keep working.
            record.status, record.outcome, record.reason = response.status_code, "error", "provider_error"
            self._charge(key, record)
            self._record(record)
            return Response(content=response.content, status_code=response.status_code,
                            media_type=response.headers.get("content-type", "application/json"),
                            headers=returned)
        pays = flight.pay()
        try:
            payload = fmts.parse_response(response.content)
        except (ValueError, RecursionError):
            payload = None
        expected = "content" if fmt is fmts.AnthropicMessages else "choices"
        if not isinstance(payload, dict) or not isinstance(payload.get(expected), list):
            # A success the gateway cannot read is never passed on unchecked.
            if pays:
                self.estimate_missing_usage(record, 0)
            else:
                self._apply_usage(record, fmts.Usage(0, 0, 0, 0))   # another caller paid
            self._charge(key, record)
            raise Denied(502, "The provider returned a response the gateway cannot read",
                         "bad_upstream_response", outcome="error", error_type="api_error")
        record.response_model = payload.get("model") if isinstance(payload.get("model"), str) else None
        usage = fmt.usage(payload)
        if pays:
            self._apply_usage(record, usage)
        else:
            self._coalesced(record, plan, usage, flight)
        stored = response.content
        served_text: Optional[str] = None
        served_usage: Optional[fmts.Usage] = None
        try:
            if pays and plan is not None and plan.verifies():
                payload = await self._verify(fmt, provider, url, headers, body, payload, plan, record, key,
                                             scope, policy)
                if plan.answer_changed:
                    stored = json.dumps(payload).encode()     # the answer that passed, before the guard
            if pays and plan is not None and plan.right_sized_from:
                payload = await self._check_right_sized(fmt, provider, url, headers, body, payload, plan, record,
                                                        key, scope, policy)
                if plan.answer_changed:
                    stored = json.dumps(payload).encode()
            if plan is not None and plan.shadow is not None:
                from znyx_runtime.gateway.optimise.verify import answer_text
                served_text = answer_text(fmt, payload)       # before restore and the guard, like the variant
                served_usage = fmt.usage(payload)
            if pays and plan is not None:
                self._add_extra_calls(record, plan)
            self._restore_pii(fmt, payload, plan, record)
            await self._enforce_contract(fmt, payload, policy, record)
            await self._guard_output(fmt, payload, scope, policy, record, context)
        except BaseException:
            # The answer was paid for even if it cannot be checked now (and is not sent).
            if pays:
                if plan is not None:
                    self._add_extra_calls(record, plan)
                self.estimate_missing_usage(record, 0)
            self._charge(key, record)
            raise
        if pays:
            self.estimate_missing_usage(record, sum(len(p.text) for p in fmt.output_parts(payload)))
        self._charge(key, record)
        record.status = 200
        returned["x-znyx-decision"] = _worst(record.input_decision, record.output_decision)
        if plan is not None and plan.escalated_from:
            returned["x-znyx-escalated-from"] = plan.escalated_from
        if plan is not None and plan.right_sized_from:
            returned["x-znyx-model-requested"] = plan.right_sized_from
            returned["x-znyx-model-served"] = record.response_model or str(body.get("model") or "")
        if (cache_key is not None and record.outcome == "ok" and record.output_decision in ("ALLOW", "WARN")
                and not record.techniques.get("output_rewritten")
                and "no-store" not in (request.headers.get("cache-control") or "").lower()):
            # Stored before the output guard: every later hit is checked again.
            self.optimiser.cache.put(cache_key, stored, self.upstream.returned_headers(response),
                                     flight.upstream_ms)
        if (plan is not None and early is None and not votes and record.outcome == "ok"
                and record.output_decision in ("ALLOW", "WARN") and not record.techniques.get("output_rewritten")
                and (plan.on("semantic_cache") or "semantic_cache" in plan.shadow_candidates)
                and "no-store" not in (request.headers.get("cache-control") or "").lower()):
            self._semantic_store(body, url, policy, plan, record, stored, self.upstream.returned_headers(response),
                                 flight.upstream_ms)
        if self._shadow_after(fmt, provider, url, headers, plan, record, key, scope, policy,
                              served_text, served_usage):
            return JSONResponse(payload, status_code=200, headers=returned)   # recorded when the test ends
        self._record(record)
        return JSONResponse(payload, status_code=200, headers=returned)

    def _shadow_after(self, fmt, provider: Provider, url: str, headers: Dict[str, str], plan, record: CallRecord,
                      key: GatewayKey, scope: Dict[str, str], policy: Dict[str, Any],
                      served_text: Optional[str], served_usage: Optional[fmts.Usage]) -> bool:
        """Start the call's shadow test, if it has one and its answer was served as it came.
        True when the test will record the call."""
        test = plan.shadow if plan is not None else None
        if test is None or record.outcome != "ok" or served_text is None:
            return False
        if test["variant"] is None and test.get("answer") is None and not test.get("build"):
            # A routing verdict only (the score kept the call where it was): nothing to send.
            effect = plan.effect(test["technique"])
            effect.shadow, effect.basis, effect.saved_usd = True, "fixed", 0.0
            effect.routed, effect.features = test.get("routed"), test.get("features")
            return False
        self._schedule(self._run_shadow(fmt, provider, url, headers, plan, record, key, scope, policy,
                                        served_text, served_usage or fmts.Usage()), record)
        return True

    # -- verification and voting (stage B) -------------------------------------------

    _RETRY_OWNER = {"structured_output": "repair", "business_rules": "business_rules",
                    "computed_claims": "computed_claims", "grounded_retry": "grounded_retry"}

    async def _faults(self, fmt, text: str, plan, scope: Dict[str, str], record: CallRecord,
                      policy: Dict[str, Any], force: bool = False):
        """What is wrong with an answer. ``force``: every check that can run does (a
        right-sized answer is always verified), whatever the key's techniques."""
        from znyx_runtime.gateway.optimise import verify
        faults = verify.Faults()
        contract = plan.contract if plan.contract is not None or not force else verify.contract_of(policy)

        def on(technique: str) -> bool:
            return force or plan.on(technique)

        # Parsing and scanning a long answer is CPU work: off the event loop.
        if contract is not None and (on("structured_output") or plan.on("repair")):
            faults.add("structured_output", await asyncio.to_thread(verify.check_contract, text, contract))
        if contract is not None and contract.rules and on("business_rules"):
            faults.add("business_rules", await asyncio.to_thread(verify.check_rules, text, contract.rules))
        if on("computed_claims"):
            faults.add("computed_claims", await asyncio.to_thread(verify.check_claims, text))
        if on("grounded_retry") and plan.sources:
            nli = None
            config = plan.grounding_config or {}
            if isinstance(config.get("nli"), dict) and config["nli"].get("enabled"):
                from znyx_core.engine.quality.nli_client import nli_scorer_from_config
                nli = nli_scorer_from_config(config, runtime_policy=policy.get("runtime_policy"), policy=policy,
                                             egress_sink=self.egress_sink,
                                             request=self._request(scope, record, "ground", text, "assistant"))
            # NLI calls out and token overlap is CPU work: off the event loop.
            faults.add("grounded_retry", await asyncio.to_thread(verify.check_grounded, text, plan.sources,
                                                                 config, nli))
        return faults

    async def _verify(self, fmt, provider: Provider, url: str, headers: Dict[str, str], body: Dict[str, Any],
                      payload: Dict[str, Any], plan, record: CallRecord, key: GatewayKey,
                      scope: Dict[str, str], policy: Dict[str, Any]) -> Dict[str, Any]:
        """Check the answer (contract, business rules, computed claims, groundedness); fix a
        simple fault in place, else ask again once, then once on the stronger model. The
        answer returned still goes through the output guard."""
        from znyx_runtime.gateway.optimise import verify
        text = verify.answer_text(fmt, payload)
        if text is None:
            return payload
        faults = await self._faults(fmt, text, plan, scope, record, policy)
        if faults.by_technique.get("structured_output") and plan.on("repair"):
            fixed = verify.simple_repair(text)
            if fixed is not None and not verify.check_contract(fixed, plan.contract):
                verify.set_answer_text(fmt, payload, fixed)
                text, plan.answer_changed = fixed, True
                record.note("answer_repaired")
                plan.effect("repair")
                faults = await self._faults(fmt, text, plan, scope, record, policy)
        owners = [self._RETRY_OWNER[t] for t in faults.by_technique
                  if t in self._RETRY_OWNER and plan.on(self._RETRY_OWNER[t])]
        if owners:
            again = await self._ask_again(fmt, provider, url, headers, body, text, faults, plan, record, owners[0],
                                          policy=policy, scope=scope)
            if again is not None:
                payload, text = again
                faults = await self._faults(fmt, text, plan, scope, record, policy)
        target = plan.opts.setting("escalate_to")
        if (faults and plan.on("escalate") and target and record.api != "azure"
                and target != (record.response_model or record.model) and key.model_allowed(target)
                and self._model_gates_pass(key, target, record, "escalation_refused")):
            again = await self._ask_again(fmt, provider, url, headers, body, text, faults, plan, record, "escalate",
                                          model=target, policy=policy, scope=scope)
            if again is not None:
                plan.escalated_from = record.model
                payload, text = again
                faults = await self._faults(fmt, text, plan, scope, record, policy)
        contract = plan.contract
        checked = {
            "structured_output": contract is not None and bool(contract.json_schema),
            "business_rules": contract is not None and bool(contract.rules),
            "computed_claims": True,
            "grounded_retry": bool(plan.sources),
        }
        for technique, ran in checked.items():
            if ran and plan.on(technique):
                plan.effect(technique).verified = "fail" if technique in faults.by_technique else "pass"
        for technique in ("repair", "escalate"):
            if technique in plan.effects:
                plan.effects[technique].verified = "fail" if faults else "pass"
        return payload

    def _model_gates_pass(self, key: GatewayKey, target: str, record: CallRecord, note: str) -> bool:
        """Another model (escalation, right-sizing, a summary) passes the same gates a request
        naming it would: allowed for the key, a price where the key's spend is limited, and
        the console's budgets (blocked, priced only)."""
        if not key.model_allowed(target) or (key.limits.spend_usd_per_day and not key.instance_key
                                             and self.prices_for(key.member).lookup(target) is None):
            record.note(note)
            return False
        try:
            member = self.bundle_manager.member_by_key(key.member or None)
            self._check_controls(key, member, target)
        except Exception:  # noqa: BLE001 - Denied, or a member no longer served
            record.note(note)
            return False
        return True

    async def _enforce_contract(self, fmt, payload: Dict[str, Any], policy: Dict[str, Any],
                                record: CallRecord) -> None:
        """A policy check, whatever techniques the key uses: an output contract set to block
        withholds every answer choice whose text does not follow it."""
        from znyx_runtime.gateway.optimise import verify
        contract = verify.contract_of(policy)
        if contract is None or str(contract.on_fail or "").lower() != "block" or not contract.json_schema:
            return
        for index, text in verify.choice_texts(fmt, payload):
            if text is None:
                continue                       # a tool call: governed by the tool policy
            if await asyncio.to_thread(verify.check_contract, text, contract):
                fmt.withhold_choice(payload, index, self.config.settings.output_block_message)
                record.outcome = "blocked_output"
                record.output_decision = "BLOCK"
                record.rule_ids.append("output_contract.invalid")

    async def _send_extra(self, provider, url, headers, body, key, record):
        """Reserve and settle an auxiliary provider call before its ledger entry is
        added. Optional optimisation calls cannot spend capacity already reserved
        by another caller, including across replicas."""
        if key is None or key.instance_key or not (key.limits.tokens_per_day or key.limits.spend_usd_per_day):
            return await self.upstream.send(provider, url, headers, body)
        ticket = CallRecord(request_id=record.request_id, key_id=key.id, member=key.member,
                            tenant_id=key.tenant_id, app_id=key.app_id, agent_id=key.agent_id,
                            api=record.api, model=body.get("model") or record.model)
        self._reserve_call(key, ticket, body, provider)
        try:
            response = await self.upstream.send(provider, url, headers, body)
            if response.status_code == 200:
                try:
                    payload = fmts.parse_response(response.content)
                    fmt = fmts.FORMATS[record.api]
                    if isinstance(payload, dict):
                        ticket.response_model = payload.get("model") if isinstance(payload.get("model"), str) else None
                        self._apply_usage(ticket, fmt.usage(payload))
                        self.estimate_missing_usage(ticket, sum(len(p.text) for p in fmt.output_parts(payload)))
                    else:
                        self.estimate_missing_usage(ticket, 0)
                except (ValueError, TypeError, RecursionError):
                    self.estimate_missing_usage(ticket, 0)
            return response
        except BaseException:
            self.estimate_missing_usage(ticket, 0)
            raise
        finally:
            self._charge(key, ticket)

    def _extra_call(self, plan, record: CallRecord, owner: str, model: Optional[str], usage: fmts.Usage,
                    estimated: bool = False, charged: bool = False) -> None:
        """An extra call the provider billed (a retry, an escalation, a vote): its usage is
        added to the call's, and its cost is the owning technique's negative saving."""
        served = model or record.model
        cost = self.prices_for(record.member).cost(served, usage)
        if cost is None:
            record.price_known = False
        plan.extra_calls.append((served, usage, cost))
        # Charged to the key's day now: it was billed whatever happens to the call after.
        tokens = (usage.input_tokens or 0) + (usage.output_tokens or 0)
        if plan.key is not None and not charged:
            self._charge_usage(plan.key, tokens, cost or 0.0)
        plan.extra_tokens += tokens
        plan.extra_cost += cost or 0.0
        if estimated:
            record.usage_estimated = True
            record.note("usage_estimated")
        effect = plan.effect(owner)
        effect.basis = "fixed"
        effect.saved_usd = round((effect.saved_usd or 0.0) - (cost or 0.0), 8)

    async def _ask_again(self, fmt, provider: Provider, url: str, headers: Dict[str, str], body: Dict[str, Any],
                         text: str, faults, plan, record: CallRecord, owner: str, model: Optional[str] = None,
                         *, policy: Dict[str, Any], scope: Dict[str, str]):
        """One more call with what was wrong (and, to escalate, on the stronger model). It
        passes the egress gate (audited) like the first, and is paid for."""
        from znyx_runtime.gateway.optimise import verify
        retry = verify.retry_body(fmt, body, text, faults)
        if retry is None:
            return None
        if model:
            retry["model"] = model
        raw = json.dumps(retry)
        try:
            await self._permit_egress(provider, policy, len(raw), scope, record)
        except Denied:
            record.note("retry_refused")
            return None
        t0 = time.perf_counter()
        estimate = fmts.Usage(estimate_tokens(len(raw)), 0)
        try:
            response = await self._send_extra(provider, url, headers, retry, plan.key, record)
        except Denied:
            record.note("retry_refused")
            return None
        except UpstreamError as exc:
            record.note("retry_failed")
            if exc.status == 504:              # timed out: the provider may have billed it
                self._extra_call(plan, record, owner, model, estimate, estimated=True, charged=True)
            return None
        expected = "content" if fmt is fmts.AnthropicMessages else "choices"
        try:
            again = fmts.parse_response(response.content) if response.status_code == 200 else None
        except (ValueError, RecursionError):
            again = None
        if not isinstance(again, dict) or not isinstance(again.get(expected), list):
            record.note("retry_failed")
            if response.status_code == 200:    # answered, but unreadable: billed all the same
                self._extra_call(plan, record, owner, model, estimate, estimated=True, charged=True)
            return None
        usage = fmt.usage(again)
        served = again.get("model") if isinstance(again.get("model"), str) else (model or record.model)
        self._extra_call(plan, record, owner, served, usage, charged=True)
        effect = plan.effect(owner)
        effect.latency_ms = (effect.latency_ms or 0) + int((time.perf_counter() - t0) * 1000)
        record.note("escalated" if model else "asked_again")
        new_text = verify.answer_text(fmt, again)
        if new_text is None:
            return None
        if model:
            record.response_model = served
        plan.answer_changed = True
        return again, new_text

    async def _vote(self, fmt, upstream, provider: Provider, url: str, headers: Dict[str, str],
                    body: Dict[str, Any], count: int, plan, record: CallRecord, *, policy: Dict[str, Any],
                    scope: Dict[str, str]):
        """Technique 22: ``count`` answers at once; the one most agree with is the answer.
        At temperature 0 the answers would not differ, so the samples use 0.7. Every sample
        beyond the first passes the egress gate (audited) on its own."""
        from znyx_runtime.gateway.optimise import verify
        sample = dict(body)
        temperature = body.get("temperature")
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or temperature <= 0:
            sample["temperature"] = 0.7
        raw = json.dumps(sample)
        for _ in range(count - 1):
            await self._permit_egress(provider, policy, len(raw), scope, record)
        results = await asyncio.gather(*(upstream.send(provider, url, headers, sample) for _ in range(count)),
                                       return_exceptions=True)
        estimate = fmts.Usage(estimate_tokens(len(raw)), 0)
        for r in results:
            if isinstance(r, UpstreamError) and r.status == 504:
                self._extra_call(plan, record, "voting", None, estimate, estimated=True)
        answered = [r for r in results if not isinstance(r, BaseException) and r.status_code == 200]
        if not answered:
            for r in results:
                if isinstance(r, BaseException):
                    raise r
            return results[0]
        payloads = []
        for r in answered:
            try:
                payloads.append(fmts.parse_response(r.content))
            except (ValueError, RecursionError):
                payloads.append(None)
        texts = [verify.answer_text(fmt, p) if isinstance(p, dict) else None for p in payloads]
        best, agreement = verify.choose_by_vote(texts)
        effect = plan.effect("voting")
        effect.basis = "fixed"
        effect.saved_usd = effect.saved_usd or 0.0
        effect.verified = "pass" if agreement > 0.5 else "fail"
        for i, p in enumerate(payloads):
            if i == best:
                continue
            if not isinstance(p, dict):
                self._extra_call(plan, record, "voting", None, estimate, estimated=True)
                continue
            served = p.get("model") if isinstance(p.get("model"), str) else None
            self._extra_call(plan, record, "voting", served, fmt.usage(p))
        record.note("voted", len(results))
        return answered[best]

    @staticmethod
    def _restore_pii(fmt, payload: Dict[str, Any], plan, record: CallRecord) -> None:
        """Technique 30: the request's own PII back in the answer, before the output guard
        checks it (an output policy that redacts PII redacts it again)."""
        if plan is None or not len(plan.vault):
            return
        from znyx_runtime.gateway.optimise.stage_b import restore_pii
        restored = restore_pii(fmt, payload, plan.vault)
        if restored:
            record.note("pii_restored", restored)

    def _add_extra_calls(self, record: CallRecord, plan) -> None:
        """The usage and cost of retries and votes, added to the call's own: the provider
        billed every one of them."""
        for _, usage, cost in plan.extra_calls:
            for name in ("input_tokens", "output_tokens", "cached_input_tokens", "cache_write_tokens"):
                value = getattr(usage, name)
                if value:
                    setattr(record, name, (getattr(record, name) or 0) + value)
            if cost is not None and record.cost_usd is not None:
                record.cost_usd = round(record.cost_usd + cost, 8)
            elif cost is not None and record.input_tokens is not None:
                record.cost_usd = round(cost, 8)        # the call itself cost nothing (a cache hit, a share)
            plan.extra_tokens_added += (usage.input_tokens or 0) + (usage.output_tokens or 0)
            plan.extra_cost_added += cost or 0.0
        plan.extra_calls = []

    async def _answer_from_cache(self, fmt, key: GatewayKey, scope: Dict[str, str], policy: Dict[str, Any],
                                 record: CallRecord, context: Optional[Dict[str, Any]], hit,
                                 technique: str = "exact_cache") -> Response:
        """Techniques 5 and 10: a stored provider answer (to this exact request, or to a
        near-identical question), checked again under the current policy. It costs nothing
        and saves what the stored call cost."""
        plan = record.plan
        record.note(f"{technique}_hit")
        payload = fmts.parse_response(hit.content)
        record.response_model = payload.get("model") if isinstance(payload.get("model"), str) else None
        usage = fmt.usage(payload)
        plan.shared_usage = usage
        effect = plan.effect(technique, basis="call", saved_input_tokens=max(
            0, (usage.input_tokens or 0) - (usage.cached_input_tokens or 0)))
        effect.saved_usd = self.prices_for(record.member).cost(record.response_model or record.model, usage)
        if hit.upstream_ms is not None:
            effect.latency_ms = -hit.upstream_ms
        self._apply_usage(record, fmts.Usage(0, 0, 0, 0))
        try:
            self._restore_pii(fmt, payload, plan, record)
            await self._enforce_contract(fmt, payload, policy, record)
            await self._guard_output(fmt, payload, scope, policy, record, context)
        finally:
            self._charge(key, record)
        record.status = 200
        returned = dict(hit.headers)
        returned.update(self.trace_headers(record))
        returned["x-znyx-cache"] = "hit" if technique == "exact_cache" else "similar"
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

    def _coalesced(self, record: CallRecord, plan, usage: fmts.Usage, flight) -> None:
        """A call that shared another's upstream call: the provider billed the other one,
        so this one costs nothing and saves what the shared call cost."""
        plan.shared_usage = usage
        # Input read from the provider's cache is counted under prompt_cache, not twice.
        effect = plan.effect("coalesce", basis="call", saved_input_tokens=max(
            0, (usage.input_tokens or 0) - (usage.cached_input_tokens or 0)))
        effect.saved_usd = self.prices_for(record.member).cost(record.response_model or record.model, usage)
        if flight.upstream_ms is not None and record.upstream_ms is not None:
            effect.latency_ms = record.upstream_ms - flight.upstream_ms
        self._apply_usage(record, fmts.Usage(0, 0, 0, 0))

    def _apply_usage(self, record: CallRecord, usage: fmts.Usage) -> None:
        record.input_tokens = usage.input_tokens
        record.output_tokens = usage.output_tokens
        record.cached_input_tokens = usage.cached_input_tokens
        record.cache_write_tokens = usage.cache_write_tokens
        prices = self.prices_for(record.member)
        record.price_known = prices.lookup(record.response_model or record.model) is not None
        record.cost_usd = prices.cost(record.response_model or record.model, usage)

    def _record(self, record: CallRecord) -> None:
        if getattr(record, "_recorded", False):
            return                             # recorded once, whatever path got here first
        record._recorded = True
        record.latency_ms = int((time.perf_counter() - record.started) * 1000)
        record.rule_ids = sorted(set(record.rule_ids))
        if record.plan is not None:
            if record.plan.extra_calls:
                self._add_extra_calls(record, record.plan)     # billed calls are never left out
            self._count_risk(record)
            try:
                record.plan.settle(record, self.prices_for(record.member))
            except Exception:  # noqa: BLE001 - the ledger must never fail a call
                logger.warning("gateway: could not settle optimisation effects for %s", record.request_id,
                               exc_info=True)
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
        if record.api == "mcp" and (record.tools_called or record.mcp_tools_changed):
            self._audit_mcp(record)

    def _member_of(self, record: CallRecord):
        try:
            return self.bundle_manager.member_by_key(record.member or None)
        except Exception:  # noqa: BLE001 - a call the runtime refused belongs to no member
            return None

    def _audit_mcp(self, record: CallRecord) -> None:
        """The governance record of an MCP tool call (part 4): which agent called which
        server's tool, for whom, and what the gateway did; metadata only."""
        from znyx_runtime.governance_audit import call_fields
        try:
            fields = call_fields(record, self._member_of(record))
            if record.tools_called:
                self.governance.record(
                    "mcp.tool_call", **fields, server=record.provider, tool=record.tools_called[0],
                    outcome=record.outcome, reason=record.reason, rule_ids=list(record.rule_ids)[:20],
                    approved=True if record.techniques.get("approval_granted") else None,
                    identity_provider=record.identity_provider)
            if record.mcp_tools_changed:
                self.governance.record("mcp.tool_changed", **fields, server=record.provider,
                                       tools=list(record.mcp_tools_changed)[:50])
        except Exception:  # noqa: BLE001 - the audit never fails a call
            logger.debug("gateway: governance record failed", exc_info=True)
