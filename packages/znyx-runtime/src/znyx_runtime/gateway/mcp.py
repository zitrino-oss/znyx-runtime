"""The MCP gateway (Phase 5, part 2): an inline proxy for agents' tool calls.

An agent connects to ``/v1/mcp/{server}`` with its gateway key as if it were the MCP server
(Streamable HTTP, JSON-RPC 2.0). The gateway forwards to the configured server after its
checks, under the same policy, stops, egress gate and usage records as the agent's model
calls, so an agent's whole run (model calls and tool calls) is one trace and one policy.

What goes to the server:

- only methods the gateway knows (``_FORWARDED`` and the guarded ones below); others are refused;
- ``initialize``: the client's ``sampling`` capability is removed unless the policy allows it, so
  a server cannot make the agent's model generate;
- ``tools/call``: only a tool the agent may see; arguments checked like a model's tool call (the
  ``tools`` policy, and content the policy blocks, such as secrets); a call the policy marks for
  approval waits for a person, and stops are checked again once approved;
- the parameters of other requests that carry data (prompt arguments, resource URIs,
  completions, the agent's answers to the server's questions) are checked for content the
  policy blocks.

What comes back to the agent:

- ``tools/list``: only allowed, unchanged tools: a tool's definition is pinned the first time
  it is seen (by the control plane in managed mode, else in the shared state) and a different
  definition is held until a person accepts it; the manifest scanner can hide one too;
- every text in a result (tool results, task results, resources, prompts, structured content,
  links, readable embedded resources, error messages) is checked like a tool result; a block
  makes the answer an error, a redaction replaces the text;
- other messages the server streams first are relayed only when they are notifications or
  questions the agent may answer (never answers to other requests, never sampling unless
  allowed), their text checked the same way.

Sessions: the server's ``Mcp-Session-Id`` is never shown to the caller; the gateway gives its
own, bound to the key that opened it. Answers are bounded in size and time.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import re
import time
import uuid
from types import SimpleNamespace
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Tuple

from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from znyx_core import approvals, identity

logger = logging.getLogger(__name__)

SESSION_HEADER = "mcp-session-id"
PROTOCOL_HEADER = "mcp-protocol-version"
CLIENT_CREDENTIAL_HEADER = "x-znyx-mcp-authorization"
SESSION_TTL = 24 * 3600
SESSIONS_PER_KEY_PER_DAY = 1000
PIN_TTL = 400 * 86400
DEFS_TTL = 300
MAX_LIST_PAGES = 10
MAX_TOOLS_PER_LIST = 500
KEEPALIVE_SECONDS = 10
MAX_ANSWER_BYTES = 10 * 1024 * 1024        # a server's answer (or SSE stream) read for one request
MAX_PLAIN_PARAMS = 4096                    # parameters of requests that carry no data of their own
MAX_RELAYED = 1000
MAX_BLOB_CHECK = 1024 * 1024
_TEXT_MIME = re.compile(r"^(text/|application/(json|xml|yaml|x-yaml|javascript)\b)")
_PROTOCOL = re.compile(r"\d{4}-\d{2}-\d{2}")
_EVENT_ID = re.compile(r"[\x21-\x7e]{1,200}")

# Requests forwarded as they are (they carry no data of the agent's own beyond small params).
_FORWARDED = frozenset({"ping", "resources/list", "resources/templates/list", "resources/subscribe",
                        "resources/unsubscribe", "prompts/list", "logging/setLevel", "tasks/get", "tasks/list",
                        "tasks/cancel"})
# Requests whose parameters carry data going to the server: checked first.
_OUTGOING = frozenset({"completion/complete"})
_GUARDED = frozenset({"initialize", "tools/list", "tools/call", "resources/read", "prompts/get", "tasks/result"})
_CLIENT_NOTIFICATIONS = frozenset({"notifications/initialized", "notifications/cancelled", "notifications/progress",
                                   "notifications/roots/list_changed"})
# What a server may send the agent besides its answer.
_SERVER_REQUESTS = frozenset({"elicitation/create", "roots/list", "ping", "sampling/createMessage"})

# JSON-RPC error codes the gateway answers with
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
SERVER_ERROR = -32000


class McpRefusal(Exception):
    """Refused before the server is reached: ``status`` is the HTTP status."""

    def __init__(self, status: int, message: str, code: str):
        super().__init__(message)
        self.status, self.message, self.code = status, message, code


_LONE_SURROGATE = re.compile("[\ud800-\udfff]")


def tool_hash(tool: Dict[str, Any]) -> str:
    """What a tool's pin is: its definition as the model sees it."""
    material = {k: tool.get(k) for k in ("name", "description", "inputSchema", "outputSchema", "annotations")}
    text = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _rpc_error(msg_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _tool_refusal(msg_id: Any, message: str) -> Dict[str, Any]:
    """A refused tool call as a tool error the model can read (not a protocol error)."""
    return {"jsonrpc": "2.0", "id": msg_id,
            "result": {"content": [{"type": "text", "text": message}], "isError": True}}


def _sse(message: Dict[str, Any]) -> bytes:
    return f"event: message\ndata: {json.dumps(message, separators=(',', ':'))}\n\n".encode()


def _valid_id(value: Any) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)) or (isinstance(value, str) and len(value) <= 128)


async def _sse_messages(response) -> AsyncIterator[Dict[str, Any]]:
    """JSON-RPC messages from a server's SSE stream (``data:`` lines joined per event). Read as
    bytes and split here, so no line is held whole before the size bound applies."""
    buffer = b""
    data: List[str] = []
    size = 0

    def finish():
        try:
            message = json.loads("\n".join(data))
        except ValueError:
            return None
        return message if isinstance(message, dict) else None

    async for chunk in response.aiter_bytes():
        size += len(chunk)
        if size > MAX_ANSWER_BYTES:
            raise McpRefusal(502, "The MCP server's answer is too large", "upstream_too_large")
        buffer += chunk
        while b"\n" in buffer:
            raw, buffer = buffer.split(b"\n", 1)
            line = raw.rstrip(b"\r").decode("utf-8", "replace")
            if line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
            elif line == "" and data:
                message = finish()
                data = []
                if message is not None:
                    yield message
    if buffer.startswith(b"data:"):
        data.append(buffer[5:].decode("utf-8", "replace").lstrip(" "))
    if data:
        message = finish()
        if message is not None:
            yield message


class McpGateway:
    def __init__(self, gateway) -> None:
        self.gw = gateway
        self._defs: Dict[str, Tuple[float, Dict[str, Dict[str, Any]]]] = {}
        self._pins_memo: Optional[tuple] = None

    # -- entry point ----------------------------------------------------------------

    async def handle(self, request: Request, server_name: str) -> Response:
        from znyx_runtime.gateway.pipeline import CallRecord, Denied
        if request.method == "GET":
            return Response(status_code=405, headers={"allow": "POST, DELETE"})
        record = CallRecord(request_id=request.headers.get("x-request-id") or f"mcp-{uuid.uuid4().hex}",
                            key_id="", tenant_id="", app_id="", agent_id="", api="mcp", provider=server_name[:200])
        record.provider_format = "mcp"
        self.gw._trace(record, request)
        try:
            ctx = await self._context(request, record, server_name)
            if request.method == "DELETE":
                return await self._end_session(request, ctx, record)
            body = await self.gw._read_body(request)
            return await self._dispatch(request, ctx, record, body)
        except McpRefusal as refusal:
            return self._refuse(record, refusal.status, refusal.message, refusal.code)
        except Denied as denied:
            return self._refuse(record, denied.status, denied.message, denied.code, denied.outcome)

    def _refuse(self, record, status: int, message: str, code: str, outcome: str = "denied") -> Response:
        record.status, record.outcome, record.reason = status, outcome, code
        self.gw._record(record)
        return JSONResponse(_rpc_error(None, SERVER_ERROR, message) | {"znyx": {"code": code}},
                            status_code=status, headers=self.gw.trace_headers(record))

    async def _context(self, request: Request, record, server_name: str) -> Dict[str, Any]:
        """Who is calling which server under which policy, or a refusal."""
        from znyx_runtime.bundle_manager import ScopeNotServed
        key, _ = self.gw._identify(request)
        record.key_id, record.tenant_id, record.app_id, record.agent_id = (
            key.id, key.tenant_id, key.app_id, key.agent_id)
        record.member = key.member
        record._reauthorize = lambda: self.gw.current_identity(request)
        try:
            member = self.gw.bundle_manager.member_by_key(key.member or None)
        except ScopeNotServed as exc:
            raise McpRefusal(403, str(exc), "project_not_served") from None
        try:
            server = self.gw.config.mcp_server_for(key, server_name)
        except LookupError as exc:
            raise McpRefusal(404, str(exc), "mcp_server_not_available") from None
        if server.blocked:
            raise McpRefusal(403, server.blocked, "mcp_server_not_allowed_here")
        ctx: Dict[str, Any] = {"key": key, "member": member, "server": server, "record": record,
                               "project": member.project_id if member is not None else "",
                               "environment": member.environment_id if member is not None else ""}
        ctx["delegation"] = self.gw.delegation_context(request, key, member)
        ctx["stops"] = self._stops_or_refuse(ctx)
        self._rate_limit(key)
        scope = self.gw._scope(key, request, member)
        try:
            policy = self.gw.bundle_manager.get_policy(tenant_id=scope["tenant_id"], app_id=scope["app_id"],
                                                       agent_id=scope["agent_id"], env=scope["env"], member=member)
        except RuntimeError:
            raise McpRefusal(503, "Policy unavailable (fail-closed)", "policy_unavailable") from None
        record.user = self.gw._user_of(request, {})
        self.gw.apply_delegation(record, ctx["delegation"])
        record._stops = ctx["stops"]
        ctx.update(scope=scope, policy=policy)
        return ctx

    def _stops_or_refuse(self, ctx):
        """The caller's stops now; a stopped caller or server is refused."""
        delegation = ctx.get("delegation")
        stops = self.gw.stops_for(ctx["key"], ctx["member"], person=delegation.person if delegation else None)
        hit = stops.stopped or self.gw.stop_set().mcp_server(ctx["server"].name, ctx["project"], ctx["environment"])
        if hit is not None:
            raise McpRefusal(403, hit.message(), "stopped")
        return stops

    def _rate_limit(self, key) -> None:
        from znyx_core.state_store import StateUnavailable, get_state_store
        from znyx_runtime.gateway.pipeline import Denied
        rpm = key.limits.requests_per_minute
        if not rpm:
            return
        try:
            allowed = get_state_store().consume_token(f"pin:gw:rpm:{self.gw._state_id(key)}",
                                                      rate_per_second=float(rpm) / 60.0, capacity=float(rpm))
        except StateUnavailable:
            raise Denied(503, "Rate limits cannot be checked right now (fail-closed)", "state_unavailable",
                         outcome="error") from None
        if not allowed:
            raise Denied(429, "Rate limit for this key reached; retry shortly", "rate_limit_exceeded")

    # -- sessions -------------------------------------------------------------------

    @staticmethod
    def _session_key(gateway_session: str) -> str:
        return f"pin:gw:mcp-session:{gateway_session}"

    def _owner(self, ctx) -> str:
        return f"{self.gw._state_id(ctx['key'])}|{ctx['server'].member}/{ctx['server'].name}"

    def _upstream_session(self, request: Request, ctx) -> Optional[str]:
        """The server's session for the caller's gateway session; refused when the session
        belongs to another key or server."""
        from znyx_core.state_store import get_state_store
        given = request.headers.get(SESSION_HEADER)
        if not given:
            return None
        valid = given.startswith("zms_") and len(given) == 36
        raw = get_state_store().get_value(self._session_key(given)) if valid else None
        try:
            entry = json.loads(raw) if raw else None
        except ValueError:
            entry = None
        if not isinstance(entry, dict) or entry.get("owner") != self._owner(ctx):
            # The spec's answer to an unknown session: the client starts a new one.
            raise McpRefusal(404, "Unknown or expired MCP session; initialize again", "mcp_session_unknown")
        return entry.get("upstream") or None

    def _new_session(self, ctx, upstream: str) -> str:
        from znyx_core.state_store import get_state_store
        store = get_state_store()
        day = time.strftime("%Y%m%d", time.gmtime())
        made = store.incr(f"pin:gw:mcp-sessions:{self.gw._state_id(ctx['key'])}:{day}", {"n": 1}, 2 * 86400)
        if made.get("n", 0) > SESSIONS_PER_KEY_PER_DAY:
            raise McpRefusal(429, "Too many MCP sessions for this key today", "mcp_sessions_exhausted")
        session = "zms_" + uuid.uuid4().hex
        store.set_value(self._session_key(session), json.dumps({"owner": self._owner(ctx), "upstream": upstream[:512]}),
                        SESSION_TTL)
        return session

    async def _end_session(self, request: Request, ctx, record) -> Response:
        from znyx_core.state_store import get_state_store
        upstream = self._upstream_session(request, ctx)
        if upstream:
            try:
                client = await self._client(ctx["server"])
                await client.delete(ctx["server"].url, headers=await self._headers(request, ctx, upstream))
            except Exception as exc:  # noqa: BLE001 - the gateway's session ends whatever the server says
                logger.info("MCP session end at %s failed: %s", ctx["server"].name, exc)
        get_state_store().delete(self._session_key(request.headers.get(SESSION_HEADER) or ""))
        record.status, record.outcome = 200, "ok"
        return Response(status_code=200, headers=self.gw.trace_headers(record))

    # -- upstream -------------------------------------------------------------------

    async def _client(self, server):
        return await self.gw.upstream.client(SimpleNamespace(
            name=f"mcp:{server.member}/{server.name}", timeout_seconds=server.timeout_seconds))

    async def _headers(self, request: Request, ctx, upstream_session: Optional[str]) -> Dict[str, str]:
        from znyx_core import usage_events
        server = ctx["server"]
        headers = {"content-type": "application/json", "accept": "application/json, text/event-stream"}
        version = request.headers.get(PROTOCOL_HEADER)
        if version and _PROTOCOL.fullmatch(version):
            headers[PROTOCOL_HEADER] = version
        event_id = request.headers.get("last-event-id")
        if event_id and _EVENT_ID.fullmatch(event_id):
            headers["last-event-id"] = event_id
        for name, value in server.headers:
            headers[name] = value
        if upstream_session:
            headers[SESSION_HEADER] = upstream_session
        record = ctx["record"]
        headers["traceparent"] = usage_events.traceparent(record.trace_id, record.span_id)
        if server.uses_client_credential:
            given = request.headers.get(CLIENT_CREDENTIAL_HEADER)
            if not given:
                raise McpRefusal(401, f"MCP server {server.name} uses your own credential: send it as "
                                      "X-Znyx-Mcp-Authorization", "mcp_credential_required")
            headers[server.auth_header] = given
        else:
            credential = await asyncio.to_thread(server.credential)
            if credential:
                headers[server.auth_header] = (f"Bearer {credential}" if server.auth_header.lower() == "authorization"
                                               else credential)
        return headers

    async def _send(self, request: Request, ctx, message: Dict[str, Any]):
        """Forward one message through the egress gate; the open response (JSON or SSE)."""
        server, record = ctx["server"], ctx["record"]
        payload = json.dumps(message, separators=(",", ":")).encode()
        await self.gw._permit_egress(server, ctx["policy"], len(payload), ctx["scope"], record)
        headers = await self._headers(request, ctx, ctx.get("upstream_session"))
        client = await self._client(server)
        t0 = time.perf_counter()
        try:
            response = await client.send(client.build_request("POST", server.url, content=payload, headers=headers),
                                          stream=True)
        except Exception as exc:  # noqa: BLE001
            raise McpRefusal(502, f"MCP server {server.name} could not be reached ({type(exc).__name__})",
                             "upstream_error") from None
        record.upstream_ms = int((time.perf_counter() - t0) * 1000)
        response.extensions["znyx_timeout"] = server.timeout_seconds
        return response

    async def _answer_of(self, response, msg_id: Any, relay: List[Dict[str, Any]],
                         keep: Optional[Callable[[Dict[str, Any]], Awaitable[Optional[Dict[str, Any]]]]] = None,
                         timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """The server's answer to ``msg_id``; other messages it sent first go to ``relay``
        when ``keep`` lets them (and as ``keep`` returns them). Bounded in size and in time
        (the server's timeout for the whole answer, not per chunk)."""
        try:
            return await asyncio.wait_for(self._read_answer(response, msg_id, relay, keep),
                                          timeout or response.extensions.get("znyx_timeout") or 600)
        except asyncio.TimeoutError:
            raise McpRefusal(504, "The MCP server did not answer in time", "upstream_timeout") from None
        finally:
            await response.aclose()

    async def _read_answer(self, response, msg_id: Any, relay: List[Dict[str, Any]], keep=None) -> Optional[Dict[str, Any]]:
        kind = response.headers.get("content-type", "")
        if response.status_code == 202:
            return None
        if "text/event-stream" in kind and response.status_code < 400:
            async for message in _sse_messages(response):
                if message.get("id") == msg_id and type(message.get("id")) is type(msg_id) \
                        and ("result" in message or "error" in message):
                    return message
                kept = await keep(message) if keep is not None else None
                if kept is not None and len(relay) < MAX_RELAYED:
                    relay.append(kept)
            return None
        chunks, size = [], 0
        async for chunk in response.aiter_bytes():
            size += len(chunk)
            if size > MAX_ANSWER_BYTES:
                raise McpRefusal(502, "The MCP server's answer is too large", "upstream_too_large")
            chunks.append(chunk)
        raw = b"".join(chunks)
        if response.status_code >= 400:
            raise McpRefusal(502 if response.status_code >= 500 else response.status_code,
                             f"The MCP server answered {response.status_code}", "upstream_error")
        try:
            body = json.loads(raw) if raw else None
        except ValueError:
            raise McpRefusal(502, "The MCP server's answer could not be read", "upstream_error") from None
        if not isinstance(body, dict):
            return None
        if body.get("id") != msg_id or type(body.get("id")) is not type(msg_id):
            raise McpRefusal(502, "The MCP server answered another request", "upstream_error")
        return body

    # -- checks ---------------------------------------------------------------------

    async def _check(self, ctx, record, text: str, label: str, name: str = "mcp"):
        """A text coming from the server, checked like a tool result."""
        from znyx_runtime.gateway.pipeline import quiet_policy
        verdict = await self.gw._check_text(text, "tool_result", "tool", name[:200], "input", ctx["scope"], record,
                                            ctx["policy"], quiet_policy(ctx["policy"]), label)
        record.rule_ids.extend(verdict.rule_ids)
        return verdict

    async def _outgoing_blocked(self, ctx, record, value: Any, label: str) -> Optional[str]:
        """Data going to the server, checked for content the policy blocks (secrets, say).
        Redactions are not applied: a tool needs the values it is given. Returns why when
        blocked."""
        from znyx_runtime.gateway.pipeline import quiet_policy
        if value in (None, {}, [], ""):
            return None
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        verdict = await self.gw._check_text(text, "text", "assistant", None, "output", ctx["scope"], record,
                                            ctx["policy"], quiet_policy(ctx["policy"]), label, approve=False)
        record.rule_ids.extend(verdict.rule_ids)
        if verdict.decision == "BLOCK":
            return verdict.user_message or "the data would send content the policy blocks"
        return None

    async def _keep_relayed(self, ctx, record, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Whether (and how) a message the server sent before its answer reaches the agent."""
        if "result" in message or "error" in message:
            record.note("mcp_relay_dropped")          # an answer to some other request: never
            return None
        method = message.get("method")
        if not isinstance(method, str):
            return None
        mcp_policy = ctx["policy"].get("mcp") if isinstance(ctx["policy"].get("mcp"), dict) else {}
        if method == "sampling/createMessage" and mcp_policy.get("allow_sampling") is not True:
            record.note("mcp_sampling_refused")
            return None
        if method not in _SERVER_REQUESTS and not method.startswith("notifications/"):
            record.note("mcp_relay_dropped")
            return None
        params = message.get("params")
        if params:
            text = json.dumps(params, ensure_ascii=False, default=str)
            verdict = await self._check(ctx, record, text, f"mcp-relay-{method[:40]}")
            if verdict.decision == "BLOCK" or verdict.sanitized_text not in (None, text):
                record.note("mcp_relay_dropped")
                return None
        return message

    async def _checked_items(self, ctx, record, items: List[Any], name: str, label: str) -> Tuple[List[Any], bool, bool]:
        """Check content and its metadata: ``(items, blocked, changed)``."""
        metadata = []
        for item in items:
            if not isinstance(item, dict):
                metadata.append(item)
                continue
            extra = {k: v for k, v in item.items() if k not in ("text", "data", "resource")}
            if isinstance(item.get("resource"), dict):
                extra["resource"] = {k: v for k, v in item["resource"].items() if k not in ("text", "blob")}
            metadata.append(extra)
        checked = await self._checked_data(ctx, record, {"result": metadata}, label + "-metadata")
        if "error" in checked:
            return [], True, False
        if items and all(isinstance(i, dict) and i.get("type") == "text"
                         and isinstance(i.get("text"), str) for i in items):
            text = "".join(i["text"] for i in items)
            verdict = await self._check(ctx, record, text, label, name)
            if verdict.decision == "BLOCK":
                return [], True, False
            if verdict.sanitized_text is not None and verdict.sanitized_text != text:
                # A rewrite may span blocks. Preserve the first block's metadata and
                # put the checked text there; no fragment of the original survives.
                return [{**items[0], "text": verdict.sanitized_text}], False, True
            return items, verdict.wants_change, False
        out: List[Any] = []
        combined: List[str] = []
        changed = False
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                out.append(item)
                continue
            item = dict(item)
            kind = item.get("type")
            texts: List[Tuple[str, str]] = []
            if kind == "text" and isinstance(item.get("text"), str):
                texts.append(("text", item["text"]))
            elif kind == "resource" and isinstance(item.get("resource"), dict):
                resource = item["resource"]
                if isinstance(resource.get("text"), str):
                    texts.append(("resource.text", resource["text"]))
                elif isinstance(resource.get("blob"), str) and _TEXT_MIME.match(str(resource.get("mimeType") or "")):
                    try:
                        if len(resource["blob"]) > 4 * ((MAX_BLOB_CHECK + 2) // 3):
                            return out, True, changed
                        decoded = base64.b64decode(resource["blob"], validate=True)
                        if len(decoded) > MAX_BLOB_CHECK:
                            return out, True, changed
                        texts.append(("resource.blob", decoded.decode("utf-8")))
                    except (ValueError, UnicodeError):
                        return out, True, changed
                elif isinstance(resource.get("blob"), str):
                    if self.gw.config.settings.media_inspection:
                        from znyx_runtime.gateway import media
                        data, _ = media.decode(resource["blob"])
                        text = await media.check(self.gw, data, resource.get("mimeType") or "application/octet-stream",
                                                 ctx["scope"], ctx["policy"], record, "output")
                        texts.append(("resource.media", text))
                    else:
                        record.note("mcp_binary_unchecked")
            elif kind == "resource_link":
                link = " ".join(str(item.get(k) or "") for k in ("name", "title", "description", "uri"))
                texts.append(("link", link))
            elif kind in ("image", "audio"):
                if self.gw.config.settings.media_inspection:
                    from znyx_runtime.gateway import media
                    data, _ = media.decode(item.get("data"))
                    text = await media.check(self.gw, data, item.get("mimeType") or kind + "/unknown",
                                             ctx["scope"], ctx["policy"], record, "output")
                    texts.append(("media", text))
                else:
                    record.note("mcp_binary_unchecked")
            elif kind not in (None,):
                texts.append(("other", json.dumps(item, ensure_ascii=False, default=str)))
            for where, text in texts:
                verdict = await self._check(ctx, record, text, f"{label}{i}", name)
                if verdict.decision == "BLOCK":
                    return out, True, changed
                if verdict.wants_change and verdict.sanitized_text is None:
                    return out, True, changed
                combined.append(verdict.sanitized_text if verdict.sanitized_text is not None else text)
                if verdict.sanitized_text is not None and verdict.sanitized_text != text:
                    changed = True
                    if where == "text":
                        item["text"] = verdict.sanitized_text
                    elif where == "resource.text":
                        item["resource"] = {**item["resource"], "text": verdict.sanitized_text}
                    elif where == "resource.blob":
                        item["resource"] = {**item["resource"], "blob": base64.b64encode(
                            verdict.sanitized_text.encode("utf-8")).decode()}
                    else:
                        item = None                 # a link or unknown item cannot be partly redacted
                        break
            if item is not None:
                out.append(item)
        if len(combined) > 1:
            text = "".join(combined)
            verdict = await self._check(ctx, record, text, label + "-combined", name)
            if (verdict.decision == "BLOCK" or verdict.wants_change
                    or verdict.sanitized_text not in (None, text)):
                # Mixed resources/links have different schemas. When a cross-item
                # rewrite cannot preserve them, withhold the result as a whole.
                return [], True, changed
        return out, False, changed

    async def _checked_data(self, ctx, record, answer: Dict[str, Any], label: str) -> Dict[str, Any]:
        """Guard text in protocol results (lists, completions, initialize, task metadata).
        Identifiers and schemas cannot safely be rewritten: any required change withholds
        the result. Text content with an editable schema uses _checked_items instead."""
        if "error" in answer:
            return await self._checked_error(ctx, record, answer, label)
        texts: List[str] = []
        pending = [answer.get("result")]
        while pending:
            value = pending.pop()
            if isinstance(value, str):
                texts.append(value)
            elif isinstance(value, dict):
                pending.extend(reversed(list(value.values())))
            elif isinstance(value, list):
                pending.extend(reversed(value))
        for text in texts + (["".join(texts)] if len(texts) > 1 else []):
            verdict = await self._check(ctx, record, text, label)
            if (verdict.decision == "BLOCK" or verdict.wants_change
                    or verdict.sanitized_text not in (None, text)):
                return self._blocked_answer(record, answer.get("id"), as_error=True)
        return answer

    def _blocked_answer(self, record, msg_id: Any, as_error: bool) -> Dict[str, Any]:
        record.outcome, record.reason = "blocked_output", "blocked_output"
        message = self.gw.config.settings.output_block_message
        if as_error:
            return _rpc_error(msg_id, SERVER_ERROR, message)
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"content": [{"type": "text", "text": message}],
                                                           "isError": True}}

    async def _checked_error(self, ctx, record, answer: Dict[str, Any], name: str) -> Dict[str, Any]:
        error = answer.get("error") if isinstance(answer.get("error"), dict) else {}
        text = " ".join([str(error.get("message") or ""), json.dumps(error.get("data"), default=str)
                         if error.get("data") is not None else ""]).strip()
        if not text:
            return answer
        verdict = await self._check(ctx, record, text, "mcp-error", name)
        if verdict.decision == "BLOCK" or (verdict.wants_change and verdict.sanitized_text is None):
            return self._blocked_answer(record, answer.get("id"), as_error=True)
        if verdict.sanitized_text is not None and verdict.sanitized_text != text:
            return _rpc_error(answer.get("id"), int(error.get("code") or SERVER_ERROR)
                              if isinstance(error.get("code"), int) else SERVER_ERROR, verdict.sanitized_text[:2000])
        return answer

    async def _checked_result(self, ctx, record, answer: Dict[str, Any], name: str) -> Dict[str, Any]:
        """A tool (or task) result as the agent may see it: every text checked like a tool
        result, ``structuredContent`` too; a block makes it an error the model reads, a
        redaction replaces the text (and drops ``structuredContent``, which cannot be redacted
        the same way)."""
        if "error" in answer:
            return await self._checked_error(ctx, record, answer, name)
        result = answer.get("result")
        if not isinstance(result, dict):
            return answer
        items, blocked, changed = await self._checked_items(ctx, record, list(result.get("content") or []), name,
                                                            "mcp-result")
        if blocked:
            return self._blocked_answer(record, answer.get("id"), as_error=False)
        out = {**result, "content": items}
        metadata = {k: v for k, v in result.items() if k not in ("content", "structuredContent")}
        if "error" in await self._checked_data(ctx, record, {"result": metadata}, "mcp-result-metadata"):
            return self._blocked_answer(record, answer.get("id"), as_error=False)
        if result.get("structuredContent") is not None:
            text = json.dumps(result["structuredContent"], ensure_ascii=False, default=str)
            verdict = await self._check(ctx, record, text, "mcp-structured", name)
            if verdict.decision == "BLOCK" or (verdict.wants_change and verdict.sanitized_text is None):
                return self._blocked_answer(record, answer.get("id"), as_error=False)
            if verdict.sanitized_text is not None and verdict.sanitized_text != text:
                changed = True
        if changed:
            out.pop("structuredContent", None)
            record.note("mcp_result_redacted")
        return {**answer, "result": out}

    # -- dispatch -------------------------------------------------------------------

    async def _dispatch(self, request: Request, ctx, record, body: Dict[str, Any]) -> Response:
        if body.get("jsonrpc") != "2.0":
            raise McpRefusal(400, "Not a JSON-RPC 2.0 message", "invalid_request")
        method = body.get("method")
        msg_id = body.get("id")
        if "id" in body and not _valid_id(msg_id):
            raise McpRefusal(400, "A JSON-RPC id is a number or a string of at most 128 characters", "invalid_request")
        if "params" in body and body["params"] is not None and not isinstance(body["params"], dict):
            raise McpRefusal(400, "params must be an object", "invalid_request")
        ctx["upstream_session"] = None if method == "initialize" else self._upstream_session(request, ctx)
        record.model = method[:200] if isinstance(method, str) else None
        if method is None:
            return await self._client_answer(request, ctx, record, body)     # the agent answers the server
        if not isinstance(method, str):
            raise McpRefusal(400, "method must be a string", "invalid_request")
        if msg_id is None:
            if method not in _CLIENT_NOTIFICATIONS:
                raise McpRefusal(400, f"Notification {method[:80]} is not forwarded by the gateway", "invalid_request")
            return await self._forward_plain(request, ctx, record, body)
        if method == "initialize":
            return await self._initialize(request, ctx, record, body)
        if method == "tools/list":
            return await self._tools_list(request, ctx, record, body)
        if method == "tools/call":
            return await self._tools_call(request, ctx, record, body)
        if method in ("resources/read", "prompts/get", "tasks/result"):
            return await self._read_content(request, ctx, record, body)
        if method in _OUTGOING:
            why = await self._outgoing_blocked(ctx, record, body.get("params"), "mcp-params")
            if why:
                return self._reply(request, record, _rpc_error(msg_id, SERVER_ERROR, f"Refused by the gateway: {why}"), [])
            return await self._forward_answer(request, ctx, record, body)
        if method in _FORWARDED:
            if len(json.dumps(body.get("params") or {}, default=str)) > MAX_PLAIN_PARAMS:
                raise McpRefusal(400, f"params of {method} are too large", "invalid_request")
            return await self._forward_answer(request, ctx, record, body)
        return self._reply(request, record, _rpc_error(msg_id, METHOD_NOT_FOUND,
                                                       f"Method {method[:80]} is not allowed through the gateway"), [])

    def _reply(self, request: Request, record, answer: Optional[Dict[str, Any]], relay: List[Dict[str, Any]],
               extra_headers: Optional[Dict[str, str]] = None) -> Response:
        headers = {**self.gw.trace_headers(record), **(extra_headers or {})}
        record.status = 200
        if record.outcome in ("", None):
            record.outcome = "ok"
        self.gw._record(record)
        if answer is None:
            return Response(status_code=202, headers=headers)
        accepts_sse = "text/event-stream" in (request.headers.get("accept") or "")
        if relay and accepts_sse:
            content = b"".join(_sse(m) for m in relay) + _sse(answer)
            return Response(content, media_type="text/event-stream", headers=headers)
        return JSONResponse(answer, headers=headers)

    async def _forward_plain(self, request: Request, ctx, record, body: Dict[str, Any]) -> Response:
        response = await self._send(request, ctx, body)
        try:
            status = response.status_code
        finally:
            await response.aclose()
        record.status, record.outcome = status, "ok" if status < 400 else "error"
        self.gw._record(record)
        return Response(status_code=202 if status < 400 else status, headers=self.gw.trace_headers(record))

    async def _client_answer(self, request: Request, ctx, record, body: Dict[str, Any]) -> Response:
        """The agent's answer to a question the server asked (elicitation, roots): checked for
        content the policy blocks before it goes."""
        if "result" not in body and "error" not in body:
            raise McpRefusal(400, "Not a JSON-RPC request, notification or answer", "invalid_request")
        why = await self._outgoing_blocked(ctx, record, body.get("result") or body.get("error"), "mcp-answer")
        if why:
            raise McpRefusal(403, f"Refused by the gateway: {why}", "blocked_input")
        return await self._forward_plain(request, ctx, record, body)

    async def _forward_answer(self, request: Request, ctx, record, body: Dict[str, Any]) -> Response:
        relay: List[Dict[str, Any]] = []
        answer = await self._answer_of(await self._send(request, ctx, body), body.get("id"), relay,
                                       lambda m: self._keep_relayed(ctx, record, m))
        if isinstance(answer, dict):
            answer = await self._checked_data(ctx, record, answer, "mcp")
        return self._reply(request, record, answer, relay)

    async def _initialize(self, request: Request, ctx, record, body: Dict[str, Any]) -> Response:
        params = dict(body.get("params") or {})
        capabilities = dict(params.get("capabilities") or {}) if isinstance(params.get("capabilities"), dict) else {}
        mcp_policy = ctx["policy"].get("mcp") if isinstance(ctx["policy"].get("mcp"), dict) else {}
        if "sampling" in capabilities and mcp_policy.get("allow_sampling") is not True:
            capabilities.pop("sampling")
            record.note("mcp_sampling_removed")
        message = {**body, "params": {**params, "capabilities": capabilities}}
        if len(json.dumps(message, default=str)) > MAX_PLAIN_PARAMS * 4:
            raise McpRefusal(400, "initialize parameters are too large", "invalid_request")
        response = await self._send(request, ctx, message)
        upstream = response.headers.get(SESSION_HEADER) or ""
        relay: List[Dict[str, Any]] = []
        answer = await self._answer_of(response, body.get("id"), relay, lambda m: self._keep_relayed(ctx, record, m))
        if isinstance(answer, dict):
            answer = await self._checked_data(ctx, record, answer, "mcp-initialize")
        extra = {SESSION_HEADER: self._new_session(ctx, upstream)}
        return self._reply(request, record, answer, relay, extra)

    # -- tools ----------------------------------------------------------------------

    def _pin_key(self, ctx, tool: str) -> str:
        server = ctx["server"]
        material = json.dumps([server.member or "-", server.name, tool])
        return "pin:gw:mcp-tool:" + hashlib.sha256(material.encode("utf-8", "surrogatepass")).hexdigest()

    def _console_pins(self) -> Dict[Tuple[str, str, str], str]:
        raw = self.gw._controls().get("mcp_pins")
        memo = self._pins_memo
        if memo is not None and memo[0] is raw:
            return memo[1]
        pins: Dict[Tuple[str, str, str], str] = {}
        for entry in raw if isinstance(raw, list) else []:
            if isinstance(entry, dict) and all(isinstance(entry.get(k), str) for k in ("server", "tool", "sha256")):
                pins[(str(entry.get("project_id") or ""), entry["server"], entry["tool"])] = entry["sha256"]
        self._pins_memo = (raw, pins)
        return pins

    def _accepted(self, ctx, tool: str, digest: str) -> bool:
        server = ctx["server"]
        if (tool, digest) in server.accept:
            return True
        for entry in self.gw._controls().get("mcp_accept") or []:
            if (isinstance(entry, dict) and entry.get("server") == server.name and entry.get("tool") == tool
                    and entry.get("sha256") == digest and entry.get("project_id") in (None, "", ctx["project"])):
                return True
        return False

    def _pin_state(self, ctx, tool: Dict[str, Any]) -> Tuple[str, str]:
        """``("ok" | "changed", hash)``. With a console, its pin (first sight anywhere, kept
        durably) is the reference; else the first sight in the shared state. A different
        definition is held unless that change was accepted."""
        from znyx_core.state_store import get_state_store
        name = str(tool.get("name") or "")
        digest = tool_hash(tool)
        console = self._console_pins().get((ctx["project"], ctx["server"].name, name))
        if console is not None:
            return ("ok" if console == digest or self._accepted(ctx, name, digest) else "changed"), digest
        store = get_state_store()
        key = self._pin_key(ctx, name)
        if store.set_value(key, digest, PIN_TTL, only_if_absent=True):
            return "ok", digest
        pinned = store.get_value(key)
        if pinned == digest:
            return "ok", digest
        if self._accepted(ctx, name, digest):
            store.set_value(key, digest, PIN_TTL)
            return "ok", digest
        return "changed", digest

    def _scanner(self, ctx):
        from znyx_core.detectors.mcp_manifest_scanner import McpManifestScannerDetector
        config = ctx["policy"].get("mcp_manifest_scanner")
        config = dict(config) if isinstance(config, dict) else {"action": "WARN"}
        return McpManifestScannerDetector({**config, "enabled": True})

    def _allowed(self, ctx, name: str) -> bool:
        stops = ctx["stops"]
        if stops.quarantined is not None or stops.tool_stop(name) is not None:
            return False
        delegation = ctx.get("delegation")
        if delegation is not None and not identity.tool_granted(delegation.tools, name):
            return False
        tools_policy = ctx["policy"].get("tools") if isinstance(ctx["policy"].get("tools"), dict) else {}
        if tools_policy.get("enabled", True):
            allowed = set(tools_policy.get("allowed") or tools_policy.get("allowed_tools") or [])
            denied = set(tools_policy.get("denied") or tools_policy.get("blocked_tools") or [])
            if (allowed and name not in allowed) or name in denied:
                return False
        return True

    def _visible(self, ctx, tool: Dict[str, Any], record, changed: List[str], scanner) -> bool:
        """Whether the agent may see (and call) this tool. Only allowed tools are pinned."""
        name = tool.get("name")
        if not isinstance(name, str) or not name or len(name) > 200 or not self._allowed(ctx, name):
            return False
        state, digest = self._pin_state(ctx, tool)
        if state == "changed":
            changed.append(f"{name}@{digest}")
            return False
        result = scanner.detect({"tools": [tool]})
        record.rule_ids.extend(h.rule_id for h in result.rule_hits)
        return result.decision.value != "BLOCK"

    async def _tools_list(self, request: Request, ctx, record, body: Dict[str, Any]) -> Response:
        relay: List[Dict[str, Any]] = []
        answer = await self._answer_of(await self._send(request, ctx, body), body.get("id"), relay,
                                       lambda m: self._keep_relayed(ctx, record, m))
        if isinstance(answer, dict) and "error" in answer:
            answer = await self._checked_error(ctx, record, answer, "mcp")
        elif isinstance(answer, dict) and isinstance(answer.get("result"), dict) \
                and isinstance(answer["result"].get("tools"), list):
            listed = [t for t in answer["result"]["tools"] if isinstance(t, dict)]
            if len(listed) > MAX_TOOLS_PER_LIST:
                record.note("mcp_tools_over_limit", len(listed) - MAX_TOOLS_PER_LIST)
            tools = listed[:MAX_TOOLS_PER_LIST]
            self._remember(ctx, tools, replace=not (body.get("params") or {}).get("cursor"))
            changed: List[str] = []
            scanner = self._scanner(ctx)
            kept = [t for t in tools if self._visible(ctx, t, record, changed, scanner)]
            if len(kept) != len(listed):
                record.note("mcp_tools_hidden", len(listed) - len(kept))
            answer = {**answer, "result": {**answer["result"], "tools": kept}}
            record.tools_offered = [str(t["name"])[:200] for t in kept][:50]
            record.mcp_tools_changed = changed[:50]
            record.mcp_tools_seen = [f"{str(t.get('name'))[:120]}@{tool_hash(t)}" for t in tools
                                     if isinstance(t.get("name"), str) and self._allowed(ctx, t["name"])][:50]
        return self._reply(request, record, answer, relay)

    def _remember(self, ctx, tools: List[Dict[str, Any]], replace: bool) -> None:
        key = f"{ctx['server'].member}/{ctx['server'].name}"
        known = {} if replace or key not in self._defs else dict(self._defs[key][1])
        known.update({str(t.get("name")): t for t in tools if t.get("name")})
        self._defs[key] = (time.monotonic(), dict(list(known.items())[:MAX_TOOLS_PER_LIST * MAX_LIST_PAGES]))

    async def _definition(self, request: Request, ctx, name: str) -> Optional[Dict[str, Any]]:
        """The tool's current definition: from the last list, else asked of the server."""
        key = f"{ctx['server'].member}/{ctx['server'].name}"
        cached = self._defs.get(key)
        if cached is not None and time.monotonic() - cached[0] < DEFS_TTL and name in cached[1]:
            return cached[1][name]
        cursor, found = None, []
        for _page in range(MAX_LIST_PAGES):
            message = {"jsonrpc": "2.0", "id": f"znyx-list-{uuid.uuid4().hex}", "method": "tools/list"}
            if cursor:
                message["params"] = {"cursor": cursor}
            answer = await self._answer_of(await self._send(request, ctx, message), message["id"], [])
            result = (answer or {}).get("result") if isinstance(answer, dict) else None
            if not isinstance(result, dict):
                break
            found += [t for t in result.get("tools") or [] if isinstance(t, dict)]
            cursor = result.get("nextCursor")
            if not cursor or len(found) >= MAX_TOOLS_PER_LIST * MAX_LIST_PAGES:
                break
        self._remember(ctx, found, replace=True)
        return next((t for t in found if t.get("name") == name), None)

    async def _tools_call(self, request: Request, ctx, record, body: Dict[str, Any]) -> Response:
        from znyx_runtime.gateway.pipeline import quiet_policy
        msg_id = body.get("id")
        params = dict(body.get("params") or {})
        name = params.get("name")
        if isinstance(name, str):
            name = _LONE_SURROGATE.sub("\ufffd", name)        # never a real tool; kept printable for the record
        arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
        if not isinstance(name, str) or not name or len(name) > 200:
            return self._reply(request, record, _rpc_error(msg_id, INVALID_REQUEST, "tools/call needs a tool name"), [])
        record.saw_tool_call(name)
        stops, policy, scope = ctx["stops"], ctx["policy"], ctx["scope"]

        def refuse(message: str, reason: str, outcome: str = "denied") -> Response:
            record.outcome, record.reason = outcome, reason
            return self._reply(request, record, _tool_refusal(msg_id, f"Refused by the gateway: {message}"), [])

        hit = stops.quarantined or stops.tool_stop(name)
        if hit is not None:
            return refuse(hit.message(), "stopped")
        definition = await self._definition(request, ctx, name)
        if definition is None:
            return refuse(f"tool {name} is not offered by {ctx['server'].name}", "mcp_tool_unknown")
        changed: List[str] = []
        if not self._visible(ctx, definition, record, changed, self._scanner(ctx)):
            record.mcp_tools_changed = changed
            why = ("its definition changed since it was approved; an administrator must accept the change"
                   if changed else "this tool is not allowed here")
            return refuse(f"tool {name}: {why}", "mcp_tool_changed" if changed else "mcp_tool_not_allowed")
        # The arguments, checked like a tool call in a model's answer (the tools policy, and
        # content the policy blocks); the other parameters (``_meta``) for blocked content.
        text = json.dumps(arguments, ensure_ascii=False)
        verdict = await self.gw._check_text(text, "tool_call", "assistant", name, "output", scope, record,
                                            policy, quiet_policy(policy), "mcp-args", approve=False)
        record.rule_ids.extend(verdict.rule_ids)
        if verdict.decision == "BLOCK" or (verdict.wants_change and verdict.sanitized_tool_args is None):
            return refuse(verdict.user_message or "the arguments were blocked by the policy", "blocked_input",
                          "blocked_input")
        if verdict.sanitized_tool_args is not None:
            arguments = verdict.sanitized_tool_args
            record.note("mcp_arguments_rewritten")
        rest = {k: v for k, v in params.items() if k not in ("name", "arguments")}
        why = await self._outgoing_blocked(ctx, record, rest, "mcp-meta")
        if why:
            return refuse(why, "blocked_input", "blocked_input")
        rules = approvals.rules_of(policy)
        annotations = definition.get("annotations") if isinstance(definition.get("annotations"), dict) else {}
        reason = rules.reason(name, annotations) if rules is not None else None
        message = {**body, "params": {**params, "arguments": arguments}}
        if reason is not None:
            return await self._call_after_approval(request, ctx, record, message, rules, reason, name, arguments)
        return await self._call(request, ctx, record, message, name)

    async def _call_after_approval(self, request: Request, ctx, record, message: Dict[str, Any], rules,
                                   reason: str, name: str, arguments: Dict[str, Any]) -> Response:
        """Hold the call while a person decides. A client that reads streams gets keep-alive
        comments meanwhile; the call itself goes on only once approved, and only if no stop
        came in meanwhile."""
        ask = self.gw.approvals.ask(rules=rules, reason=reason, tool=name, arguments=arguments,
                                    server=ctx["server"].name, record=record, member=ctx["member"])

        async def decided(approved: bool, why: str) -> Response:
            if not approved:
                return self._approval_refused(request, record, message.get("id"), why)
            try:
                key, member, delegation = self.gw.current_caller(record)
                scope = self.gw._scope(key, request, member)
                policy = self.gw.bundle_manager.get_policy(
                    tenant_id=scope["tenant_id"], app_id=scope["app_id"], agent_id=scope["agent_id"],
                    env=scope["env"], member=member)
                server = self.gw.config.mcp_server_for(key, ctx["server"].name)
                if server.blocked:
                    raise McpRefusal(403, server.blocked, "mcp_server_not_allowed_here")
                ctx.update(key=key, member=member, delegation=delegation, scope=scope, policy=policy, server=server)
                self.gw.apply_delegation(record, delegation)
                stops = self._stops_or_refuse(ctx)
                ctx["stops"] = record._stops = stops
            except McpRefusal as refusal:
                return self._approval_refused(request, record, message.get("id"), f"stopped meanwhile: {refusal.message}")
            except Exception:
                return self._approval_refused(request, record, message.get("id"), "the caller or server is no longer authorized")
            hit = stops.quarantined or stops.tool_stop(name)
            if hit is not None:
                return self._approval_refused(request, record, message.get("id"), f"stopped meanwhile: {hit.message()}")
            if not self._allowed(ctx, name):
                return self._approval_refused(request, record, message.get("id"), "this tool is no longer allowed")
            from znyx_runtime.gateway.pipeline import quiet_policy
            verdict = await self.gw._check_text(json.dumps(arguments), "tool_call", "assistant", name,
                                                "output", scope, record, quiet_policy(policy), quiet_policy(policy),
                                                "mcp-approved", approve=False)
            if (verdict.decision == "BLOCK" or verdict.wants_change
                    or verdict.sanitized_tool_args not in (None, arguments)):
                return self._approval_refused(request, record, message.get("id"), "the current policy refuses these arguments")
            return await self._call(request, ctx, record, message, name)

        if "text/event-stream" not in (request.headers.get("accept") or ""):
            return await decided(*(await ask))

        async def stream() -> AsyncIterator[bytes]:
            task = asyncio.ensure_future(ask)
            try:
                while not task.done():
                    done, _ = await asyncio.wait({task}, timeout=KEEPALIVE_SECONDS)
                    if not done:
                        yield b": waiting for approval\n\n"
                response = await decided(*task.result())
                yield response.body if response.media_type == "text/event-stream" else _sse(json.loads(response.body))
            finally:
                if not task.done():
                    task.cancel()
                self.gw._record(record)

        return StreamingResponse(stream(), media_type="text/event-stream", headers=self.gw.trace_headers(record))

    def _approval_refused(self, request: Request, record, msg_id: Any, why: str) -> Response:
        record.outcome, record.reason = "denied", "not_approved"
        record.rule_ids.append(approvals.RULE_NOT_APPROVED)
        return self._reply(request, record, _tool_refusal(msg_id, f"Refused by the gateway: not approved ({why})"),
                           [])

    async def _call(self, request: Request, ctx, record, message: Dict[str, Any], name: str) -> Response:
        relay: List[Dict[str, Any]] = []
        answer = await self._answer_of(await self._send(request, ctx, message), message.get("id"), relay,
                                       lambda m: self._keep_relayed(ctx, record, m))
        if isinstance(answer, dict):
            answer = await self._checked_result(ctx, record, answer, name)
        return self._reply(request, record, answer, relay)

    # -- resources, prompts, task results -------------------------------------------

    async def _read_content(self, request: Request, ctx, record, body: Dict[str, Any]) -> Response:
        msg_id = body.get("id")
        method = body["method"]
        params = body.get("params") or {}
        why = await self._outgoing_blocked(ctx, record, params, "mcp-params")
        if why:
            return self._reply(request, record, _rpc_error(msg_id, SERVER_ERROR, f"Refused by the gateway: {why}"), [])
        relay: List[Dict[str, Any]] = []
        answer = await self._answer_of(await self._send(request, ctx, body), msg_id, relay,
                                       lambda m: self._keep_relayed(ctx, record, m))
        if not isinstance(answer, dict):
            return self._reply(request, record, answer, relay)
        label = str(params.get("name") or params.get("uri") or params.get("taskId") or method)[:200]
        if method == "tasks/result":
            answer = await self._checked_result(ctx, record, answer, label)
            return self._reply(request, record, answer, relay)
        if "error" in answer:
            return self._reply(request, record, await self._checked_error(ctx, record, answer, label), relay)
        result = answer.get("result") if isinstance(answer.get("result"), dict) else {}
        if method == "resources/read":
            contents = [{"type": "resource", "resource": c} for c in result.get("contents") or [] if isinstance(c, dict)]
            items, blocked, changed = await self._checked_items(ctx, record, contents, label, "mcp-content")
            if blocked:
                return self._reply(request, record, self._blocked_answer(record, msg_id, as_error=True), relay)
            result = {**result, "contents": [i["resource"] for i in items]}
        else:                                                       # prompts/get
            original = [m for m in result.get("messages") or [] if isinstance(m, dict)]
            items, blocked, changed = await self._checked_items(
                ctx, record, [m.get("content") for m in original], label, "mcp-prompt")
            # A rewrite spanning messages cannot safely be assigned to their roles.
            if blocked or len(items) != len(original):
                return self._reply(request, record, self._blocked_answer(record, msg_id, as_error=True), relay)
            messages = [{**m, "content": item} for m, item in zip(original, items)]
            if result.get("description"):
                verdict = await self._check(ctx, record, str(result["description"]), "mcp-prompt-description", label)
                if verdict.decision == "BLOCK":
                    return self._reply(request, record, self._blocked_answer(record, msg_id, as_error=True), relay)
                if verdict.sanitized_text is not None:
                    result = {**result, "description": verdict.sanitized_text}
            result = {**result, "messages": messages}
        return self._reply(request, record, {**answer, "result": result}, relay)
