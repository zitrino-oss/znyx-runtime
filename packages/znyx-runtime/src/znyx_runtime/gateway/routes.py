"""Gateway routes: the provider APIs, on the instance's one port.

    POST /v1/chat/completions                                  OpenAI-compatible
    POST /openai/deployments/{deployment}/chat/completions     Azure OpenAI
    POST /v1/messages                                          Anthropic
    GET  /v1/models                                            what this key may use
    GET  /v1/znyx/batches/{id}                                 a low-urgency call's answer
    POST /v1/mcp/{server}                                      MCP (Streamable HTTP), brokered
    GET  /v1/znyx/approvals[/{id}], POST .../{id}/decide       approval requests (approver key)

An application switches by changing its SDK's base URL to this instance (and its
API key to a gateway key): ``OpenAI(base_url="http://znyx:8080/v1")``,
``AzureOpenAI(azure_endpoint="http://znyx:8080")``, ``Anthropic(base_url="http://znyx:8080")``.
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

router = APIRouter(include_in_schema=False)


def _gateway(request: Request):
    return getattr(request.app.state, "gateway", None)


def _unavailable(api: str) -> Response:
    from znyx_runtime.gateway.formats import FORMATS
    status, body = FORMATS[api].error(
        503, "The gateway is not configured on this instance: add providers in gateway.yaml, "
             "or set OPENAI_API_KEY / ANTHROPIC_API_KEY", code="gateway_not_configured",
        error_type="api_error")
    return JSONResponse(body, status_code=status)


@router.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    gw = _gateway(request)
    if gw is None or not gw.config.enabled:
        return _unavailable("openai")
    return await gw.handle(request, "openai")


@router.post("/openai/deployments/{deployment}/chat/completions")
async def azure_chat_completions(deployment: str, request: Request) -> Response:
    gw = _gateway(request)
    if gw is None or not gw.config.enabled:
        return _unavailable("azure")
    return await gw.handle(request, "azure", deployment=deployment)


@router.post("/v1/messages")
async def messages(request: Request) -> Response:
    gw = _gateway(request)
    if gw is None or not gw.config.enabled:
        return _unavailable("anthropic")
    return await gw.handle(request, "anthropic")


@router.get("/v1/znyx/batches/{job_id}")
async def batch_job(job_id: str, request: Request) -> Response:
    """A queued low-urgency call (``X-Znyx-Urgency: low``): its state, and the checked
    answer once the provider's batch returned it."""
    gw = _gateway(request)
    if gw is None or not gw.config.enabled:
        return _unavailable("openai")
    return await gw.batch_status(request, job_id)


@router.get("/v1/models")
async def list_models(request: Request) -> Response:
    """The models this key is approved for (OpenAI list shape). With no allowlist the
    gateway cannot know the provider's catalogue, so the list is empty."""
    from znyx_runtime.gateway.formats import OpenAIChat
    from znyx_runtime.gateway.pipeline import CallRecord, Denied
    gw = _gateway(request)
    if gw is None or not gw.config.enabled:
        return _unavailable("openai")
    try:
        key, _ = gw._identify(request)
    except Denied as denied:
        record = CallRecord(request_id="models", key_id="", tenant_id="", app_id="", agent_id="",
                            api="openai")
        return gw._finish(record, OpenAIChat, denied)
    created = int(time.time())
    data = [{"id": m, "object": "model", "created": created, "owned_by": "znyx-gateway"}
            for m in key.allowed_models if not any(c in m for c in "*?[")]
    return JSONResponse({"object": "list", "data": data})


@router.get("/v1/znyx/channels")
async def channels(request: Request) -> Response:
    """Capabilities for this key; configured extraction is distinct from service health."""
    import os
    from znyx_runtime.gateway.pipeline import Denied
    from znyx_runtime.gateway.channels import CHANNELS
    from znyx_runtime.gateway.defence import active
    gw = _gateway(request)
    if gw is None or not gw.config.enabled:
        return _unavailable("openai")
    try:
        key, _ = gw._identify(request)
    except Denied as exc:
        return JSONResponse({"error": {"code": exc.code, "message": exc.message}}, status_code=exc.status)
    providers = [provider for name, provider in gw.config._providers_of(key).items()
                 if key.may_use(name) and not provider.blocked]
    openai = any(provider.format in ("openai", "azure_openai") for provider in providers)
    return JSONResponse({"http_channels": (["chat/completions", *CHANNELS] if openai else []) +
        (["messages"] if any(p.format == "anthropic" for p in providers) else []),
        "realtime": "websocket-buffered-turns" if any(p.format == "openai" for p in providers) else None,
        "responses_streaming": "buffered" if openai else None,
        "media_inspection": gw.config.settings.media_inspection,
        "media_extractor_configured": bool(os.getenv("ZNYX_MEDIA_EXTRACT_URL") or os.getenv("ZNYX_INFERENCE_URL")),
        "shared_defence": active(gw._controls().get("shared_defence") or {})})


@router.api_route("/v1/mcp/{server}", methods=["POST", "GET", "DELETE"])
async def mcp(server: str, request: Request) -> Response:
    """An MCP server, brokered: agents connect here with their gateway key (see
    ``gateway.mcp``)."""
    gw = _gateway(request)
    if gw is None:
        return JSONResponse({"jsonrpc": "2.0", "id": None,
                             "error": {"code": -32000, "message": "The gateway is not configured on this instance"}},
                            status_code=503)
    return await gw.mcp.handle(request, server)


def _approver_only(request: Request):
    """The approval routes are for the people deciding, with the approver key
    (``ZNYX_APPROVER_KEY``): never a gateway or instance key, which agents hold."""
    import hmac
    import os
    gw = _gateway(request)
    if gw is None:
        return None, JSONResponse({"error": "gateway not configured"}, status_code=503)
    expected = os.getenv("ZNYX_APPROVER_KEY") or ""
    if len(expected) < 16:
        return None, JSONResponse({"error": "Set ZNYX_APPROVER_KEY (at least 16 characters) to view and decide "
                                            "approval requests on this runtime"}, status_code=403)
    given = request.headers.get("x-znyx-approver-key") or ""
    if not hmac.compare_digest(given.encode(), expected.encode()):
        return None, JSONResponse({"error": "Approval requests are read and decided with the approver key"},
                                  status_code=403)
    return gw, None


@router.get("/v1/znyx/approvals")
async def approvals_list(request: Request, status: str = "") -> Response:
    """Approval requests this runtime asked (with their full arguments: inside the network)."""
    import asyncio
    gw, refused = _approver_only(request)
    if refused is not None:
        return refused
    entries = await asyncio.to_thread(gw.approvals.local_list, status or None)
    return JSONResponse({"approvals": entries})


@router.get("/v1/znyx/defence/contributions")
async def defence_contributions(request: Request) -> Response:
    import asyncio
    gw, refused = _approver_only(request)
    if refused is not None:
        return refused
    try:
        cursor = int(request.query_params.get("cursor", "0"))
        if not 0 <= cursor < 2**63:
            raise ValueError()
        page = await asyncio.to_thread(gw.defence.inspect_page, cursor)
    except ValueError:
        return JSONResponse({"error": "cursor must be a valid record byte offset"}, status_code=400)
    return JSONResponse(page)


@router.get("/v1/znyx/approvals/{approval_id}")
async def approval_get(approval_id: str, request: Request) -> Response:
    import asyncio
    gw, refused = _approver_only(request)
    if refused is not None:
        return refused
    entry = await asyncio.to_thread(gw.approvals.local_get, approval_id)
    if entry is None:
        return JSONResponse({"error": "No such approval request"}, status_code=404)
    return JSONResponse(entry)


@router.post("/v1/znyx/approvals/{approval_id}/decide")
async def approval_decide(approval_id: str, request: Request) -> Response:
    """Decide a request (no console): ``{"decision": "approve" | "deny", "note"}``."""
    import asyncio
    gw, refused = _approver_only(request)
    if refused is not None:
        return refused
    try:
        body = await request.json()
    except ValueError:
        body = None
    decision = body.get("decision") if isinstance(body, dict) else None
    if decision not in ("approve", "deny"):
        return JSONResponse({"error": "decision must be approve or deny"}, status_code=422)
    note = body.get("note") if isinstance(body.get("note"), str) else None
    try:
        entry = await asyncio.to_thread(gw.approvals.local_decide, approval_id, decision, "approver key", note)
    except LookupError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except PermissionError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    return JSONResponse(entry)



@router.post("/v1/znyx/token")
async def delegated_token(request: Request) -> Response:
    """Token exchange (part 3): the agent's gateway key and a person's identity-provider token
    (``subject_token``) give a short-lived token that acts for that person."""
    from znyx_runtime.bundle_manager import ScopeNotServed
    from znyx_runtime.gateway.identity import TokenRefused
    from znyx_runtime.gateway.pipeline import Denied
    gw = _gateway(request)
    no_store = {"cache-control": "no-store", "pragma": "no-cache"}
    if gw is None:
        return JSONResponse({"error": "temporarily_unavailable", "error_description": "gateway not configured"},
                            status_code=503, headers=no_store)
    try:
        key, _ = gw._identify(request)
        if key.instance_key or getattr(request.state, "znyx_delegation", None) is not None:
            raise TokenRefused(401, "invalid_client", "authenticate with the agent's gateway key")
        gw.mcp._rate_limit(key)
        member = gw.bundle_manager.member_by_key(key.member or None)
        kind = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
        if kind == "application/x-www-form-urlencoded":
            body = dict(await request.form())
        else:
            try:
                body = await request.json()
            except ValueError:
                body = None
        if not isinstance(body, dict):
            raise TokenRefused(400, "invalid_request", "send a JSON object or a form")
        out = gw.delegations.exchange(key, member, body)
    except TokenRefused as refused:
        return JSONResponse({"error": refused.error, "error_description": refused.description},
                            status_code=refused.status,
                            headers={**no_store, **({"retry-after": "60"} if refused.status == 429 else {})})
    except Denied as denied:
        error = "invalid_client" if denied.status == 401 else "access_denied"
        return JSONResponse({"error": error, "error_description": denied.message}, status_code=denied.status,
                            headers=no_store)
    except ScopeNotServed as exc:
        return JSONResponse({"error": "access_denied", "error_description": str(exc)}, status_code=403,
                            headers=no_store)
    return JSONResponse(out, headers=no_store)


@router.delete("/v1/znyx/token")
async def revoke_token(request: Request) -> Response:
    """Revoke the delegated token sent as the bearer token."""
    gw = _gateway(request)
    auth = request.headers.get("authorization") or ""
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    if gw is None or not token:
        return JSONResponse({"error": "invalid_request"}, status_code=400)
    revoked = gw.delegations.revoke(token)
    return JSONResponse({"revoked": revoked}, status_code=200 if revoked else 404)
