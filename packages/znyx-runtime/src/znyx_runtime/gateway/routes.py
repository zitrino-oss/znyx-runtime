"""Gateway routes: the provider APIs, on the instance's one port.

    POST /v1/chat/completions                                  OpenAI-compatible
    POST /openai/deployments/{deployment}/chat/completions     Azure OpenAI
    POST /v1/messages                                          Anthropic
    GET  /v1/models                                            what this key may use

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
